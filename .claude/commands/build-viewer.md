# Build Three.js Robot/Device Viewer from Blender Scene

Build a config JSON and export a GLB for the interactive Three.js FK/IK viewer from an armature in the current Blender scene. The viewer (`threejs_scene.html`) is generic and config-driven — **do NOT modify it**. All you need to produce is a config JSON and a GLB file.

User arguments (optional): $ARGUMENTS
- If the user specifies an armature name, use that. Otherwise auto-detect the first armature in the scene.
- If the user specifies a config filename, use that. Otherwise default to `meca500_config.json`.
- If the user specifies a GLB filename, use that. Otherwise derive from the config name.

## Step 1: Inspect the Blender Scene

Use `mcp__blender__get_scene_info` to get the scene overview. Identify:
- The armature object (type: ARMATURE)
- All mesh objects and their names
- Any objects to exclude (cameras, lights, curves, empties)

## Step 2: Extract Armature Bone Data

Use `mcp__blender__execute_blender_code` to extract bone transforms. **CRITICAL: Use the correct C matrix** — `C_gltf = [[1,0,0],[0,0,1],[0,-1,0]]`. The inverse matrix `[[1,0,0],[0,0,-1],[0,1,0]]` produces wrong transforms.

```python
import bpy, json, mathutils

arm_obj = bpy.data.objects['ARMATURE_NAME']  # Replace with actual name
arm = arm_obj.data

# CORRECT coordinate conversion: Blender Z-up to glTF/Three.js Y-up
# Maps: Blender X→X, Blender Y→-Z, Blender Z→Y
C = mathutils.Matrix((
    (1, 0, 0, 0),
    (0, 0, 1, 0),
    (0, -1, 0, 0),
    (0, 0, 0, 1),
))
C_inv = C.inverted()

bones_data = []
for bone in arm.bones:
    if bone.parent:
        local_mat = C @ bone.parent.matrix_local.inverted() @ bone.matrix_local @ C_inv
    else:
        local_mat = C @ bone.matrix_local @ C_inv

    pos = local_mat.to_translation()
    quat = local_mat.to_quaternion()  # Blender quats are (w,x,y,z)

    bones_data.append({
        'name': bone.name,
        'parent': bone.parent.name if bone.parent else None,
        'pos': [round(pos.x, 5), round(pos.y, 5), round(pos.z, 5)],
        'quat_wxyz': [round(quat.w, 5), round(quat.x, 5), round(quat.y, 5), round(quat.z, 5)],
    })

print(json.dumps(bones_data, indent=2))
```

## Step 3: Determine Joint Rotation Axes and Fixed Joints

Use `mcp__blender__execute_blender_code` to read quaternion locks. In Blender QUATERNION rotation mode, `lock_rotation_w` and `lock_rotation[0:2]` correspond to W, X, Y, Z components. The **free** (unlocked) component determines the rotation axis.

```python
import bpy

arm_obj = bpy.data.objects['ARMATURE_NAME']
pose = arm_obj.pose

for pb in pose.bones:
    lock_w = pb.lock_rotation_w
    lock_xyz = list(pb.lock_rotation)
    free = []
    if not lock_w: free.append('W')
    if not lock_xyz[0]: free.append('X')
    if not lock_xyz[1]: free.append('Y')
    if not lock_xyz[2]: free.append('Z')
    print(f"{pb.name}: locks W={lock_w} X={lock_xyz[0]} Y={lock_xyz[1]} Z={lock_xyz[2]}  free={free}")
```

### Interpreting the results:

**Fixed joints:** All WXYZ locked (free=[]) → set `"fixed": true` in config. These are kinematic links that don't rotate.

**Movable joints:** Exactly one component free (plus W which is always free for unit quaternions). The free component after C_gltf conversion gives the Three.js rotation axis:

| Blender free component | Three.js axis |
|----------------------|---------------|
| X free (W,X free)    | `[1, 0, 0]`  |
| Y free (W,Y free)    | `[0, 0, -1]` |
| Z free (W,Z free)    | `[0, 1, 0]`  |

**Why Y maps to [0,0,-1]:** C_gltf @ R_y(θ) @ C_gltf_inv = R_z(-θ), so Blender Y rotation becomes negative Z in Three.js.

**All components free:** This is a kinematic link (all DOF unlocked = no constraint), treat as `"fixed": true`.

If the bone uses a different rotation mode or constraints don't reveal axes, ask the user.

## Step 4: Determine Mesh-to-Bone Parenting

Use `mcp__blender__execute_blender_code` to find which meshes are parented to which bones, including mesh-to-mesh parenting chains:

```python
import bpy, json

arm_obj = bpy.data.objects['ARMATURE_NAME']

# Direct bone parenting
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        parent_name = obj.parent.name if obj.parent else None
        parent_type = obj.parent_type if obj.parent else None
        parent_bone = obj.parent_bone if obj.parent_bone else None
        print(f"{obj.name}: parent={parent_name}, type={parent_type}, bone={parent_bone}")
```

**Important:** Meshes can be parented to bones (direct) or to other meshes (indirect). For the config `links` array:
- Bone-parented meshes → map to the joint index of that bone
- Mesh-parented meshes (e.g., nozzle→detector_arm→delta) → map to the same joint as their ultimate bone-parented ancestor
- Unparented meshes → these are static scene objects, don't include in `links`

**Name sanitization:** Three.js GLTFLoader sanitizes node names: spaces → underscores, dots removed. Config link names MUST use sanitized versions (e.g., `"detector_arm_back"` not `"detector arm back"`).

## Step 5: Determine Joint Limits

Check if Blender bones have rotation limits set:

```python
import bpy

arm_obj = bpy.data.objects['ARMATURE_NAME']
for pb in arm_obj.pose.bones:
    if hasattr(pb, 'use_ik_limit_x') and pb.use_ik_limit_x:
        print(f"{pb.name}: IK X limits [{pb.ik_min_x:.3f}, {pb.ik_max_x:.3f}] rad")
    if hasattr(pb, 'use_ik_limit_y') and pb.use_ik_limit_y:
        print(f"{pb.name}: IK Y limits [{pb.ik_min_y:.3f}, {pb.ik_max_y:.3f}] rad")
    if hasattr(pb, 'use_ik_limit_z') and pb.use_ik_limit_z:
        print(f"{pb.name}: IK Z limits [{pb.ik_min_z:.3f}, {pb.ik_max_z:.3f}] rad")
```

If no limits are set, use defaults: `[-180, 180]` for most joints, `[-360, 360]` for continuous rotation joints. Fixed joints use `[0, 0]`.

## Step 6: Export GLB from Blender (Skin-Free)

**CRITICAL:** Bone-parented meshes cause Blender to add a `skins` array to the GLB. Three.js GLTFLoader handles skeleton children differently, making nested mesh nodes inaccessible via `model.traverse()`. **Always export with `export_skins=False` after temporarily unparenting bone-parented meshes.**

```python
import bpy, os

arm_obj = bpy.data.objects['ARMATURE_NAME']
output_path = 'OUTPUT_PATH'  # e.g., '/path/to/robot_scene.glb'

# Hide render-hidden meshes from viewport temporarily (auxiliary objects like samples, detectors, debug spheres)
visibility_backup = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.hide_render and not obj.hide_viewport:
        visibility_backup.append(obj)
        obj.hide_viewport = True

# Backup and unparent bone-parented meshes to avoid skins in GLB
parenting_backup = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.parent == arm_obj and obj.parent_bone:
        parenting_backup.append({
            'obj': obj,
            'parent_bone': obj.parent_bone,
            'parent_type': obj.parent_type,
            'matrix_world': obj.matrix_world.copy()
        })
        mat = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mat

# Export
bpy.ops.export_scene.gltf(
    filepath=output_path,
    export_format='GLB',
    export_apply=True,
    use_visible=True,
    export_yup=True,
    export_animations=False,
    export_skins=False,
)

# Restore parenting
for info in parenting_backup:
    obj = info['obj']
    obj.parent = arm_obj
    obj.parent_bone = info['parent_bone']
    obj.parent_type = info['parent_type']
    obj.matrix_world = info['matrix_world']

# Restore visibility
for obj in visibility_backup:
    obj.hide_viewport = False

print(f"Exported to {output_path}")
print(f"Unparented/restored {len(parenting_backup)} bone-parented meshes")
print(f"Hidden/restored {len(visibility_backup)} render-hidden meshes")
```

### Browser cache busting:
If re-exporting an updated model, change the GLB filename (e.g., `model_v2.glb`) to bypass browser cache. Update the config's `"model"` field to match.

## Step 7: Generate the Config JSON

Create a config JSON file. The `threejs_scene.html` viewer loads this config dynamically. **Do NOT modify threejs_scene.html.**

### Config structure:
```json
{
  "name": "Device Name",
  "model": "device_scene.glb",
  "joints": [
    {
      "name": "joint_name",
      "restPos": [x, y, z],
      "restQuat": [w, x, y, z],
      "axis": [x, y, z],
      "limits": [-180, 180],
      "parent": 0
    }
  ],
  "links": [
    { "name": "mesh_name", "label": "Display Name", "joint": 0 }
  ],
  "demoPose": [0, 45, -90]
}
```

### Joint fields:
- `restPos`, `restQuat`: from Step 2 (C_gltf converted). `restQuat` is [w,x,y,z]
- `axis`: from Step 3 (quaternion lock analysis)
- `limits`: from Step 5, in degrees. Fixed joints use `[0, 0]`
- `parent`: index of parent joint in this array, or `-1` for root joints
- `fixed` (optional): `true` for non-movable kinematic links

### Branching kinematic chains:
For devices with multiple independent chains (e.g., i16 diffractometer has gamma and mu chains):
- Each chain starts from a root joint with `"parent": -1`
- If chains share a physical root bone, create separate root joint entries (one per chain) with the same transform
- Sub-branches (e.g., merlin detector branching from stokes) use the `parent` field to point to the branch point joint

### Link fields:
- `name`: sanitized mesh name (spaces→underscores, no dots) matching what Three.js GLTFLoader produces
- `label`: human-readable display name
- `joint`: index into the `joints` array that this mesh moves with

### demoPose:
Array of joint angles in degrees, one per joint. Fixed joints should be `0`. Choose angles that show the device in an interesting non-home pose.

## Step 8: Add Config to Model Selector

The viewer has a model selector dropdown. To make the new config available, add its filename to the `configFiles` array in `js/panel.js`. Search for `configFiles` in that file and append the new config filename to the array.

## Step 9: Start HTTP Server and Test

```bash
python3 -m http.server 8000
```

Tell the user to open `http://localhost:8000/threejs_scene.html?config=CONFIG_FILE.json` and verify:
1. All meshes load and appear correctly
2. FK sliders move the correct joints
3. Each joint rotates about the correct axis
4. Labels toggle shows correct mesh names
5. Demo pose looks reasonable

### Troubleshooting:
- **Meshes not found:** Check GLTFLoader name sanitization. Log `glTF nodes:` in browser console and match against config link names.
- **Wrong rotation direction:** Flip the axis sign (e.g., `[0,0,-1]` → `[0,0,1]`).
- **Meshes detached from kinematics:** Likely a skins/skeleton issue. Re-export with the skin-free method from Step 6.
- **Browser shows old model:** Change GLB filename for cache busting.
- **Dark/black materials:** Blender materials with complex node trees may export with near-black Base Color. Check the Principled BSDF Base Color values. If the actual colors are in `.001` variant materials or linked nodes, swap to the colored materials before export and restore after.
