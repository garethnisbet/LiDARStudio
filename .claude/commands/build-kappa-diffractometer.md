# Build Kappa Diffractometer Viewer from Blender Scene

Build a config JSON and export a GLB for a kappa diffractometer in the interactive Three.js FK/IK viewer. The viewer (`threejs_scene.html`) auto-detects kappa geometry when joints named `theta`, `kappa`, and `phi` are present, and automatically provides virtual chi/theta/phi angle control with compensation.

**Do NOT modify `threejs_scene.html`** — all you need to produce is a config JSON and a GLB file.

User arguments (optional): $ARGUMENTS
- If the user specifies an armature name, use that. Otherwise auto-detect.
- If the user specifies a config filename, use that. Otherwise ask the user for a name.

---

## Prerequisites: Kappa Geometry Naming Convention

The viewer auto-detects kappa geometry by searching for joints with **exact names**:
- `theta` — the theta rotation axis (outermost of the three kappa axes)
- `kappa` — the kappa rotation axis (tilted relative to theta by angle alpha)
- `phi` — the phi rotation axis (innermost, carries the sample)

These three joints **must** use these exact names in the config JSON. Other joints in the diffractometer (gamma, delta, mu, stokes, etc.) can use any name.

The viewer will automatically:
1. Measure the kappa alpha angle from the geometry at startup
2. Build a compensation lookup table (theta/phi offsets vs kappa angle)
3. Provide virtual chi/theta/phi sliders with kappa sign toggle

---

## Step 1: Inspect the Blender Scene

Use `mcp__blender__get_scene_info` to identify:
- The armature object (type: ARMATURE)
- All mesh objects and their names
- The bone hierarchy (which bones form the kappa chain)

Identify the kappa-specific bones: the three bones whose rotations correspond to theta, kappa, and phi. The kappa bone's rotation axis should be tilted relative to the theta bone's axis — this tilt angle is the **kappa alpha** (typically ~50 degrees for most kappa diffractometers, but measured automatically from geometry).

## Step 2: Extract Armature Bone Data

Use `mcp__blender__execute_blender_code` to extract bone transforms. **CRITICAL: Use the correct C matrix** — `C_gltf = [[1,0,0],[0,0,1],[0,-1,0]]`.

```python
import bpy, json, mathutils

arm_obj = bpy.data.objects['ARMATURE_NAME']  # Replace with actual name
arm = arm_obj.data

# CORRECT coordinate conversion: Blender Z-up to glTF/Three.js Y-up
# Maps: Blender X->X, Blender Y->-Z, Blender Z->Y
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

Use `mcp__blender__execute_blender_code` to read quaternion locks:

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

**Fixed joints:** All WXYZ locked (free=[]) or all unlocked (no constraint) -> set `"fixed": true`.

**Movable joints:** The free component after C_gltf conversion gives the Three.js rotation axis:

| Blender free component | Three.js axis |
|----------------------|---------------|
| X free (W,X free)    | `[1, 0, 0]`  |
| Y free (W,Y free)    | `[0, 0, -1]` |
| Z free (W,Z free)    | `[0, 1, 0]`  |

**Why Y maps to [0,0,-1]:** C_gltf @ R_y(theta) @ C_gltf_inv = R_z(-theta), so Blender Y rotation becomes negative Z in Three.js.

## Step 4: Determine Mesh-to-Bone Parenting

```python
import bpy, json

arm_obj = bpy.data.objects['ARMATURE_NAME']

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        parent_name = obj.parent.name if obj.parent else None
        parent_type = obj.parent_type if obj.parent else None
        parent_bone = obj.parent_bone if obj.parent_bone else None
        print(f"{obj.name}: parent={parent_name}, type={parent_type}, bone={parent_bone}")
```

**Name sanitization:** Three.js GLTFLoader sanitizes names: spaces -> underscores, dots removed. Config link names MUST use sanitized versions.

## Step 5: Determine Joint Limits

```python
import bpy, math

arm_obj = bpy.data.objects['ARMATURE_NAME']
for pb in arm_obj.pose.bones:
    limits = []
    if hasattr(pb, 'use_ik_limit_x') and pb.use_ik_limit_x:
        limits.append(f"X [{math.degrees(pb.ik_min_x):.1f}, {math.degrees(pb.ik_max_x):.1f}]")
    if hasattr(pb, 'use_ik_limit_y') and pb.use_ik_limit_y:
        limits.append(f"Y [{math.degrees(pb.ik_min_y):.1f}, {math.degrees(pb.ik_max_y):.1f}]")
    if hasattr(pb, 'use_ik_limit_z') and pb.use_ik_limit_z:
        limits.append(f"Z [{math.degrees(pb.ik_min_z):.1f}, {math.degrees(pb.ik_max_z):.1f}]")
    if limits:
        print(f"{pb.name}: {', '.join(limits)}")
```

If no limits are set, use sensible defaults for diffractometers:
- theta: `[-200, 180]`
- kappa: `[-150, 150]` (typical for most kappa heads)
- phi: `[-360, 360]` (continuous rotation)
- gamma/delta: check the specific instrument documentation
- Fixed joints: `[0, 0]`

## Step 6: Export GLB from Blender (Skin-Free)

**CRITICAL:** Bone-parented meshes cause Blender to add a `skins` array to the GLB, which breaks Three.js mesh traversal. Always unparent before export.

```python
import bpy, os

arm_obj = bpy.data.objects['ARMATURE_NAME']
output_path = 'OUTPUT_PATH'  # e.g., '/path/to/diffractometer.glb'

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

print(f"Exported to {output_path}")
print(f"Unparented/restored {len(parenting_backup)} bone-parented meshes")
```

### Browser cache busting:
If re-exporting, change the GLB filename (e.g., `model_v2.glb`) to bypass browser cache.

## Step 7: Build the Kinematic Chain Map

Kappa diffractometers typically have **branching kinematic chains**. Map out the full bone hierarchy before writing the config.

### Typical kappa diffractometer chain structure:

```
Chain 1: Detector arm (gamma/delta side)
  root_gamma (fixed, parent: -1)
    -> gamma (movable) -> [fixed links] -> delta (movable)
      -> [fixed links] -> stokes (movable) -> detector_end (movable)
      -> [branching to other detectors]

Chain 2: Sample environment (mu/theta/kappa/phi side)
  root_mu (fixed, parent: -1)
    -> [fixed links] -> mu (movable)
      -> [fixed links] -> theta (movable)        <-- KAPPA CHAIN START
        -> [fixed links] -> kappa (movable)       <-- tilted axis
          -> [fixed links] -> phi (movable)       <-- sample rotation

Optional: virtual axis reference bones (fixed, for visualization only)
```

**Important:** Both root joints share the same physical origin but are separate entries with `"parent": -1`. This allows the viewer to handle branching correctly.

### Identifying fixed kinematic links:
Bones between movable joints (e.g., `ktheta.001`, `kkappa.001`) are **fixed kinematic links**. They encode the physical geometry (offsets, tilts) of the mechanism. They must appear in the config with `"fixed": true` and `"limits": [0, 0]`.

## Step 8: Generate the Config JSON

### Critical requirements for kappa auto-detection:

1. **Joint naming:** The config MUST have joints named exactly `"theta"`, `"kappa"`, and `"phi"` for the viewer to activate kappa geometry mode.

2. **Fixed links between kappa joints:** The fixed bones between theta->kappa and kappa->phi encode the kappa alpha angle. Their rest transforms must be extracted accurately — this is what determines alpha.

3. **Joint ordering:** Joints in the `joints` array define the FK chain. The theta->kappa->phi chain must be properly connected via `parent` indices through any intermediate fixed links.

### Config structure:

```json
{
  "name": "Diffractometer Name",
  "model": "diffractometer_scene.glb",
  "joints": [
    {
      "name": "root_gamma",
      "restPos": [0, 0, 0],
      "restQuat": [w, x, y, z],
      "axis": [0, 0, 1],
      "limits": [0, 0],
      "parent": -1,
      "fixed": true
    },
    {
      "name": "gamma",
      "restPos": [x, y, z],
      "restQuat": [w, x, y, z],
      "axis": [1, 0, 0],
      "limits": [-10, 120],
      "parent": 0
    },
    "... intermediate fixed links ...",
    {
      "name": "theta",
      "restPos": [x, y, z],
      "restQuat": [w, x, y, z],
      "axis": [1, 0, 0],
      "limits": [-200, 180],
      "parent": "index_of_parent"
    },
    "... fixed links encoding kappa tilt ...",
    {
      "name": "kappa",
      "restPos": [x, y, z],
      "restQuat": [w, x, y, z],
      "axis": [0, 0, 1],
      "limits": [-150, 150],
      "parent": "index_of_parent"
    },
    "... fixed links ...",
    {
      "name": "phi",
      "restPos": [x, y, z],
      "restQuat": [w, x, y, z],
      "axis": [0, 0, -1],
      "limits": [-360, 360],
      "parent": "index_of_parent"
    }
  ],
  "links": [
    { "name": "mesh_name", "label": "Display Name", "joint": 0 }
  ],
  "demoPose": [0, 30, 0, 45, ...]
}
```

### Joint fields:
- `restPos`, `restQuat`: from Step 2 (C_gltf converted). `restQuat` is [w,x,y,z]
- `axis`: from Step 3 (quaternion lock analysis)
- `limits`: from Step 5, in degrees. Fixed joints use `[0, 0]`
- `parent`: index of parent joint in this array, or `-1` for root joints
- `fixed` (optional): `true` for non-movable kinematic links

### Link fields:
- `name`: sanitized mesh name (spaces->underscores, no dots)
- `label`: human-readable display name
- `joint`: index into the `joints` array that this mesh moves with

### demoPose:
Array of joint angles in degrees, one per joint. Fixed joints should be `0`. For kappa diffractometers, a good demo pose exercises the kappa mechanism — e.g., set theta=30, kappa=50, phi=-60 to show the chi tilt.

## Step 9: Add Config to Model Selector

Add the new config filename to the `configFiles` array in `threejs_scene.html`. Search for `configFiles` and add the new entry.

## Step 10: Start HTTP Server and Test

```bash
python3 -m http.server 8000
```

Tell the user to open `http://localhost:8000/threejs_scene.html?config=CONFIG_FILE.json` and verify:

### Standard checks:
1. All meshes load and appear correctly
2. FK sliders move the correct joints
3. Each joint rotates about the correct axis
4. Labels toggle shows correct mesh names

### Kappa-specific checks:
5. **Virtual angles panel appears** — a "Virtual Angles" section should appear below the FK sliders with theta, chi, phi sliders and a kappa sign toggle button
6. **Chi slider works** — dragging chi should move kappa while compensating theta and phi to maintain orientation
7. **Kappa alpha is correct** — check the browser console for the log line: `Kappa geometry: alpha=XX.X deg, chi=90 deg -> kappa=YYY.Y deg, ktheta=ZZ.Z deg, kphi=ZZ.Z deg`. The alpha angle should match the known physical kappa tilt angle of the instrument
8. **Kappa sign toggle** — clicking the kappa sign button should switch between the two solutions (positive and negative kappa for the same chi)
9. **Virtual angle readback** — moving physical theta/kappa/phi sliders should update the virtual angle displays consistently
10. **Demo button** — a "Demo Pose" button should appear, setting theta/kappa/phi to show the kappa mechanism in action

### Troubleshooting:

- **Virtual angles panel missing:** Check that joints are named exactly `theta`, `kappa`, and `phi` (case-sensitive, no prefix/suffix)
- **Wrong alpha angle:** The fixed bones between theta and kappa encode the tilt. Verify their rest transforms were extracted with the correct C matrix
- **Chi compensation is wrong:** The compensation table is built numerically from actual geometry. If the fixed link transforms are wrong, compensation will be wrong. Re-extract bone data with the correct C_gltf matrix
- **Kappa sign doesn't flip correctly:** Both solutions should produce the same chi angle. If they diverge, check that the theta and phi axis directions are correct
- **Meshes not found:** Check GLTFLoader name sanitization. Log `glTF nodes:` in browser console
- **Meshes detached from kinematics:** Re-export with the skin-free method from Step 6
- **Browser shows old model:** Change GLB filename for cache busting
