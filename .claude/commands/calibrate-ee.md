# Calibrate End-Effector Crosshair for Robot Arms

Compute the end-effector position offset and crosshair axis directions from a Blender armature, then update the device config JSON. The viewer (`threejs_scene.html`) reads `eeOffset` and `eeAxes` from the config to position and orient the crosshair at the tool flange.

**Do NOT modify `threejs_scene.html`** — only update the config JSON.

User arguments (optional): $ARGUMENTS
- If the user specifies a config filename, use that. Otherwise auto-detect from context.
- If the user specifies the robot's coordinate convention (e.g., "Z-down at flange"), use that to set axis directions.

---

## Step 1: Identify the Armature and Last Bone

Use `mcp__blender__get_scene_info` to find the armature. Then read the config JSON to identify which bone is the last joint in the kinematic chain (the joint with the highest index that is not fixed).

## Step 2: Compute the EE Offset

The crosshair should sit at the **tail** of the last bone (the tool flange face), not at the bone's head (joint center). Extract the tail-to-head offset in the last bone's local frame.

```python
import bpy, mathutils

arm_obj = bpy.data.objects['ARMATURE_NAME']
arm = arm_obj.data

C = mathutils.Matrix((
    (1, 0, 0, 0),
    (0, 0, 1, 0),
    (0, -1, 0, 0),
    (0, 0, 0, 1),
))
C_inv = C.inverted()

# Replace with the actual last bone name
bone = arm.bones['LAST_BONE_NAME']

# Tail offset in Blender armature space
offset_blender = bone.tail_local - bone.head_local

# Convert to Three.js world space
offset_threejs = C @ mathutils.Vector((*offset_blender, 0))

# Convert to the bone's local frame (accounting for all parent transforms)
bone_world = C @ bone.matrix_local @ C_inv
bone_rot_inv = bone_world.to_quaternion().to_matrix().to_4x4().inverted()
offset_local = bone_rot_inv @ mathutils.Vector((offset_threejs.x, offset_threejs.y, offset_threejs.z, 0))

print(f"eeOffset: [{offset_local.x:.5f}, {offset_local.y:.5f}, {offset_local.z:.5f}]")
```

Add the result to the config as `"eeOffset": [x, y, z]`.

## Step 3: Determine the Crosshair Axis Directions

The crosshair has three colored axes:
- **Blue** (index 0) — typically Robot X or tool approach direction
- **Green** (index 1) — typically Robot Y or tool normal
- **Red** (index 2) — typically Robot Z or tool sliding direction

The axes are specified in the **last joint's local frame** (the frame defined by the joint's restQuat in the config).

### Finding the correct directions:

1. **Ask the user** what the robot's tool frame convention is. Common conventions:
   - **Z-down at flange**: Z points into the workpiece (common for Meca500, UR, KUKA)
   - **Z-along-tool**: Z points along the tool direction
   - **Manufacturer-specific**: check the robot's documentation

2. **Map the robot convention to Three.js local axes.** At the last bone in rest pose, determine which Three.js local direction corresponds to each robot axis. Use the bone's orientation and the C_gltf conversion:

```python
import bpy, mathutils, json

arm_obj = bpy.data.objects['ARMATURE_NAME']
arm = arm_obj.data

C = mathutils.Matrix((
    (1, 0, 0, 0),
    (0, 0, 1, 0),
    (0, -1, 0, 0),
    (0, 0, 0, 1),
))
C_inv = C.inverted()

bone = arm.bones['LAST_BONE_NAME']
bone_world = C @ bone.matrix_local @ C_inv
rot = bone_world.to_quaternion().to_matrix()

# Show what each Three.js local axis maps to in world space
for i, label in enumerate(['Local X', 'Local Y', 'Local Z']):
    axis = rot.col[i].to_3d().normalized()
    print(f"{label}: world ({axis.x:.3f}, {axis.y:.3f}, {axis.z:.3f})")
```

3. **Set the axes** based on the desired mapping. Each axis is a unit vector `[x, y, z]` in the last joint's local frame. The axes should form a right-handed coordinate system.

### Common configurations:

| Robot Convention | Blue (0) | Green (1) | Red (2) |
|-----------------|----------|-----------|---------|
| Default (XYZ)   | [1,0,0]  | [0,-1,0]  | [0,0,1] |
| Z-down flange   | [0,0,-1] | [0,-1,0]  | [1,0,0] |
| Z-along-tool    | [1,0,0]  | [0,1,0]   | [0,0,1] |

The user may need to rotate axes interactively to match their convention. Rotations about an axis follow the right-hand rule:

| Rotation | X maps to | Y maps to | Z maps to |
|----------|-----------|-----------|-----------|
| +90° about X | X | -Z | Y |
| -90° about X | X | Z | -Y |
| +90° about Y | -Z | Y | X |
| -90° about Y | Z | Y | -X |
| +90° about Z | Y | -X | Z |
| -90° about Z | -Y | X | Z |

Add the result to the config as `"eeAxes": [[bx,by,bz],[gx,gy,gz],[rx,ry,rz]]`.

## Step 4: Update the Config JSON

Add or update these fields in the config:

```json
{
  "eeOffset": [x, y, z],
  "eeAxes": [[bx,by,bz], [gx,gy,gz], [rx,ry,rz]]
}
```

- `eeOffset`: position of the EE point relative to the last joint, in the last joint's local frame (meters)
- `eeAxes`: three unit vectors `[blue, green, red]` in the last joint's local frame. If omitted, defaults to `[[1,0,0],[0,-1,0],[0,0,1]]`

## Step 5: Test

Tell the user to refresh the viewer and verify:
1. **Crosshair position** — the crosshair center should be at the tool flange face (end of the last link mesh)
2. **Blue axis** — points in the expected direction for the robot's X convention
3. **Green axis** — points in the expected direction for the robot's Y convention
4. **Red axis** — points in the expected direction for the robot's Z convention
5. **IK readout** — if IK is supported, the position/orientation readout should match the robot's convention when the crosshair is moved

### Interactive adjustment:

If the axes aren't right, the user can request rotations like:
- "rotate red and blue 90° about green" — applies a rotation to the two axes while keeping the third fixed
- "flip blue" — negates the blue axis vector
- "swap red and blue" — exchanges the two axis vectors

Apply these as vector transformations to the `eeAxes` values in the config.
