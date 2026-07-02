# LidarStudio

A browser-based studio for **generating and editing LiDAR point clouds and 3D Gaussian splats**, built on a Three.js + GaussianSplats3D viewer with a Python backend. Generate clouds/splats from raw scan bags, then clean, crop, recolour, erase, and re-place them interactively — all non-destructively, with results saved back into your project.

LidarStudio is a fork of a config-driven robot/device viewer; that engine (mesh import, transform gizmos, multi-object scene, remote API) is retained and documented below the LiDAR sections, since the editor builds on it.

> **Note:** the LiDAR features are driven by a Python backend (`lidar_jobs.py`, `process_pointcloud.py`, `process_splat.py`, `edit_ops.py`, `cloud_ops.py`, `splat_io.py`) wired into `server.py`. The browser JS is served fresh on every reload, but the Python modules are imported once and cached — **restart `server.py` after any `.py` change** for it to take effect.

## LiDAR Workflow

The **LiDAR Workflow** panel (top of the right-hand control panel) drives the whole pipeline. Run with `python3 server.py` and open `http://localhost:8080`.

> **Network exposure:** the server binds `127.0.0.1` by default. Pass `--host 0.0.0.0` to serve the viewer to other devices (e.g. a VR headset) — the LiDAR `/api/*` endpoints, which browse and read/write the local filesystem, stay loopback-only even then unless you also pass `--allow-remote-fs`. Static serving is allowlisted (`js/`, `node_modules/`, and specific root-level asset types); the Python source, `.git`, and dotfiles are never served.

### Generate

Turn a raw scan folder (LiDAR + IMU + image `.bag` files) into a cloud or splat:

- **Point cloud** — KISS-ICP registration + multi-view photo colouring, voxel-downsampled.
- **Splat** — three modes:
  - **surfel** — fast, derives surface-aligned gaussians directly from the coloured cloud.
  - **trained** — GPU-trained 3D Gaussian Splatting (needs CUDA + the scan trajectory).
  - **bootstrap** — CPU fallback, one gaussian per point. Adjustable **blob size (m)** controls the gaussian radius (smaller = finer, less blobby).

Progress streams live into the panel; the result loads straight into the scene and appears in the Library.

### Library

Lists the clouds/splats already in the project (`pointclouds/` and `splats/`). **Load** brings one into the scene; long names truncate with the full name on hover.

### Editing (non-destructive)

All edits operate on the selected object, write a sibling `*_edited.ply` (the original is untouched), and reload in place. **Revert to original** reloads the source.

- **Decimate** — keep 1-in-N points.
- **Denoise** — statistical outlier removal (SOR; neighbours + std ratio).
- **Crop** — an oriented 3D box (translate/rotate/scale via gizmo or **T/R/S** keys); keep inside or outside.
- **Recolour** — re-project the scan photos onto an edited cloud using its saved trajectory.
- **Save transformed (bake pose)** — bake the object's in-scene transform into the file.
- Decimate and Denoise can be **limited to the visibility box region**.

### Visibility box

A non-destructive oriented box that hides geometry inside *or* outside it (live GPU clip) so internal structure can be inspected while editing. It can also **delete the shown side** in one click. Both the visibility and crop boxes are transformable with **T/R/S** and switch focus on click.

### Eraser (Cube / Sphere / Cylinder)

Add a primitive and use it as an eraser volume:

- **Point clouds** — turn the **live eraser** on, then drag a primitive through the cloud: points vanish in real time as the volume sweeps over them. Per-stroke **undo**. Saved when you **Save As**.
- **Splats** — a one-shot **Erase now** deletes points inside the current primitives.

### Save As & Workflow

- **Save As** — export the selected object under a new name. Optionally **bake the transform into the coordinates** for a self-contained, portable file; otherwise a lossless copy plus a pose sidecar that reloads correctly here.
- **Save / Load Workflow** — remember the panel's settings (paths, parameters, options) in the browser and restore them on demand.

### How placement is preserved

Edits keep the file in its original local frame and store the object's pose in a `<file>.pose.json` sidecar, reapplied on load — so an edited/erased object reloads in the same place and orientation (and it works for splats, which carry an internal viewer transform). "Save transformed" / the Save As *bake* option instead write world coordinates for use in other tools.

## Supported Devices (device-viewer engine)

| Device | Config | Type | Description |
|--------|--------|------|-------------|
| Meca500 R3 | `meca500_config.json` | 6-DOF serial | Compact industrial manipulator |
| i16 Diffractometer | `i16_config.json` | Branching (10 movable) | Diamond Light Source 6-circle diffractometer with merlin and crystal detectors |
| i19 Kappa Diffractometer | `i19_config.json` | Branching | Diamond Light Source kappa diffractometer (2θ / θ / κ / φ chain) |
| Yaskawa GP225 | `gp225_config.json` | 6-DOF serial | Heavy-payload industrial robot |
| Yaskawa GP280 | `gp280_config.json` | 6-DOF serial | Heavy-payload industrial robot |
| Yaskawa GP180-120 | `gp180_config.json` | 6-DOF serial | Heavy-payload industrial robot |
| Yaskawa MotoMini | `motomini_config.json` | 6-DOF serial | Compact industrial robot |
| Hexapod | `hexapod_config.json` | Stewart platform (6 legs) | 6-DOF parallel kinematic platform with Damped Track leg IK |

New devices can be added from Blender scenes using `import_robot.py` (serial robots) or `import_hexapod.py` (Stewart platforms). See [Adding New Devices](#adding-new-devices).

## Features

- **Config-driven viewer** — a single generic `threejs_scene.html` viewer loads any device via JSON config
- **Multi-device scene** — load multiple devices simultaneously from the add-device dropdown; click a device in the list or click its mesh to switch active device
- **Device renaming** — double-click a device name in the device list to rename it
- **Device origin transform** — move and rotate device origins with translate/rotate gizmo modes; World/Local space toggle for gizmo axis alignment; numeric X/Y/Z (mm) and Rx/Ry/Rz (deg) inputs for precise positioning, synced live with the gizmo
- **Device parenting** — parent a device to a link on another device so it follows the kinematic chain
- **Auto-fit camera** — camera automatically frames the loaded model on startup
- **Forward Kinematics** — joint angle sliders for all movable joints (fixed kinematic links are hidden)
- **Inverse Kinematics** — 6-DOF damped least-squares solver (position + ZYX Euler orientation)
- **Branching kinematic chains** — supports devices with multiple independent chains and sub-branches (e.g., i16 has gamma and mu chains with merlin/crystal sub-branches)
- **Stewart platform / hexapod support** — parallel kinematic platform with 6 two-bar linkage legs using Damped Track IK; platform pose control via translation (X/Y/Z mm) and rotation (Rx/Ry/Rz deg) sliders
- **Hexapod FK solver** — Newton-Raphson solver computes platform pose from 6 leg lengths; interactive leg length sliders drive the platform in real time with bidirectional sync to pose sliders
- **Drag Platform mode** — attach TransformControls gizmo directly to the hexapod platform for interactive 3D dragging with real-time leg tracking, slider sync, and limit clamping (keyboard: T for translate, R for rotate)
- **Draggable IK target** — move the green sphere with the gizmo or use XYZ / alpha-beta-gamma sliders
- **Orientation gizmo** — visual end-effector orientation indicator showing the current tool frame axes
- **Double-click to type** — double-click any slider value label to enter a number directly
- **Mesh labels toggle** — show/hide object name labels on all meshes
- **Mesh import** — load external STL, OBJ, PLY, and GLB/GLTF files into the scene with auto-scaling, labels, and per-object colour
- **Primitive objects** — add cube, sphere, and cylinder primitives directly from the toolbar
- **Object duplication** — duplicate any imported or primitive object with a single click
- **Scene save/load** — save the scene to a file and reload it via the Save Scene / Load Scene buttons (LidarStudio does not auto-restore the scene on page load; clouds/splats are loaded explicitly from the Library)
- **Object manipulation** — click objects to select, then move, rotate, or scale with transform gizmos (keyboard: T/R/S, Escape to deselect); World/Local space toggle for gizmo axis alignment
- **Parent-child linking** — attach objects to device links so they follow the kinematic chain
- **Self-collision detection** — BVH-accelerated triangle-level intersection testing between device links, using kinematic adjacency to skip physically connected parts
- **Collision detection** — intersection testing between device links and imported scene objects, with red highlight on colliding meshes
- **Screenshot** — one-click PNG capture of the WebGL view composited with the control panel overlay
- **Unified Euler convention** — viewer, WebSocket state, and the Python `GNKinematics` library all report end-effector orientation as the same ZYX Euler triple (α=Rx, β=Ry, γ=Rz), with matching gimbal-lock branches
- **Remote control API** — two-way WebSocket API for controlling any device from Python or any WebSocket client
- **Session routing** — each browser tab gets a unique session ID; controllers can target a specific tab or broadcast to all
- **Connection info panel** — click the API status indicator (top-left) to see the session ID, connection command, and download `RemoteAPI.zip` (IPython client + `GNKinematics` library + robot definitions)

## Quick Start

**LidarStudio (LiDAR generation + editing):**
```bash
pip3 install aiohttp websockets numpy open3d scipy   # plus torch + gsplat for GPU-trained splats
python3 server.py
```
Open `http://localhost:8080` and use the **LiDAR Workflow** panel. The `/api/*` endpoints (generation, editing, save) require `server.py` — the plain static server below won't drive them. Remember to restart `server.py` after editing any backend `.py` file.

**Device viewer only — standalone (no backend):**
```bash
python3 -m http.server 8000
```
Open `http://localhost:8000/threejs_scene.html` in a browser. Use the dropdown to switch devices, or specify a config: `?config=i16_config.json`. (LiDAR generation/editing is unavailable without `server.py`.)

**With remote control API:**
```bash
pip3 install aiohttp websockets ipython
python3 server.py --config meca500_config.json
```
Open `http://localhost:8080` in a browser. The status indicator in the top-left shows the WebSocket connection state and the session ID for this tab. Click it to open the connection info panel, which shows the full connection command, WebSocket URL, and a link to download `RemoteAPI.zip` — a self-contained bundle with `robot_ipython.py`, the `GNKinematics` Python library, and `RobotDefinitions.py`.

## Adding New Devices

### Importing from Blender

The `import_robot.py` script extracts bone transforms, rotation axes, mesh parenting, and joint limits from a Blender armature, then exports a config JSON and skin-free GLB.

#### Blender Setup

1. **Armature**: create an armature with one bone per joint/kinematic link. Bones should form a parent-child chain matching the kinematic chain.
2. **Mesh parenting**: parent each mesh to its corresponding bone (select mesh, then bone in Pose Mode, `Ctrl+P` → Bone).
3. **Rotation mode**: set all pose bones to **Quaternion** rotation mode.
4. **Quaternion locks**: for each bone, lock all quaternion components except the one that corresponds to the rotation axis:
   - Lock W, Y, Z → free X (rotation about bone-local X)
   - Lock W, X, Z → free Y (rotation about bone-local Y)
   - Lock W, X, Y → free Z (rotation about bone-local Z)
   - Lock all (W, X, Y, Z) → fixed joint (no rotation)
   - Leave all unlocked → fixed joint (treated as structural)
5. **Joint limits**: set IK limits on the free axis for each movable bone (`Bone Properties → Inverse Kinematics`). If no IK limits are set, the importer defaults to [-180, 180].
6. **Render visibility**: hide any helper meshes by disabling their render visibility (camera icon in outliner). These will be excluded from the GLB export.

#### Running the Importer

In Blender's Script Editor (or Python console):

```python
exec(open('/path/to/RobotVisualisation/import_robot.py').read())
```

To override defaults, set variables before `exec()`:

```python
ARMATURE_NAME = 'MyArmature'     # default: auto-detect first armature
DEVICE_NAME = 'My Robot'         # default: armature name
CONFIG_FILE = 'my_robot_config.json'  # default: derived from device name
GLB_FILE = 'my_robot_scene.glb'  # default: derived from config name
exec(open('/path/to/RobotVisualisation/import_robot.py').read())
```

The script:
1. Finds the armature and extracts bone rest transforms (converted from Blender Z-up to Three.js Y-up)
2. Determines rotation axes from quaternion locks
3. Maps meshes to bones via parent chains
4. Extracts IK joint limits
5. Exports a skin-free GLB (temporarily unparents bone-parented meshes to avoid glTF skins)
6. Generates a config JSON with joints, links, eeOffset, and eeAxes
7. Registers the config in `js/panel.js` if not already present

#### Post-Import Checklist

Open `http://localhost:8000/threejs_scene.html?config=my_robot_config.json` and verify:

1. All meshes load and appear correctly
2. FK sliders move the correct joints
3. Each joint rotates about the correct axis and in the correct direction
4. Labels toggle shows correct mesh names

If rotation directions are wrong for specific joints, negate the `axis` array in the config JSON (e.g. `[0, -1, 0]` → `[0, 1, 0]`). If meshes are missing, check GLTFLoader name sanitization in the browser console — the importer sanitizes names (spaces → underscores, dots removed) but the GLB mesh names must match.

#### apiSign

If the robot manufacturer's joint angle convention is opposite to the viewer's for specific joints, add `"apiSign": -1` to those joints in the config. This flips the sign on the slider display and the WebSocket API without changing the physical rotation axis. For example, the Meca500 J4 has `"apiSign": -1` because the manufacturer defines positive J4 in the opposite direction.

### Importing a Hexapod from Blender

The `import_hexapod.py` script extracts pivot positions, mesh assignments, and platform limits from a Blender armature with Damped Track leg constraints.

#### Blender Setup

1. **Armature**: create an armature with bones for each leg segment plus a control bone for the platform.
2. **Lower leg bones** (LL): one per leg, positioned at the base pivot. Each has a **Damped Track** constraint targeting its corresponding upper leg bone.
3. **Upper leg bones** (UL): one per leg, positioned at the platform pivot. Each has a **Damped Track** constraint targeting its lower leg bone, plus a **Child Of** constraint targeting the control bone.
4. **Control bone**: a single bone (e.g., `ControlHandle`) that the upper leg bones follow. Moving it moves the platform.
5. **Mesh parenting**: parent each lower leg mesh to its LL bone, each upper leg mesh to its UL bone, and the top plate mesh to the control bone. The base plate can be unparented.
6. **Platform limits** (optional): add **Limit Location** and **Limit Rotation** constraints to the control bone to define the platform's travel range. The importer reads these and converts to mm/degrees. If no constraints are found, reasonable defaults are used.

#### Running the Importer

```python
exec(open('/path/to/RobotVisualisation/import_hexapod.py').read())
```

To override defaults:

```python
ARMATURE_NAME = 'MyHexapod'
CONTROL_BONE  = 'ControlHandle'
DEVICE_NAME   = 'Hexapod'
BASE_MESH     = 'BasePlate'
PLATFORM_MESH = 'TopPlate'
exec(open('/path/to/RobotVisualisation/import_hexapod.py').read())
```

The script:
1. Finds the armature and auto-detects the hexapod structure from Damped Track + Child Of constraints
2. Identifies lower/upper leg bone pairs and the control bone
3. Extracts rest-pose pivot positions (converted from Blender Z-up to Three.js Y-up)
4. Maps meshes to bones (legs, platform, base)
5. Reads Limit Location / Limit Rotation constraints for platform limits
6. Exports a skin-free GLB
7. Generates a `type: "hexapod"` config JSON
8. Registers the config in `js/panel.js`

#### Post-Import Checklist

Open `http://localhost:8000/threejs_scene.html?config=hexapod_config.json` and verify:

1. All meshes load and appear correctly
2. Platform translation sliders move the platform
3. Rotation sliders tilt the platform correctly
4. Legs track correctly (no pass-through or separation)
5. Demo pose button works
6. FK solver (leg length sliders) converges
7. Drag Platform mode works with TransformControls

If pivots are wrong, check bone head positions in Blender edit mode. If limits are wrong, add Limit Location / Limit Rotation constraints to the control bone, or edit the limits in the config JSON.

### Adding Robot Kinematics Definitions

To enable IK via the Python `GNKinematics` library (used by `robot_ipython.py`), add a kinematics definition to `RobotDefinitions.py`:

```python
from GNKinematics import kinematics

MyRobot_kin = kinematics.kinematics.from_home_positions(
    v0=np.array([0, 0, 135.0]),       # base to J2 (mm)
    v1=np.array([0, 0, 270.0]),       # J2 to J3
    v2=np.array([60, 0, 308]),        # J3 to J4
    v3=np.array([120, 0, 308]),       # J4 to J5
    v4=np.array([190, 0, 308]),       # J5 to J6 flange
    motor_limits=np.array([[-175, 175], [-70, 90], [-135, 70],
                            [-170, 120], [-90, 115], [-360, 360]]),
    centre_offset=[0, 0, 0],
    tool_offset=[0, 0, 0],
    strategy='minimum_movement',
    weighting=[6, 5, 4, 3, 2, 1],
)
```

The `v0`–`v4` vectors are the joint positions at home (all joints at zero), in the robot's Z-up coordinate frame. `motor_limits` are per-joint angle limits in degrees.

Then add the new object to the IPython namespace in `robot_ipython.py`:

1. Import it at the top: `from RobotDefinitions import ..., MyRobot_kin`
2. Add it to the `user_ns` dict passed to `IPython.start_ipython()`:
   ```python
   user_ns={
       ...
       "MyRobot_kin": MyRobot_kin,
   }
   ```

### Config File Structure

```json
{
  "name": "Device Name",
  "model": "device_scene.glb",
  "joints": [
    {
      "name": "joint_name",
      "bone": "blender_bone_name",
      "restPos": [x, y, z],
      "restQuat": [w, x, y, z],
      "axis": [x, y, z],
      "limits": [-180, 180],
      "parent": 0,
      "fixed": false,
      "apiSign": 1
    }
  ],
  "links": [
    { "name": "mesh_name", "label": "Display Name", "joint": 0 }
  ],
  "eeOffset": [0, 0, -0.05],
  "eeAxes": [[0, 0, -1], [0, -1, 0], [1, 0, 0]],
  "demoPose": [0, 45, -90]
}
```

Key fields:
- **joints**: one entry per bone — `restPos`/`restQuat` from Blender (C matrix converted), `axis` from quaternion lock analysis, `parent` index (-1 for roots), `fixed` for non-movable kinematic links, optional `apiSign` (-1 to flip slider/API convention)
- **links**: maps GLB mesh names (sanitized: spaces→underscores, dots removed) to joint indices
- **eeOffset**: displacement from last joint to end-effector point (derived from last bone length)
- **eeAxes**: end-effector crosshair axes — use `[[0,0,-1],[0,-1,0],[1,0,0]]` for all robots
- **demoPose**: joint angles in degrees for the demo button (one per joint, fixed joints = 0)

### Hexapod Config File Structure

```json
{
  "name": "Hexapod",
  "type": "hexapod",
  "model": "hexapod_scene.glb",
  "platform": {
    "mesh": "TopPlate",
    "restPosition": [0.0, 0.2, 0.0]
  },
  "base": {
    "mesh": "BasePlate"
  },
  "legs": [
    {
      "basePivot": [0.125, 0.014, 0.029],
      "platformPivotLocal": [0.058, -0.013, 0.068],
      "lowerMesh": "LowerLeg_1",
      "upperMesh": "UpperLeg_1"
    }
  ],
  "limits": {
    "x": [-30, 30], "y": [-30, 30], "z": [-20, 20],
    "rx": [-11, 11], "ry": [-11, 11], "rz": [-20, 20]
  },
  "demoPose": [0, 0, 5, 5, 0, 10]
}
```

Key fields:
- **type**: must be `"hexapod"` to trigger the parallel kinematics loader
- **platform.mesh**: GLB mesh name for the top plate (reparented to the platform group)
- **platform.restPosition**: platform centre in Three.js Y-up coordinates (metres) at home pose
- **base.mesh**: GLB mesh name for the base plate
- **legs**: array of 6 leg entries — `basePivot` is the lower pivot in world coordinates (metres), `platformPivotLocal` is the upper pivot relative to the platform rest position
- **limits**: platform travel limits — translation in mm (`x`, `y`, `z`) and rotation in degrees (`rx`, `ry`, `rz`)
- **demoPose**: `[x, y, z, rx, ry, rz]` for the demo button (mm and degrees)

## Objects

### Mesh Import

Click **Import Mesh** to load files into the scene. Supported formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| STL | `.stl` | Binary or ASCII; auto-scaled if bounding box > 1 m |
| OBJ | `.obj` | Geometry only; no MTL material files |
| PLY | `.ply` | Binary and ASCII; vertex colours preserved if present |
| GLB / GLTF | `.glb`, `.gltf` | Full scene hierarchy, materials, and textures |

### Primitives

Click **Cube**, **Sphere**, or **Cylinder** to add a primitive shape. Primitives behave identically to imported objects — they can be moved, coloured, parented, and are persisted across reloads.

### Transform Gizmos

| Key | Mode |
|-----|------|
| `T` | Move (translate) |
| `R` | Rotate |
| `S` | Scale |
| `Esc` | Deselect |

Both device and object gizmos have a **World/Local** toggle button. In World mode the gizmo axes align with the scene axes; in Local mode they align with the object's own axes. The toggle resets to World when the gizmo is deactivated.

### Parent-Child Linking

Use the **Parent** dropdown to attach objects to device links. Parented objects follow the kinematic chain. Local transforms are preserved when reparenting.

## Collision Detection

Toggle **Collision: ON/OFF** in the panel. The viewer tests for triangle-level intersections using:

1. **Broad phase** — AABB check to eliminate distant pairs
2. **Narrow phase** — BVH-accelerated triangle-triangle intersection via [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh)

Self-collision between device links uses kinematic adjacency analysis — links sharing the same joint or connected through a parent-child relationship are skipped. Links on separate branches (e.g., gamma chain vs mu chain on the i16) are always checked.

## Remote Control

The WebSocket API at `ws://localhost:8080/ws` allows any client to control devices, manage multi-device scenes, manipulate objects, and control the camera in real time. All commands target the active device by default; include `"device": "<name>"` to target a specific device.

### Session Routing

Each browser tab that connects to the viewer is assigned a unique **session ID** (an 8-character hex string, e.g. `ab12cd34`). The ID is shown in the status bar (`API: connected [ab12cd34]`) and in the connection info panel.

Controllers can target a specific viewer tab or broadcast to all:

| Connection | Routes to |
|-----------|-----------|
| `ws://localhost:8080/ws` | All connected viewer tabs |
| `ws://localhost:8080/ws?session=ab12cd34` | Only the tab with that session ID |

Active sessions can be listed via the HTTP endpoint:
```
GET http://localhost:8080/sessions
→ [{"id": "ab12cd34", "viewers": 1}, ...]
```

### Interactive Client (IPython)

```bash
python3 robot_ipython.py --config meca500_config.json                        # broadcast to all tabs
python3 robot_ipython.py --config meca500_config.json --session ab12cd34     # target a specific tab
python3 robot_ipython.py --config i16_config.json --session ab12cd34       # i16 diffractometer
python3 robot_ipython.py --url ws://192.168.1.100:8080/ws --session ab12cd34  # remote server
```

The client launches an IPython terminal with a pre-connected `robot` object. It supports two syntaxes — **Python method calls** for full programmatic control, and **space-separated commands** (via IPython magics) for quick interactive use. The prompt, help text, and tab completion adapt to the loaded device. The `robot_ipython.py` file can also be downloaded from the connection info panel in the viewer.

**Both syntaxes work side by side:**

```python
meca500 [1]: home                                    # space-separated
meca500 [2]: robot.home()                            # Python method
meca500 [3]: joints 0 30 60 0 45 90                  # space-separated
meca500 [4]: robot.joints(0, 30, 60, 0, 45, 90)     # Python method
meca500 [5]: for a in range(0, 91, 10):              # full Python syntax
         ...:     robot.joint('J1', a)
         ...:     time.sleep(0.1)
```

**Position queries** (return values for programmatic use):

| Property / Method | Example | Description |
|---------|---------|-------------|
| `robot.pos` | `x, y, z = robot.pos` | End-effector position [x, y, z] mm |
| `robot.ori` | `robot.ori` | End-effector orientation [a, b, g] degrees |
| `robot.angles` | `robot.angles` | All joint angles (list) |
| `robot.get_joint('J1')` | `a = robot.get_joint('J1')` | Single joint angle by name |
| `robot.mode` | `robot.mode` | Current mode ('FK' or 'IK') |
| `robot.get_device_pos('GP225')` | `d = robot.get_device_pos()` | Any device: pos, rot, joints, EE |
| `robot.get_obj_pos('cube_1')` | `o = robot.get_obj_pos(0)` | Any object: pos, rot, scale, BB |
| `robot.platform_pose` | `robot.platform_pose` | Hexapod platform pose [x,y,z,rx,ry,rz] |
| `robot.leg_lengths` | `robot.leg_lengths` | Hexapod leg lengths [l1..l6] mm |

**Device commands:**

| Space-separated | Python | Description |
|---------|---------|-------------|
| `state` | `robot.state()` | Request current device state |
| `devices` | `robot.devices()` | List all loaded devices |
| `device i16` | `robot.device('i16')` | Switch active device by name |
| `sessions` | `robot.sessions()` | List viewer session IDs |
| `home` | `robot.home()` | All joints to 0 |
| `fk` | `robot.fk()` | Switch to FK mode |
| `ik` | `robot.ik()` | Switch to IK mode |
| `joints 45 -90 0 0 30 0` | `robot.joints(45, -90, 0, 0, 30, 0)` | Set all movable joint angles (degrees) |
| `joint gamma 45` | `robot.joint('gamma', 45)` | Set a single joint by name |
| `pos meca500 [0,0,0,0,0,0]` | `robot.set_pos('meca500', [0,0,0,0,0,0])` | Set joints on a named device (accepts list, numpy array, or callable) |
| `pos meca500 {2: 45}` | `robot.set_pos('meca500', {2: 45})` | Set individual axes only (key = joint index or name) |
| `pos i19 {'v:chi': 45}` | `robot.set_pos('i19', {'v:chi': 45})` | Set kappa virtual angle(s) — keys `v:chi`, `v:theta`, `v:phi` |
| `inc meca500 [0,0,0,0,0,10]` | `robot.inc_pos('meca500', [0,0,0,0,0,10])` | Increment joints on a named device relative to current |
| `inc meca500 {5: 10}` | `robot.inc_pos('meca500', {5: 10})` | Increment individual axes only (key = joint index or name) |
| `inc i19 {'v:chi': 5}` | `robot.inc_pos('i19', {'v:chi': 5})` | Increment kappa virtual angle(s) |
| `move 150 100 300 45 0 0` | `robot.move(150, 100, 300, 45, 0, 0)` | IK move to position (mm) + orientation (deg) |
| `target 190 0 308` | `robot.target(190, 0, 308)` | Set IK target without switching mode |
| `demo` | `robot.demo()` | Run the config's demo pose |

**Device transform commands:**

| Python | Description |
|--------|-------------|
| `robot.devpose([x,y,z], [rx,ry,rz])` | Set device origin position (mm) and/or rotation (deg); also accepts a single 6-element list |
| `robot.devtranslate(dx, dy, dz, space='parent')` | Translate device origin by delta (mm) in parent/local/world frame |
| `robot.devrotate(rx, ry, rz, space='parent')` | Rotate device origin by delta (deg) in parent/local/world frame |

`devpose` sets the absolute position and/or rotation of the device origin in its parent frame (the same values shown in the numeric inputs when **Move Device Origin** is active). Both arguments are optional — pass only `position` or only `rotation` to change one without affecting the other.

```python
r.devpose([100, 0, 0])                  # position only
r.devpose([100, 0, 0], [0, 0, 90])      # position + rotation
r.devpose([100, 0, 0, 0, 0, 90])        # full pose as one list
r.devpose(rotation=[0, 0, 90])          # rotation only
r.devpose([0, 0, 0], device='GP225')    # specific device
```

**Hexapod (Stewart platform) commands:**

| Property / Method | Description |
|---------|-------------|
| `robot.platform([x,y,z,rx,ry,rz])` | Set platform pose (mm, degrees) |
| `robot.platform_pose` | Current platform pose `[x, y, z, rx, ry, rz]` |
| `robot.leg_lengths` | Current leg lengths `[l1..l6]` in mm |
| `robot.hexapod_fk([pose])` | FK: pose → leg lengths (query only). Omit pose to use current |
| `robot.hexapod_ik([l1..l6])` | IK: leg lengths (mm) → platform pose (query only) |
| `robot.get_leg_lengths()` | Get current leg lengths with per-leg detail |
| `robot.set_leg_lengths(l1..l6)` | Set platform pose by specifying desired leg lengths (mm) |

```python
hexapod [1]: robot.platform([0, 0, 10, 0, 0, 0])     # move platform up 10mm
hexapod [2]: robot.platform_pose                        # read back current pose
hexapod [3]: robot.leg_lengths                          # read current leg lengths
hexapod [4]: result = robot.hexapod_fk([5, 0, 10, 2, 0, 0])  # compute leg lengths for a pose
hexapod [5]: result = robot.hexapod_ik([150, 150, 150, 150, 150, 150])  # solve pose from leg lengths
hexapod [6]: robot.set_leg_lengths([152, 148, 150, 150, 151, 149])      # drive platform via leg lengths
```

**Coordinate transform:**

| Python | Description |
|--------|-------------|
| `robot.worldToLocal([x,y,z,rx,ry,rz])` | Transform a world-frame pose into the active device's local frame |
| `robot.worldToLocal([x,y,z], [rx,ry,rz])` | Same, with separate position and orientation arguments |
| `robot.worldToLocal(pos, ori, device='GP225')` | Transform relative to a specific device |

`worldToLocal` converts a world-frame pose (position in mm, orientation as XYZ intrinsic Euler angles in degrees) into the coordinate frame of a device's origin. This is useful when a robot is mounted at an arbitrary position/rotation and you need to express a world target in the robot's own coordinate system — for example, to feed into an IK solver that expects local coordinates.

The orientation convention is `R = Rx(α)·Ry(β)·Rz(γ)` in a right-handed Z-up frame (X right, Y into scene, Z up). Note that Y points *into* the scene, which is the negation of the viewer's Y readback from `dev_pose()`.

When called with a 6-element list, returns a 6-element list `[x, y, z, rx, ry, rz]`. When called with separate position and orientation arguments, returns a `(position, orientation)` tuple.

```python
# Robot mounted at world position [500, 200, 0] with 45° rotation
meca500 [1]: robot.worldToLocal([600, 200, 300, 0, 0, 0])
# → [70.71, 70.71, 300.0, 0.0, 0.0, -45.0]   (position and orientation in robot's local frame)

meca500 [2]: p, o = robot.worldToLocal([600, 200, 300], [0, 0, 0])
# p = [70.71, 70.71, 300.0], o = [0.0, 0.0, -45.0]

meca500 [3]: robot.worldToLocal([600, 200, 300], [0, 0, 0], device='GP225')
# Transform relative to a different device
```

**Motion planning commands:**

| Space-separated | Python | Description |
|---------|---------|-------------|
| `plan --start 0 0 0 --end 45 90 0` | `robot.plan([0,0,0], [45,90,0])` | Path plan between two poses |
| `plan --start mu=0 --end mu=45` | `robot.plan({'mu': 0}, {'mu': 45})` | Named axes |
| `scan theta 0 90 5` | `robot.scan(('theta', 0, 90, 5))` | 1D scan |
| `scan theta 0 90 5 phi 0 30 2` | `robot.scan(('theta',0,90,5), ('phi',0,30,2))` | 2D grid scan |
| `scan theta 0 90 5 phi 0 1` | `robot.scan(('theta',0,90,5), ('phi',0,1))` | Coupled scan |
| `scan DevA:J1 0 50 5 DevB:J1 0 30 5` | `robot.scan(('DevA:J1',0,50,5), ('DevB:J1',0,30,5))` | Multi-device scan |
| `scan v:chi 0 90 5` | `robot.scan(('v:chi', 0, 90, 5))` | Kappa virtual-axis scan (chi/theta/phi) |
| `scan v:chi 0 45 5 v:phi 0 30 5` | `robot.scan(('v:chi',0,45,5), ('v:phi',0,30,5))` | Virtual-axis grid/coupled scan |
| `scan ee:x 150 250 10` | `robot.scan(('ee:x', 150, 250, 10))` | Cartesian end-effector scan (Python IK) |
| `scan ee:x 150 250 5 ee:y -50 50 5` | `robot.scan(('ee:x',150,250,5), ('ee:y',-50,50,5))` | Cartesian grid/coupled scan |
| `scan ee:z 200 400 10 --space world` | `robot.scan(('ee:z',200,400,10), space='world')` | Cartesian scan in world frame |
| `scan ee:x ee:y ee:z waypoints` | `robot.scan('ee:x','ee:y','ee:z', waypoints)` | Cartesian array scan (rows = poses) |
| `scan GP180_120:ee:z 200 400 10` | `robot.scan(('GP180_120:ee:z', 200, 400, 10))` | Cartesian scan on a named device |
| `scan GP180_120:ee:z 200 400 10 Meca500:ee:x 150 250 10` | `robot.scan(('GP180_120:ee:z',200,400,10), ('Meca500:ee:x',150,250,10))` | Multi-device Cartesian scan |
| `scan GP180_120:ee:y 354 400 10 I16_diff:delta 0 120 10` | `robot.scan(('GP180_120:ee:y',354,400,10), ('I16_diff:delta',0,120,10))` | Mixed Cartesian + joint scan |
| `scan GP180_120 scanpoints` | `robot.scan('GP180_120', scanpoints)` | Full-vector array scan (device name expands to all joints) |
| `scan GP180_120 Meca500 combined_pts` | `robot.scan('GP180_120', 'Meca500', combined_pts)` | Multi-device vector scan (cols = joints of each device) |
| `scan robot1 robot2 my_func()` | `robot.scan('robot1', 'robot2', my_func)` | Multi-device vector scan with callable |
| — | `robot.scan(('@Cube:tx', 0, 100, 10))` | Object translation scan |
| — | `robot.scan(('@Cube:rz', 0, 360, 10))` | Object rotation scan |
| — | `robot.scan(('@Cube:tx',0,100,10), space='world')` | Object scan in world coords |
| — | `robot.scan(('J1',0,90,5), ('@Cube:tz',0,50,5))` | Mixed joint + object scan |

Object scan axes use `@ObjectName:component` syntax where component is `tx`, `ty`, `tz`, `rx`, `ry`, or `rz`. The `space` parameter (`'local'` or `'world'`) controls the coordinate frame for object transforms (default: `'local'`).

Kappa virtual axes use a `v:` prefix (`v:chi`, `v:theta`, `v:phi`) to disambiguate from the physical `theta`/`phi` joints. Virtual scans target the active kappa device and cannot be mixed with physical-joint axes in the same scan. The same `v:<axis>` keys work as dict keys in `robot.set_pos` / `robot.inc_pos`.

Cartesian end-effector axes use an `ee:` prefix — `x`, `y`, `z` (mm) and `a`, `b`, `g` (ZYX Euler degrees). Each target pose is solved to joint angles by the analytical Python IK (`GNKinematics`) and streamed as joint waypoints, so the on-screen pose matches the analytical solution. Start/end/step are absolute coordinates and unlisted axes hold their current value. `--space`/`space=` selects the frame: `local` (default, robot base frame) or `world` (converted per waypoint via `worldToLocal`, so the end-effector tracks world axes regardless of how the device is mounted). Supported on Meca500, GP180_120, GP225, GP280 and MotoMini; cannot be mixed with joint, virtual, or object axes in the same scan.

By default an `ee:` scan targets the active device. Prefix the axis with a device name — `Device:ee:<axis>` (e.g. `GP180_120:ee:z`) — to target a specific device, or list several to scan multiple arms in one command. Each device is solved with its own IK and base pose, and the per-step joint solutions are streamed together (grid axes form a product across devices, coupled axes lock-step with the primary), mirroring multi-device joint scans. Device-prefixed axes are supported in the range form only; for an array scan, switch to the device first with `robot.device('Name')`.

Cartesian `ee:` axes can be combined with ordinary joint axes on *other* devices in the same scan (e.g. `scan GP180_120:ee:y 354 400 10 I16_diff:delta 0 120 10`) — the Cartesian device is IK-solved while the joint axis is set directly, and all devices step together. A single device cannot mix `ee:` and joint axes, and `ee:` cannot be combined with object (`@`) or virtual (`v:`) axes.

**Object commands:**

| Space-separated | Python | Description |
|---------|---------|-------------|
| `objects` | `robot.objects()` | List all imported objects |
| `obj MyPart` | `robot.obj('MyPart')` | Get object details |
| `objpos MyPart 100 50 0` | `robot.objpos('MyPart', 100, 50, 0)` | Set object position (mm) |
| `objrot #0 0 0 45` | `robot.objrot('#0', 0, 0, 45)` | Set object rotation (degrees) |
| `objscale MyPart 2` | `robot.objscale('MyPart', 2)` | Set uniform scale |
| `objvis MyPart on` | `robot.objvis('MyPart', True)` | Show/hide object |
| `collision on` | `robot.collision(True)` | Enable/disable collision detection |
| `collisions` | `robot.collisions()` | Get current collision pairs |

Object transforms accept a `space` parameter (`'parent'`, `'local'`, or `'world'`):

| Python | Description |
|--------|-------------|
| `robot.objtranslate('Cube', dx, dy, dz, space='parent')` | Translate object by delta (mm) |
| `robot.objrotate('Cube', rx, ry, rz, space='parent')` | Rotate object by delta (deg) |

**Visualization and camera:**

| Python | Description |
|--------|-------------|
| `robot.labels()` / `robot.labels(False)` | Show/hide joint labels |
| `robot.origins()` / `robot.origins(False)` | Show/hide joint origin axes |
| `robot.chain()` / `robot.chain(False)` | Show/hide kinematic chain |
| `robot.ortho()` / `robot.ortho(False)` | Orthographic/perspective camera |
| `robot.camera(position=[500,500,500])` | Set camera position/target |
| `robot.snap('iso')` | Snap to preset view |
| `robot.virtual_angles(chi=45)` | Set kappa virtual angles (diffractometers) |

### API Protocol (JSON over WebSocket)

All positions are in mm, angles in degrees, using Z-up robot convention. Most commands accept an optional `"device"` field to target a specific device by name or ID; if omitted, the active device is used.

**Device management:**
```json
{"cmd": "getState"}
{"cmd": "listDevices"}
{"cmd": "getDevice"}
{"cmd": "addDevice", "config": "i16_config.json"}
{"cmd": "removeDevice", "device": "Meca500"}
{"cmd": "renameDevice", "name": "MyRobot"}
{"cmd": "setActiveDevice", "device": "i16"}
{"cmd": "setDeviceOrigin", "position": [100, 0, 0], "rotation": [0, 0, 45]}
{"cmd": "translateDevice", "delta": [10, 0, 0], "space": "parent"}
{"cmd": "rotateDevice", "delta": [0, 0, 45], "space": "local"}
{"cmd": "setDeviceParent", "parent": "dev_0:L3"}
{"cmd": "listConfigs"}
```

**Joint control:**
```json
{"cmd": "setJoints", "angles": [0, -30, 60, 0, 45, 90]}
{"cmd": "setSingleJoint", "index": 1, "angle": -30}
{"cmd": "home"}
{"cmd": "demoPose"}
```

**Hexapod platform control:**
```json
{"cmd": "setPlatformPose", "pose": [0, 0, 5, 5, 0, 10]}
{"cmd": "hexapodFK", "pose": [0, 0, 5, 5, 0, 10]}
{"cmd": "hexapodIK", "legLengths": [150.1, 150.1, 150.1, 150.1, 150.1, 150.1]}
{"cmd": "getLegLengths"}
{"cmd": "setLegLengths", "legLengths": [150.1, 150.1, 150.1, 150.1, 150.1, 150.1]}
```
- `setPlatformPose` — set the platform pose `[x, y, z, rx, ry, rz]` (mm, degrees) and update the visualisation
- `hexapodFK` — forward kinematics: compute leg lengths from a pose (query only, does not move the platform). If `pose` is omitted, uses the current platform pose
- `hexapodIK` — inverse kinematics: compute the platform pose from 6 leg lengths in mm (query only)
- `getLegLengths` — return the current leg lengths for the current platform pose
- `setLegLengths` — solve IK for the given leg lengths and apply the resulting pose

The `getState` response for hexapod devices includes `platformPose`, `legLengths`, and `platformPosition` instead of `joints`.

**Kappa virtual angles** (diffractometer geometry):
```json
{"cmd": "setVirtualAngles", "chi": 45, "theta": 10, "phi": 20}
{"cmd": "getVirtualAngles"}
{"cmd": "setKappaSign", "positive": true}
```

**IK control:**
```json
{"cmd": "setMode", "mode": "IK"}
{"cmd": "setIKTarget", "position": [190, 0, 308], "orientation": [0, 0, 0]}
{"cmd": "moveTo", "position": [150, 100, 300], "orientation": [45, 0, 0]}
```

**Object commands:**
```json
{"cmd": "listObjects"}
{"cmd": "getObject", "object": "MyPart"}
{"cmd": "setObject", "object": "MyPart", "position": [100, 50, 0], "rotation": [0, 0, 45]}
{"cmd": "setObject", "object": "MyPart", "position": [100, 50, 0], "space": "world"}
{"cmd": "setObject", "object": "MyPart", "color": "#ff0000", "parent": "dev_0:L3"}
{"cmd": "translateObject", "name": "MyPart", "delta": [10, 0, 0], "space": "parent"}
{"cmd": "rotateObject", "name": "MyPart", "delta": [0, 0, 45], "space": "local"}
{"cmd": "addPrimitive", "type": "cube"}
{"cmd": "removeObject", "object": "MyPart"}
{"cmd": "duplicateObject", "object": "MyPart"}
{"cmd": "resetObjectRotation", "object": "MyPart"}
{"cmd": "resetObjectScale", "index": 0}
```

`setObject`, `translateObject`, and `rotateObject` accept `"space": "parent"|"local"|"world"` (default: `"parent"`). The `getObject` response includes both local (`position`, `rotation`) and world-frame (`worldPosition`, `worldRotation`) coordinates. `translateDevice` and `rotateDevice` use the same space convention.

**Collision commands:**
```json
{"cmd": "setCollision", "enabled": true}
{"cmd": "getCollisions"}
```

**Visualization toggles:**
```json
{"cmd": "setLabels", "enabled": true}
{"cmd": "setOrigins", "enabled": true}
{"cmd": "setChain", "enabled": true}
{"cmd": "setOrtho", "enabled": true}
```

**Camera control:**
```json
{"cmd": "getCamera"}
{"cmd": "setCamera", "position": [500, 500, 500], "target": [0, 0, 150]}
{"cmd": "snapCamera", "view": "front"}
```
Snap views: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`, `top`, `bottom`, `front`, `back`, `left`, `right`, `iso`.

**Scene persistence:**
```json
{"cmd": "getSceneState"}
{"cmd": "saveScene"}
{"cmd": "help"}
```

**State response (serial robot):**
```json
{
  "type": "state",
  "device": "Meca500",
  "joints": [0, -30, 60, 0, 45, 90],
  "eePosition": [190.0, 0.0, 308.0],
  "eeOrientation": [0.0, 0.0, 0.0],
  "mode": "FK",
  "ikError": null,
  "collisionEnabled": true,
  "collision": false,
  "collisions": [],
  "chi": 45.0
}
```

The `chi` field is only present for kappa-geometry devices (diffractometers).

**State response (hexapod):**
```json
{
  "type": "state",
  "device": "Hexapod",
  "deviceType": "hexapod",
  "platformPose": [0, 0, 5, 5, 0, 10],
  "legLengths": [150.12, 149.88, 150.34, 149.66, 150.21, 149.79],
  "platformPosition": [0.0, 0.0, 205.0],
  "collisionEnabled": false,
  "collision": false,
  "collisions": []
}
```

**Hexapod FK response:**
```json
{
  "type": "hexapodFK",
  "device": "Hexapod",
  "pose": [0, 0, 5, 5, 0, 10],
  "legLengths": [150.12, 149.88, 150.34, 149.66, 150.21, 149.79]
}
```

**Hexapod IK response:**
```json
{
  "type": "hexapodIK",
  "device": "Hexapod",
  "pose": [0.0, 0.0, 5.0, 5.0, 0.0, 10.0],
  "legLengths": [150.12, 149.88, 150.34, 149.66, 150.21, 149.79]
}
```

### Custom Client Example (Python)

```python
import asyncio, json, websockets

async def main():
    # Target a specific viewer tab by session ID (omit ?session=... to broadcast to all)
    async with websockets.connect("ws://localhost:8080/ws?session=ab12cd34") as ws:
        # Set joint angles
        await ws.send(json.dumps({"cmd": "setJoints", "angles": [0, -30, 60, 0, 45, 90]}))
        state = json.loads(await ws.recv())
        print(state["eePosition"])

        # Set a single joint by index
        await ws.send(json.dumps({"cmd": "setSingleJoint", "index": 1, "angle": -45}))

        # Multi-device: add a second device and position it
        await ws.send(json.dumps({"cmd": "addDevice", "config": "i16_config.json"}))
        await ws.recv()
        await ws.send(json.dumps({"cmd": "setDeviceOrigin", "device": "i16", "position": [500, 0, 0]}))

        # Camera: snap to front view
        await ws.send(json.dumps({"cmd": "snapCamera", "view": "front"}))

        # Kappa virtual angles (diffractometer only)
        await ws.send(json.dumps({"cmd": "setActiveDevice", "device": "i16"}))
        await ws.recv()
        await ws.send(json.dumps({"cmd": "setVirtualAngles", "chi": 45, "theta": 10}))

        # Translate a device in world coordinates
        await ws.send(json.dumps({"cmd": "translateDevice", "delta": [100, 0, 0], "space": "world"}))

        # Move an object in world space
        await ws.send(json.dumps({"cmd": "translateObject", "name": "Cube", "delta": [50, 0, 0], "space": "world"}))

        # Enable collision detection
        await ws.send(json.dumps({"cmd": "setCollision", "enabled": True}))

asyncio.run(main())
```

## Deployment

### Docker

```bash
docker build -t robot-visualisation .
docker run -p 8080:8080 robot-visualisation
```

### Kubernetes (Helm)

```bash
helm install robot-vis ./helm
```

See `helm/values.yaml` for configuration.

## Project Structure

```
threejs_scene.html       HTML shell — loads viewer.css and js/main.js
viewer.css               All viewer styles
js/
  main.js                Entry point — animate loop, event handlers, initialisation
  state.js               Shared mutable state (scene, cameras, controls, devices)
  scene.js               Three.js scene setup, cameras, lights, ground, nav gizmo
  device.js              Device loading, GLB import, slider/IK sync
  hexapod.js             Hexapod loader, Damped Track IK, FK solver, platform sync
  kinematics.js          FK, IK solver, kappa geometry math
  panel.js               Control panel UI, device list, parent dropdowns
  stl.js                 Mesh import/export, primitives, point-cloud/splat add, eraser/clip shaders
  lidar.js               LiDAR Workflow panel — generate, library, edit, boxes, eraser, save/workflow
  collision.js           BVH-accelerated collision detection (Web Worker + main-thread fallback)
  collision-worker.js    Background thread for collision math
  websocket.js           WebSocket client for remote control API
lidar_jobs.py            LiDAR /api/* endpoints — generation jobs, edit/apply, save_as, outputs
process_pointcloud.py    Scan bags → registered, coloured point cloud (KISS-ICP + multi-view colour)
process_splat.py         Splat generation — surfel / GPU-trained / CPU bootstrap (with --splat-size)
edit_ops.py              Unified edit ops (decimate/denoise/crop/erase/drop/transform) for clouds + splats
cloud_ops.py             open3d point-cloud helpers (SOR, oriented-box keep, primitive erase masks)
splat_io.py              Format-preserving splat PLY load/save + transform (bake) of gaussians
import_robot.py          Blender import script — extracts serial robot armature to config JSON + GLB
import_hexapod.py        Blender import script — extracts hexapod (Damped Track legs) to config JSON + GLB
server.py                WebSocket + HTTP server for remote control API
robot_ipython.py         IPython remote control client (any device)
GNKinematics/            Python forward/inverse kinematics library (matches viewer's ZYX Euler)
RobotDefinitions.py      Robot DH / geometry parameters for GNKinematics
RemoteAPI.zip            Bundled client (ipython client + GNKinematics + RobotDefinitions); served from viewer
meca500_config.json      Meca500 R3 device config
i16_config.json          i16 diffractometer device config
i19_config.json          i19 kappa diffractometer device config
gp225_config.json        Yaskawa GP225 device config
gp280_config.json        Yaskawa GP280 device config
gp180_config.json        Yaskawa GP180-120 device config
robot_scene.glb          Meca500 GLB model
i16_scene.glb            i16 diffractometer GLB model
i19_scene.glb            i19 kappa diffractometer GLB model
gp225_scene.glb          Yaskawa GP225 GLB model
gp280_scene.glb          Yaskawa GP280 GLB model
gp180_scene.glb          Yaskawa GP180-120 GLB model
hexapod_config.json      Hexapod Stewart platform device config
hexapod_scene.glb        Hexapod GLB model
Dockerfile               Multi-stage container build
helm/                    Kubernetes Helm chart
```

## IK Solver (Serial Robots)

The viewer uses a 6×N geometric Jacobian with damped least-squares (DLS):

- **Position error**: difference between target and end-effector world position
- **Orientation error**: rotation vector from quaternion error (target × current⁻¹)
- **Convention**: ZYX Euler angles (alpha=Rz, beta=Ry, gamma=Rx)
- Orientation is weighted at 0.3× relative to position to prioritise reach accuracy
- For N < 6 joints: underdetermined for full 6-DOF; for N > 6: redundancy handled naturally by DLS

## Hexapod Kinematics

The hexapod uses two kinematics approaches:

**Leg IK (Damped Track)**: given the platform pose [x, y, z, rx, ry, rz], each leg's lower and upper segments are oriented to track toward their respective pivot points using `Quaternion.setFromUnitVectors()` — the Three.js equivalent of Blender's Damped Track constraint. This runs every frame and is the primary visual update.

**Platform FK (Newton-Raphson)**: given 6 target leg lengths, solves for the platform pose that produces those lengths. Uses a numerical 6×6 Jacobian (central finite differences) and Gaussian elimination with partial pivoting. Convergence is typically 3–5 iterations from a nearby starting pose. The solver clamps the result to the config's platform limits. This drives the leg length sliders in the UI — dragging a leg length slider triggers FK, which updates the pose and all other sliders bidirectionally.
