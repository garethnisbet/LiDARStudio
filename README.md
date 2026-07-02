# LidarStudio

A browser-based studio for **generating and editing LiDAR point clouds and 3D Gaussian splats**, built on a Three.js + GaussianSplats3D viewer with a Python backend. Generate clouds/splats from raw scan bags, then clean, crop, recolour, erase, and re-place them interactively — all non-destructively, with results saved back into your project.

> **Note:** the LiDAR features are driven by a Python backend (`lidar_jobs.py`, `process_pointcloud.py`, `process_splat.py`, `edit_ops.py`, `cloud_ops.py`, `splat_io.py`) wired into `server.py`. The browser JS is served fresh on every reload, but the Python modules are imported once and cached — **restart `server.py` after any `.py` change** for it to take effect.

## Quick Start

```bash
pip3 install aiohttp numpy plyfile open3d scipy   # plus torch + gsplat for GPU-trained splats
python3 server.py
```

Open `http://localhost:8080` and use the **LiDAR Workflow** panel.

> **Network exposure:** the server binds `127.0.0.1` by default. Pass `--host 0.0.0.0` to serve the viewer to other devices (e.g. a VR headset) — the LiDAR `/api/*` endpoints, which browse and read/write the local filesystem, stay loopback-only even then unless you also pass `--allow-remote-fs`. Static serving is allowlisted (`js/`, `node_modules/`, and specific root-level asset types); the Python source, `.git`, and dotfiles are never served.

## LiDAR Workflow

The **LiDAR Workflow** panel (top of the right-hand control panel) drives the whole pipeline.

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

## Viewer & Editor Engine

The scene around the LiDAR tools is a general-purpose 3D editor:

- **Mesh import** — load STL, OBJ (+MTL), PLY, GLB/GLTF, and splat (`.ply`, `.splat`, `.ksplat`, `.spz`) files via the Import button or drag-and-drop, with auto-scaling, per-object colour and opacity.
- **Primitives** — add cube / sphere / cylinder objects (also used as eraser volumes by the LiDAR tools).
- **Object manipulation** — click to select, then move/rotate/scale with transform gizmos (keyboard **T/R/S**, **Esc** to deselect); World/Local gizmo space toggle; numeric X/Y/Z (mm), Rx/Ry/Rz (deg) and scale inputs synced live with the gizmo; lockable aspect ratio.
- **Object list** — per-object visibility, colour, opacity, rename, duplicate, and remove.
- **Scene save/load** — save the scene (objects + camera + floor) to a JSON file and reload it later; the scene also auto-saves to IndexedDB every 30 s. Clouds/splats are loaded explicitly from the Library — the page starts empty by design.
- **Camera** — orbit controls, Blender-style navigation gizmo with axis snapping, orthographic toggle, auto-orbit ("Fly Around"), splat foreground clip slider.
- **Rendering** — on-demand render loop (draws only when something changes), BVH-accelerated raycasting, studio lighting, shader floor grid with adjustable size.
- **Screenshot** — one-click PNG capture of the WebGL view composited with the control panel.
- **VR (WebXR)** — enter VR/AR on a headset (e.g. Meta Quest): grab and place objects, teleport locomotion, snap turn, passthrough toggle, in-headset control panel, persistent room anchors so the scene stays put between sessions.

## Architecture

**Client (browser, `js/`)** — all rendering and real-time interaction: Three.js scene, gizmos, live eraser and visibility-box clipping, splat rendering (depth-sorted in a worker via `SharedArrayBuffer` — the server sends COOP/COEP headers to enable this), WebXR.

**Server (Python)** — everything that touches files or needs numpy/CUDA:

| Module | Role |
|--------|------|
| `server.py` | aiohttp server: static assets (allowlisted), LiDAR API mounting, loopback guard |
| `lidar_jobs.py` | `/api/*` handlers: projects, scan validation, job orchestration + SSE progress, file proxy |
| `process_pointcloud.py` | bag reading, IMU deskew, KISS-ICP registration, photo colouring (run as subprocess per job) |
| `process_splat.py` | surfel / trained (CUDA) / bootstrap splat generation |
| `edit_ops.py`, `cloud_ops.py`, `splat_io.py` | decimate, SOR, crop, recolour, transform bake, PLY I/O, pose sidecars |

The browser calls the backend over REST + Server-Sent Events; there is no other coupling, so the viewer also works as a plain static page (without generation/editing).

## Project Structure

```
server.py                aiohttp server (static + LiDAR API)
lidar_jobs.py            LiDAR workflow API handlers
process_pointcloud.py    bag → coloured, registered point cloud
process_splat.py         cloud → gaussian splat (surfel / trained / bootstrap)
edit_ops.py              edit operations (decimate, SOR, crop, recolour, bake)
cloud_ops.py             point-cloud helpers (open3d)
splat_io.py              splat PLY read/write
threejs_scene.html       the app page
viewer.css               styles
js/
  main.js                init, event handlers, render loop
  scene.js               scene, cameras, lights, floor, nav gizmo
  state.js               shared mutable state
  stl.js                 import, object list, selection, scene save/load
  lidar.js               LiDAR workflow panel + editing UI
  storage.js             IndexedDB persistence
  vr.js                  WebXR support
Dockerfile, helm/        container + k8s deployment
```

## Deployment

**Docker** — the image serves the viewer and the editing API (`numpy` + `plyfile`); the heavy generation dependencies (open3d, kiss-icp, torch/gsplat) are intentionally not included — generation runs on a workstation with the scan data.

```bash
docker build -t lidarstudio .
docker run -p 8080:8080 lidarstudio
```

**Helm** — `helm/` contains a chart (deployment + service + ingress). Set `image.repository` and `ingress.host` in `values.yaml` for your cluster.
