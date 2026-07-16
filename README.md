[![CI](https://github.com/garethnisbet/LiDARStudio/actions/workflows/ci.yml/badge.svg)](https://github.com/garethnisbet/LiDARStudio/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# LidarStudio

A browser-based studio for **generating and editing LiDAR point clouds and 3D Gaussian splats**, built on a Three.js + GaussianSplats3D viewer with a Python backend. Generate clouds/splats from raw scan bags, then clean, crop, recolour, erase, and re-place them interactively — all non-destructively, with results saved back into your project.

> **Note:** the LiDAR features are driven by the Python backend (`src/lidarstudio/`). The browser JS is served fresh on every reload, but the Python modules are imported once per process — **restart the server after any backend `.py` change** for it to take effect.

## Quick Start

The Python side follows the [DLS python-copier-template](https://github.com/DiamondLightSource/python-copier-template) layout and is managed with [uv](https://docs.astral.sh/uv/):

```bash
npm install                              # front-end deps (three.js, GaussianSplats3D)
uv sync                                  # create .venv and install lidarstudio + dev tools
uv run lidarstudio                       # start the server
```

Generation needs the pipeline extra (`uv sync --extra pipeline` — open3d, kiss-icp, rosbags, OpenCV, scipy; the trained-splat mode additionally needs CUDA + torch + gsplat). Editing/serving alone needs no extras.

Open `http://localhost:8080` and use the **LiDAR Workflow** panel.

> **Low on home-directory space?** `source scratch-env.sh` before installing to redirect the uv/npm/pip caches, the downloaded Python, and the `.venv` onto scratch (`/scratch/$USER` by default, or set `SCRATCH` yourself) — nothing but a few dotfiles then touches `$HOME`. Source it in every shell where you use the project. Note that scratch is usually auto-purged, so re-run `uv sync` if the venv disappears, and keep generated clouds/splats somewhere persistent. If `node_modules/` is also a problem, clone the repo onto scratch too.

> **Network exposure:** the server binds `127.0.0.1` by default. Pass `--host 0.0.0.0` to serve the viewer to other devices (e.g. a VR headset) — the LiDAR `/api/*` endpoints, which browse and read/write the local filesystem, stay loopback-only even then unless you also pass `--allow-remote-fs`. Static serving is allowlisted (`js/`, `node_modules/`, and specific root-level asset types); the Python source, `.git`, and dotfiles are never served.

## LiDAR Workflow

The **LiDAR Workflow** panel (top of the right-hand control panel) drives the whole pipeline.

### Generate

Turn a raw scan folder (LiDAR + IMU + image `.bag` files) into a cloud, splat, or mesh:

- **Point cloud** — KISS-ICP registration + multi-view photo colouring, voxel-downsampled.
- **Splat** — GPU-trained 3D Gaussian Splatting (needs CUDA + torch/gsplat), seeded from the coloured cloud. A draft→max **quality** slider drives resolution / iterations / gaussian cap; expert overrides (anisotropy cap, render-patch size, SfM poses) sit below it. Full-resolution runs auto-generate SfM camera poses (COLMAP/GLOMAP, sim(3)-aligned to the LiDAR trajectory). Training renders **full frames** with an **annealed position LR** and **antialiased (Mip-Splatting) rasterisation** — on the reference scan this recipe reaches 26.7 dB train-view PSNR, at parity with commercial reconstructions of the same scanner class. An **ultra** preset adds a LiDAR-anchored monocular depth prior (Depth Anything V2) for slightly crisper fine detail — budget for a much longer run: the per-frame prior computation alone takes hours (~6 h on the reference scan) before training starts, plus a ~30 GB temporary cache.
- **Mesh** — screened-Poisson surface reconstruction from the dense cloud, photo-coloured (best for rooms and flat surfaces; thin structures stay the splat's job).

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

| Module (`src/lidarstudio/`) | Role |
|--------|------|
| `server.py` | aiohttp server: static assets (allowlisted), LiDAR API mounting, loopback guard |
| `lidar_jobs.py` | `/api/*` handlers: projects, scan validation, job orchestration + SSE progress, file proxy |
| `process_pointcloud.py` | bag reading, IMU deskew, KISS-ICP registration, photo colouring (run as subprocess per job) |
| `process_splat.py` | GPU 3DGS training (gsplat/CUDA): full-frame rendering, MCMC densification, annealed position LR, antialiased rasterisation |
| `process_sfm.py` | COLMAP/GLOMAP camera poses from the scan photos, sim(3)-aligned to the LiDAR trajectory (auto-run for full-res splats) |
| `process_mesh.py` | screened-Poisson mesh from the dense cloud, photo-coloured |
| `edit_ops.py`, `cloud_ops.py`, `splat_io.py` | decimate, SOR, crop, recolour, transform bake, PLY I/O, pose sidecars |

The browser calls the backend over REST + Server-Sent Events; there is no other coupling, so the viewer also works as a plain static page (without generation/editing).

## Project Structure

```
pyproject.toml           packaging, deps, ruff/pyright/pytest/tox config (DLS template)
src/lidarstudio/
  __main__.py            `python -m lidarstudio` / `lidarstudio` entry point
  server.py              aiohttp server (static + LiDAR API)
  lidar_jobs.py          LiDAR workflow API handlers
  process_pointcloud.py  bag → coloured, registered point cloud
  process_splat.py       cloud → GPU-trained gaussian splat (gsplat/CUDA)
  process_sfm.py         scan photos → SfM camera poses (COLMAP/GLOMAP)
  process_mesh.py        dense cloud → photo-coloured Poisson mesh
  edit_ops.py            edit operations (decimate, SOR, crop, recolour, bake)
  cloud_ops.py           point-cloud helpers (open3d)
  splat_io.py            splat PLY read/write
tests/                   pytest suite
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
.github/, .devcontainer/ CI workflows + dev container (from the DLS template)
```

## Development

```bash
uv sync            # dev environment (.venv)
uv run tox -p      # run everything CI runs: pre-commit (ruff), pyright, pytest
```

The repo was generated against the [DLS python-copier-template](https://github.com/DiamondLightSource/python-copier-template); pull future template updates with `uvx copier update`.

## Deployment

**Docker** — the image serves the viewer and the editing API (`numpy` + `plyfile`); the heavy generation dependencies (open3d, kiss-icp, torch/gsplat) are intentionally not included — generation runs on a workstation with the scan data.

```bash
docker build -t lidarstudio .
docker run -p 8080:8080 lidarstudio
```

**Helm** — `helm/` contains a chart (deployment + service + ingress). Set `image.repository` and `ingress.host` in `values.yaml` for your cluster.
