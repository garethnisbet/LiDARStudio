# Client-Side Migration Plan

Migrate LidarStudio's heavy lifting from the Python backend to the browser, stage by stage. The end state: the server shrinks to a static file host, and all point-cloud/splat processing runs client-side in Web Workers / WASM / WebGPU.

**Usage:** `/client-side-migration [stage]` — run one stage at a time, in order. With no argument, determine the next incomplete stage (check the criteria at the end of each stage) and propose it before starting.

## Context (verified 2026-07-02)

- The page is already **cross-origin isolated** (`server.py` sends COOP/COEP headers), so `SharedArrayBuffer`, multithreaded WASM, and WebGPU compute are available now.
- The browser already does real-time point editing (live eraser, visibility-box GPU clip in `js/lidar.js`) — the migration extends that pattern to persistence and generation.
- Server pieces being replaced (all under `src/lidarstudio/`): `lidar_jobs.py` (API + file proxy), `edit_ops.py`/`cloud_ops.py`/`splat_io.py` (edit ops + PLY I/O), `process_pointcloud.py` (bag → cloud), `process_splat.py` (cloud → splat).
- Keep the Python pipeline working in parallel throughout — it remains the headless/batch path and the fallback for non-Chromium browsers and VR headsets (weak clients want server-side processing; consider a "process locally / on server" toggle rather than deleting the server path).

## Stage 1 — Edit ops + Save in the browser (do this first; proves the pattern)

- Move decimate, crop (oriented box), transform-bake, and eraser persistence into a Web Worker operating on typed arrays (port the logic from `edit_ops.py`; the eraser volumes already exist client-side).
- Write results with the **File System Access API**: `showDirectoryPicker()` on project open, persist the handle (IndexedDB), write `*_edited.ply` + `.pose.json` sidecars directly. Fallback for non-Chromium: download the file.
- PLY read/write in JS (typed-array slicing; GaussianSplats3D already parses PLY for rendering — reuse or mirror its layout handling, including splat PLY fields from `splat_io.py`).
- Keep `/api/edit/apply` as fallback; UI prefers the local path when the API is available in the browser.
- **Done when:** an edit → save → reload round-trip works with the server stopped after initial page load (except static serving), and output PLYs are byte-compatible with `edit_ops.py` output for the same op (write a comparison script).

## Stage 2 — Library + file access without the proxy

- List `pointclouds/` and `splats/` from the directory handle instead of `/api/project/outputs`; load files as `File` objects instead of `/api/scan/file`.
- Project create/open (`project.json`, subfolders) via directory handles — replaces the tkinter browse dialog entirely on Chromium.
- **Done when:** full library browse/load/save works with `/api/*` unreachable.

## Stage 3 — SOR denoise + recolour client-side

- SOR: voxel-hash grid k-NN in a worker (or WebGPU compute); port parameters (neighbours, std ratio) from `edit_ops.py` and validate outputs against it on a reference cloud.
- Recolour: decode scan photos with `createImageBitmap()`, project points into images on GPU (WebGL/WebGPU pass) using the saved trajectory — port the projection math from `edit_ops.recolour` / `process_pointcloud.py` colouring.
- **Done when:** results match the Python versions within tolerance on a reference scan.

## Stage 4 — Point-cloud generation in the browser (the big lift)

- Bag parsing: `@foxglove/rosbag` streaming reader in a worker (never load whole bags; multi-GB inputs, WASM 4 GB cap).
- Port IMU deskew + voxel downsample (typed-array math, straightforward from `process_pointcloud.py`).
- **KISS-ICP**: compile the kiss-icp C++ core (Eigen + voxel hash map, small and portable) to WASM with Emscripten, threaded build (isolation already in place). This is the long pole — spike it before committing to the rest of the stage. Fallback if the port stalls: point-to-plane ICP + voxel grid implemented directly in WASM/WebGPU.
- Photo colouring: reuse Stage 3's GPU projection.
- **Done when:** a reference scan produces a cloud comparable to the Python pipeline (registration error within tolerance) at acceptable speed (~2–3× native is expected).

## Stage 5 — Splat generation

- Surfel mode: k-NN (reuse Stage 3 grid) + per-point 3×3 PCA eigendecomposition in WASM/worker; port from `process_splat.py`.
- Bootstrap mode: trivial port (one gaussian per point).
- Trained mode: evaluate **Brush** (Rust/WGPU in-browser 3DGS trainer) — if unsuitable, keep trained mode as the one documented server-side exception.

## Cross-cutting

- Every stage: heavy work in workers, never the main thread; stream progress to the existing panel UI.
- Static hosting end-state needs COOP/COEP — if moving off `server.py`, use the `coi-serviceworker` shim.
- Feature-detect: File System Access API and WebGPU are Chromium(+recent Safari for WebGPU) — degrade to the server path elsewhere, and default VR headsets to the server path.
- Update README (client/server split section) as each stage lands.
