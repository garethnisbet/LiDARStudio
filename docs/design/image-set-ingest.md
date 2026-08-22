# Design: `image_set` ingest — photogrammetry from phone photos

**Status:** implemented, phases 0-3 (2026-08-21). Phases 4-5 remain open.
**Date:** 2026-08-21
**Scope:** add a second scan source (a folder of phone photos or a video) alongside the existing LiDAR bag source, producing point clouds and splats through the pipeline that already exists.

---

## 0. Phase 0 results — what the probe changed

Run against the interpreter `_sfm_python()` locates (`/usr/bin/python3`,
pycolmap **4.0.4**). Two assumptions in this document were wrong and the
implementation differs accordingly:

- **`refine_focal_length` is not on the mapper options.** It lives at
  `GlobalPipelineOptions.mapper.bundle_adjustment.refine_focal_length` for
  GLOMAP, and as `IncrementalPipelineOptions.ba_refine_focal_length` for the
  incremental fallback. Both are set (§4.5) only when the camera is
  uncalibrated.
- **`pycolmap.has_cuda is False` on this build.** Dense MVS (§7.5) is therefore
  *not available here at all*, which settles risk 2: the sparse SfM cloud is the
  only seed the reconstruction itself can produce. Meshing does not wait on MVS —
  it takes its density from the trained splat instead (§4.12).

Confirmed as assumed: `match_exhaustive`, `match_vocabtree`, `match_spatial`
and `match_image_pairs` all exist; `SIMPLE_RADIAL` is available.

One thing the probe added that this document had not anticipated:
**pycolmap ships no vocabulary tree** — `VocabTreePairingOptions.vocab_tree_path`
defaults to `.`. Vocab-tree matching is therefore never selected automatically
(§4.5.2); large unordered sets fall back to sequential+quadratic, and a user
who has downloaded a tree can pass `--matching vocabtree --vocab-tree <path>`.

Verified end-to-end on a synthetic 60-frame room-corner capture with known
camera poses: all 60 frames registered, camera centres recovered to 0.57% of
scene extent after sim(3) alignment to ground truth, and the solved focal
length converged to 699.4 px against a true 700.0 px. A splat trained from that
capture reached 27.0 dB in 800 iterations.

---

## 1. Context

LidarStudio currently accepts exactly one kind of input: a scan folder containing
`LIDAR_*.bag`, `IMAGE_*.bag`, and a `calibration/` directory from a 3DMakerpro-class
scanner. Everything from `_scan_contents()` onward assumes that shape.

But the back half of the pipeline is already source-agnostic, and one module in
particular is already a photogrammetry engine:

- **`process_sfm.py`** runs pycolmap SIFT extraction, sequential + loop-pair matching,
  and GLOMAP global mapping. It exists to *refine* LiDAR poses, but the SfM itself
  never touches LiDAR data — only stage C (`align_to_lidar`) does.
- **`process_splat.py:232`** already accepts an `sfm_poses` npz that **replaces** every
  frame's odometry pose. When that argument is supplied, the LiDAR trajectory is
  bypassed entirely and frames without an SfM pose are dropped
  (`process_splat.py:333`).
- The viewer, Library, editing ops (`edit_ops.py`, `cloud_ops.py`), pose sidecars,
  eraser, Save As, and VR all operate on PLY files and know nothing about bags.

So the work is not "build photogrammetry". It is "let the front of the pipeline
accept a different input and produce the same three artifacts the LiDAR path
produces".

### Why in this repo rather than a new one

Forking would duplicate ~11k lines of viewer JS plus the editing and export stack,
and then watch the copy rot. The dependency argument for a split is real but is
solved by an optional extra, the same way the existing `pipeline` extra works.
The only genuine cost is that the product name stops describing the whole product.

---

## 2. Goals and non-goals

**Goals**

1. Point the app at a folder of phone photos (or a phone video) and get a splat.
2. Reuse `process_sfm.py`, `process_splat.py`, the Library, and every editing tool
   without forking them.
3. Keep the LiDAR path byte-for-byte unchanged in behaviour.
4. Let a photogrammetry-only user install without kiss-icp, rosbags, or open3d.

**Non-goals (this design)**

- Fusing a phone capture and a LiDAR capture into one reconstruction.
- Metric scale from phone sensors (ARKit/ARCore pose export). Section 8 keeps the
  hook open; the work is out of scope here.
- Meshing a phone capture *directly from SfM points*. The mesh route goes
  through the trained splat instead (§4.12); a dense-MVS seed stays deferred (§7.5).

---

## 3. Core design: converge on the existing artifacts

The image_set path produces the **same three artifacts** as the LiDAR path, which is
what lets everything downstream stay untouched.

| Artifact | LiDAR source | image_set source |
| --- | --- | --- |
| `sfm/sfm_viewmats_<id>.npz` | `process_sfm.py` + Umeyama alignment to the LiDAR trajectory | `process_sfm.py`, stage C skipped |
| `pointclouds/pointcloud_<id>.ply` | KISS-ICP registration + multi-view photo colouring | SfM sparse points (later: MVS dense) |
| Iterable posed frames | `ppc.read_images(IMAGE_*.bag)` | normalized `images/` directory + manifest |

One useful consequence: full-resolution training currently **requires** SfM poses
because odometry poses carry ~0.7° of error (`lidar_jobs.py:929`). For image sets,
SfM poses are the *only* poses — so the image_set path lands directly on the
already-proven high-quality configuration, with no odometry fallback to support.

---

## 4. Component changes

### 4.1 Source detection — `_scan_contents()` (`lidar_jobs.py:142`)

Add a `source` discriminator to the returned dict:

```python
"source": "lidar_bag" | "image_set" | None,
"images": {"count": N, "dir": ..., "formats": [...], "video": ..., "sparse": bool},
```

Detection rule: no `LIDAR_*.bag` present, **and** ≥4 images (`.jpg .jpeg .png .heic`)
either at the top level or inside a single subdirectory (`images/`, `DCIM/`, or the
sole subdirectory present) → `image_set`. A lone `.mp4`/`.mov` also qualifies.

Two thresholds, not one: 4 is geometry's hard floor (`_MIN_IMAGES`), 20 is the
count a capture *should* have (`_GOOD_IMAGES` / `REC_FRAMES`). Between them the
set is detected, previewable and ingestable, with `"sparse": true` and an
ingest warning — a 12-photo orbit of a small object often does reconstruct, and
a folder that silently fails to be recognised is worse than one that warns.

`valid` changes from `bool(image_bag and lidar_bag)` to `source is not None`.

Care needed: the existing loop puts unrecognised entries into `extras`, and a
`thumbnail/` directory already exists on LiDAR scans — exclude it from the image count.

**Video is the better phone capture mode** and should be treated as first-class, not a
fallback: it gives denser coverage and guarantees constant intrinsics across frames.

### 4.2 New module: `process_ingest.py`

A normalization pass, run as its own job so it can be inspected before an hour of SIFT.

Output goes **into the project**, never the source folder — a mounted phone or a
camera roll may be read-only:

```
<project>/scans/<name>/images/NNNN.jpg
<project>/scans/<name>/manifest.csv    # name, key_ns, source_file, exif_focal, exif_ts
```

Steps:

1. **Decode** — HEIC via `pillow-heif`; video via cv2 at a configurable fps.
2. **Apply EXIF orientation** — `cv2.imread` silently ignores it, which would otherwise
   feed SfM a set of images with inconsistent rotation.
3. **Downscale** to a max long edge (~2000–3000 px). 48 MP phone stills add nothing to
   SfM accuracy and dominate memory.
4. **Sharpness filter** — Laplacian variance, drop the worst decile. The same measure
   already used at `process_splat.py:420`.
5. **Write** sequential `NNNN.jpg` plus the manifest.

Two conditions are surfaced as warnings rather than silent failures:

- **Mixed EXIF focal lengths** (ultrawide + main lens in one capture) break
  `CameraMode.SINGLE` at `process_sfm.py:115`. Needs per-lens camera groups.
- **Too few images, or weak overlap** — tell the user before they burn an hour, not after.

Progress is emitted as `PROGRESS:<pct>:<msg>` lines, so `_stream_proc`
(`lidar_jobs.py:449`) streams it into the existing panel with no frontend work.

### 4.3 New module: `frame_source.py`

```python
def open_frames(scan: Path, camera: str, rot: str) -> Iterator[tuple[int, np.ndarray]]
```

Dispatches to `ppc.read_images` for bags, or to the manifest for image sets.

The yielded `key_ns` is the **join key** between the poses npz and the trainer.
For image sets it must come from the **persisted manifest**, not from enumeration
order — otherwise a single frame dropped between the SfM run and the training run
silently shifts every subsequent pose association. `process_sfm.py:86` already writes
exactly this kind of manifest (`timestamps.csv`); this generalizes it and makes it the
contract rather than an internal detail.

### 4.4 Job wiring

Add `"ingest"` to the type allowlist at `lidar_jobs.py:397` and an `_job_ingest`
branch in `_run_job` (`lidar_jobs.py:415`).

### 4.5 SfM without LiDAR — `process_sfm.py`

Three guarded changes:

1. **`fisheye_prior()` → `camera_prior()`** — returns `("OPENCV_FISHEYE", params)` when
   `calibration/` exists, else `("SIMPLE_RADIAL", None)` so COLMAP seeds the focal
   length from EXIF. Enable `refine_focal_length` and `refine_extra_params` in the
   mapper options for the image_set case; the LiDAR case keeps its factory prior.
2. **Matching strategy becomes a parameter** of `extract_and_match()`:
   - sequential (overlap 20, as today) for video and ordered captures
   - exhaustive for < 300 unordered photos
   - vocab-tree above that
3. **`--traj` becomes optional** in `main()`. When absent, skip `align_to_lidar` and
   export poses in the SfM's own frame.

**⚠️ Loop closure needs rescaling.** `build_loop_pairs` (`process_sfm.py:270`) itself
works fine on unaligned poses — it reads only camera centres and optical axes, and
derives the frame index from `int(n.split(".")[0])`, which the `NNNN.jpg` naming
preserves. But `MAX_DIST_M = 2.0` and the 0.05 m floor inside `align_to_lidar` are
**metric constants**, and an unscaled SfM frame has no metres. Either normalize the
thresholds by the median inter-camera distance, or gate stage D off until scale is
established (§8).

**npz contract change** — add `camera_model` and `camera_params`. Keep the existing
`fisheye_params` key so previously generated files still load.

### 4.6 Seed cloud

`process_splat.py` requires a coloured seed cloud (`load_coloured_ply`, `--pointcloud`).

Export `rec.points3D` from the reconstruction — each point carries an RGB colour — as
`pointclouds/pointcloud_<id>.ply`. Typically 50k–500k points, thin next to LiDAR's
millions, but MCMC densification grows from it. Using the existing naming convention
means `_list_outputs()` (`lidar_jobs.py:213`) picks it up with no changes.

Dense MVS is the better seed and is deferred to §7.5.

### 4.7 Training — `build_camera_frames` (`process_splat.py:222`)

Four branches on source type:

| Concern | Current (line) | image_set |
| --- | --- | --- |
| Frame iteration | `IMAGE_*.bag` glob (`:306`) | `frame_source.open_frames` |
| Intrinsics | `ppc.load_calibration` (`:272`) | npz `camera_model` / `camera_params` |
| Undistortion | `cv2.fisheye.initUndistortRectifyMap` (`:356`) | `cv2.initUndistortRectifyMap` (pinhole radial) |
| Pose source | `T @ inv(P)` from trajectory (`:380`) | `sfm_vm` only |

`traj_path` and the `T` extrinsic become unused on this path — make `traj_path`
optional and require `sfm_poses`.

### 4.8 `_job_splat` (`lidar_jobs.py:887`)

- The output stem derives from `contents["lidar_bag"]` (`:904`) — needs an image_set
  fallback for naming.
- The auto-resolve block at `:1045` tries to generate poses by aligning to a
  `.traj.npz` sidecar. For image sets this must be bypassed: the poses either already
  exist from the ingest+SfM run, or the job cannot run at all.

### 4.9 Features disabled on image sets

Each needs an explicit, readable message rather than an obscure downstream failure:

- `--mono` and `edit_ops.recolour` — both need a trajectory sidecar.
- The **mesh job on the SfM cloud** — screened-Poisson on sparse SfM points produces
  garbage. It runs on the splat instead (§4.12).
- Panel previews `_scan_photo_index` (`lidar_jobs.py:1492`) and `scan_sweeps`
  (`:1590`) — both are bag readers. The sweep preview has no image_set analogue; the
  photo preview should be reimplemented against the manifest.
- `cam_time_offset` and IMU levelling — no second clock, no IMU.

### 4.10 Interpreter probing — `_sfm_python()` (`lidar_jobs.py:855`)

Currently probes `import pycolmap, rosbags, scipy, cv2`. A photogrammetry-only install
has no `rosbags`, so this would reject a perfectly good interpreter. Split into two
capability probes and record which the located interpreter satisfies.

### 4.11 Packaging

```toml
photogrammetry = [
    "pycolmap",
    "opencv-python-headless",
    "pillow",
    "pillow-heif",
    "numpy",
    "scipy",
]
```

No kiss-icp, rosbags, or open3d — a phone-only user installs a substantially
lighter stack than `pipeline`.

---

### 4.12 Meshing a phone capture: Poisson on the splat

Screened Poisson needs dense, normal-bearing geometry. Without MVS the reconstruction
gives only a sparse SfM cloud — but the **trained splat** is dense: training migrates
the gaussian means onto real surfaces, densifies thin structure and prunes whatever no
photo supports. Exporting the centres (`edit_ops.splat_to_cloud`, already built for the
LiDAR path) turns the splat into a cloud Poisson can eat.

Three substitutions make `process_mesh.py` run without any LiDAR artifact:

| LiDAR route | Phone route |
| --- | --- |
| `.traj.npz` trajectory | `--sfm-poses sfm/sfm_viewmats_<ts>.npz` (world→cam viewmats inverted to camera centres) |
| photo re-projection from `IMAGE_*.bag` | the cloud's own per-point RGB, from the splat's SH DC term |
| metric, gravity-levelled frame | the SfM frame: **unitless and unlevelled** |

Poses are used for one thing only: orienting normals toward the camera that saw each
point. That inherits the LiDAR route's known limitation — a surface no camera looked at
gets its normal flipped toward the nearest camera anyway — which matters more here,
where an orbit around a closed object never sees the base it stands on.

`_job_mesh` resolves the seed itself: an explicit seed pins the mesh, otherwise it
reuses an export newer than the splat, or makes one. A stale export (from a splat
retrained since) is rebuilt rather than silently reused.

---

## 5. What LiDAR was giving us for free

Three things vanish when the LiDAR is removed, and each needs a deliberate answer:

| Lost | Consequence | Answer |
| --- | --- | --- |
| **Metric scale** | Reconstruction is unitless; measurements meaningless; metric constants in stage D break | §8 — two-point reference measurement |
| **Levelled world frame** | Cloud loads sideways in the viewer | §8 — RANSAC dominant plane, manual override |
| **Dense geometry** | Seed cloud is 50k points rather than millions | §7.5 — MVS, or accept MCMC densification |

---

## 6. Phasing

Each phase is independently verifiable, so the work can stop at any boundary and
still leave the tree working.

### Phase 0 — verify assumptions

pycolmap is not in the default venv; it lives in whatever interpreter
`_sfm_python()` locates. Before writing code, confirm on that interpreter:

- pycolmap version and whether `match_exhaustive` / `match_vocabtree` exist
- whether the build is CUDA-enabled (decides whether §7.5 is possible at all)

`SequentialPairingOptions` (`process_sfm.py:120`) is a recent API, so the installed
version is already fairly new — but MVS support is a separate build flag.

### Phase 1 — ingest, no SfM

§4.1 source detection, §4.2 `process_ingest.py`, §4.3 `frame_source.py`, §4.4 job
wiring, and the panel UI.

*Verifiable on its own:* point at a photo folder, get a normalized `images/` directory
and manifest with live progress in the panel. No SfM involved.

### Phase 2 — SfM without LiDAR

§4.5 and §4.6. Produces a poses npz and a sparse seed cloud that appears in the Library
and loads in the viewer (sideways and unscaled — that's Phase 4).

### Phase 3 — training

§4.7, §4.8, §4.9. First splat from phone photos.

### Phase 4 — scale and level

§8. Makes the output usable rather than merely correct.

### Phase 5 — dense MVS

§7.5. Better seed, and a mesh built from scanned rather than trained geometry.

---

## 7. Risks and open questions

1. **pycolmap API drift** — the matching and mapping entry points have changed shape
   across versions. Phase 0 exists to pin this down.
2. **MVS may be unavailable.** pycolmap wheels frequently ship without CUDA. Must
   feature-detect and fall back to the sparse seed rather than failing the job.
3. **Sparse seed quality is unproven here.** The splat recipe was tuned against a dense
   LiDAR seed with accurate normals (`process_splat.py:657` seeds anisotropy from
   geometry). A 50k-point SfM seed with no normals may need different MCMC settings.
   This is the largest technical unknown in the plan.
4. **Mixed-lens captures.** Modern phones switch lenses mid-capture without asking.
   Warning-only in Phase 1; per-lens camera groups if it proves common.
5. **Rolling shutter.** Phone video has it; COLMAP's model does not. Likely acceptable
   at handheld speeds, but it sets a quality ceiling.

### 7.5 Dense MVS (deferred)

`undistort_images` → `patch_match_stereo` → `stereo_fusion`, producing a dense cloud
that improves the splat seed and gives §4.12 a mesh source that is measured rather
than trained. Gated on a CUDA-enabled COLMAP build (risk 2): pycolmap 4.0.4 here
exposes both entry points but reports `has_cuda: False`, and there is no `colmap`
CLI installed, so patch-match stereo refuses despite the local RTX 3090.

---

## 8. Scale and levelling (Phase 4)

**Scale.** SfM is unitless. Add a viewer tool: pick two points, type the real-world
distance, write the resulting factor into the pose sidecar. This is the minimum viable
answer and works with any phone.

The alternative — exporting ARKit/ARCore poses alongside the images — gives metric
scale *and* pose priors that make SfM substantially more robust, and on an iPhone Pro
would reuse much more of the existing LiDAR machinery. Out of scope here, but the
manifest format should reserve columns for per-frame pose so it can slot in later
without a migration.

**Up axis.** The SfM world frame is arbitrary. RANSAC the dominant plane, align its
normal to the viewer's up axis, and provide a manual override for scenes where the
dominant plane isn't the ground.

---

## 9. Capture guidance

These cause most photogrammetry failures and belong in the UI, not just the docs:

- **Lock AE and AF before shooting.** Exposure drift wrecks splat colour consistency;
  autofocus changes intrinsics frame to frame, which the single-camera model cannot
  represent.
- **Prefer video to stills** — denser coverage, constant intrinsics.
- **Don't let the phone switch to the ultrawide** mid-capture.
- **Move slowly.** Motion blur is dropped by the sharpness filter, which means blurry
  frames become missing coverage.

---

## 10. Naming

With this landed, "LidarStudio" no longer describes the whole product. Options, in
increasing disruption: keep the name and subtitle it; rename the product but keep the
repo; rename both. Not blocking — the decision can wait until Phase 3 proves the
approach works.
