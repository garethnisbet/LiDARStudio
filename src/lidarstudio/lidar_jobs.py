#!/usr/bin/env python3
"""
LiDAR Scan Processing Workflow Server

Usage:
    pip install aiohttp
    python lidar_server.py [--port 8090]

Open http://localhost:8090/ in your browser.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import threading
import uuid
from pathlib import Path

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("lidar-editor")

ROOT = Path(__file__).parent

# Active jobs: job_id -> {"queue": Queue, "done": bool}
jobs: dict = {}


# ── Native folder picker ───────────────────────────────────────────────────────


def _tk_browse_folder(title: str = "Select Folder", initial: str = "") -> str | None:
    """Open a native folder picker dialog. Returns the chosen path or None."""
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return None  # headless — no display

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    result: dict = {"path": None}

    def _run():
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        root.update()
        chosen = filedialog.askdirectory(
            title=title,
            initialdir=initial or os.path.expanduser("~"),
            parent=root,
        )
        root.destroy()
        result["path"] = chosen or None

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=120)
    return result["path"]


# ── Scan folder inspection ─────────────────────────────────────────────────────


def _scan_contents(scan_path: Path) -> dict:
    """Return a description of the files found inside a scan folder."""
    image_bag = lidar_bag = params_json = None
    calibration_dir = camera_dir = thumbnail_dir = False
    extras: list = []

    try:
        for item in scan_path.iterdir():
            n = item.name
            if item.is_file():
                if n.startswith("IMAGE_") and n.endswith(".bag"):
                    image_bag = n
                elif n.startswith("LIDAR_") and n.endswith(".bag"):
                    lidar_bag = n
                elif n == "project_parameters.json":
                    params_json = n
                else:
                    extras.append(n)
            elif item.is_dir():
                if n == "calibration":
                    calibration_dir = True
                elif n == "camera":
                    camera_dir = True
                elif n == "thumbnail":
                    thumbnail_dir = True
                else:
                    extras.append(n + "/")
    except PermissionError as exc:
        return {"error": str(exc), "valid": False}

    # Collect thumbnail assets (preview images + PLY)
    thumbnails: dict = {}
    if thumbnail_dir:
        th_path = scan_path / "thumbnail"
        for f in th_path.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                stem = f.stem.lower()
                if stem.startswith("image_"):
                    thumbnails["camera_image"] = str(f)
                elif stem.startswith("lidar_"):
                    thumbnails["lidar_image"] = str(f)
            elif f.suffix.lower() == ".ply":
                thumbnails["preview_ply"] = str(f)

    # Read device info from calibration
    device_info: dict = {}
    if calibration_dir:
        calib_file = scan_path / "calibration" / "calib.json"
        if calib_file.exists():
            try:
                cal = json.loads(calib_file.read_text())
                device_info = cal.get("device", {})
                cam_names = list(cal.get("camera_info", {}).keys())
                device_info["cameras"] = cam_names
            except Exception:
                pass

    return {
        "image_bag": image_bag,
        "lidar_bag": lidar_bag,
        "project_parameters": params_json,
        "calibration": calibration_dir,
        "camera": camera_dir,
        "thumbnail": thumbnail_dir,
        "valid": bool(image_bag and lidar_bag),
        "extras": extras,
        "thumbnails": thumbnails,
        "device": device_info,
    }


def _list_outputs(project_path: Path) -> dict:
    result: dict = {"pointclouds": [], "splats": []}
    for kind in ("pointclouds", "splats"):
        d = project_path / kind
        if d.exists():
            result[kind] = [
                {
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1_048_576, 1),
                }
                for f in sorted(d.glob("*.ply"))
                # the dense cloud is an internal splat input (speckled, not for
                # the on-screen cloud view), so keep it out of the list
                if not (kind == "pointclouds" and f.stem.endswith("_dense"))
            ]
    return result


# ── HTTP handlers ──────────────────────────────────────────────────────────────


async def browse_handler(request):
    """POST /api/browse — open native folder picker; return chosen path."""
    data: dict = {}
    if request.content_length:
        try:
            data = await request.json()
        except Exception:
            pass

    title = data.get("title", "Select Folder")
    initial = data.get("initial", "")

    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, lambda: _tk_browse_folder(title, initial))

    if path is not None:
        return web.json_response({"path": path})

    # Check whether tkinter is available but user cancelled vs. not available
    try:
        import tkinter  # noqa: F401

        return web.json_response({"path": None, "cancelled": True})
    except ImportError:
        return web.json_response(
            {"path": None, "error": "tkinter not available — type the path manually"}
        )


async def browse_dir_handler(request):
    """POST /api/browse/dir — list subdirectories for fallback directory picker."""
    data = await request.json()
    raw = data.get("path", os.path.expanduser("~"))
    p = Path(raw)

    if not p.exists() or not p.is_dir():
        return web.json_response({"error": "Not a directory"}, status=400)

    try:
        items = [
            {"name": item.name, "path": str(item)}
            for item in sorted(p.iterdir())
            if item.is_dir() and not item.name.startswith(".")
        ]
    except PermissionError:
        return web.json_response({"error": "Permission denied"}, status=403)

    return web.json_response(
        {
            "path": str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "items": items,
        }
    )


async def project_create_handler(request):
    """POST /api/project/create — create a new project folder."""
    data = await request.json()
    folder = data.get("folder", "").strip()
    name = data.get("name", "").strip()

    if not folder:
        return web.json_response({"error": "folder is required"}, status=400)
    if not name:
        return web.json_response({"error": "name is required"}, status=400)

    safe_name = name.replace("/", "_").replace("\\", "_")
    project_path = Path(folder) / safe_name

    if project_path.exists():
        return web.json_response(
            {"error": f"Already exists: {project_path}"}, status=409
        )

    try:
        project_path.mkdir(parents=True)
        for sub in ("pointclouds", "splats", "exports"):
            (project_path / sub).mkdir()
        meta = {"version": "1.0", "name": safe_name}
        (project_path / "project.json").write_text(json.dumps(meta, indent=2))
        return web.json_response({"path": str(project_path), "name": safe_name})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def project_open_handler(request):
    """POST /api/project/open — validate and return an existing project."""
    data = await request.json()
    path = data.get("path", "").strip()

    if not path:
        return web.json_response({"error": "path is required"}, status=400)

    p = Path(path)
    if not p.exists() or not p.is_dir():
        return web.json_response({"error": "Folder not found"}, status=400)

    meta: dict = {}
    meta_file = p / "project.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            pass

    return web.json_response(
        {
            "path": str(p),
            "name": p.name,
            "meta": meta,
            "outputs": _list_outputs(p),
        }
    )


async def scan_validate_handler(request):
    """POST /api/scan/validate — inspect a scan folder."""
    data = await request.json()
    path = data.get("path", "").strip()

    if not path:
        return web.json_response({"error": "path is required"}, status=400)

    p = Path(path)
    if not p.exists() or not p.is_dir():
        return web.json_response({"error": "Folder not found"}, status=400)

    contents = _scan_contents(p)

    params: dict = {}
    if contents.get("project_parameters"):
        try:
            params = json.loads((p / contents["project_parameters"]).read_text())
        except Exception as exc:
            params = {"_parse_error": str(exc)}

    return web.json_response(
        {
            "path": str(p),
            "name": p.name,
            "contents": contents,
            "parameters": params,
        }
    )


# ── Processing jobs ────────────────────────────────────────────────────────────


async def process_start_handler(request):
    """POST /api/process/start — launch a background processing job."""
    data = await request.json()
    job_type = data.get("type")
    project_path = data.get("project_path")
    scan_path = data.get("scan_path")
    options = data.get("options", {})

    if job_type not in ("pointcloud", "splat"):
        return web.json_response(
            {"error": "type must be 'pointcloud' or 'splat'"}, status=400
        )
    if not project_path or not scan_path:
        return web.json_response(
            {"error": "project_path and scan_path required"}, status=400
        )

    job_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {"queue": queue, "done": False}

    asyncio.create_task(_run_job(job_id, job_type, project_path, scan_path, options))

    return web.json_response({"job_id": job_id})


async def _run_job(job_id, job_type, project_path, scan_path, options):
    queue = jobs[job_id]["queue"]
    try:
        if job_type == "pointcloud":
            await _job_pointcloud(project_path, scan_path, options, queue)
        else:
            await _job_splat(project_path, scan_path, options, queue)
    except Exception as exc:
        await queue.put({"event": "error", "message": str(exc)})
    finally:
        jobs[job_id]["done"] = True
        await queue.put(None)  # sentinel


async def _stream_proc(proc, queue) -> int:
    """Stream stdout of a subprocess to the job queue. Returns exit code."""
    async for raw in proc.stdout:
        text = raw.decode(errors="replace").rstrip()
        if not text:
            continue
        if text.startswith("PROGRESS:"):
            parts = text.split(":", 2)
            pct = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            msg = parts[2] if len(parts) > 2 else ""
            await queue.put({"event": "progress", "percent": pct, "message": msg})
        else:
            await queue.put({"event": "log", "message": text})
    await proc.wait()
    return proc.returncode


def _pointcloud_cmd(scan, contents, output_file, voxel):
    """Build the process_pointcloud.py argv for a scan."""
    cmd = [
        sys.executable,
        str(ROOT / "process_pointcloud.py"),
        "--lidar-bag",
        str(scan / contents["lidar_bag"]),
        "--image-bag",
        str(scan / contents["image_bag"]),
        "--output",
        str(output_file),
        "--voxel-size",
        str(voxel),
    ]
    if contents.get("project_parameters"):
        cmd += ["--params", str(scan / "project_parameters.json")]
    if contents.get("calibration"):
        cmd += ["--calibration", str(scan / "calibration")]
    return cmd


async def _job_pointcloud(project_path, scan_path, options, queue):
    proj = Path(project_path)
    scan = Path(scan_path)
    contents = _scan_contents(scan)

    await queue.put(
        {
            "event": "progress",
            "percent": 0,
            "message": "Starting point cloud generation…",
        }
    )

    if not contents.get("valid"):
        await queue.put(
            {"event": "error", "message": "Scan folder is missing required bag files"}
        )
        return

    out_dir = proj / "pointclouds"
    out_dir.mkdir(exist_ok=True)

    # Derive timestamp from e.g. LIDAR_20260615182333.bag
    stem = Path(contents["lidar_bag"]).stem
    ts = stem.split("_", 1)[1] if "_" in stem else stem
    output_file = out_dir / f"pointcloud_{ts}.ply"

    cmd = _pointcloud_cmd(scan, contents, output_file, options.get("voxel_size", 0.05))

    await queue.put(
        {
            "event": "progress",
            "percent": 5,
            "message": "Launching process_pointcloud.py…",
        }
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        await queue.put(
            {
                "event": "error",
                "message": "process_pointcloud.py not found — see the stub script.",
            }
        )
        return

    rc = await _stream_proc(proc, queue)

    if rc == 0 and output_file.exists():
        size_mb = round(output_file.stat().st_size / 1_048_576, 1)
        await queue.put({"event": "progress", "percent": 100, "message": "Complete!"})
        await queue.put(
            {
                "event": "result",
                "type": "pointcloud",
                "path": str(output_file),
                "filename": output_file.name,
                "size_mb": size_mb,
            }
        )
    else:
        await queue.put(
            {
                "event": "error",
                "message": f"process_pointcloud.py exited with code {rc}",
            }
        )


async def _job_splat(project_path, scan_path, options, queue):
    proj = Path(project_path)
    scan = Path(scan_path)
    contents = _scan_contents(scan)

    await queue.put(
        {
            "event": "progress",
            "percent": 0,
            "message": "Starting Gaussian Splat generation…",
        }
    )

    out_dir = proj / "splats"
    out_dir.mkdir(exist_ok=True)

    stem = Path(contents.get("lidar_bag") or "LIDAR_output.bag").stem
    ts = stem.split("_", 1)[1] if "_" in stem else stem
    output_file = out_dir / f"splat_{ts}.ply"

    mode = options.get("splat_mode", "surfel")
    pc_dir = proj / "pointclouds"

    # Explicit seed cloud (e.g. an edited one) — pins the splat to that file
    # instead of the job's default (surfel: the scan's own dense cloud;
    # trained/bootstrap: the latest project cloud).
    seed = (options.get("pointcloud") or "").strip()
    seed_p = Path(seed) if seed else None
    if seed_p and not seed_p.exists():
        await queue.put(
            {"event": "error", "message": f"Selected seed cloud not found: {seed}"}
        )
        return

    # Optional externally-computed SfM camera poses (the campaign's biggest
    # quality lever). Validate up front so a typo fails fast, not 40 min in.
    sfm_poses = (options.get("sfm_poses") or "").strip()
    if sfm_poses and not Path(sfm_poses).exists():
        await queue.put(
            {"event": "error", "message": f"SfM poses file not found: {sfm_poses}"}
        )
        return

    if mode == "surfel":
        # Surfels need a *dense* cloud (small splats only look good when the
        # cloud is dense), so the splat job keeps its own fine-voxel cloud,
        # separate from the coarser one used for the on-screen cloud view.
        if not contents.get("valid"):
            await queue.put(
                {
                    "event": "error",
                    "message": "Scan folder is missing required bag files",
                }
            )
            return
        voxel = float(options.get("splat_voxel", 0.01))
        dense = seed_p or pc_dir / f"pointcloud_{ts}_dense.ply"
        if not dense.exists():
            pc_dir.mkdir(exist_ok=True)
            await queue.put(
                {
                    "event": "progress",
                    "percent": 5,
                    "message": f"Building dense cloud for splat (voxel {voxel} m)…",
                }
            )
            pc_cmd = _pointcloud_cmd(scan, contents, dense, voxel)
            try:
                pc_proc = await asyncio.create_subprocess_exec(
                    *pc_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except FileNotFoundError:
                await queue.put(
                    {"event": "error", "message": "process_pointcloud.py not found."}
                )
                return
            prc = await _stream_proc(pc_proc, queue)
            if prc != 0 or not dense.exists():
                await queue.put(
                    {
                        "event": "error",
                        "message": f"Dense cloud step failed (code {prc})",
                    }
                )
                return
        else:
            await queue.put(
                {"event": "log", "message": f"Using dense cloud: {dense.name}"}
            )

        cmd = [
            sys.executable,
            str(ROOT / "process_splat.py"),
            "--scan",
            str(scan),
            "--output",
            str(output_file),
            "--pointcloud",
            str(dense),
            "--surfel",
            "--surfel-flatten",
            str(options.get("surfel_flatten", 0.2)),
            "--surfel-sor",
            str(options.get("surfel_sor", 2.0)),
        ]
    else:
        # Legacy GPU-trained / bootstrap path (explicit seed cloud if chosen,
        # else the latest coloured cloud in the project).
        if seed_p:
            pc_files = [seed_p]
        else:
            pc_files = (
                sorted(pc_dir.glob("pointcloud_*.ply")) if pc_dir.exists() else []
            )
            pc_files = [
                p for p in pc_files if not p.stem.endswith("_dense")
            ] or pc_files
        cmd = [
            sys.executable,
            str(ROOT / "process_splat.py"),
            "--scan",
            str(scan),
            "--output",
            str(output_file),
            "--iterations",
            str(options.get("iterations", 7000)),
        ]
        if mode == "bootstrap":
            cmd += ["--bootstrap", "--splat-size", str(options.get("splat_size", 0.05))]
        elif mode == "trained":
            # Quality knobs driven by the UI 'quality' slider (draft→max). The
            # shape/opacity/pose-association recipe constants (opacity-reg,
            # flat-reg, min-opacity, min-scale, cam-time-offset) now come from
            # process_splat.py's own champion defaults, so we only pass the
            # resolution/count/sharpness knobs that scale with quality.
            cmd += [
                "--downscale",
                str(int(options.get("downscale", 1))),
                "--max-init-points",
                str(int(options.get("max_init_points", 3_000_000))),
                "--cap-max",
                str(int(options.get("cap_max", 6_000_000))),
                "--undistort-scale",
                str(float(options.get("undistort_scale", 1.0))),
                "--drop-blurry",
                str(float(options.get("drop_blurry", 0.15))),
            ]
            if sfm_poses:
                cmd += ["--sfm-poses", sfm_poses]
        if pc_files:
            cmd += ["--pointcloud", str(pc_files[-1])]
            await queue.put(
                {
                    "event": "log",
                    "message": f"Using existing point cloud: {pc_files[-1].name}",
                }
            )

    await queue.put(
        {"event": "progress", "percent": 5, "message": "Launching process_splat.py…"}
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        await queue.put(
            {
                "event": "error",
                "message": "process_splat.py not found — see the stub script.",
            }
        )
        return

    rc = await _stream_proc(proc, queue)

    if rc == 0 and output_file.exists():
        size_mb = round(output_file.stat().st_size / 1_048_576, 1)
        await queue.put({"event": "progress", "percent": 100, "message": "Complete!"})
        await queue.put(
            {
                "event": "result",
                "type": "splat",
                "path": str(output_file),
                "filename": output_file.name,
                "size_mb": size_mb,
            }
        )
    else:
        await queue.put(
            {"event": "error", "message": f"process_splat.py exited with code {rc}"}
        )


async def process_events_handler(request):
    """GET /api/process/events/{job_id} — SSE stream for job progress."""
    job_id = request.match_info["job_id"]

    if job_id not in jobs:
        return web.Response(status=404, text="Job not found")

    queue = jobs[job_id]["queue"]

    resp = web.StreamResponse()
    resp.headers["Content-Type"] = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    await resp.prepare(request)

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60)
            except TimeoutError:
                await resp.write(b": keep-alive\n\n")
                continue

            if event is None:
                await resp.write(b"event: done\ndata: {}\n\n")
                break

            await resp.write(f"data: {json.dumps(event)}\n\n".encode())
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        jobs.pop(job_id, None)

    return resp


async def scan_file_handler(request):
    """GET /api/scan/file?path=... — proxy for scan folder files (images, PLY, JSON)."""
    raw = request.query.get("path", "")
    if not raw:
        return web.Response(status=400, text="path required")

    p = Path(raw)
    allowed_ext = {".png", ".jpg", ".jpeg", ".json", ".ply"}
    if p.suffix.lower() not in allowed_ext:
        return web.Response(status=403, text="File type not served")
    if not p.exists() or not p.is_file():
        return web.Response(status=404, text="File not found")

    # Determine content type
    ct_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
        ".ply": "application/octet-stream",
    }
    ct = ct_map.get(p.suffix.lower(), "application/octet-stream")
    return web.FileResponse(p, headers={"Content-Type": ct})


async def project_outputs_handler(request):
    """POST /api/project/outputs — list generated files in a project."""
    data = await request.json()
    path = data.get("path", "").strip()
    if not path:
        return web.json_response({"error": "path required"}, status=400)
    p = Path(path)
    if not p.exists():
        return web.json_response({"error": "Project not found"}, status=404)
    return web.json_response(_list_outputs(p))


async def edit_apply_handler(request):
    """POST /api/edit/apply — run one edit op on a cloud/splat PLY.

    Body: {path, op, params, [output]}.  Writes a sibling *_edited.ply with a
    unique numbered name (never overwriting an earlier edit) and returns the
    edit summary so the UI can reload it.
    """
    data = await request.json()
    path = (data.get("path") or "").strip()
    op = data.get("op")
    params = data.get("params", {})
    if not path or not op:
        return web.json_response({"error": "path and op required"}, status=400)
    src = Path(path)
    if not src.exists():
        return web.json_response({"error": "file not found"}, status=404)

    out = data.get("output") or _unique_output(src, "edited")

    try:
        from lidarstudio import edit_ops

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, edit_ops.apply_edit, str(src), out, op, params
        )
        _write_pose_sidecar(out, data.get("pose"))
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def edit_recolour_handler(request):
    """POST /api/edit/recolour — re-project a scan's photos onto a cloud/splat."""
    data = await request.json()
    path = (data.get("path") or "").strip()
    scan = (data.get("scan_path") or "").strip()
    if not path or not scan:
        return web.json_response({"error": "path and scan_path required"}, status=400)
    src = Path(path)
    if not src.exists():
        return web.json_response({"error": "file not found"}, status=404)
    out = data.get("output") or _unique_output(src, "recoloured")
    try:
        from lidarstudio import edit_ops

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, edit_ops.recolour, str(src), out, scan
        )
        _write_pose_sidecar(out, data.get("pose"))
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# Per-scan photo index: scan folder → list of (bag_ts_ns, connection topic).
# Built once by iterating message headers (no JPEG decoding); lets the frame
# endpoint jump straight to one message via a start/stop time window.
_photo_index: dict = {}


def _scan_photo_index(scan: Path):
    if str(scan) in _photo_index:
        return _photo_index[str(scan)]
    from rosbags.highlevel import AnyReader

    contents = _scan_contents(scan)
    if not contents.get("image_bag"):
        raise FileNotFoundError("no IMAGE_*.bag in scan folder")
    bag = scan / contents["image_bag"]
    ts = []
    with AnyReader([bag]) as reader:
        conns = [
            c for c in reader.connections if "image" in c.topic and "camera" in c.topic
        ] or list(reader.connections)
        topic = conns[0].topic
        conns = [c for c in conns if c.topic == topic]
        for _conn, t, _raw in reader.messages(connections=conns):
            ts.append(t)
    _photo_index[str(scan)] = (bag, topic, ts)
    return _photo_index[str(scan)]


async def scan_photos_handler(request):
    """POST /api/scan/photos — index the scan's camera images: {count, ts}."""
    data = await request.json()
    scan = Path((data.get("path") or "").strip())
    if not scan.is_dir():
        return web.json_response({"error": "scan folder not found"}, status=404)
    try:
        loop = asyncio.get_event_loop()
        _bag, topic, ts = await loop.run_in_executor(None, _scan_photo_index, scan)
        return web.json_response({"count": len(ts), "topic": topic})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


def _read_photo(scan: Path, index: int) -> bytes:
    from rosbags.highlevel import AnyReader

    bag, topic, ts = _scan_photo_index(scan)
    if not 0 <= index < len(ts):
        raise IndexError(f"frame {index} out of range 0..{len(ts) - 1}")
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _t, raw in reader.messages(
            connections=conns, start=ts[index], stop=ts[index] + 1
        ):
            msg = reader.deserialize(raw, conn.msgtype)
            return bytes(msg.data)  # CompressedImage payload is already JPEG
    raise RuntimeError("frame not found in bag")


def _prep_photo(jpeg: bytes, rot: str, width: int) -> bytes:
    """Optionally rotate to portrait and downscale for display."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    rot_map = {
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    if rot in rot_map:
        img = cv2.rotate(img, rot_map[rot])
    if width and img.shape[1] > width:
        h = round(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 87])
    return out.tobytes() if ok else jpeg


async def scan_photo_handler(request):
    """GET /api/scan/photo?path=<scan>&index=N[&rot=ccw&width=1600] — one
    camera JPEG from the bag, optionally rotated/downscaled for display."""
    scan = Path(request.query.get("path", ""))
    rot = request.query.get("rot", "")
    try:
        index = int(request.query.get("index", "0"))
        width = int(request.query.get("width", "0"))
    except ValueError:
        return web.Response(status=400, text="bad index/width")
    if not scan.is_dir():
        return web.Response(status=404, text="scan folder not found")
    try:
        loop = asyncio.get_event_loop()
        jpeg = await loop.run_in_executor(None, _read_photo, scan, index)
        if rot or width:
            jpeg = await loop.run_in_executor(None, _prep_photo, jpeg, rot, width)
        return web.Response(body=jpeg, content_type="image/jpeg")
    except Exception as exc:
        return web.Response(status=500, text=str(exc))


# Per-scan LiDAR sweep index, mirroring the photo index.
_sweep_index: dict = {}


def _scan_sweep_index(scan: Path):
    if str(scan) in _sweep_index:
        return _sweep_index[str(scan)]
    from rosbags.highlevel import AnyReader

    contents = _scan_contents(scan)
    if not contents.get("lidar_bag"):
        raise FileNotFoundError("no LIDAR_*.bag in scan folder")
    bag = scan / contents["lidar_bag"]
    ts = []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if "PointCloud2" in c.msgtype]
        if not conns:
            raise RuntimeError("no PointCloud2 topic in LIDAR bag")
        topic = conns[0].topic
        conns = [c for c in conns if c.topic == topic]
        for _conn, t, _raw in reader.messages(connections=conns):
            ts.append(t)
    _sweep_index[str(scan)] = (bag, topic, ts)
    return _sweep_index[str(scan)]


async def scan_sweeps_handler(request):
    """POST /api/scan/sweeps — index the scan's LiDAR sweeps: {count, topic}."""
    data = await request.json()
    scan = Path((data.get("path") or "").strip())
    if not scan.is_dir():
        return web.json_response({"error": "scan folder not found"}, status=404)
    try:
        loop = asyncio.get_event_loop()
        _bag, topic, ts = await loop.run_in_executor(None, _scan_sweep_index, scan)
        return web.json_response({"count": len(ts), "topic": topic})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


def _render_sweep(scan: Path, index: int, half_range: float = 12.0) -> bytes:
    """Render sweep N as a PNG: top-down (x/y) and side (x/z), coloured by z."""
    import cv2
    import numpy as np
    from rosbags.highlevel import AnyReader

    bag, topic, ts = _scan_sweep_index(scan)
    if not 0 <= index < len(ts):
        raise IndexError(f"sweep {index} out of range 0..{len(ts) - 1}")
    xyz = None
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _t, raw in reader.messages(
            connections=conns, start=ts[index], stop=ts[index] + 1
        ):
            msg = reader.deserialize(raw, conn.msgtype)
            fields = {f.name: f for f in msg.fields}
            offs = [fields[n].offset for n in ("x", "y", "z")]
            n_pts = msg.width * msg.height
            rawa = np.frombuffer(msg.data, np.uint8).reshape(n_pts, msg.point_step)
            xyz = np.stack(
                [
                    np.frombuffer(rawa[:, o : o + 4].tobytes(), np.float32)
                    for o in offs
                ],
                axis=1,
            )
            break
    if xyz is None:
        raise RuntimeError("sweep not found in bag")
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz) < half_range).all(1)
    xyz = xyz[ok]
    S = 640
    canvas = np.full((S, 2 * S + 8, 3), 16, np.uint8)
    # Height colouring, robust range.
    z = xyz[:, 2]
    zlo, zhi = (np.percentile(z, 2), np.percentile(z, 98)) if len(z) else (0, 1)
    tn = np.clip((z - zlo) / max(zhi - zlo, 1e-6) * 255, 0, 255).astype(np.uint8)
    colours = cv2.applyColorMap(tn.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    def paint(u, v, xoff):
        m = (u >= 0) & (u < S) & (v >= 0) & (v < S)
        canvas[v[m], u[m] + xoff] = colours[m]

    scale = S / (2 * half_range)
    # Top-down: x right, y up.
    paint(
        (xyz[:, 0] * scale + S / 2).astype(np.int32),
        (S / 2 - xyz[:, 1] * scale).astype(np.int32),
        0,
    )
    # Side: x right, z up.
    paint(
        (xyz[:, 0] * scale + S / 2).astype(np.int32),
        (S / 2 - xyz[:, 2] * scale).astype(np.int32),
        S + 8,
    )
    for xoff, label in ((0, "top-down (x/y)"), (S + 8, "side (x/z)")):
        cv2.drawMarker(canvas, (xoff + S // 2, S // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 12, 1)
        cv2.putText(canvas, label, (xoff + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
    cv2.putText(canvas, f"{len(xyz):,} pts  +/-{half_range:.0f} m",
                (8, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    ok2, png = cv2.imencode(".png", canvas)
    if not ok2:
        raise RuntimeError("png encode failed")
    return png.tobytes()


async def scan_sweep_handler(request):
    """GET /api/scan/sweep?path=<scan>&index=N — rendered LiDAR sweep PNG."""
    scan = Path(request.query.get("path", ""))
    try:
        index = int(request.query.get("index", "0"))
    except ValueError:
        return web.Response(status=400, text="bad index")
    if not scan.is_dir():
        return web.Response(status=404, text="scan folder not found")
    try:
        loop = asyncio.get_event_loop()
        png = await loop.run_in_executor(None, _render_sweep, scan, index)
        return web.Response(body=png, content_type="image/png")
    except Exception as exc:
        return web.Response(status=500, text=str(exc))


def _unique_output(src: Path, tag: str) -> str:
    """First free sibling name ``<stem>_<tag>.ply``, ``<stem>_<tag>2.ply``, … .

    Never overwrites an earlier output. Chained edits (re-editing a file that
    already carries the tag) collapse back to the base stem, so a second crop
    of ``cloud_edited.ply`` becomes ``cloud_edited2.ply``, not
    ``cloud_edited_edited.ply``.
    """
    stem = re.sub(rf"_{tag}\d*$", "", src.stem)
    n = 1
    while True:
        name = f"{stem}_{tag}.ply" if n == 1 else f"{stem}_{tag}{n}.ply"
        cand = src.with_name(name)
        if not cand.exists():
            return str(cand)
        n += 1


def _write_pose_sidecar(out_path, pose):
    """Persist the object's in-scene pose next to an edited file (``<out>.pose.json``)
    so reloading it from the Library restores its placement. The edit keeps the
    file in its original local frame, so without this it would reload at identity
    (wrong orientation). ``pose`` is {position, rotation, scale, visible,
    parentLink}; a falsy/identity pose removes any stale sidecar so the two stay
    consistent (e.g. after a 'transform' bake)."""
    sidecar = Path(str(out_path) + ".pose.json")
    try:
        if pose:
            sidecar.write_text(json.dumps(pose))
        elif sidecar.exists():
            sidecar.unlink()
    except Exception as exc:
        log.warning("pose sidecar write failed for %s: %s", out_path, exc)


async def edit_save_as_handler(request):
    """POST /api/edit/save_as — save a cloud/splat to a new file under a chosen name.

    Two modes:
      • copy (default): lossless byte copy, preserving all fields/format, with the
        object's pose stored in a sidecar so it reloads in the same place here.
      • bake: when ``matrix`` (the object's world matrix, column-major 16) is
        given, the transform is baked into the coordinates, producing a
        self-contained file that's correctly oriented in any viewer (no sidecar).

    Body: {path, output, [pose], [matrix]}. `output` must be a .ply path.
    """
    data = await request.json()
    src = Path((data.get("path") or "").strip())
    out = (data.get("output") or "").strip()
    if not src.name or not out:
        return web.json_response({"error": "path and output required"}, status=400)
    if not src.exists():
        return web.json_response({"error": "source not found"}, status=404)
    out_path = Path(out)
    if out_path.suffix.lower() != ".ply":
        return web.json_response({"error": "output must be a .ply file"}, status=400)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = data.get("matrix")
        if matrix is not None:
            from lidarstudio import edit_ops

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                edit_ops.apply_edit,
                str(src),
                str(out_path),
                "transform",
                {"matrix": matrix},
            )
            _write_pose_sidecar(str(out_path), None)  # baked → no sidecar
        else:
            shutil.copy2(src, out_path)
            _write_pose_sidecar(str(out_path), data.get("pose"))
        return web.json_response({"output": str(out_path), "baked": matrix is not None})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Route registration ──────────────────────────────────────────────────────
# LidarStudio hosts the viewer at "/" and owns the static handler, so we attach
# only the LiDAR workflow's /api/* endpoints onto the shared aiohttp app.


def register_routes(app: web.Application) -> None:
    app.router.add_post("/api/edit/apply", edit_apply_handler)
    app.router.add_post("/api/edit/recolour", edit_recolour_handler)
    app.router.add_post("/api/edit/save_as", edit_save_as_handler)
    app.router.add_post("/api/browse", browse_handler)
    app.router.add_post("/api/browse/dir", browse_dir_handler)
    app.router.add_post("/api/project/create", project_create_handler)
    app.router.add_post("/api/project/open", project_open_handler)
    app.router.add_post("/api/scan/validate", scan_validate_handler)
    app.router.add_post("/api/process/start", process_start_handler)
    app.router.add_get("/api/process/events/{job_id}", process_events_handler)
    app.router.add_post("/api/project/outputs", project_outputs_handler)
    app.router.add_get("/api/scan/file", scan_file_handler)
    app.router.add_post("/api/scan/photos", scan_photos_handler)
    app.router.add_get("/api/scan/photo", scan_photo_handler)
    app.router.add_post("/api/scan/sweeps", scan_sweeps_handler)
    app.router.add_get("/api/scan/sweep", scan_sweep_handler)
