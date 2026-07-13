#!/usr/bin/env python3
"""
LiDAR Scan Processing Workflow Server

Usage:
    pip install aiohttp
    python lidar_server.py [--port 8090]

Open http://localhost:8090/ in your browser.
"""

import asyncio
import functools
import json
import logging
import os
import re
import shutil
import subprocess
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


def _has_display() -> bool:
    """True if a graphical session is available to open a native dialog on."""
    if sys.platform != "linux":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _tk_available() -> bool:
    """True if tkinter can be imported (the last-resort native dialog backend)."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def _desktop_browse_folder(title: str, initial: str) -> tuple[bool, str | None]:
    """Open the desktop's own folder picker via zenity (GTK) or kdialog (KDE).

    These render the modern system file navigator that matches the user's
    desktop, unlike tkinter's dated Motif-style chooser. Returns
    ``(handled, path)``: ``handled`` is True when such a tool ran to a definite
    answer (``path`` = the choice, or None if cancelled), and False when no tool
    is installed so the caller should fall back to tkinter. This split keeps a
    cancelled zenity dialog from immediately popping a second tkinter dialog.
    """
    start = (initial or os.path.expanduser("~")).rstrip("/") or "/"

    zenity = shutil.which("zenity") or shutil.which("qarma")
    if zenity:
        try:
            out = subprocess.run(
                [zenity, "--file-selection", "--directory",
                 f"--title={title}", f"--filename={start}/"],
                capture_output=True, text=True, timeout=300,
            )
            # 0 = selected, 1 = cancelled; both are definitive answers.
            if out.returncode in (0, 1):
                return True, (out.stdout.strip() or None)
        except Exception:
            pass

    kdialog = shutil.which("kdialog")
    if kdialog:
        try:
            out = subprocess.run(
                [kdialog, "--getexistingdirectory", start, "--title", title],
                capture_output=True, text=True, timeout=300,
            )
            if out.returncode in (0, 1):
                return True, (out.stdout.strip() or None)
        except Exception:
            pass

    return False, None


def _native_browse_folder(title: str = "Select Folder", initial: str = "") -> str | None:
    """Open a native folder picker: the desktop chooser if possible, else tkinter."""
    if not _has_display():
        return None  # headless — no display
    handled, chosen = _desktop_browse_folder(title, initial)
    if handled:
        return chosen
    return _tk_browse_folder(title, initial)


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
    result: dict = {"pointclouds": [], "splats": [], "meshes": []}
    for kind in ("pointclouds", "splats", "meshes"):
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
    path = await loop.run_in_executor(None, lambda: _native_browse_folder(title, initial))

    if path is not None:
        return web.json_response({"path": path})

    # No path came back. Distinguish "the native picker isn't usable here"
    # (headless server, or no picker tool → the client should fall back to the
    # in-browser navigator) from "the dialog opened and the user cancelled".
    native_ok = _has_display() and (
        bool(shutil.which("zenity") or shutil.which("qarma") or shutil.which("kdialog"))
        or _tk_available()
    )

    if native_ok:
        return web.json_response({"path": None, "cancelled": True})
    return web.json_response(
        {"path": None, "error": "no native picker — use the in-browser navigator"}
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

    if job_type not in ("pointcloud", "splat", "mesh"):
        return web.json_response(
            {"error": "type must be 'pointcloud', 'splat' or 'mesh'"}, status=400
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
    job = jobs[job_id]
    queue = job["queue"]
    try:
        if job_type == "pointcloud":
            await _job_pointcloud(project_path, scan_path, options, queue, job)
        elif job_type == "mesh":
            await _job_mesh(project_path, scan_path, options, queue, job)
        else:
            await _job_splat(project_path, scan_path, options, queue, job)
    except Exception as exc:
        await queue.put({"event": "error", "message": str(exc)})
    finally:
        job["done"] = True
        await queue.put(None)  # sentinel


async def process_cancel_handler(request):
    """POST /api/process/cancel — stop a running job by killing its subprocess."""
    data = await request.json()
    job_id = (data.get("job_id") or "").strip()
    job = jobs.get(job_id)
    if not job:
        return web.json_response({"error": "job not found"}, status=404)
    job["cancelled"] = True
    proc = job.get("proc")
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    return web.json_response({"stopped": True})


async def _stream_proc(proc, queue):
    """Stream a subprocess's stdout to the job queue.

    Returns ``(exit_code, last_error_line)`` — the most recent ``ERROR:`` line,
    so the caller can report *why* a job failed rather than just its exit code.
    """
    last_error = None
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
            if text.lstrip().startswith("ERROR"):
                last_error = text.strip()
            await queue.put({"event": "log", "message": text})
    await proc.wait()
    return proc.returncode, last_error


def _pointcloud_cmd(scan, contents, output_file, voxel, mono=False, keep_self_view=False):
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
    if mono:
        # Monochromatic: skip photo colouring (much faster), shade by intensity.
        cmd += ["--mono"]
    if keep_self_view:
        # Default removes the scanner's own handle/mount; this keeps it.
        cmd += ["--keep-self-view"]
    if contents.get("project_parameters"):
        cmd += ["--params", str(scan / "project_parameters.json")]
    if contents.get("calibration"):
        cmd += ["--calibration", str(scan / "calibration")]
    return cmd


async def _job_pointcloud(project_path, scan_path, options, queue, job=None):
    job = job if job is not None else {}
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

    cmd = _pointcloud_cmd(
        scan,
        contents,
        output_file,
        options.get("voxel_size", 0.05),
        mono=bool(options.get("mono", False)),
        keep_self_view=bool(options.get("keep_self_view", False)),
    )

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

    job["proc"] = proc
    if job.get("cancelled"):
        proc.terminate()

    rc, err = await _stream_proc(proc, queue)
    job["proc"] = None

    if job.get("cancelled"):
        await queue.put(
            {"event": "cancelled", "message": "Point cloud generation stopped."}
        )
    elif rc == 0 and output_file.exists():
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
                "message": err or f"process_pointcloud.py exited with code {rc}",
            }
        )


def _mesh_cmd(scan, contents, cloud_file, output_file, options):
    """Build the process_mesh.py argv for a scan + seed cloud."""
    cmd = [
        sys.executable,
        str(ROOT / "process_mesh.py"),
        "--cloud",
        str(cloud_file),
        "--image-bag",
        str(scan / contents["image_bag"]),
        "--output",
        str(output_file),
        "--camera",
        str(options.get("camera", "front")),
        "--depth",
        str(int(options.get("depth", 10))),
        "--density-quantile",
        str(float(options.get("density_quantile", 0.03))),
        "--decimate",
        str(int(options.get("decimate", 0))),
        "--outlier-neighbors",
        str(int(options.get("outlier_neighbors", 20))),
        "--outlier-std",
        str(float(options.get("outlier_std", 2.0))),
        "--pre-voxel",
        str(float(options.get("pre_voxel", 0.0))),
        "--smooth",
        str(int(options.get("smooth", 15))),
        "--colour-range",
        str(float(options.get("colour_range", 20.0))),
    ]
    if contents.get("calibration"):
        cmd += ["--calibration", str(scan / "calibration")]
    return cmd


async def _job_mesh(project_path, scan_path, options, queue, job=None):
    job = job if job is not None else {}
    proj = Path(project_path)
    scan = Path(scan_path)
    contents = _scan_contents(scan)

    await queue.put(
        {"event": "progress", "percent": 0, "message": "Starting mesh generation…"}
    )

    if not contents.get("valid"):
        await queue.put(
            {"event": "error", "message": "Scan folder is missing required bag files"}
        )
        return

    out_dir = proj / "meshes"
    out_dir.mkdir(exist_ok=True)

    stem = Path(contents.get("lidar_bag") or "LIDAR_output.bag").stem
    ts = stem.split("_", 1)[1] if "_" in stem else stem
    output_file = out_dir / f"mesh_{ts}.ply"

    # Seed cloud: an explicit one (e.g. an edited cloud) pins the mesh; otherwise
    # pick the latest project cloud that has a trajectory sidecar (required for
    # camera-oriented normals + photo colouring). Unlike splats, a *dense* cloud
    # is preferred here — Poisson is only as good as its point density.
    from lidarstudio import edit_ops

    seed = (options.get("pointcloud") or "").strip()
    seed_p = Path(seed) if seed else None
    if seed_p and not seed_p.exists():
        await queue.put(
            {"event": "error", "message": f"Selected seed cloud not found: {seed}"}
        )
        return

    if seed_p:
        cloud_file = seed_p
    else:
        pc_dir = proj / "pointclouds"
        pc_files = sorted(pc_dir.glob("pointcloud_*.ply")) if pc_dir.exists() else []
        pc_files = [p for p in pc_files if edit_ops._find_traj(p)]
        if not pc_files:
            await queue.put(
                {
                    "event": "error",
                    "message": "No point cloud with a trajectory sidecar found — "
                    "generate a point cloud first.",
                }
            )
            return
        dense = [p for p in pc_files if p.stem.endswith("_dense")]
        cloud_file = (dense or pc_files)[-1]

    if edit_ops._find_traj(cloud_file) is None:
        await queue.put(
            {
                "event": "error",
                "message": f"{cloud_file.name} has no trajectory sidecar (.traj.npz); "
                "mesh colouring needs camera poses. Re-generate the cloud.",
            }
        )
        return

    await queue.put(
        {"event": "log", "message": f"Meshing from cloud: {cloud_file.name}"}
    )

    cmd = _mesh_cmd(scan, contents, cloud_file, output_file, options)

    await queue.put(
        {"event": "progress", "percent": 5, "message": "Launching process_mesh.py…"}
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        await queue.put(
            {"event": "error", "message": "process_mesh.py not found."}
        )
        return

    job["proc"] = proc
    if job.get("cancelled"):
        proc.terminate()

    rc, err = await _stream_proc(proc, queue)
    job["proc"] = None

    if job.get("cancelled"):
        await queue.put({"event": "cancelled", "message": "Mesh generation stopped."})
    elif rc == 0 and output_file.exists():
        size_mb = round(output_file.stat().st_size / 1_048_576, 1)
        await queue.put({"event": "progress", "percent": 100, "message": "Complete!"})
        await queue.put(
            {
                "event": "result",
                "type": "mesh",
                "path": str(output_file),
                "filename": output_file.name,
                "size_mb": size_mb,
            }
        )
    else:
        await queue.put(
            {
                "event": "error",
                "message": err or f"process_mesh.py exited with code {rc}",
            }
        )


@functools.lru_cache(maxsize=1)
def _system_health() -> dict:
    """Grade CPU / RAM / GPU for splat work as green|amber|red. Cached (static)."""

    def grade(v, good, ok):
        return "green" if v >= good else "amber" if v >= ok else "red"

    # CPU: logical cores (point-cloud + undistort are CPU-bound).
    cores = os.cpu_count() or 0
    cpu = {"grade": grade(cores, 8, 4), "detail": f"{cores} logical cores"}

    # RAM: full-res frame caches are memory-hungry (32 GB comfortable, 16 tight).
    ram_gb = 0.0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram_gb = int(line.split()[1]) / 1024 / 1024
                    break
    except OSError:
        pass
    ram = {"grade": grade(ram_gb, 32, 16), "detail": f"{ram_gb:.0f} GB"}

    # GPU: needs an NVIDIA card + the torch/gsplat stack; VRAM sets the ceiling.
    name, vram_gb = None, 0.0
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            nm, mem = r.stdout.strip().splitlines()[0].split(",")
            name, vram_gb = nm.strip(), float(mem) / 1024  # MiB → GiB
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    stack_ok = False
    try:
        rr = subprocess.run(
            [_splat_python(), "-c",
             "import torch, gsplat; print(int(torch.cuda.is_available()))"],
            capture_output=True, text=True, timeout=120,
        )
        stack_ok = rr.returncode == 0 and rr.stdout.strip().endswith("1")
    except (OSError, subprocess.SubprocessError):
        pass
    if not stack_ok or name is None:
        gpu = {
            "grade": "red",
            "detail": (f"{name} — no CUDA/torch stack" if name else "no CUDA GPU"),
        }
    else:
        gpu = {
            "grade": grade(vram_gb, 12, 8),
            "detail": f"{name}, {vram_gb:.0f} GB VRAM",
        }
    return {"cpu": cpu, "ram": ram, "gpu": gpu}


async def system_health_handler(request):
    """GET /api/system — CPU/RAM/GPU health grades for the startup panel."""
    return web.json_response(await asyncio.to_thread(_system_health))


@functools.lru_cache(maxsize=1)
def _splat_python() -> str:
    """A Python interpreter that has the splat-training stack (torch + gsplat).

    The server can run in a venv without the heavy CUDA deps, so process_splat
    may need a different interpreter than the server itself. Checked once and
    cached. Override with $LIDARSTUDIO_SPLAT_PYTHON.
    """
    seen = set()
    for cand in (
        os.environ.get("LIDARSTUDIO_SPLAT_PYTHON"),
        sys.executable,
        shutil.which("python3"),
        "/usr/bin/python3",
    ):
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            r = subprocess.run(
                [cand, "-c", "import torch, gsplat"],
                capture_output=True,
                timeout=120,
            )
            if r.returncode == 0:
                if cand != sys.executable:
                    logging.info("splat training will use %s (has torch/gsplat)", cand)
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    # None found — fall back to the server's own; process_splat then emits its
    # own clear "torch/gsplat not installed" error.
    return sys.executable


@functools.lru_cache(maxsize=1)
def _sfm_python():
    """A Python interpreter with the SfM stack (pycolmap + bag/align deps),
    used by process_sfm.py to auto-generate SfM poses. Same split-interpreter
    pattern as _splat_python. Returns None when no candidate qualifies —
    callers then tell the user how to proceed instead of crashing mid-job.
    Override with $LIDARSTUDIO_SFM_PYTHON.
    """
    seen = set()
    for cand in (
        os.environ.get("LIDARSTUDIO_SFM_PYTHON"),
        sys.executable,
        shutil.which("python3"),
        "/usr/bin/python3",
    ):
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            r = subprocess.run(
                [cand, "-c", "import pycolmap, rosbags, scipy, cv2"],
                capture_output=True,
                timeout=120,
            )
            if r.returncode == 0:
                if cand != sys.executable:
                    logging.info("SfM pose generation will use %s (has pycolmap)", cand)
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    return None


async def _job_splat(project_path, scan_path, options, queue, job=None):
    job = job if job is not None else {}
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

    pc_dir = proj / "pointclouds"

    # Explicit seed cloud (e.g. an edited one) pins the splat to that file;
    # otherwise the latest trainable project cloud is used.
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

    # Full-resolution training REQUIRES SfM poses. Odometry poses carry ~0.7°
    # of error — a >10px misregistration at downscale-1 focal lengths — so the
    # photometric loss can never converge: the model limps at opacity ~0.03
    # until MCMC relocation stops (0.8·iters) and then collapses to invisible
    # 1mm gaussians (observed 2026-07-12: 30k/ds1 run, 13.2 dB, dead splat
    # after 4 h). When the field is empty we resolve them below (after the
    # seed cloud is known, since alignment needs its trajectory sidecar):
    # reuse the project's generated npz, or run process_sfm.py to build one.

    # GPU-trained splat: explicit seed cloud if chosen, else the latest
    # trainable coloured cloud in the project.
    from lidarstudio import edit_ops

    if seed_p:
        pc_files = [seed_p]
    else:
        pc_files = sorted(pc_dir.glob("pointcloud_*.ply")) if pc_dir.exists() else []
        pc_files = [p for p in pc_files if not p.stem.endswith("_dense")] or pc_files
        # Training needs a trajectory sidecar; the latest cloud may be an edited
        # one without it. Prefer a trainable auto-seed when one exists.
        trainable = [p for p in pc_files if edit_ops._find_traj(p)]
        if trainable:
            pc_files = trainable

    # Monochromatic seed (the fast --mono generator, or an import without photo
    # colour): a grey seed trains a grey splat, so colour it from the scan
    # photos first — same occlusion-aware projection the generator uses — and
    # train on the coloured copy. Detection is by content (r==g==b everywhere),
    # since mono clouds share the generated-cloud naming.
    if pc_files:
        seed_cloud = pc_files[-1]
        loop = asyncio.get_event_loop()
        try:
            mono = await loop.run_in_executor(
                None, edit_ops.is_monochrome, str(seed_cloud)
            )
        except Exception:
            mono = False  # unreadable seed → let training surface the real error
        if mono:
            if edit_ops._find_traj(seed_cloud) is None:
                await queue.put(
                    {
                        "event": "error",
                        "message": (
                            f"{seed_cloud.name} is monochromatic and has no "
                            "trajectory sidecar (.traj.npz), so it can't be "
                            "coloured from the scan photos. Pick a generated "
                            "cloud, or regenerate it from the scan."
                        ),
                    }
                )
                return
            recoloured = seed_cloud.with_name(seed_cloud.stem + "_recoloured.ply")
            if (
                recoloured.exists()
                and recoloured.stat().st_mtime >= seed_cloud.stat().st_mtime
            ):
                await queue.put(
                    {
                        "event": "log",
                        "message": (
                            f"Monochromatic seed — reusing existing coloured "
                            f"copy {recoloured.name}"
                        ),
                    }
                )
            else:
                await queue.put(
                    {
                        "event": "progress",
                        "percent": 3,
                        "message": "Colouring monochromatic seed from scan photos…",
                    }
                )
                await queue.put(
                    {
                        "event": "log",
                        "message": (
                            f"{seed_cloud.name} has no photo colour — projecting "
                            "the scan's photos onto it before training (can take "
                            "a few minutes)…"
                        ),
                    }
                )
                try:
                    result = await loop.run_in_executor(
                        None,
                        edit_ops.recolour,
                        str(seed_cloud),
                        str(recoloured),
                        str(scan),
                    )
                except Exception as exc:
                    await queue.put(
                        {
                            "event": "error",
                            "message": f"Colouring the mono seed failed: {exc}",
                        }
                    )
                    return
                await queue.put(
                    {
                        "event": "log",
                        "message": (
                            f"Coloured {result['coloured']:,}/{result['total']:,} "
                            f"points → {recoloured.name}"
                        ),
                    }
                )
            pc_files[-1] = recoloured

    # Resolve SfM poses when the field was left empty: a previously generated
    # project file is reused at any downscale (better poses never hurt); at
    # downscale 1, where they are REQUIRED, a missing file is generated now
    # via process_sfm.py (COLMAP/GLOMAP + LiDAR alignment — CPU-heavy, on the
    # order of an hour or two, but a one-off per scan).
    if not sfm_poses:
        canonical = proj / "sfm" / f"sfm_viewmats_{ts}.npz"
        if canonical.exists():
            sfm_poses = str(canonical)
            await queue.put(
                {
                    "event": "log",
                    "message": f"Using the project's generated SfM poses: {canonical.name}",
                }
            )
        elif int(options.get("downscale", 1)) < 2:
            traj = edit_ops._find_traj(pc_files[-1]) if pc_files else None
            if traj is None:
                await queue.put(
                    {
                        "event": "error",
                        "message": (
                            "Downscale 1 needs SfM poses, and they can't be "
                            "auto-generated: the seed cloud has no trajectory "
                            "sidecar (.traj.npz) to align them to. Pick a "
                            "generated cloud, or use downscale 2+."
                        ),
                    }
                )
                return
            sfm_py = await asyncio.to_thread(_sfm_python)
            if sfm_py is None:
                await queue.put(
                    {
                        "event": "error",
                        "message": (
                            "Downscale 1 needs SfM poses, and no Python with "
                            "pycolmap was found to generate them (pip install "
                            "pycolmap, or set $LIDARSTUDIO_SFM_PYTHON). Either "
                            "install it, set an SfM poses .npz in the quality "
                            "panel, or use downscale 2+."
                        ),
                    }
                )
                return
            await queue.put(
                {
                    "event": "progress",
                    "percent": 1,
                    "message": (
                        "No SfM poses for this scan — generating them first "
                        "(COLMAP + LiDAR alignment, CPU-bound, can take "
                        "an hour or two; reused by every later run)…"
                    ),
                }
            )
            sfm_cmd = [
                sfm_py,
                str(ROOT / "process_sfm.py"),
                "--scan",
                str(scan),
                "--traj",
                str(traj),
                "--output",
                str(canonical),
                "--workspace",
                str(proj / "sfm" / f"work_{ts}"),
            ]
            proc = await asyncio.create_subprocess_exec(
                *sfm_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            job["proc"] = proc
            if job.get("cancelled"):
                proc.terminate()
            rc, err = await _stream_proc(proc, queue)
            job["proc"] = None
            if job.get("cancelled"):
                await queue.put(
                    {"event": "cancelled", "message": "Splat generation stopped."}
                )
                return
            if rc != 0 or not canonical.exists():
                await queue.put(
                    {
                        "event": "error",
                        "message": err
                        or f"SfM pose generation failed (exit code {rc})",
                    }
                )
                return
            sfm_poses = str(canonical)

    cmd = [
        _splat_python(),
        str(ROOT / "process_splat.py"),
        "--scan",
        str(scan),
        "--output",
        str(output_file),
        "--iterations",
        str(options.get("iterations", 7000)),
        # Quality knobs driven by the UI 'quality' slider (draft→max). The
        # shape/opacity/pose-association recipe constants (opacity-reg, flat-reg,
        # min-opacity, min-scale, cam-time-offset) come from process_splat.py's
        # own champion defaults, so we only pass the knobs that scale with quality.
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
        # Anisotropy cap (s_max/s_min). 20:1 champion; raise for thin structures.
        "--aniso-cap",
        str(float(options.get("aniso_cap", 20.0))),
        # Largest-axis cap in WORLD metres. Bounds per-gaussian pixel footprint
        # (hence gsplat's tile-intersection VRAM) independent of scene extent —
        # the old normalised-unit clamp let spread-out scenes grow 25 cm splats
        # and OOM a 24 GB card mid-run.
        "--max-scale",
        str(float(options.get("max_scale", 0.12))),
        # Depth-prior loss (Depth Anything V2 anchored to the LiDAR cloud),
        # aimed at near-field softness. Experimental: off by default until an
        # A/B run proves it; enable per-run via options.depth_loss ≈ 0.05.
        "--depth-loss",
        str(float(options.get("depth_loss", 0.0))),
        # In-training pose-opt defaults ON but diverges on these SfM/odometry
        # poses (the campaign retired it); keep it off for app runs.
        "--no-pose-opt",
        # Patch training: render random crops so GPU memory is bounded by the
        # patch, not the (downscale-1) frame size — lets high-quality runs fit
        # without OOM. Frames smaller than this render whole (no-op).
        "--patch-size",
        str(int(options.get("patch_size", 1600))),
        # On-disk frame cache: the undistorted uint8 frames can exceed host RAM
        # at downscale-1 / high undistort-scale; memmap them so RSS stays flat.
        "--memmap-frames",
    ]
    if sfm_poses:
        cmd += ["--sfm-poses", sfm_poses]
        # process_sfm.py stores the bundle-adjusted OPENCV_FISHEYE intrinsics
        # alongside the poses; undistorting with the same lens model the poses
        # were optimised under avoids baking a ~2px radial warp into training.
        try:
            import numpy as np

            with np.load(sfm_poses) as d:
                fisheye = d["fisheye_params"] if "fisheye_params" in d.files else None
        except Exception:
            fisheye = None  # hand-supplied npz without the extra key — fine
        if fisheye is not None and len(fisheye) == 8:
            cmd += [
                "--fisheye-intrinsics",
                ",".join(f"{float(v):.10g}" for v in fisheye),
            ]
            await queue.put(
                {
                    "event": "log",
                    "message": "Using SfM-refined fisheye intrinsics from the pose file.",
                }
            )
    if pc_files:
        cmd += ["--pointcloud", str(pc_files[-1])]
        await queue.put(
            {
                "event": "log",
                "message": f"Using existing point cloud: {pc_files[-1].name}",
            }
        )
        # Training needs the seed's trajectory sidecar; warn up front if missing.
        if edit_ops._find_traj(pc_files[-1]) is None:
            await queue.put(
                {
                    "event": "log",
                    "message": (
                        f"⚠ {pc_files[-1].name} has no trajectory sidecar "
                        "(.traj.npz), which training requires. Re-run the edit "
                        "(edits now carry the trajectory) or pick a generated cloud."
                    ),
                }
            )

    await queue.put(
        {"event": "progress", "percent": 5, "message": "Launching process_splat.py…"}
    )

    # expandable_segments cuts CUDA fragmentation, which is what tips heavy
    # (downscale-1 / 6M) runs into OOM on smaller-VRAM cards.
    splat_env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=splat_env,
        )
    except FileNotFoundError:
        await queue.put(
            {
                "event": "error",
                "message": "process_splat.py not found — see the stub script.",
            }
        )
        return

    # Register the process so a /cancel can kill it; honour a cancel that raced in.
    job["proc"] = proc
    if job.get("cancelled"):
        proc.terminate()

    rc, err = await _stream_proc(proc, queue)
    job["proc"] = None

    if job.get("cancelled"):
        await queue.put({"event": "cancelled", "message": "Splat generation stopped."})
    elif rc == 0 and output_file.exists():
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
            {
                "event": "error",
                "message": err or f"process_splat.py exited with code {rc}",
            }
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


async def project_delete_handler(request):
    """POST /api/project/delete — remove an output ``.ply`` and its sidecars.

    Restricted to files inside a project's ``pointclouds``/``splats`` folder, so
    it can't be used to delete arbitrary paths.
    """
    data = await request.json()
    path = (data.get("path") or "").strip()
    if not path:
        return web.json_response({"error": "path required"}, status=400)
    p = Path(path)
    if p.suffix.lower() != ".ply" or p.parent.name not in ("pointclouds", "splats"):
        return web.json_response(
            {"error": "only output .ply files can be deleted"}, status=403
        )
    if not p.exists():
        return web.json_response({"error": "file not found"}, status=404)
    removed = []
    # The .ply plus the sidecars that ride with it (viewer pose + trajectory).
    for f in (p, Path(str(p) + ".pose.json"), Path(str(p) + ".traj.npz")):
        try:
            if f.exists():
                f.unlink()
                removed.append(f.name)
        except OSError as exc:
            return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"deleted": removed})


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
            from lidarstudio import edit_ops

            shutil.copy2(src, out_path)
            _write_pose_sidecar(str(out_path), data.get("pose"))
            # Copy mode is frame-preserving (byte copy), so the source cloud's
            # trajectory sidecar still applies — carry it so a Save-As'd cloud
            # stays GPU-trainable, just like an edit-save does. (Bake mode above
            # changes the frame and deliberately does NOT carry it.) No-ops if
            # the source has no sidecar.
            edit_ops._copy_traj(str(src), str(out_path))
        trainable = edit_ops._find_traj(out_path) is not None if matrix is None else False
        return web.json_response(
            {"output": str(out_path), "baked": matrix is not None, "trainable": trainable}
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def edit_import_handler(request):
    """POST /api/edit/import?dir=<libdir>&name=<basename> — persist a
    browser-imported cloud/splat PLY to disk so the server-side edit ops can
    run on it. The request body is the raw .ply bytes.

    Files opened via the picker / drag-drop live only as a browser buffer, so
    delete/crop/etc. (which operate on a real file) have nothing to point at.
    The first edit uploads the bytes here, into the Library folder alongside
    project outputs, so an imported object becomes a normal editable,
    re-loadable artifact — identical to a Library-loaded file from then on.
    Returns {output}.
    """
    body = await request.read()
    if not body:
        return web.json_response({"error": "empty body"}, status=400)
    # Sanitise to a bare .ply basename — never trust the client with a path.
    raw = (request.query.get("name") or "imported").strip()
    base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(raw).name) or "imported"
    if not base.lower().endswith(".ply"):
        base += ".ply"
    d = (request.query.get("dir") or "").strip()
    out_dir = Path(d) if d else Path.home()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / base
        # Never clobber an existing Library file with the same name.
        if out_path.exists():
            out_path = Path(_unique_output(out_path, "imported"))
        out_path.write_bytes(body)
        return web.json_response({"output": str(out_path)})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Route registration ──────────────────────────────────────────────────────
# LidarStudio hosts the viewer at "/" and owns the static handler, so we attach
# only the LiDAR workflow's /api/* endpoints onto the shared aiohttp app.


def register_routes(app: web.Application) -> None:
    app.router.add_post("/api/edit/apply", edit_apply_handler)
    app.router.add_post("/api/edit/import", edit_import_handler)
    app.router.add_post("/api/edit/recolour", edit_recolour_handler)
    app.router.add_post("/api/edit/save_as", edit_save_as_handler)
    app.router.add_post("/api/browse", browse_handler)
    app.router.add_post("/api/browse/dir", browse_dir_handler)
    app.router.add_post("/api/project/create", project_create_handler)
    app.router.add_post("/api/project/open", project_open_handler)
    app.router.add_post("/api/scan/validate", scan_validate_handler)
    app.router.add_get("/api/system", system_health_handler)
    app.router.add_post("/api/process/start", process_start_handler)
    app.router.add_post("/api/process/cancel", process_cancel_handler)
    app.router.add_get("/api/process/events/{job_id}", process_events_handler)
    app.router.add_post("/api/project/outputs", project_outputs_handler)
    app.router.add_post("/api/project/delete", project_delete_handler)
    app.router.add_get("/api/scan/file", scan_file_handler)
    app.router.add_post("/api/scan/photos", scan_photos_handler)
    app.router.add_get("/api/scan/photo", scan_photo_handler)
    app.router.add_post("/api/scan/sweeps", scan_sweeps_handler)
    app.router.add_get("/api/scan/sweep", scan_sweep_handler)
