"""Job output is teed to <project>/logs so it can be read after the browser
has consumed the live stream."""

import asyncio

from lidarstudio.lidar_jobs import _job_log_path, _LoggedQueue


def _run(coro):
    return asyncio.run(coro)  # closes its loop; a leaked one fails a later test


def test_every_event_is_written_once_and_still_queued(tmp_path):
    path = _job_log_path(tmp_path, "splat", "abc123")
    assert path.parent == tmp_path / "logs"
    assert path.name.startswith("splat_") and path.name.endswith("_abc123.log")

    async def go():
        q = _LoggedQueue(path)
        await q.put({"event": "log", "message": "  frame store: 179 in VRAM"})
        await q.put({"event": "progress", "percent": 42, "message": "Training"})
        await q.put({"event": "error", "message": "boom"})
        q.put_nowait({"event": "result", "filename": "x.ply"})
        await q.put(None)  # sentinel: queued, not logged
        q.close()
        return [q.get_nowait() for _ in range(q.qsize())]

    got = _run(go())
    assert len(got) == 5 and got[-1] is None

    lines = path.read_text().splitlines()
    assert len(lines) == 4  # no double write via put() → put_nowait()
    assert lines[0].endswith("  frame store: 179 in VRAM")
    assert lines[1].endswith("[42%] Training")
    assert lines[2].endswith("[error] boom")
    assert lines[3].endswith('[result]  {"filename": "x.ply"}')


def test_unwritable_project_never_fails_the_job(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")  # logs/ can't be made under a regular file

    async def go():
        q = _LoggedQueue(blocker / "logs" / "j.log")
        await q.put({"event": "log", "message": "hi"})
        q.close()
        return q.log_path, q.get_nowait()

    log_path, ev = _run(go())
    assert log_path is None and ev["message"] == "hi"
