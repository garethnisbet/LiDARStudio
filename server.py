#!/usr/bin/env python3
"""
Robot Visualisation — WebSocket + HTTP Server

Serves the Three.js viewer and provides a two-way WebSocket API
for remote robot control.

Usage:
    pip install aiohttp
    python server.py [--port 8000] [--config meca500_config.json]

WebSocket endpoint: ws://localhost:8000/ws

API Protocol (JSON over WebSocket):
────────────────────────────────────
Commands (send TO the viewer):
  {"cmd": "getState"}
  {"cmd": "setJoints", "angles": [j1, j2, ...jN]}                # degrees (all joints)
  {"cmd": "setSingleJoint", "index": i, "angle": deg}           # set one joint by index
  {"cmd": "home"}                                                 # all joints to 0
  {"cmd": "setMode", "mode": "FK"}                                # "FK" or "IK"
  {"cmd": "setIKTarget", "position": [x,y,z], "orientation": [a,b,g]}  # mm, deg (Z-up)
  {"cmd": "moveTo", "position": [x,y,z], "orientation": [a,b,g]}       # switch to IK + set target
  {"cmd": "translateDevice", "delta": [dx,dy,dz], "space": "parent"}   # mm; space: parent|local|world
  {"cmd": "rotateDevice", "delta": [rx,ry,rz], "space": "parent"}      # deg; space: parent|local|world
  {"cmd": "translateObject", "name": "Cube", "delta": [dx,dy,dz], "space": "parent"}
  {"cmd": "rotateObject", "name": "Cube", "delta": [rx,ry,rz], "space": "parent"}
  {"cmd": "setPlatformPose", "pose": [x,y,z,rx,ry,rz]}                 # hexapod: set pose (mm, deg)
  {"cmd": "hexapodFK", "pose": [x,y,z,rx,ry,rz]}                       # hexapod FK: pose → leg lengths
  {"cmd": "hexapodIK", "legLengths": [l1..l6]}                         # hexapod IK: leg lengths (mm) → pose
  {"cmd": "getLegLengths"}                                               # hexapod: get current leg lengths
  {"cmd": "setLegLengths", "legLengths": [l1..l6]}                     # hexapod: set pose via leg lengths

State (sent FROM the viewer):
  {
    "type": "state",
    "joints": [j1..jN],          # degrees (all joints)
    "eePosition": [x, y, z],    # mm, Z-up
    "eeOrientation": [a, b, g], # degrees, Euler (a=Rx, b=Ry, g=Rz) relative to home (home = [0, 90, 0])
    "mode": "FK" | "IK",
    "ikError": null | <mm>
  }

Session routing:
  Viewers connect with ?role=viewer&session=<id>
  Controllers connect with ?role=controller&session=<id>
  If no session is given for a controller, messages are broadcast to all viewers.
  GET /sessions  →  list active session IDs
"""

import argparse
import asyncio
import json
import logging
import ssl
import subprocess
import tempfile
import uuid
from pathlib import Path

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("robot-server")

ROOT = Path(__file__).parent

# sessions[session_id] = set of viewer websockets
sessions: dict = {}
# session_controllers[session_id | None] = set of controller websockets
# None key = "all sessions" (no session filter)
session_controllers: dict = {}


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _viewer_count() -> int:
    return sum(len(v) for v in sessions.values())


def _controller_count() -> int:
    return sum(len(v) for v in session_controllers.values())


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    role = request.query.get("role", "controller")
    peer = request.remote

    if role == "viewer":
        session_id = request.query.get("session") or _short_id()
        sessions.setdefault(session_id, set()).add(ws)
        log.info(
            f"viewer [{session_id}] connected from {peer}"
            f"  (sessions={len(sessions)}, viewers={_viewer_count()}, controllers={_controller_count()})"
        )
        # Confirm assigned session ID back to the viewer
        await ws.send_json({"type": "session", "id": session_id})
    else:
        session_id = request.query.get("session") or None
        session_controllers.setdefault(session_id, set()).add(ws)
        label = session_id if session_id else "*"
        log.info(
            f"controller [{label}] connected from {peer}"
            f"  (sessions={len(sessions)}, viewers={_viewer_count()}, controllers={_controller_count()})"
        )

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"error": "invalid JSON"})
                    continue

                if role == "controller":
                    # Route to the targeted session's viewers, or all viewers if no session filter
                    if session_id:
                        target_viewers = list(sessions.get(session_id, set()))
                    else:
                        target_viewers = [v for vset in sessions.values() for v in vset]
                    for v in target_viewers:
                        try:
                            await v.send_json(data)
                        except Exception:
                            for s in sessions.values():
                                s.discard(v)

                elif role == "viewer":
                    # Forward state to controllers targeting this session + unfiltered controllers
                    target_controllers = list(
                        session_controllers.get(session_id, set()) |
                        session_controllers.get(None, set())
                    )
                    for c in target_controllers:
                        try:
                            await c.send_json(data)
                        except Exception:
                            for s in session_controllers.values():
                                s.discard(c)

            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break
    finally:
        if role == "viewer":
            s = sessions.get(session_id, set())
            s.discard(ws)
            if not s:
                sessions.pop(session_id, None)
            log.info(
                f"viewer [{session_id}] disconnected from {peer}"
                f"  (sessions={len(sessions)}, viewers={_viewer_count()}, controllers={_controller_count()})"
            )
        else:
            label = session_id if session_id else "*"
            s = session_controllers.get(session_id, set())
            s.discard(ws)
            if not s:
                session_controllers.pop(session_id, None)
            log.info(
                f"controller [{label}] disconnected from {peer}"
                f"  (sessions={len(sessions)}, viewers={_viewer_count()}, controllers={_controller_count()})"
            )

    return ws


async def healthz_handler(request):
    return web.Response(text="ok")


async def sessions_handler(request):
    """Return list of active viewer sessions."""
    data = [
        {"id": sid, "viewers": len(vws)}
        for sid, vws in sessions.items()
        if vws
    ]
    return web.json_response(data)


@web.middleware
async def cross_origin_isolation_mw(request, handler):
    """Add COOP/COEP headers so the page is cross-origin isolated.

    This makes SharedArrayBuffer available, which lets the Gaussian-splat
    viewer run its depth sort zero-copy in a worker (big win on mobile/VR).
    All assets here are same-origin, so isolation does not break any loads.
    """
    resp = await handler(request)
    # The WebSocket upgrade response is already sent during handler execution;
    # don't try to mutate its headers.
    if not isinstance(resp, web.WebSocketResponse):
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return resp


def create_app(config_path=None):
    app = web.Application(middlewares=[cross_origin_isolation_mw])

    # LidarStudio boots into an empty editor scene by default ("none" = no robot).
    config_name = Path(config_path).name if config_path else "none"

    async def index_handler(request):
        if "config" not in request.query:
            raise web.HTTPFound(f"/?config={config_name}")
        return web.FileResponse(ROOT / "threejs_scene.html")

    app.router.add_get("/healthz", healthz_handler)
    app.router.add_get("/sessions", sessions_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", index_handler)

    # LiDAR workflow API (cloud/splat generation, projects) — registered before
    # the static catch-all so the /api/* routes win.
    try:
        import lidar_jobs
        lidar_jobs.register_routes(app)
    except Exception as e:  # pragma: no cover - keep the viewer usable if jobs fail to load
        logging.getLogger("server").warning(f"LiDAR workflow routes unavailable: {e}")

    app.router.add_static("/", ROOT, show_index=False)
    return app


def _make_ssl_context(certfile=None, keyfile=None):
    """Create an SSL context, generating a self-signed cert if none provided."""
    if certfile and keyfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        return ctx

    cert_dir = Path(tempfile.mkdtemp(prefix="robot-ssl-"))
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ], check=True, capture_output=True)
    log.info(f"Generated self-signed certificate in {cert_dir}")

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


async def _run_https(app, host, port, ssl_ctx):
    """Run HTTPS on `port`."""
    runner = web.AppRunner(app)
    await runner.setup()
    https_site = web.TCPSite(runner, host, port, ssl_context=ssl_ctx)
    await https_site.start()

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Robot Visualisation WebSocket Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--config", default="none",
                        help="Optional robot config JSON; default 'none' = empty editor scene")
    parser.add_argument("--ssl", action="store_true", default=False,
                        help="Enable HTTPS (auto-generates self-signed cert if --cert/--key not given)")
    parser.add_argument("--cert", default=None, help="Path to SSL certificate file")
    parser.add_argument("--key", default=None, help="Path to SSL private key file")
    args = parser.parse_args()

    ssl_ctx = None
    if args.ssl or args.cert:
        ssl_ctx = _make_ssl_context(args.cert, args.key)

    app = create_app(config_path=args.config)
    config_name = Path(args.config).name

    if ssl_ctx:
        https_port = args.port
        log.info(f"Starting HTTPS server on https://{args.host}:{https_port}")
        log.info(f"Open https://localhost:{https_port}/?config={config_name} in your browser")
        log.info(f"WebSocket API at wss://localhost:{https_port}/ws")
        log.info(f"Robot config: {args.config}")
        asyncio.run(_run_https(app, args.host, https_port, ssl_ctx))
    else:
        log.info(f"Starting server on http://{args.host}:{args.port}")
        log.info(f"Open http://localhost:{args.port}/?config={config_name} in your browser")
        log.info(f"WebSocket API at ws://localhost:{args.port}/ws")
        log.info(f"Active sessions at http://localhost:{args.port}/sessions")
        log.info(f"Robot config: {args.config}")
        web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
