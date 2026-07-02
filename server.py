#!/usr/bin/env python3
"""
LidarStudio — HTTP server

Serves the Three.js point-cloud / splat editor and the LiDAR workflow API
(cloud/splat generation, editing, projects — see lidar_jobs.py).

Usage:
    pip install aiohttp
    python server.py [--port 8080]

Network exposure:
    Binds 127.0.0.1 by default. Pass --host 0.0.0.0 to serve the viewer to
    other machines (e.g. a VR headset). The LiDAR /api/* endpoints read and
    write the local filesystem, so they stay loopback-only even then unless
    --allow-remote-fs is also given.
"""

import argparse
import ipaddress
import logging
import ssl
import subprocess
import tempfile
from pathlib import Path

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("lidarstudio")

ROOT = Path(__file__).parent


async def healthz_handler(request):
    return web.Response(text="ok")


def _is_loopback_peer(remote) -> bool:
    """True if the request came from this machine (or a non-IP transport)."""
    if not remote:
        return True
    try:
        addr = ipaddress.ip_address(remote)
    except ValueError:
        return False
    if addr.is_loopback:
        return True
    mapped = getattr(addr, "ipv4_mapped", None)  # e.g. ::ffff:127.0.0.1
    return bool(mapped and mapped.is_loopback)


@web.middleware
async def local_fs_api_guard_mw(request, handler):
    """Keep the LiDAR /api/* endpoints loopback-only unless --allow-remote-fs.

    Those endpoints browse directories and read/write files anywhere the
    server's user can, so they must not be reachable from the network just
    because the viewer is (e.g. --host 0.0.0.0 for a VR headset)."""
    if (
        request.path.startswith("/api/")
        and not request.app.get("allow_remote_fs")
        and not _is_loopback_peer(request.remote)
    ):
        return web.json_response(
            {"error": "filesystem API is loopback-only; "
                      "start the server with --allow-remote-fs to enable remote clients"},
            status=403,
        )
    return await handler(request)


@web.middleware
async def cross_origin_isolation_mw(request, handler):
    """Add COOP/COEP headers so the page is cross-origin isolated.

    This makes SharedArrayBuffer available, which lets the Gaussian-splat
    viewer run its depth sort zero-copy in a worker (big win on mobile/VR).
    All assets here are same-origin, so isolation does not break any loads.
    """
    resp = await handler(request)
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return resp


# Root-level files the viewer may fetch (styles, scene JSON, models, …).
# Everything else at the repo root — Python source, .git, dotfiles — is not served.
_ROOT_ASSET_EXTENSIONS = {".css", ".html", ".ico", ".json", ".glb", ".gltf",
                          ".stl", ".ply", ".png", ".jpg", ".jpeg", ".zip"}


async def root_asset_handler(request):
    """Serve an allowlisted asset file sitting directly in the project root."""
    name = request.match_info["name"]
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise web.HTTPNotFound()
    p = ROOT / name
    if (
        p.suffix.lower() not in _ROOT_ASSET_EXTENSIONS
        or not p.is_file()
        or p.resolve().parent != ROOT.resolve()
    ):
        raise web.HTTPNotFound()
    return web.FileResponse(p)


def create_app(allow_remote_fs=False):
    app = web.Application(middlewares=[local_fs_api_guard_mw, cross_origin_isolation_mw])
    app["allow_remote_fs"] = allow_remote_fs

    async def index_handler(request):
        return web.FileResponse(ROOT / "threejs_scene.html")

    app.router.add_get("/healthz", healthz_handler)
    app.router.add_get("/", index_handler)

    # LiDAR workflow API (cloud/splat generation, editing, projects) —
    # registered before the static catch-all so the /api/* routes win.
    try:
        import lidar_jobs
        lidar_jobs.register_routes(app)
    except Exception as e:  # pragma: no cover - keep the viewer usable if jobs fail to load
        logging.getLogger("server").warning(f"LiDAR workflow routes unavailable: {e}")

    # Static assets, allowlisted: the JS modules, npm packages, and specific
    # root-level file types. Deliberately NOT a blanket add_static("/", ROOT),
    # which would expose the server source, .git, and everything else here.
    app.router.add_static("/js/", ROOT / "js")
    app.router.add_static("/node_modules/", ROOT / "node_modules")
    app.router.add_get("/{name}", root_asset_handler)
    return app


def _make_ssl_context(certfile=None, keyfile=None):
    """Create an SSL context, generating a self-signed cert if none provided."""
    if certfile and keyfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        return ctx

    cert_dir = Path(tempfile.mkdtemp(prefix="lidarstudio-ssl-"))
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


def main():
    parser = argparse.ArgumentParser(description="LidarStudio server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to serve the LAN)")
    parser.add_argument("--allow-remote-fs", action="store_true", default=False,
                        help="Let non-loopback clients use the LiDAR /api/* endpoints "
                             "(they read/write the local filesystem; needed for e.g. a VR "
                             "headset running the full workflow)")
    parser.add_argument("--ssl", action="store_true", default=False,
                        help="Enable HTTPS (auto-generates self-signed cert if --cert/--key not given)")
    parser.add_argument("--cert", default=None, help="Path to SSL certificate file")
    parser.add_argument("--key", default=None, help="Path to SSL private key file")
    args = parser.parse_args()

    ssl_ctx = None
    if args.ssl or args.cert:
        ssl_ctx = _make_ssl_context(args.cert, args.key)

    try:
        host_is_local = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        host_is_local = args.host in ("localhost",)
    if not host_is_local:
        log.warning(f"Binding {args.host}: the viewer is network-visible.")
        if args.allow_remote_fs:
            log.warning("--allow-remote-fs: remote clients can browse and read/write "
                        "this machine's filesystem via the LiDAR API.")
        else:
            log.info("LiDAR filesystem APIs stay loopback-only "
                     "(pass --allow-remote-fs to enable remote clients).")

    app = create_app(allow_remote_fs=args.allow_remote_fs)

    scheme = "https" if ssl_ctx else "http"
    log.info(f"Starting server on {scheme}://{args.host}:{args.port}")
    log.info(f"Open {scheme}://localhost:{args.port}/ in your browser")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    main()
