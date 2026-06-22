from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy_blur import (  # noqa: E402
    configure,
    initialize_runtime,
    protect_video_track,
    register_reference_face,
    runtime_stats,
    shutdown_runtimes,
)

LOGGER = logging.getLogger("privacy_blur.webrtc_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local WebRTC privacy blur test site."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reference", default=str(PROJECT_ROOT / "me.jpeg"))
    parser.add_argument("--face-imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=4.0)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback for local development. Production fails fast without CUDA by default.",
    )
    return parser.parse_args()


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(ROOT / "index.html")


def create_app(args: argparse.Namespace) -> web.Application:
    register_reference_face(args.reference)
    config = configure(
        face_imgsz=args.face_imgsz,
        device=args.device,
        require_gpu=not args.allow_cpu,
        use_half=args.half,
        max_batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    runtime = initialize_runtime(config)
    app = web.Application()
    app["pcs"] = set()
    app["runtime"] = runtime

    async def offer(request: web.Request) -> web.Response:
        params = await request.json()
        peer_connection = RTCPeerConnection()
        app["pcs"].add(peer_connection)

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            LOGGER.info("connection state: %s", peer_connection.connectionState)
            if peer_connection.connectionState in {"failed", "closed", "disconnected"}:
                await peer_connection.close()
                app["pcs"].discard(peer_connection)

        @peer_connection.on("track")
        def on_track(track) -> None:
            LOGGER.info("track received: %s", track.kind)
            if track.kind == "video":
                protected_track = protect_video_track(track)
                peer_connection.addTrack(protected_track)

        offer_description = RTCSessionDescription(
            sdp=params["sdp"], type=params["type"]
        )
        await peer_connection.setRemoteDescription(offer_description)

        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {
                    "sdp": peer_connection.localDescription.sdp,
                    "type": peer_connection.localDescription.type,
                }
            ),
        )

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ready",
                "peers": len(app["pcs"]),
                "runtimes": runtime_stats(),
            }
        )

    async def on_shutdown(application: web.Application) -> None:
        coroutines = [pc.close() for pc in application["pcs"]]
        if coroutines:
            await asyncio.gather(*coroutines)
        application["pcs"].clear()
        shutdown_runtimes()

    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    web.run_app(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
