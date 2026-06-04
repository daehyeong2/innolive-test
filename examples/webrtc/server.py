#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from privacy_blur import (  # noqa: E402
    PrivacyFilterSettings,
    get_privacy_filter,
    protect_video_track,
)

logger = logging.getLogger("webrtc_example")
ROOT = Path(__file__).resolve().parent
pcs: set[RTCPeerConnection] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local WebRTC camera test for privacy_blur."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--profile",
        choices=["privacy", "balanced", "fast"],
        default="fast",
    )
    parser.add_argument("--asset", help="Privacy filter asset path.")
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


async def index(request: web.Request) -> web.Response:
    return web.FileResponse(ROOT / "index.html")


async def offer(request: web.Request) -> web.Response:
    params = await request.json()
    offer_description = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"],
    )

    pc = RTCPeerConnection()
    pcs.add(pc)
    privacy_filter = request.app["privacy_filter"]

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Received %s track.", track.kind)
        if track.kind == "video":
            pc.addTrack(protect_video_track(track, privacy_filter))

    await pc.setRemoteDescription(offer_description)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        ),
    )


async def on_shutdown(app: web.Application) -> None:
    await asyncio.gather(*(pc.close() for pc in pcs), return_exceptions=True)
    pcs.clear()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    app = web.Application()
    app["privacy_filter"] = get_privacy_filter(
        PrivacyFilterSettings(
            profile=args.profile,
            asset_path=args.asset,
            accelerator=args.accelerator,
            debug=args.debug,
        )
    )
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
