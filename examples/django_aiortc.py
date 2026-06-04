from __future__ import annotations

from pathlib import Path
import sys

from aiortc import RTCPeerConnection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_blur import get_privacy_filter, protect_video_track


pc = RTCPeerConnection()
privacy_filter = get_privacy_filter()


@pc.on("track")
def on_track(track):
    if track.kind == "video":
        pc.addTrack(protect_video_track(track, privacy_filter))
