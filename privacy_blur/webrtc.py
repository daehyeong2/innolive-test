from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import PrivacyBlurNotReadyError

if TYPE_CHECKING:
    from .core import FacePrivacyFilter


class PrivacyVideoTrack:
    def __new__(cls, source_track: Any, privacy_filter: FacePrivacyFilter):
        track_type = _privacy_video_track_type()
        return track_type(source_track, privacy_filter)


def _privacy_video_track_type():
    try:
        from aiortc import VideoStreamTrack
    except ImportError as exc:
        raise PrivacyBlurNotReadyError("aiortc is required to protect a WebRTC video track.") from exc

    class _PrivacyVideoTrack(VideoStreamTrack):
        kind = "video"

        def __init__(self, source_track: Any, privacy_filter: FacePrivacyFilter):
            super().__init__()
            self.source_track = source_track
            self.privacy_filter = privacy_filter

        async def recv(self):
            frame = await self.source_track.recv()
            return await self.privacy_filter.apply_async(frame)

        def stop(self):
            stop = getattr(self.source_track, "stop", None)
            if callable(stop):
                stop()
            self.privacy_filter.close()
            return super().stop()

    return _PrivacyVideoTrack
