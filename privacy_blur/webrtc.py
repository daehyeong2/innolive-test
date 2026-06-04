from __future__ import annotations

from .errors import PrivacyFilterNotReadyError
from .public import PrivacyFilter, get_privacy_filter


class PrivacyVideoTrack:
    def __new__(cls, source_track, privacy_filter: PrivacyFilter | None = None):
        track_type = _privacy_video_track_type()
        return track_type(source_track, privacy_filter)


def protect_video_track(source_track, privacy_filter: PrivacyFilter | None = None):
    return PrivacyVideoTrack(
        source_track=source_track,
        privacy_filter=privacy_filter,
    )


def _privacy_video_track_type():
    try:
        from aiortc import VideoStreamTrack
    except ImportError as exc:
        raise PrivacyFilterNotReadyError(
            "aiortc is required to protect a video track."
        ) from exc

    class _PrivacyVideoTrack(VideoStreamTrack):
        def __init__(
            self,
            source_track,
            privacy_filter: PrivacyFilter | None = None,
        ):
            super().__init__()
            self.source_track = source_track
            self.privacy_filter = privacy_filter or get_privacy_filter()

        async def recv(self):
            frame = await self.source_track.recv()
            return await self.privacy_filter.apply_async(frame)

    return _PrivacyVideoTrack
