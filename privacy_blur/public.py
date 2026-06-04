from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import threading
from typing import Any

import numpy as np

from .errors import (
    PrivacyBlurError,
    PrivacyFilterNotReadyError,
    UnsupportedFrameTypeError,
)
from ._internal.runtime import PrivacyRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrivacyFilterSettings:
    profile: str = "privacy"
    asset_path: str | None = None
    accelerator: str = "auto"
    worker_count: int = 1
    realtime: bool = True
    inplace: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        if self.profile not in {"privacy", "balanced", "fast"}:
            raise PrivacyBlurError(
                "profile must be one of: privacy, balanced, fast"
            )
        if self.accelerator not in {"auto", "cpu", "gpu"}:
            raise PrivacyBlurError(
                "accelerator must be one of: auto, cpu, gpu"
            )
        if self.worker_count < 1:
            raise PrivacyBlurError("worker_count must be at least 1")


class PrivacyFilter:
    def __init__(self, settings: PrivacyFilterSettings | None = None):
        self.settings = settings or PrivacyFilterSettings()
        self._runtime = PrivacyRuntime(self.settings)
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.worker_count,
            thread_name_prefix="privacy-filter",
        )

    def apply(self, frame):
        """
        Accepts:
        - numpy.ndarray image
        - av.VideoFrame

        Returns:
        - same style as input where possible
        """
        try:
            if isinstance(frame, np.ndarray):
                return self._runtime.process_ndarray(frame)

            video_frame_type = _video_frame_type()
            if video_frame_type is not None and isinstance(frame, video_frame_type):
                return self._apply_video_frame(frame)

            raise UnsupportedFrameTypeError(
                f"Unsupported frame type: {type(frame).__name__}"
            )
        except PrivacyBlurError:
            raise
        except Exception as exc:
            raise PrivacyFilterNotReadyError(
                "Privacy filter could not process the frame."
            ) from exc

    async def apply_async(self, frame):
        """
        Async wrapper for WebRTC event loop.
        Must not block the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.apply, frame)

    def reset(self) -> None:
        self._runtime.reset()

    def stats(self) -> dict:
        return self._runtime.stats()

    def _apply_video_frame(self, frame):
        video_frame_type = _video_frame_type()
        if video_frame_type is None:
            raise UnsupportedFrameTypeError("av.VideoFrame is not available")

        image = frame.to_ndarray(format="bgr24")
        processed = self._runtime.process_ndarray(image)
        protected = video_frame_type.from_ndarray(processed, format="bgr24")
        protected.pts = frame.pts
        protected.time_base = frame.time_base
        return protected


_default_filter: PrivacyFilter | None = None
_default_lock = threading.Lock()


def create_privacy_filter(
    settings: PrivacyFilterSettings | None = None,
) -> PrivacyFilter:
    return PrivacyFilter(settings=settings)


def get_privacy_filter(
    settings: PrivacyFilterSettings | None = None,
) -> PrivacyFilter:
    """
    Process-level singleton.
    Django worker process 안에서 한 번만 생성되도록 한다.
    """
    global _default_filter
    with _default_lock:
        if _default_filter is None:
            _default_filter = create_privacy_filter(settings=settings)
        elif settings is not None and settings != _default_filter.settings:
            logger.debug("Existing privacy filter singleton is being reused.")
        return _default_filter


def _video_frame_type() -> Any | None:
    try:
        from av import VideoFrame
    except ImportError:
        return None
    return VideoFrame
