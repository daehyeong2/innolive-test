from __future__ import annotations

from .api import (
    configure,
    get_default_config,
    new_filter,
    process_frame,
    protect_video_track,
    register_reference_face,
)
from .config import PrivacyBlurConfig
from .errors import PrivacyBlurError, PrivacyBlurNotReadyError, UnsupportedFrameTypeError

__all__ = [
    "FacePrivacyEngine",
    "FacePrivacyFilter",
    "PrivacyBlurConfig",
    "PrivacyBlurError",
    "PrivacyBlurNotReadyError",
    "UnsupportedFrameTypeError",
    "configure",
    "get_default_config",
    "new_filter",
    "process_frame",
    "protect_video_track",
    "register_reference_face",
]


def __getattr__(name: str):
    if name == "FacePrivacyEngine":
        from .core import FacePrivacyEngine

        return FacePrivacyEngine
    if name == "FacePrivacyFilter":
        from .core import FacePrivacyFilter

        return FacePrivacyFilter
    raise AttributeError(f"module 'privacy_blur' has no attribute {name!r}")
