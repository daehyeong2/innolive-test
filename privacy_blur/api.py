from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

from .config import PrivacyBlurConfig

if TYPE_CHECKING:
    from .core import FacePrivacyFilter


_default_config = PrivacyBlurConfig()
_default_lock = threading.Lock()


def register_reference_face(path: str | Path) -> PrivacyBlurConfig:
    """Register the face image that should be excluded from mosaic/blur."""
    global _default_config
    reference_path = Path(path).expanduser().resolve()
    with _default_lock:
        _default_config = replace(
            _default_config,
            enable_identity_exclusion=True,
            me_image_path=reference_path,
        )
        return _default_config


def configure(**overrides: Any) -> PrivacyBlurConfig:
    global _default_config
    with _default_lock:
        _default_config = replace(_default_config, **overrides)
        return _default_config


def get_default_config() -> PrivacyBlurConfig:
    with _default_lock:
        return _default_config


def new_filter(config: PrivacyBlurConfig | None = None, **overrides: Any) -> FacePrivacyFilter:
    from .core import create_filter

    base = config or get_default_config()
    if overrides:
        base = replace(base, **overrides)
    return create_filter(base)


def protect_video_track(
    source_track: Any,
    privacy_filter: FacePrivacyFilter | None = None,
    config: PrivacyBlurConfig | None = None,
    **overrides: Any,
) -> Any:
    from .webrtc import PrivacyVideoTrack

    filter_to_use = privacy_filter or new_filter(config, **overrides)
    return PrivacyVideoTrack(source_track, filter_to_use)


def process_frame(frame: Any, privacy_filter: FacePrivacyFilter | None = None) -> Any:
    filter_to_use = privacy_filter or new_filter()
    return filter_to_use.apply(frame)
