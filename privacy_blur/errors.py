from __future__ import annotations


class PrivacyBlurError(Exception):
    """Base exception for privacy blur failures."""


class PrivacyBlurNotReadyError(PrivacyBlurError):
    """Raised when models or runtime dependencies are missing."""


class UnsupportedFrameTypeError(PrivacyBlurError):
    """Raised when a frame type cannot be processed."""

