from __future__ import annotations

import numpy as np

from .config import InternalConfig

try:
    import cv2
except ImportError:
    cv2 = None


def apply_privacy_blur(
    frame: np.ndarray,
    mask: np.ndarray,
    config: InternalConfig,
) -> np.ndarray:
    output = frame if config.inplace else frame.copy()
    if mask.size == 0 or not mask.any():
        return output

    if cv2 is None:
        output[mask > 0] = 0
        return output

    sigma = max(12, int(min(frame.shape[:2]) * 0.04))
    blurred = cv2.GaussianBlur(output, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = blurred.astype(np.float32) * alpha + output.astype(np.float32) * (1.0 - alpha)
    output[...] = np.clip(blended, 0, 255).astype(output.dtype)
    return output
