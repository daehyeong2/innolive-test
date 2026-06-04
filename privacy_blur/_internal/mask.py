from __future__ import annotations

import numpy as np

from .config import InternalConfig
from .types import Detection

try:
    import cv2
except ImportError:
    cv2 = None


def build_privacy_mask(
    frame_shape: tuple[int, ...],
    detections: list[Detection],
    config: InternalConfig,
) -> np.ndarray:
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for detection in detections:
        x1, y1, x2, y2 = _expanded_box(detection, width, height, config)
        if x2 <= x1 or y2 <= y1:
            continue

        if cv2 is not None:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            mask[y1:y2, x1:x2] = 255

    if cv2 is not None and mask.any():
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5)

    return mask


def _expanded_box(
    detection: Detection,
    width: int,
    height: int,
    config: InternalConfig,
) -> tuple[int, int, int, int]:
    box_w = detection.x2 - detection.x1
    box_h = detection.y2 - detection.y1
    cx = detection.x1 + box_w / 2
    cy = detection.y1 + box_h / 2
    expanded_w = box_w * config.box_expand_w
    expanded_h = box_h * config.box_expand_h

    x1 = max(0, int(round(cx - expanded_w / 2)))
    y1 = max(0, int(round(cy - expanded_h / 2)))
    x2 = min(width, int(round(cx + expanded_w / 2)))
    y2 = min(height, int(round(cy + expanded_h / 2)))
    return x1, y1, x2, y2
