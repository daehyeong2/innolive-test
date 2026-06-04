from __future__ import annotations

import numpy as np

from .blur import apply_privacy_blur
from .config import InternalConfig
from .detector import FaceDetectionService
from .mask import build_privacy_mask
from .types import Detection


class FacePrivacyProcessor:
    def __init__(self, config: InternalConfig, detection_service: FaceDetectionService):
        self.config = config
        self._detection_service = detection_service
        self._frame_index = 0
        self._held: list[Detection] = []
        self._hold_remaining = 0

    def process(self, frame: np.ndarray) -> np.ndarray:
        self._frame_index += 1
        if self._should_refresh():
            self._held = self._detection_service.detect(frame)
            self._hold_remaining = self.config.hold_frames
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1

        mask = build_privacy_mask(frame.shape, self._held, self.config)
        return apply_privacy_blur(frame, mask, self.config)

    def reset(self) -> None:
        self._frame_index = 0
        self._held = []
        self._hold_remaining = 0

    def _should_refresh(self) -> bool:
        interval = max(1, self.config.detect_every_n_frames)
        if self._frame_index == 1:
            return True
        if self._hold_remaining <= 0:
            return True
        return (self._frame_index - 1) % interval == 0
