from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from privacy_blur.errors import PrivacyFilterNotReadyError
from .config import InternalConfig
from .types import Detection

logger = logging.getLogger(__name__)


class YoloFaceDetector:
    def __init__(self, config: InternalConfig):
        self.config = config
        self._engine: Any | None = None
        self._active_device: str | None = None
        self._load_count = 0
        self._warned_accelerator_fallback = False

    def detect(self, frame: np.ndarray) -> list[Detection]:
        engine = self._ensure_engine()
        try:
            results = engine.predict(
                source=frame,
                imgsz=self.config.imgsz,
                conf=self.config.conf_threshold,
                device=self._active_device,
                verbose=False,
            )
        except Exception as exc:
            raise PrivacyFilterNotReadyError(
                "Privacy filter could not analyze the frame."
            ) from exc

        if not results:
            return []
        return self._extract(results[0])

    def reset(self) -> None:
        self._engine = None
        self._active_device = None

    def stats(self) -> dict:
        return {
            "ready": self._engine is not None,
            "device": self._active_device,
            "load_count": self._load_count,
        }

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine

        if not os.path.exists(self.config.asset_path):
            raise PrivacyFilterNotReadyError(
                f"Privacy filter asset was not found: {self.config.asset_path}"
            )

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise PrivacyFilterNotReadyError(
                "Privacy filter runtime dependency is not available."
            ) from exc

        self._active_device = self._resolve_device()
        try:
            self._engine = YOLO(self.config.asset_path)
        except Exception as exc:
            raise PrivacyFilterNotReadyError(
                "Privacy filter asset could not be loaded."
            ) from exc

        self._load_count += 1
        logger.debug("Privacy filter asset loaded.")
        return self._engine

    def _resolve_device(self) -> str:
        if self.config.device == "cpu":
            return "cpu"

        has_cuda = _cuda_available()
        if self.config.device == "cuda":
            if has_cuda:
                return "cuda"
            if not self._warned_accelerator_fallback:
                logger.warning("Requested accelerator is unavailable; using CPU.")
                self._warned_accelerator_fallback = True
            return "cpu"

        return "cuda" if has_cuda else "cpu"

    def _extract(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        if xyxy is None or xyxy.size == 0:
            return []

        scores = _to_numpy(getattr(boxes, "conf", None))
        if scores is None or len(scores) != len(xyxy):
            scores = np.ones((len(xyxy),), dtype=np.float32)

        detections: list[Detection] = []
        for box, score in zip(xyxy, scores):
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            if min(x2 - x1, y2 - y1) < self.config.min_face_size:
                continue
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                )
            )
        return detections


class FaceDetectionService:
    def __init__(self, config: InternalConfig):
        self._lock = threading.RLock()
        self._detector = YoloFaceDetector(config)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        with self._lock:
            return self._detector.detect(frame)

    def reset(self) -> None:
        with self._lock:
            self._detector.reset()

    def stats(self) -> dict:
        with self._lock:
            return self._detector.stats()


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
