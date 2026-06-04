from __future__ import annotations

import logging
import os
from queue import Queue
import threading
from typing import Any

import numpy as np

from privacy_blur.errors import PrivacyFilterNotReadyError
from .config import InternalConfig
from .detector import FaceDetectionService
from .processor import FacePrivacyProcessor

logger = logging.getLogger(__name__)

PROFILE_DEFAULTS = {
    "privacy": {
        "imgsz": 1280,
        "conf_threshold": 0.20,
        "min_face_size": 6,
        "box_expand_w": 1.30,
        "box_expand_h": 1.30,
        "detect_every_n_frames": 1,
        "hold_frames": 12,
    },
    "balanced": {
        "imgsz": 960,
        "conf_threshold": 0.24,
        "min_face_size": 8,
        "box_expand_w": 1.35,
        "box_expand_h": 1.55,
        "detect_every_n_frames": 1,
        "hold_frames": 10,
    },
    "fast": {
        "imgsz": 736,
        "conf_threshold": 0.28,
        "min_face_size": 10,
        "box_expand_w": 1.30,
        "box_expand_h": 1.45,
        "detect_every_n_frames": 2,
        "hold_frames": 8,
    },
}


class PrivacyRuntime:
    def __init__(self, settings: Any):
        self.settings = settings
        self.config = self._build_internal_config(settings)
        self._detection_service = FaceDetectionService(self.config)
        self._processed_frames = 0
        self._counter_lock = threading.Lock()
        self._worker_count = max(1, int(settings.worker_count))

        if self._worker_count == 1:
            self._processor = FacePrivacyProcessor(
                self.config,
                self._detection_service,
            )
            self._lock = threading.RLock()
            self._pool: Queue[FacePrivacyProcessor] | None = None
        else:
            self._processor = None
            self._lock = None
            self._pool = Queue(maxsize=self._worker_count)
            for _ in range(self._worker_count):
                self._pool.put(
                    FacePrivacyProcessor(
                        self.config,
                        self._detection_service,
                    )
                )

    def process_ndarray(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise PrivacyFilterNotReadyError("Expected a numpy image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise PrivacyFilterNotReadyError("Expected a BGR image with 3 channels.")

        if self._worker_count == 1:
            assert self._processor is not None
            assert self._lock is not None
            with self._lock:
                result = self._processor.process(frame)
        else:
            assert self._pool is not None
            processor = self._pool.get()
            try:
                result = processor.process(frame)
            finally:
                self._pool.put(processor)

        with self._counter_lock:
            self._processed_frames += 1
        return result

    def reset(self) -> None:
        if self._worker_count == 1:
            assert self._processor is not None
            assert self._lock is not None
            with self._lock:
                self._processor.reset()
        else:
            assert self._pool is not None
            processors: list[FacePrivacyProcessor] = []
            for _ in range(self._worker_count):
                processor = self._pool.get()
                processor.reset()
                processors.append(processor)
            for processor in processors:
                self._pool.put(processor)

        self._detection_service.reset()
        with self._counter_lock:
            self._processed_frames = 0

    def stats(self) -> dict:
        with self._counter_lock:
            processed_frames = self._processed_frames
        runtime_stats = self._detection_service.stats()
        return {
            "profile": self.settings.profile,
            "workers": self._worker_count,
            "processed_frames": processed_frames,
            "ready": runtime_stats["ready"],
            "accelerator": _public_accelerator(runtime_stats["device"]),
            "asset_load_count": runtime_stats["load_count"],
        }

    def _build_internal_config(self, settings: Any) -> InternalConfig:
        profile = PROFILE_DEFAULTS[settings.profile]
        device = {
            "auto": "auto",
            "gpu": "cuda",
            "cpu": "cpu",
        }[settings.accelerator]

        if settings.debug:
            logging.getLogger("privacy_blur").setLevel(logging.DEBUG)

        return InternalConfig(
            asset_path=_resolve_asset_path(settings),
            device=device,
            imgsz=profile["imgsz"],
            conf_threshold=profile["conf_threshold"],
            min_face_size=profile["min_face_size"],
            box_expand_w=profile["box_expand_w"],
            box_expand_h=profile["box_expand_h"],
            detect_every_n_frames=profile["detect_every_n_frames"],
            hold_frames=profile["hold_frames"],
            realtime=settings.realtime,
            inplace=settings.inplace,
            debug=settings.debug,
        )


def _resolve_asset_path(settings: Any) -> str:
    if settings.asset_path:
        return settings.asset_path

    django_path = _django_asset_path()
    if django_path:
        return django_path

    env_path = os.getenv("PRIVACY_FILTER_ASSET_PATH")
    if env_path:
        return env_path

    return "models/yolov8n-face.pt"


def _django_asset_path() -> str | None:
    try:
        from django.conf import settings as django_settings
    except Exception:
        return None

    try:
        if not django_settings.configured:
            return None
        value = getattr(django_settings, "PRIVACY_FILTER", None)
    except Exception:
        return None

    if isinstance(value, dict):
        asset_path = value.get("ASSET_PATH")
        if asset_path:
            return str(asset_path)
    return None


def _public_accelerator(value: str | None) -> str:
    if value == "cuda":
        return "gpu"
    if value == "cpu":
        return "cpu"
    return "auto"
