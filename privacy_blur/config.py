from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PrivacyBlurConfig:
    face_model_path: Path = Path("models/yolov8n-face.pt")
    face_imgsz: int = 640
    face_conf_threshold: float = 0.22
    face_iou_threshold: float = 0.45
    face_max_detections: int = 300
    device: str | None = None
    require_gpu: bool = False
    use_half: bool = True

    max_batch_size: int = 8
    batch_wait_ms: float = 4.0
    max_pending_frames: int = 128

    min_face_size: int = 8
    max_face_aspect_ratio: float = 2.2

    enable_identity_exclusion: bool = True
    me_image_path: Path = Path("me.jpeg")
    face_detection_model_path: Path = Path("models/face_detection_yunet_2023mar.onnx")
    face_recognition_model_path: Path = Path(
        "models/face_recognition_sface_2021dec.onnx"
    )
    identity_detector_size: int = 320
    identity_detector_confidence: float = 0.55
    face_match_cosine_threshold: float = 0.40
    face_match_lost_cosine_threshold: float = 0.30
    identity_recheck_interval: int = 12
    identity_min_face_size: int = 36

    detect_every_n_frames: int = 1
    mask_hold_frames: int = 18
    track_iou_threshold: float = 0.08
    track_center_distance_scale: float = 1.10

    box_expand_w: float = 1.35
    box_expand_h: float = 1.60
    small_face_threshold: int = 40
    small_box_expand_w: float = 1.75
    small_box_expand_h: float = 2.10
    center_y_shift: float = -0.06

    blur_kernel: int = 101
    feather: int = 21

    def __post_init__(self) -> None:
        object.__setattr__(self, "face_model_path", Path(self.face_model_path))
        object.__setattr__(self, "me_image_path", Path(self.me_image_path))
        object.__setattr__(
            self, "face_detection_model_path", Path(self.face_detection_model_path)
        )
        object.__setattr__(
            self, "face_recognition_model_path", Path(self.face_recognition_model_path)
        )

        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms cannot be negative")
        if self.max_pending_frames < self.max_batch_size:
            raise ValueError("max_pending_frames must be at least max_batch_size")
        if self.detect_every_n_frames < 1:
            raise ValueError("detect_every_n_frames must be at least 1")
        if self.identity_detector_size < 32 or self.identity_detector_size % 32:
            raise ValueError("identity_detector_size must be a multiple of 32")

    def with_overrides(self, **kwargs: Any) -> "PrivacyBlurConfig":
        return replace(self, **kwargs)
