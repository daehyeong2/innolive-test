from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PrivacyBlurConfig:
    face_model_path: Path = Path("models/yolov8n-face.pt")
    face_imgsz: int = 1280
    face_conf_threshold: float = 0.22
    face_iou_threshold: float = 0.45
    face_max_detections: int = 300
    device: str | None = None

    min_face_size: int = 8
    max_face_aspect_ratio: float = 2.2

    enable_identity_exclusion: bool = True
    me_image_path: Path = Path("me.jpeg")
    face_detection_model_path: Path = Path("models/face_detection_yunet_2023mar.onnx")
    face_recognition_model_path: Path = Path("models/face_recognition_sface_2021dec.onnx")
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

    worker_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "face_model_path", Path(self.face_model_path))
        object.__setattr__(self, "me_image_path", Path(self.me_image_path))
        object.__setattr__(self, "face_detection_model_path", Path(self.face_detection_model_path))
        object.__setattr__(self, "face_recognition_model_path", Path(self.face_recognition_model_path))

        if self.worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        if self.detect_every_n_frames < 1:
            raise ValueError("detect_every_n_frames must be at least 1")

    def with_overrides(self, **kwargs: Any) -> "PrivacyBlurConfig":
        return replace(self, **kwargs)

