from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InternalConfig:
    asset_path: str
    device: str
    imgsz: int
    conf_threshold: float
    min_face_size: int
    box_expand_w: float
    box_expand_h: float
    detect_every_n_frames: int
    hold_frames: int
    realtime: bool
    inplace: bool
    debug: bool
