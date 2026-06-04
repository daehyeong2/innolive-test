from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
