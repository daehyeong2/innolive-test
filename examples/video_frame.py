from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
from av import VideoFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_blur import get_privacy_filter


def main() -> int:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    video_frame = VideoFrame.from_ndarray(image, format="bgr24")
    video_frame.pts = 100
    video_frame.time_base = Fraction(1, 90000)

    privacy_filter = get_privacy_filter()

    protected_frame = privacy_filter.apply(video_frame)

    assert protected_frame.pts == video_frame.pts
    assert protected_frame.time_base == video_frame.time_base
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
