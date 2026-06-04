from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_blur.public import get_privacy_filter


async def main() -> int:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    privacy_filter = get_privacy_filter()

    protected_frame = await privacy_filter.apply_async(frame)

    assert protected_frame.shape == frame.shape
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
