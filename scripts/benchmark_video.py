#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_blur.public import PrivacyFilterSettings, create_privacy_filter

logger = logging.getLogger("benchmark_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark privacy protection throughput.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument(
        "--profile",
        choices=["privacy", "balanced", "fast"],
        default="privacy",
    )
    parser.add_argument("--asset", help="Privacy filter asset path.")
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    privacy_filter = create_privacy_filter(
        PrivacyFilterSettings(
            profile=args.profile,
            asset_path=args.asset,
            accelerator=args.accelerator,
            debug=args.debug,
        )
    )

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"Could not open input video: {args.input}")

    count = 0
    start = time.perf_counter()
    try:
        while True:
            if args.max_frames and count >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            privacy_filter.apply(frame)
            count += 1
    finally:
        capture.release()

    elapsed = max(time.perf_counter() - start, 1e-9)
    logger.info("Frames: %s", count)
    logger.info("Elapsed: %.3fs", elapsed)
    logger.info("Throughput: %.2f FPS", count / elapsed)
    logger.info("Stats: %s", privacy_filter.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
