#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_blur.public import PrivacyFilterSettings, create_privacy_filter

logger = logging.getLogger("process_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply privacy protection to a video.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument(
        "--browser-output",
        help="Optional browser-friendly output path.",
    )
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
    parser.add_argument(
        "--no-h264",
        action="store_true",
        help="Deprecated. H.264 is already skipped unless --h264 is used.",
    )
    parser.add_argument(
        "--h264",
        action="store_true",
        help="Try H.264 output before falling back to mp4v.",
    )
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

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _open_writer(args.output, fps, width, height, args.h264)
    browser_writer = None
    if args.browser_output:
        browser_writer = _open_writer(args.browser_output, fps, width, height, False)

    start = time.perf_counter()
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            protected = privacy_filter.apply(frame)
            writer.write(protected)
            if browser_writer is not None:
                browser_writer.write(protected)
            count += 1
    finally:
        capture.release()
        writer.release()
        if browser_writer is not None:
            browser_writer.release()

    elapsed = max(time.perf_counter() - start, 1e-9)
    logger.info("Processed %s frames at %.2f FPS.", count, count / elapsed)
    return 0


def _open_writer(path: str, fps: float, width: int, height: int, prefer_h264: bool):
    codes = ["avc1", "H264", "mp4v"] if prefer_h264 else ["mp4v"]
    for code in codes:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*code),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise SystemExit(f"Could not open output video: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
