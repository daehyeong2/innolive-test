from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_blur.public import PrivacyFilterSettings, get_privacy_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protect a single image loaded as a BGR ndarray."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--profile",
        choices=["privacy", "balanced", "fast"],
        default="privacy",
        help="Protection profile.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Could not read image: {args.input}")

    privacy_filter = get_privacy_filter(
        PrivacyFilterSettings(profile=args.profile)
    )

    protected_image = privacy_filter.apply(image)

    if not cv2.imwrite(args.output, protected_image):
        raise SystemExit(f"Could not write image: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
