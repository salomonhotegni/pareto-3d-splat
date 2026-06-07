#!/usr/bin/env python3
"""Validate the official NeRF Synthetic Lego dataset."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.datasets.nerf_synthetic import (  # noqa: E402
    DatasetValidationError,
    OFFICIAL_LEGO_IMAGE_MODE,
    OFFICIAL_LEGO_IMAGE_SIZE,
    OFFICIAL_LEGO_SPLIT_COUNTS,
    validate_nerf_synthetic_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate NeRF Synthetic pose files and referenced PNG images."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=ROOT_DIR / "data" / "nerf_synthetic" / "lego",
        help="dataset directory (default: data/nerf_synthetic/lego)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summaries = validate_nerf_synthetic_dataset(
            args.dataset,
            expected_split_counts=OFFICIAL_LEGO_SPLIT_COUNTS,
            expected_image_size=OFFICIAL_LEGO_IMAGE_SIZE,
            expected_image_mode=OFFICIAL_LEGO_IMAGE_MODE,
        )
    except DatasetValidationError as error:
        print(f"Dataset validation failed: {error}", file=sys.stderr)
        return 1

    angle_degrees = math.degrees(summaries[0].camera_angle_x)
    print(f"Validated NeRF Synthetic Lego dataset: {args.dataset.resolve()}")
    for summary in summaries:
        print(f"  {summary.name}: {summary.frame_count} cameras")
    print(
        f"  images: {summaries[0].image_size[0]}x{summaries[0].image_size[1]} "
        f"{summaries[0].image_mode} PNG"
    )
    print(f"  horizontal field of view: {angle_degrees:.3f} degrees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

