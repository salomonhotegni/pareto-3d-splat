#!/usr/bin/env python3
"""Validate an official NeRF Synthetic scene."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.datasets.nerf_synthetic import (  # noqa: E402
    DatasetValidationError,
    OFFICIAL_NERF_SYNTHETIC_IMAGE_MODE,
    OFFICIAL_NERF_SYNTHETIC_IMAGE_SIZE,
    OFFICIAL_NERF_SYNTHETIC_SPLIT_COUNTS,
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
    parser.add_argument(
        "--train-views",
        type=int,
        default=OFFICIAL_NERF_SYNTHETIC_SPLIT_COUNTS["train"],
        help="expected train camera count",
    )
    parser.add_argument(
        "--validation-views",
        type=int,
        default=OFFICIAL_NERF_SYNTHETIC_SPLIT_COUNTS["val"],
        help="expected validation camera count",
    )
    parser.add_argument(
        "--test-views",
        type=int,
        default=OFFICIAL_NERF_SYNTHETIC_SPLIT_COUNTS["test"],
        help="expected test camera count",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summaries = validate_nerf_synthetic_dataset(
            args.dataset,
            expected_split_counts={
                "train": args.train_views,
                "val": args.validation_views,
                "test": args.test_views,
            },
            expected_image_size=OFFICIAL_NERF_SYNTHETIC_IMAGE_SIZE,
            expected_image_mode=OFFICIAL_NERF_SYNTHETIC_IMAGE_MODE,
        )
    except DatasetValidationError as error:
        print(f"Dataset validation failed: {error}", file=sys.stderr)
        return 1

    angle_degrees = math.degrees(summaries[0].camera_angle_x)
    print(f"Validated NeRF Synthetic scene: {args.dataset.resolve()}")
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
