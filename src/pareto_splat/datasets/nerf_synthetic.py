"""Validation for NeRF Synthetic datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError


OFFICIAL_NERF_SYNTHETIC_SPLIT_COUNTS = {
    "train": 100,
    "val": 100,
    "test": 200,
}
OFFICIAL_NERF_SYNTHETIC_IMAGE_SIZE = (800, 800)
OFFICIAL_NERF_SYNTHETIC_IMAGE_MODE = "RGBA"


class DatasetValidationError(ValueError):
    """Raised when a dataset does not match the NeRF Synthetic contract."""


@dataclass(frozen=True)
class SplitSummary:
    """Validated properties of one camera split."""

    name: str
    frame_count: int
    camera_angle_x: float
    image_size: tuple[int, int]
    image_mode: str
    image_paths: frozenset[Path]


def _validate_transform(matrix: object, context: str) -> None:
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise DatasetValidationError(f"{context}: transform_matrix must be 4x4")

    numeric_matrix: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4:
            raise DatasetValidationError(f"{context}: transform_matrix must be 4x4")
        try:
            numeric_row = [float(value) for value in row]
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"{context}: transform_matrix must contain numbers"
            ) from error
        if not all(math.isfinite(value) for value in numeric_row):
            raise DatasetValidationError(
                f"{context}: transform_matrix contains a non-finite value"
            )
        numeric_matrix.append(numeric_row)

    expected_last_row = (0.0, 0.0, 0.0, 1.0)
    if any(
        not math.isclose(actual, expected, abs_tol=1e-5)
        for actual, expected in zip(numeric_matrix[3], expected_last_row)
    ):
        raise DatasetValidationError(
            f"{context}: transform_matrix has an invalid homogeneous row"
        )


def _image_path(dataset_dir: Path, frame_path: object, context: str) -> Path:
    if not isinstance(frame_path, str) or not frame_path:
        raise DatasetValidationError(f"{context}: file_path must be a string")

    relative_path = Path(frame_path)
    if relative_path.is_absolute():
        raise DatasetValidationError(f"{context}: file_path must be relative")

    image_path = (dataset_dir / f"{frame_path}.png").resolve()
    try:
        image_path.relative_to(dataset_dir.resolve())
    except ValueError as error:
        raise DatasetValidationError(
            f"{context}: file_path escapes the dataset directory"
        ) from error
    return image_path


def _validate_split(
    dataset_dir: Path,
    split: str,
    expected_count: int | None,
    expected_image_size: tuple[int, int] | None,
    expected_image_mode: str | None,
) -> SplitSummary:
    transforms_path = dataset_dir / f"transforms_{split}.json"
    if not transforms_path.is_file():
        raise DatasetValidationError(f"missing {transforms_path}")

    try:
        contents = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(
            f"could not read valid JSON from {transforms_path}"
        ) from error

    try:
        camera_angle_x = float(contents["camera_angle_x"])
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetValidationError(
            f"{transforms_path}: camera_angle_x must be numeric"
        ) from error
    if not math.isfinite(camera_angle_x) or not 0.0 < camera_angle_x < math.pi:
        raise DatasetValidationError(
            f"{transforms_path}: camera_angle_x must be between 0 and pi"
        )

    frames = contents.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DatasetValidationError(
            f"{transforms_path}: frames must be a non-empty list"
        )
    if expected_count is not None and len(frames) != expected_count:
        raise DatasetValidationError(
            f"{transforms_path}: expected {expected_count} frames, found {len(frames)}"
        )

    image_paths: set[Path] = set()
    observed_size: tuple[int, int] | None = None
    observed_mode: str | None = None

    for index, frame in enumerate(frames):
        context = f"{transforms_path}: frame {index}"
        if not isinstance(frame, dict):
            raise DatasetValidationError(f"{context}: frame must be an object")

        _validate_transform(frame.get("transform_matrix"), context)
        image_path = _image_path(dataset_dir, frame.get("file_path"), context)
        if image_path in image_paths:
            raise DatasetValidationError(
                f"{context}: duplicate image reference {image_path}"
            )
        if not image_path.is_file():
            raise DatasetValidationError(f"{context}: missing image {image_path}")

        try:
            with Image.open(image_path) as image:
                image_size = image.size
                image_mode = image.mode
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise DatasetValidationError(
                f"{context}: could not decode PNG image {image_path}"
            ) from error

        if expected_image_size is not None and image_size != expected_image_size:
            raise DatasetValidationError(
                f"{context}: expected image size {expected_image_size}, "
                f"found {image_size}"
            )
        if expected_image_mode is not None and image_mode != expected_image_mode:
            raise DatasetValidationError(
                f"{context}: expected image mode {expected_image_mode}, "
                f"found {image_mode}"
            )
        if observed_size is not None and image_size != observed_size:
            raise DatasetValidationError(
                f"{context}: image size differs from earlier frames"
            )
        if observed_mode is not None and image_mode != observed_mode:
            raise DatasetValidationError(
                f"{context}: image mode differs from earlier frames"
            )

        observed_size = image_size
        observed_mode = image_mode
        image_paths.add(image_path)

    assert observed_size is not None
    assert observed_mode is not None
    return SplitSummary(
        name=split,
        frame_count=len(frames),
        camera_angle_x=camera_angle_x,
        image_size=observed_size,
        image_mode=observed_mode,
        image_paths=frozenset(image_paths),
    )


def validate_nerf_synthetic_dataset(
    dataset_dir: Path,
    *,
    expected_split_counts: Mapping[str, int] | None = None,
    expected_image_size: tuple[int, int] | None = None,
    expected_image_mode: str | None = None,
) -> tuple[SplitSummary, ...]:
    """Validate pose files and referenced PNGs for train, val, and test splits."""

    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise DatasetValidationError(f"dataset directory does not exist: {dataset_dir}")

    split_names = ("train", "val", "test")
    summaries = tuple(
        _validate_split(
            dataset_dir,
            split,
            None if expected_split_counts is None else expected_split_counts[split],
            expected_image_size,
            expected_image_mode,
        )
        for split in split_names
    )

    reference_angle = summaries[0].camera_angle_x
    for summary in summaries[1:]:
        if not math.isclose(summary.camera_angle_x, reference_angle, abs_tol=1e-7):
            raise DatasetValidationError(
                f"{summary.name}: camera_angle_x differs from the train split"
            )

    seen_paths: set[Path] = set()
    for summary in summaries:
        overlap = seen_paths.intersection(summary.image_paths)
        if overlap:
            raise DatasetValidationError(
                f"{summary.name}: image paths overlap another split"
            )
        seen_paths.update(summary.image_paths)

    return summaries
