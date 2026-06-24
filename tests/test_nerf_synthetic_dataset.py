import json
from pathlib import Path

import pytest
from PIL import Image

from pareto_splat.datasets.nerf_synthetic import (
    DatasetValidationError,
    OFFICIAL_NERF_SYNTHETIC_IMAGE_MODE,
    OFFICIAL_NERF_SYNTHETIC_IMAGE_SIZE,
    OFFICIAL_NERF_SYNTHETIC_SPLIT_COUNTS,
    validate_nerf_synthetic_dataset,
)


IDENTITY_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def create_tiny_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        image_dir = root / split
        image_dir.mkdir(parents=True)
        Image.new("RGBA", (4, 4), (255, 255, 255, 0)).save(
            image_dir / "r_0.png"
        )
        transforms = {
            "camera_angle_x": 0.6911112070083618,
            "frames": [
                {
                    "file_path": f"./{split}/r_0",
                    "transform_matrix": IDENTITY_TRANSFORM,
                }
            ],
        }
        (root / f"transforms_{split}.json").write_text(
            json.dumps(transforms),
            encoding="utf-8",
        )


def test_validate_nerf_synthetic_dataset(tmp_path: Path) -> None:
    create_tiny_dataset(tmp_path)

    summaries = validate_nerf_synthetic_dataset(
        tmp_path,
        expected_split_counts={"train": 1, "val": 1, "test": 1},
        expected_image_size=(4, 4),
        expected_image_mode="RGBA",
    )

    assert [summary.name for summary in summaries] == ["train", "val", "test"]
    assert all(summary.frame_count == 1 for summary in summaries)


def test_validator_rejects_missing_referenced_image(tmp_path: Path) -> None:
    create_tiny_dataset(tmp_path)
    (tmp_path / "test" / "r_0.png").unlink()

    with pytest.raises(DatasetValidationError, match="missing image"):
        validate_nerf_synthetic_dataset(tmp_path)


def test_validator_accepts_symlinked_images_outside_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    derived = tmp_path / "derived"
    create_tiny_dataset(source)
    create_tiny_dataset(derived)

    for split in ("train", "val", "test"):
        derived_image = derived / split / "r_0.png"
        derived_image.unlink()
        derived_image.symlink_to((source / split / "r_0.png").resolve())

    summaries = validate_nerf_synthetic_dataset(
        derived,
        expected_split_counts={"train": 1, "val": 1, "test": 1},
        expected_image_size=(4, 4),
        expected_image_mode="RGBA",
    )

    assert all(summary.frame_count == 1 for summary in summaries)


def test_official_contract_applies_to_all_synthetic_scenes() -> None:
    assert OFFICIAL_NERF_SYNTHETIC_SPLIT_COUNTS == {
        "train": 100,
        "val": 100,
        "test": 200,
    }
    assert OFFICIAL_NERF_SYNTHETIC_IMAGE_SIZE == (800, 800)
    assert OFFICIAL_NERF_SYNTHETIC_IMAGE_MODE == "RGBA"
