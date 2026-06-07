import json
from pathlib import Path

import pytest
from PIL import Image

from pareto_splat.datasets.nerf_synthetic import (
    DatasetValidationError,
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

