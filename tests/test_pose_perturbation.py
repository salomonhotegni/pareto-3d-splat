import json
from pathlib import Path

import numpy as np

from pareto_splat.pose_perturbation import (
    PosePerturbationSettings,
    axis_angle_to_rotation_matrix,
    prepare_perturbed_nerf_synthetic_dataset,
    perturb_transforms_payload,
)


def transform_payload() -> dict:
    return {
        "camera_angle_x": 0.6911112070083618,
        "frames": [
            {
                "file_path": "./test/r_0",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            {
                "file_path": "./test/r_1",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_axis_angle_to_rotation_matrix_is_valid_rotation() -> None:
    rotation = axis_angle_to_rotation_matrix(np.array([0.0, 0.0, np.pi / 2.0]))

    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])


def test_perturb_transforms_payload_is_deterministic() -> None:
    payload = transform_payload()
    settings = PosePerturbationSettings(
        rotation_degrees=0.5,
        translation_std=0.01,
        seed=42,
    )

    first, first_metadata = perturb_transforms_payload(payload, settings)
    second, second_metadata = perturb_transforms_payload(payload, settings)

    assert first == second
    assert first_metadata == second_metadata
    assert first != payload
    assert first_metadata["frame_count"] == 2
    first_rotation = np.array(first["frames"][0]["transform_matrix"])[:3, :3]
    assert np.allclose(first_rotation.T @ first_rotation, np.eye(3), atol=1e-12)


def test_zero_perturbation_keeps_transforms_unchanged() -> None:
    payload = transform_payload()
    settings = PosePerturbationSettings(
        rotation_degrees=0.0,
        translation_std=0.0,
        seed=42,
    )

    perturbed, metadata = perturb_transforms_payload(payload, settings)

    assert perturbed == payload
    assert metadata["mean_rotation_angle_degrees"] == 0.0
    assert metadata["mean_translation_norm"] == 0.0


def test_prepare_perturbed_dataset_perturbs_selected_split_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "train").mkdir(parents=True)
    (source / "test").mkdir()
    (source / "test" / "r_0.png").write_bytes(b"not-a-real-png")
    train_payload = transform_payload()
    test_payload = transform_payload()
    write_json(source / "transforms_train.json", train_payload)
    write_json(source / "transforms_test.json", test_payload)

    output = tmp_path / "output"
    result = prepare_perturbed_nerf_synthetic_dataset(
        source,
        output,
        PosePerturbationSettings(
            rotation_degrees=0.0,
            translation_std=0.01,
            seed=0,
            splits=("test",),
        ),
        image_policy="copy",
    )

    assert result.metadata_path.is_file()
    assert (output / "test" / "r_0.png").is_file()
    assert json.loads((output / "transforms_train.json").read_text()) == train_payload
    assert json.loads((output / "transforms_test.json").read_text()) != test_payload
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["splits"] == ["test"]
    assert metadata["split_summaries"]["test"]["frame_count"] == 2
