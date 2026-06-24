import json
from pathlib import Path

import numpy as np
from PIL import Image

from pareto_splat.input_degradation import (
    InputDegradationSettings,
    degrade_image,
    evenly_spaced_indices,
    prepare_degraded_nerf_synthetic_dataset,
)


def transform_payload(split: str, count: int) -> dict:
    return {
        "camera_angle_x": 0.6911112070083618,
        "frames": [
            {
                "file_path": f"./{split}/r_{index}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, float(index)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
            for index in range(count)
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_rgba(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4), color).save(path)


def make_dataset(root: Path, train_count: int = 4) -> None:
    for split, count in (("train", train_count), ("val", 2), ("test", 3)):
        write_json(root / f"transforms_{split}.json", transform_payload(split, count))
        for index in range(count):
            write_rgba(root / split / f"r_{index}.png", (100, 120, 140, 200))


def test_evenly_spaced_indices_preserve_coverage() -> None:
    assert evenly_spaced_indices(10, 4) == (0, 3, 6, 9)
    assert evenly_spaced_indices(5, 5) == (0, 1, 2, 3, 4)


def test_gaussian_noise_is_deterministic_and_preserves_alpha() -> None:
    image = Image.new("RGBA", (4, 4), (100, 120, 140, 200))
    settings = InputDegradationSettings(
        degradation="gaussian-noise",
        noise_std=0.05,
        seed=7,
    )

    first = degrade_image(image, settings, np.random.default_rng(settings.seed))
    second = degrade_image(image, settings, np.random.default_rng(settings.seed))

    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first)[..., :3], np.asarray(image)[..., :3])
    assert np.array_equal(np.asarray(first)[..., 3], np.asarray(image)[..., 3])


def test_brightness_scales_rgb_and_preserves_alpha() -> None:
    image = Image.new("RGBA", (2, 2), (100, 120, 140, 200))
    settings = InputDegradationSettings(
        degradation="brightness",
        brightness_factor=0.5,
        seed=0,
    )

    output = degrade_image(image, settings, np.random.default_rng(0))
    pixels = np.asarray(output)

    assert tuple(pixels[0, 0, :3]) == (50, 60, 70)
    assert pixels[0, 0, 3] == 200


def test_prepare_degraded_dataset_subsets_train_and_keeps_test_clean(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    make_dataset(source, train_count=5)
    output = tmp_path / "output"

    result = prepare_degraded_nerf_synthetic_dataset(
        source,
        output,
        InputDegradationSettings(
            degradation="clean",
            train_view_count=3,
            seed=0,
        ),
        image_policy="copy",
    )

    train = json.loads((output / "transforms_train.json").read_text())
    test = json.loads((output / "transforms_test.json").read_text())
    metadata = json.loads(result.metadata_path.read_text())

    assert len(train["frames"]) == 3
    assert [frame["file_path"] for frame in train["frames"]] == [
        "./train/r_0",
        "./train/r_2",
        "./train/r_4",
    ]
    assert len(test["frames"]) == 3
    assert (output / "test" / "r_0.png").is_file()
    assert metadata["split_summaries"]["train"]["output_frame_count"] == 3
    assert metadata["split_summaries"]["test"]["degradation"] == "clean"
