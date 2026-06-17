import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pareto_splat.pruning import (
    PruningError,
    logit,
    prune_mask,
    prune_ply,
    sigmoid,
)


def write_tiny_ply(path: Path) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
    ]
    vertices = np.array(
        [
            (0.0, 0.0, 0.0, logit(0.10), 1.0),
            (1.0, 0.0, 0.0, logit(0.40), 2.0),
            (2.0, 0.0, 0.0, logit(0.80), 3.0),
            (3.0, 0.0, 0.0, logit(0.95), 4.0),
        ],
        dtype=dtype,
    )
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def read_vertices(path: Path) -> np.ndarray:
    return PlyData.read(path)["vertex"].data


def test_sigmoid_and_logit_are_inverse_for_opacity_values() -> None:
    values = np.array([0.1, 0.4, 0.8, 0.95])

    assert np.allclose(sigmoid(np.array([logit(value) for value in values])), values)


def test_top_k_pruning_keeps_largest_opacity_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    output = tmp_path / "pruned" / "point_cloud.ply"
    write_tiny_ply(source)

    result = prune_ply(
        source,
        output,
        "top-k",
        keep_count=2,
    )
    vertices = read_vertices(output)

    assert result.input_count == 4
    assert result.output_count == 2
    assert vertices["x"].tolist() == pytest.approx([2.0, 3.0])
    assert vertices["scale_0"].tolist() == pytest.approx([3.0, 4.0])


def test_opacity_threshold_pruning_uses_activated_opacity(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    output = tmp_path / "pruned" / "point_cloud.ply"
    write_tiny_ply(source)

    result = prune_ply(
        source,
        output,
        "opacity-threshold",
        opacity_threshold=0.5,
    )
    vertices = read_vertices(output)
    metadata = json.loads((output.parent / "pruning_metadata.json").read_text())

    assert result.output_count == 2
    assert vertices["x"].tolist() == pytest.approx([2.0, 3.0])
    assert metadata["strategy_parameters"]["opacity_threshold"] == 0.5
    assert metadata["input_count"] == 4
    assert metadata["output_count"] == 2


def test_random_pruning_is_seed_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    output_a = tmp_path / "a" / "point_cloud.ply"
    output_b = tmp_path / "b" / "point_cloud.ply"
    write_tiny_ply(source)

    prune_ply(source, output_a, "random", keep_fraction=0.5, seed=7)
    prune_ply(source, output_b, "random", keep_fraction=0.5, seed=7)

    assert read_vertices(output_a)["x"].tolist() == read_vertices(output_b)[
        "x"
    ].tolist()


def test_invalid_pruning_arguments_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    write_tiny_ply(source)
    vertices = read_vertices(source)

    with pytest.raises(PruningError, match="exactly one"):
        prune_mask(vertices, "top-k", keep_count=1, keep_fraction=0.5)
    with pytest.raises(PruningError, match="opacity_threshold"):
        prune_mask(vertices, "opacity-threshold")
