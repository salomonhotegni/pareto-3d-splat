"""Post-training Gaussian pruning for GraphDeCo-compatible PLY models."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from plyfile import PlyData, PlyElement


PruningStrategy = Literal["random", "opacity-threshold", "top-k"]
PRUNING_SCHEMA_VERSION = 1
GRAPHDECO_METADATA_FILES = ("cfg_args", "cameras.json", "exposure.json")


class PruningError(ValueError):
    """Raised when pruning inputs or settings are invalid."""


@dataclass(frozen=True)
class PruningResult:
    """Summary of one pruning operation."""

    strategy: PruningStrategy
    input_path: Path
    output_path: Path
    input_count: int
    output_count: int
    keep_fraction: float
    pruned_fraction: float
    metadata: dict[str, Any]


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute sigmoid stably for raw GraphDeCo opacity logits."""

    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def logit(probability: float) -> float:
    """Convert an opacity threshold from probability to raw logit space."""

    if not 0.0 < probability < 1.0:
        raise PruningError("opacity threshold must be in the open interval (0, 1)")
    return math.log(probability / (1.0 - probability))


def load_vertex_data(path: Path) -> tuple[PlyData, np.ndarray]:
    """Load the vertex table from a PLY file and validate core fields."""

    try:
        ply = PlyData.read(path)
    except Exception as error:  # noqa: BLE001 - plyfile raises several types.
        raise PruningError(f"could not read PLY file: {path}") from error

    if "vertex" not in ply:
        raise PruningError(f"PLY file has no vertex element: {path}")
    vertices = ply["vertex"].data
    if vertices.ndim != 1:
        raise PruningError(f"PLY vertex data must be one-dimensional: {path}")
    if len(vertices) == 0:
        raise PruningError(f"PLY file has no vertices: {path}")
    if vertices.dtype.names is None or "opacity" not in vertices.dtype.names:
        raise PruningError(f"PLY vertex data must contain an opacity field: {path}")

    opacities = np.asarray(vertices["opacity"], dtype=np.float64)
    if not np.all(np.isfinite(opacities)):
        raise PruningError(f"PLY opacity field contains non-finite values: {path}")
    return ply, vertices


def resolve_keep_count(
    total_count: int,
    *,
    keep_count: int | None = None,
    keep_fraction: float | None = None,
) -> int:
    """Resolve a requested retention level to an integer count."""

    if total_count <= 0:
        raise PruningError("total count must be positive")
    if (keep_count is None) == (keep_fraction is None):
        raise PruningError("provide exactly one of keep_count or keep_fraction")

    if keep_count is not None:
        if keep_count <= 0 or keep_count > total_count:
            raise PruningError(
                f"keep_count must be in [1, {total_count}], found {keep_count}"
            )
        return keep_count

    assert keep_fraction is not None
    if not 0.0 < keep_fraction <= 1.0:
        raise PruningError("keep_fraction must be in the interval (0, 1]")
    return max(1, min(total_count, int(round(total_count * keep_fraction))))


def random_mask(total_count: int, keep_count: int, seed: int) -> np.ndarray:
    """Select a uniformly random subset of fixed cardinality."""

    if keep_count <= 0 or keep_count > total_count:
        raise PruningError("invalid random pruning keep count")
    rng = np.random.default_rng(seed)
    selected = rng.choice(total_count, size=keep_count, replace=False)
    mask = np.zeros(total_count, dtype=bool)
    mask[selected] = True
    return mask


def top_k_opacity_mask(vertices: np.ndarray, keep_count: int) -> np.ndarray:
    """Keep the Gaussians with largest activated opacity."""

    if keep_count <= 0 or keep_count > len(vertices):
        raise PruningError("invalid top-k pruning keep count")
    scores = sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))
    order = np.argsort(scores, kind="stable")
    selected = order[-keep_count:]
    mask = np.zeros(len(vertices), dtype=bool)
    mask[selected] = True
    return mask


def opacity_threshold_mask(
    vertices: np.ndarray,
    opacity_threshold: float,
) -> np.ndarray:
    """Keep Gaussians with activated opacity above a threshold."""

    threshold_logit = logit(opacity_threshold)
    return np.asarray(vertices["opacity"], dtype=np.float64) >= threshold_logit


def prune_mask(
    vertices: np.ndarray,
    strategy: PruningStrategy,
    *,
    keep_count: int | None = None,
    keep_fraction: float | None = None,
    opacity_threshold: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a pruning mask and strategy-specific metadata."""

    total_count = len(vertices)
    if strategy in {"random", "top-k"}:
        resolved_keep_count = resolve_keep_count(
            total_count,
            keep_count=keep_count,
            keep_fraction=keep_fraction,
        )
        if opacity_threshold is not None:
            raise PruningError(
                "opacity_threshold is only valid for opacity-threshold pruning"
            )
        if strategy == "random":
            mask = random_mask(total_count, resolved_keep_count, seed)
            strategy_metadata: dict[str, Any] = {
                "seed": seed,
                "requested_keep_count": keep_count,
                "requested_keep_fraction": keep_fraction,
            }
        else:
            mask = top_k_opacity_mask(vertices, resolved_keep_count)
            strategy_metadata = {
                "score": "opacity",
                "requested_keep_count": keep_count,
                "requested_keep_fraction": keep_fraction,
            }
    elif strategy == "opacity-threshold":
        if keep_count is not None or keep_fraction is not None:
            raise PruningError(
                "keep_count and keep_fraction are not valid for "
                "opacity-threshold pruning"
            )
        if opacity_threshold is None:
            raise PruningError(
                "opacity-threshold pruning requires opacity_threshold"
            )
        mask = opacity_threshold_mask(vertices, opacity_threshold)
        threshold_logit = logit(opacity_threshold)
        strategy_metadata = {
            "opacity_threshold": opacity_threshold,
            "opacity_threshold_logit": threshold_logit,
        }
    else:
        raise PruningError(f"unsupported pruning strategy: {strategy}")

    if not mask.any():
        raise PruningError("pruning would remove every Gaussian")
    return mask, strategy_metadata


def write_vertex_data(
    output_path: Path,
    vertices: np.ndarray,
    source_ply: PlyData,
) -> None:
    """Write a GraphDeCo-compatible PLY while preserving vertex fields."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_element = PlyElement.describe(vertices, "vertex")
    PlyData(
        [vertex_element],
        text=source_ply.text,
        byte_order=source_ply.byte_order,
        comments=source_ply.comments,
        obj_info=source_ply.obj_info,
    ).write(output_path)


def copy_graphdeco_metadata(source_model_path: Path, output_model_path: Path) -> None:
    """Copy lightweight GraphDeCo model metadata into a pruned model directory."""

    for name in GRAPHDECO_METADATA_FILES:
        source = source_model_path / name
        if source.is_file():
            output = output_model_path / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)


def metadata_path_for_output(output_path: Path) -> Path:
    """Return the JSON metadata path associated with a pruned PLY file."""

    return output_path.parent / "pruning_metadata.json"


def prune_ply(
    input_path: Path,
    output_path: Path,
    strategy: PruningStrategy,
    *,
    keep_count: int | None = None,
    keep_fraction: float | None = None,
    opacity_threshold: float | None = None,
    seed: int = 0,
    source_model_path: Path | None = None,
    output_model_path: Path | None = None,
) -> PruningResult:
    """Prune a GraphDeCo point-cloud PLY and write metadata."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    source_ply, vertices = load_vertex_data(input_path)
    mask, strategy_metadata = prune_mask(
        vertices,
        strategy,
        keep_count=keep_count,
        keep_fraction=keep_fraction,
        opacity_threshold=opacity_threshold,
        seed=seed,
    )

    pruned_vertices = vertices[mask].copy()
    write_vertex_data(output_path, pruned_vertices, source_ply)

    input_count = len(vertices)
    output_count = len(pruned_vertices)
    keep_ratio = output_count / input_count
    metadata: dict[str, Any] = {
        "schema_version": PRUNING_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_count": input_count,
        "output_count": output_count,
        "keep_fraction": keep_ratio,
        "pruned_fraction": 1.0 - keep_ratio,
        "strategy_parameters": strategy_metadata,
        "property_names": list(vertices.dtype.names or ()),
    }

    if source_model_path is not None:
        metadata["source_model_path"] = str(source_model_path.resolve())
    if output_model_path is not None:
        metadata["output_model_path"] = str(output_model_path.resolve())
    if source_model_path is not None and output_model_path is not None:
        copy_graphdeco_metadata(
            source_model_path.resolve(),
            output_model_path.resolve(),
        )

    metadata_path_for_output(output_path).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return PruningResult(
        strategy=strategy,
        input_path=input_path,
        output_path=output_path,
        input_count=input_count,
        output_count=output_count,
        keep_fraction=keep_ratio,
        pruned_fraction=1.0 - keep_ratio,
        metadata=metadata,
    )
