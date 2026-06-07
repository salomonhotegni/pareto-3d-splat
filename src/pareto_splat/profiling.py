"""Helpers for reproducible baseline efficiency profiling."""

from __future__ import annotations

import math
import statistics
from pathlib import Path


class ProfilingInputError(ValueError):
    """Raised when a profiling artifact does not satisfy the expected format."""


def read_ply_vertex_count(path: Path) -> int:
    """Read the vertex count from a PLY header without loading the point cloud."""

    try:
        with path.open("rb") as ply_file:
            first_line = ply_file.readline()
            if first_line.strip() != b"ply":
                raise ProfilingInputError(f"not a PLY file: {path}")

            for _ in range(10_000):
                line = ply_file.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith(b"element vertex "):
                    try:
                        vertex_count = int(stripped.split()[-1])
                    except ValueError as error:
                        raise ProfilingInputError(
                            f"invalid PLY vertex count in {path}"
                        ) from error
                    if vertex_count <= 0:
                        raise ProfilingInputError(
                            f"PLY vertex count must be positive in {path}"
                        )
                    return vertex_count
                if stripped == b"end_header":
                    break
    except OSError as error:
        raise ProfilingInputError(f"could not read PLY file: {path}") from error

    raise ProfilingInputError(f"PLY vertex count not found in {path}")


def percentile(values: list[float], quantile: float) -> float:
    """Compute a linearly interpolated percentile for finite values."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], found {quantile}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("latency values must be finite")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Summarize positive per-frame GPU render latencies."""

    if not latencies_ms:
        raise ValueError("no render latencies were provided")
    if not all(
        math.isfinite(latency) and latency > 0.0 for latency in latencies_ms
    ):
        raise ValueError("render latencies must be finite and positive")

    mean_latency = statistics.fmean(latencies_ms)
    return {
        "mean_ms": mean_latency,
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": percentile(latencies_ms, 0.95),
        "standard_deviation_ms": statistics.pstdev(latencies_ms),
        "minimum_ms": min(latencies_ms),
        "maximum_ms": max(latencies_ms),
        "frames_per_second": 1000.0 / mean_latency,
    }
