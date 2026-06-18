"""Pareto dominance and non-dominated sorting utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


MAXIMIZE = "maximize"
MINIMIZE = "minimize"
VALID_DIRECTIONS = {MAXIMIZE, MINIMIZE}


class ParetoError(ValueError):
    """Raised when Pareto inputs are invalid."""


@dataclass(frozen=True)
class Objective:
    """One scalar objective used in Pareto comparisons."""

    name: str
    direction: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ParetoError("objective name must not be empty")
        if self.direction not in VALID_DIRECTIONS:
            raise ParetoError(
                f"objective direction must be {MAXIMIZE!r} or {MINIMIZE!r}"
            )

    def oriented_value(self, point: Mapping[str, Any]) -> float:
        """Return a value where larger is always better."""

        try:
            value = float(point[self.name])
        except KeyError as error:
            raise ParetoError(f"missing objective value: {self.name}") from error
        except (TypeError, ValueError) as error:
            raise ParetoError(
                f"objective value must be numeric: {self.name}"
            ) from error

        if self.direction == MAXIMIZE:
            return value
        return -value


QUALITY_OBJECTIVES = (
    Objective("psnr", MAXIMIZE),
    Objective("ssim", MAXIMIZE),
    Objective("lpips_vgg", MINIMIZE),
)
EFFICIENCY_OBJECTIVES = (
    Objective("fps", MAXIMIZE),
    Objective("mean_latency_ms", MINIMIZE),
    Objective("serialized_mib", MINIMIZE),
    Objective("peak_allocated_mib", MINIMIZE),
)
QUALITY_EFFICIENCY_OBJECTIVES = (
    Objective("psnr", MAXIMIZE),
    Objective("fps", MAXIMIZE),
    Objective("serialized_mib", MINIMIZE),
)


def _validate_objectives(objectives: Sequence[Objective]) -> tuple[Objective, ...]:
    if not objectives:
        raise ParetoError("at least one objective is required")
    return tuple(objectives)


def dominates(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    objectives: Sequence[Objective],
    *,
    tolerance: float = 0.0,
) -> bool:
    """Return True when candidate Pareto-dominates reference.

    A candidate dominates when it is no worse for every objective and strictly
    better for at least one objective. Minimize objectives are internally
    negated so the comparison is always "larger is better".
    """

    if tolerance < 0.0:
        raise ParetoError("tolerance must be non-negative")

    objective_tuple = _validate_objectives(objectives)
    strictly_better = False
    for objective in objective_tuple:
        candidate_value = objective.oriented_value(candidate)
        reference_value = objective.oriented_value(reference)
        if candidate_value < reference_value - tolerance:
            return False
        if candidate_value > reference_value + tolerance:
            strictly_better = True
    return strictly_better


def non_dominated_sort(
    points: Sequence[Mapping[str, Any]],
    objectives: Sequence[Objective],
    *,
    tolerance: float = 0.0,
) -> list[list[int]]:
    """Sort points into Pareto fronts and return point indices per front."""

    _validate_objectives(objectives)
    if not points:
        return []

    remaining = set(range(len(points)))
    fronts: list[list[int]] = []
    while remaining:
        front: list[int] = []
        for index in sorted(remaining):
            if not any(
                other != index
                and dominates(
                    points[other],
                    points[index],
                    objectives,
                    tolerance=tolerance,
                )
                for other in remaining
            ):
                front.append(index)
        fronts.append(front)
        remaining.difference_update(front)
    return fronts


def pareto_ranks(
    points: Sequence[Mapping[str, Any]],
    objectives: Sequence[Objective],
    *,
    tolerance: float = 0.0,
) -> list[int]:
    """Return zero-based Pareto rank for each point."""

    ranks = [-1] * len(points)
    for rank, front in enumerate(
        non_dominated_sort(points, objectives, tolerance=tolerance)
    ):
        for index in front:
            ranks[index] = rank
    return ranks


def annotate_pareto_ranks(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Objective],
    *,
    rank_key: str = "pareto_rank",
    tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Return copied rows with an added Pareto rank field."""

    ranks = pareto_ranks(rows, objectives, tolerance=tolerance)
    return [
        {
            **dict(row),
            rank_key: ranks[index],
        }
        for index, row in enumerate(rows)
    ]
