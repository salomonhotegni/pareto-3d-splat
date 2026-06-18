import pytest

from pareto_splat.pareto import (
    MAXIMIZE,
    MINIMIZE,
    QUALITY_EFFICIENCY_OBJECTIVES,
    Objective,
    ParetoError,
    annotate_pareto_ranks,
    dominates,
    non_dominated_sort,
    pareto_ranks,
)


def test_objective_rejects_invalid_direction() -> None:
    with pytest.raises(ParetoError, match="direction"):
        Objective("psnr", "larger")


def test_dominates_handles_mixed_maximize_and_minimize_objectives() -> None:
    objectives = (
        Objective("psnr", MAXIMIZE),
        Objective("fps", MAXIMIZE),
        Objective("size", MINIMIZE),
    )
    candidate = {"psnr": 31.0, "fps": 500.0, "size": 30.0}
    reference = {"psnr": 30.0, "fps": 500.0, "size": 35.0}

    assert dominates(candidate, reference, objectives)
    assert not dominates(reference, candidate, objectives)


def test_equal_points_do_not_dominate_each_other() -> None:
    objectives = (Objective("psnr", MAXIMIZE), Objective("latency", MINIMIZE))
    point = {"psnr": 30.0, "latency": 2.0}

    assert not dominates(point, dict(point), objectives)


def test_non_dominated_sort_builds_multiple_fronts() -> None:
    objectives = (Objective("quality", MAXIMIZE), Objective("speed", MAXIMIZE))
    points = [
        {"id": "a", "quality": 10.0, "speed": 10.0},
        {"id": "b", "quality": 9.0, "speed": 9.0},
        {"id": "c", "quality": 8.0, "speed": 12.0},
        {"id": "d", "quality": 7.0, "speed": 8.0},
        {"id": "e", "quality": 6.0, "speed": 7.0},
    ]

    assert non_dominated_sort(points, objectives) == [[0, 2], [1], [3], [4]]
    assert pareto_ranks(points, objectives) == [0, 1, 0, 2, 3]


def test_annotate_pareto_ranks_copies_rows_and_marks_dominated_points() -> None:
    rows = [
        {
            "variant_id": "baseline",
            "psnr": 35.916,
            "fps": 282.98,
            "serialized_mib": 70.91,
        },
        {
            "variant_id": "top_k_keep_075",
            "psnr": 34.496,
            "fps": 456.45,
            "serialized_mib": 53.18,
        },
        {
            "variant_id": "random_keep_075_seed_0",
            "psnr": 31.857,
            "fps": 360.79,
            "serialized_mib": 53.18,
        },
    ]

    annotated = annotate_pareto_ranks(rows, QUALITY_EFFICIENCY_OBJECTIVES)

    assert "pareto_rank" not in rows[0]
    assert annotated[0]["pareto_rank"] == 0
    assert annotated[1]["pareto_rank"] == 0
    assert annotated[2]["pareto_rank"] == 1


def test_missing_objective_values_are_reported() -> None:
    objectives = (Objective("psnr", MAXIMIZE),)

    with pytest.raises(ParetoError, match="missing"):
        dominates({}, {"psnr": 1.0}, objectives)
