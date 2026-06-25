import json
from pathlib import Path

import pytest

from pareto_splat.demo import (
    DemoError,
    build_demo_payload,
    discover_summary_paths,
    write_demo_html,
)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_discover_summary_paths_finds_generated_summaries(tmp_path: Path) -> None:
    first = (
        tmp_path
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "summary.json"
    )
    second = (
        tmp_path
        / "results"
        / "importance_ablation"
        / "lego"
        / "study_30000"
        / "summary"
        / "summary.json"
    )
    ignored = tmp_path / "results" / "pruning" / "lego" / "summary.json"
    write_summary(second, [{"scene": "lego", "variant_id": "b"}])
    write_summary(first, [{"scene": "lego", "variant_id": "a"}])
    write_summary(ignored, [{"scene": "lego", "variant_id": "ignored"}])

    paths = discover_summary_paths(tmp_path)

    assert paths == (second, first)


def test_build_demo_payload_normalizes_pareto_rows(tmp_path: Path) -> None:
    summary = (
        tmp_path
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "summary.json"
    )
    write_summary(
        summary,
        [
            {
                "scene": "lego",
                "variant_id": "baseline",
                "strategy": "baseline",
                "psnr": 35.0,
                "fps": 300.0,
                "serialized_mib": 70.0,
                "gaussian_count": 300_000,
                "pareto_rank": 0,
            },
            {
                "scene": "lego",
                "variant_id": "top_k_keep_050",
                "strategy": "top-k",
                "psnr": 30.0,
                "fps": 600.0,
                "serialized_mib": 35.0,
                "gaussian_count": 150_000,
                "pareto_rank": 1,
                "optional_null": None,
            },
        ],
    )

    payload = build_demo_payload(tmp_path)
    study = payload["studies"][0]
    rows = study["rows"]

    assert payload["objective"]["vector"] == ["psnr", "fps", "-serialized_mib"]
    assert study["study_id"] == "pruning/lego/study_30000"
    assert study["study_type"] == "pruning"
    assert study["scene"] == "lego"
    assert study["row_count"] == 2
    assert study["pareto_row_count"] == 1
    assert rows[0]["is_pareto_rank_zero"] is True
    assert rows[1]["is_pareto_rank_zero"] is False
    assert "optional_null" not in rows[1]


def test_write_demo_html_embeds_payload(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "objective": {"vector": ["psnr", "fps", "-serialized_mib"]},
        "studies": [
            {
                "study_id": "pruning/lego/study_30000",
                "study_type": "pruning",
                "scene": "lego",
                "run": "study_30000",
                "summary_path": "results/pruning/lego/study_30000/summary/summary.json",
                "row_count": 1,
                "pareto_row_count": 1,
                "rows": [
                    {
                        "variant_id": "top_k_keep_050",
                        "strategy": "top-k",
                        "psnr": 30.0,
                        "fps": 600.0,
                        "serialized_mib": 35.0,
                        "pareto_rank": 0,
                    }
                ],
            }
        ],
    }

    output_path = write_demo_html(payload, tmp_path / "demo" / "index.html")
    html = output_path.read_text(encoding="utf-8")

    assert output_path.is_file()
    assert "Pareto-Splat Demo" in html
    assert 'type="application/json"' in html
    assert "top_k_keep_050" in html
    assert "f(x) = [PSNR, FPS, -Size]" in html


def test_build_demo_payload_rejects_non_list_summary(tmp_path: Path) -> None:
    summary = (
        tmp_path
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(DemoError, match="list of rows"):
        build_demo_payload(tmp_path, summary_paths=[summary])
