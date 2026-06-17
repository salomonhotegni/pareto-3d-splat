import json
from pathlib import Path

import pytest
import yaml

from pareto_splat.pruning_study import (
    PruningStudyError,
    collect_summary_rows,
    load_pruning_study_config,
    write_summary_outputs,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def metric_payload(psnr: float) -> dict:
    return {
        "metrics": {
            "psnr": {"mean": psnr},
            "ssim": {"mean": 0.98},
            "lpips_vgg": {"mean": 0.02},
        }
    }


def profile_payload(gaussian_count: int, fps: float) -> dict:
    return {
        "rendering": {
            "frames_per_second": fps,
            "mean_ms": 1000.0 / fps,
            "p95_ms": 4.0,
        },
        "model": {
            "gaussian_count": gaussian_count,
            "serialized_mib": 10.0,
        },
        "gpu_memory": {
            "peak_allocated_mib": 100.0,
        },
    }


def pruning_payload(input_count: int, output_count: int) -> dict:
    keep_fraction = output_count / input_count
    return {
        "input_count": input_count,
        "output_count": output_count,
        "keep_fraction": keep_fraction,
        "pruned_fraction": 1.0 - keep_fraction,
    }


def test_pruning_study_config_builds_expected_variants() -> None:
    config = load_pruning_study_config(
        ROOT_DIR / "configs" / "pruning_lego.yaml",
        ROOT_DIR,
    )

    variant_ids = [variant.variant_id for variant in config.variants]

    assert config.scene == "lego"
    assert len(config.variants) == 9
    assert "random_keep_050_seed_0" in variant_ids
    assert "top_k_keep_075" in variant_ids
    assert "opacity_threshold_030" in variant_ids
    assert "iteration_30000" in str(config.variants[0].point_cloud_path)


def test_pruning_study_config_rejects_invalid_fraction(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "pruning_lego.yaml").read_text(encoding="utf-8")
    )
    raw["pruning"]["top-k"]["keep_fractions"] = [1.5]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(PruningStudyError, match="interval"):
        load_pruning_study_config(path, ROOT_DIR)


def test_collect_summary_rows_and_write_outputs(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "pruning_lego.yaml").read_text(encoding="utf-8")
    )
    raw["study"]["output_root"] = "tmp_outputs/pruning"
    raw["baseline"]["model_path"] = "tmp_outputs/baseline"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    project_root = tmp_path

    config = load_pruning_study_config(path, project_root)
    method = config.method_name
    write_json(
        config.baseline_model_path / "metrics" / method / "metrics.json",
        metric_payload(35.0),
    )
    write_json(
        config.baseline_model_path / "profile" / method / "profile.json",
        profile_payload(100, 300.0),
    )
    for index, variant in enumerate(config.variants):
        write_json(
            variant.model_path / "metrics" / method / "metrics.json",
            metric_payload(30.0 - index),
        )
        write_json(
            variant.model_path / "profile" / method / "profile.json",
            profile_payload(50, 400.0),
        )
        write_json(
            variant.point_cloud_path.parent / "pruning_metadata.json",
            pruning_payload(100, 50),
        )

    rows = collect_summary_rows(config)
    json_path, csv_path = write_summary_outputs(rows, tmp_path / "summary")

    assert len(rows) == 10
    assert rows[0]["variant_id"] == "baseline"
    assert rows[0]["keep_fraction"] == 1.0
    assert rows[1]["keep_fraction"] == 0.5
    assert json_path.is_file()
    assert csv_path.is_file()
