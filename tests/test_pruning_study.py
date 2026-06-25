import json
from pathlib import Path

import pytest
import yaml

from pareto_splat.pruning_study import (
    PruningStudyError,
    collect_summary_rows,
    load_pruning_study_config,
    write_summary_outputs,
    write_tradeoff_plots,
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
    assert len(config.variants) == 12
    assert "random_keep_050_seed_0" in variant_ids
    assert "top_k_keep_075" in variant_ids
    assert "visibility_top_k_keep_075" in variant_ids
    assert "opacity_threshold_030" in variant_ids
    assert "iteration_30000" in str(config.variants[0].point_cloud_path)


def test_importance_ablation_config_builds_score_mode_variants() -> None:
    config = load_pruning_study_config(
        ROOT_DIR / "configs" / "importance_ablation_lego.yaml",
        ROOT_DIR,
    )

    variant_ids = [variant.variant_id for variant in config.variants]
    modes = {
        variant.variant_id: variant.importance_mode
        for variant in config.variants
        if variant.strategy == "visibility-top-k"
    }

    assert config.name == "lego_importance_ablation"
    assert len(config.variants) == 12
    assert "top_k_keep_050" in variant_ids
    assert "visibility_top_k_keep_050_opacity_visibility" in variant_ids
    assert "visibility_top_k_keep_050_visibility" in variant_ids
    assert "visibility_top_k_keep_050_opacity_count" in variant_ids
    assert modes["visibility_top_k_keep_050_opacity_count"] == "opacity_count"


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
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(rows) == 13
    assert rows[0]["variant_id"] == "baseline"
    assert rows[0]["keep_fraction"] == 1.0
    assert rows[1]["keep_fraction"] == 0.5
    assert summary[0]["pareto_rank"] == 0
    assert "pareto_rank" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert json_path.is_file()
    assert csv_path.is_file()


def test_write_tradeoff_plots_includes_pareto_fronts(tmp_path: Path) -> None:
    rows = [
        {
            "variant_id": "baseline",
            "strategy": "baseline",
            "psnr": 35.0,
            "lpips_vgg": 0.02,
            "fps": 300.0,
            "gaussian_count": 300_000,
            "serialized_mib": 70.0,
        },
        {
            "variant_id": "top_k_keep_050",
            "strategy": "top-k",
            "psnr": 30.0,
            "lpips_vgg": 0.04,
            "fps": 600.0,
            "gaussian_count": 150_000,
            "serialized_mib": 35.0,
        },
        {
            "variant_id": "random_keep_050_seed_0",
            "strategy": "random",
            "psnr": 27.0,
            "lpips_vgg": 0.07,
            "fps": 450.0,
            "gaussian_count": 150_000,
            "serialized_mib": 35.0,
        },
    ]

    paths = write_tradeoff_plots(rows, tmp_path)
    filenames = {path.name for path in paths}

    assert {
        "psnr_vs_gaussians.png",
        "psnr_vs_fps.png",
        "lpips_vs_size.png",
        "pareto_psnr_vs_fps.png",
        "pareto_psnr_vs_size.png",
        "pareto_psnr_fps_size_3d.png",
    } == filenames
    assert all(path.is_file() for path in paths)
