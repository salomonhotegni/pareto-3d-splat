import json
from pathlib import Path

import pytest
import yaml

from pareto_splat.input_sensitivity import (
    InputSensitivityError,
    collect_summary_rows,
    load_input_sensitivity_config,
    write_input_sensitivity_plots,
    write_summary_outputs,
    write_variant_experiment_config,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def metric_payload(psnr: float, ssim: float = 0.98, lpips: float = 0.02) -> dict:
    return {
        "metrics": {
            "psnr": {"mean": psnr},
            "ssim": {"mean": ssim},
            "lpips_vgg": {"mean": lpips},
        }
    }


def degradation_metadata(train_count: int) -> dict:
    return {
        "split_summaries": {
            "train": {
                "input_frame_count": 100,
                "output_frame_count": train_count,
                "selected_indices": list(range(train_count)),
                "degradation": "clean",
            }
        }
    }


def write_baseline_config(tmp_path: Path) -> Path:
    baseline = yaml.safe_load(
        (ROOT_DIR / "configs" / "baseline.yaml").read_text(encoding="utf-8")
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "baseline.yaml"
    path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
    return path


def test_input_sensitivity_config_builds_expected_variants() -> None:
    config = load_input_sensitivity_config(
        ROOT_DIR / "configs" / "input_sensitivity_lego.yaml",
        ROOT_DIR,
    )
    variant_ids = [variant.variant_id for variant in config.variants]

    assert config.scene == "lego"
    assert len(config.variants) == 6
    assert "noise_std_0p02" in variant_ids
    assert "train_views_025" in variant_ids
    assert config.variants[0].settings.degradation == "gaussian-noise"
    assert config.variants[-1].settings.train_view_count == 25


def test_input_sensitivity_config_rejects_invalid_variant(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "input_sensitivity_lego.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["input_degradation"]["variants"][0]["noise_std"] = -0.1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(InputSensitivityError, match="non-negative"):
        load_input_sensitivity_config(path, ROOT_DIR)


def test_write_variant_experiment_config_points_to_variant_dataset(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "input_sensitivity_lego.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["study"]["output_root"] = "outputs/input"
    raw["study"]["dataset_root"] = "outputs/datasets"
    write_baseline_config(tmp_path)
    raw["baseline"]["config"] = "configs/baseline.yaml"
    raw["baseline"]["model_path"] = "outputs/baseline"
    raw["baseline"]["source_path"] = "outputs/source"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_input_sensitivity_config(path, tmp_path)
    variant = config.variants[-1]
    variant_config_path = write_variant_experiment_config(config, variant)
    variant_config = yaml.safe_load(variant_config_path.read_text(encoding="utf-8"))

    assert variant_config["paths"]["source_path"].endswith(
        "outputs/datasets/train_views_025"
    )
    assert variant_config["paths"]["model_path"].endswith(
        "outputs/input/train_views_025"
    )
    assert variant_config["dataset"]["train_views"] == 25
    assert variant_config["dataset"]["test_views"] == 200


def test_collect_summary_rows_and_write_outputs(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "input_sensitivity_lego.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["study"]["output_root"] = "outputs/input"
    raw["study"]["dataset_root"] = "outputs/datasets"
    write_baseline_config(tmp_path)
    raw["baseline"]["config"] = "configs/baseline.yaml"
    raw["baseline"]["model_path"] = "outputs/baseline"
    raw["baseline"]["source_path"] = "outputs/source"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_input_sensitivity_config(path, tmp_path)
    method = config.method_name
    write_json(
        config.baseline_model_path / "metrics" / method / "metrics.json",
        metric_payload(35.0, ssim=0.98, lpips=0.02),
    )
    for index, variant in enumerate(config.variants):
        write_json(
            variant.model_path / "metrics" / method / "metrics.json",
            metric_payload(34.0 - index, ssim=0.97, lpips=0.03 + index * 0.01),
        )
        write_json(
            variant.degradation_metadata_path,
            degradation_metadata(train_count=variant.train_view_count or 100),
        )

    rows = collect_summary_rows(config)
    json_path, csv_path = write_summary_outputs(rows, tmp_path / "summary")
    plot_paths = write_input_sensitivity_plots(rows, tmp_path / "summary")
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(rows) == 7
    assert rows[0]["variant_id"] == "baseline"
    assert rows[0]["train_view_count"] == 100
    assert rows[1]["psnr_drop"] == 1.0
    assert rows[1]["lpips_increase"] == pytest.approx(0.01)
    assert "psnr_drop" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert summary[-1]["train_view_count"] == 25
    assert {path.name for path in plot_paths} == {
        "psnr_by_variant.png",
        "psnr_drop_by_variant.png",
        "lpips_increase_by_variant.png",
    }
    assert all(path.is_file() for path in plot_paths)
