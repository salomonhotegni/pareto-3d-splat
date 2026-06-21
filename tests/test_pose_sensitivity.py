import json
from pathlib import Path

import pytest
import yaml

from pareto_splat.pose_sensitivity import (
    PoseSensitivityError,
    collect_summary_rows,
    load_pose_sensitivity_config,
    prepare_pose_variant_model,
    write_pose_sensitivity_plots,
    write_summary_outputs,
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


def perturbation_metadata(
    mean_rotation: float = 0.1,
    mean_translation: float = 0.002,
) -> dict:
    return {
        "split_summaries": {
            "test": {
                "mean_rotation_angle_degrees": mean_rotation,
                "max_rotation_angle_degrees": mean_rotation * 2.0,
                "mean_translation_norm": mean_translation,
                "max_translation_norm": mean_translation * 2.0,
            }
        }
    }


def test_pose_sensitivity_config_builds_expected_variants() -> None:
    config = load_pose_sensitivity_config(
        ROOT_DIR / "configs" / "pose_sensitivity_lego.yaml",
        ROOT_DIR,
    )
    variant_ids = [variant.variant_id for variant in config.variants]

    assert config.scene == "lego"
    assert len(config.variants) == 5
    assert "rot_0p25deg" in variant_ids
    assert "rot_0p50deg_trans_0p010" in variant_ids
    assert config.variants[0].settings.splits == ("test",)
    assert "iteration_30000" in str(config.variants[0].point_cloud_path)


def test_pose_sensitivity_config_rejects_negative_noise(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "pose_sensitivity_lego.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["pose_perturbation"]["variants"][0]["translation_std"] = -0.1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(PoseSensitivityError, match="non-negative"):
        load_pose_sensitivity_config(path, ROOT_DIR)


def test_prepare_pose_variant_model_links_baseline_point_cloud(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "pose_sensitivity_lego.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["study"]["output_root"] = "outputs/pose"
    raw["study"]["dataset_root"] = "outputs/datasets"
    raw["baseline"]["model_path"] = "baseline"
    raw["baseline"]["source_path"] = "source"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    source_ply = (
        tmp_path / "baseline" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    )
    source_ply.parent.mkdir(parents=True)
    source_ply.write_text("ply\n", encoding="utf-8")
    (tmp_path / "baseline" / "cfg_args").write_text("Namespace()\n", encoding="utf-8")

    config = load_pose_sensitivity_config(path, tmp_path)
    variant = config.variants[0]
    metadata_path = prepare_pose_variant_model(config, variant)

    assert metadata_path.is_file()
    assert variant.point_cloud_path.is_symlink()
    assert variant.point_cloud_path.resolve() == source_ply.resolve()
    assert (variant.model_path / "cfg_args").is_file()


def test_collect_summary_rows_and_write_outputs(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "pose_sensitivity_lego.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["study"]["output_root"] = "outputs/pose"
    raw["study"]["dataset_root"] = "outputs/datasets"
    raw["baseline"]["model_path"] = "outputs/baseline"
    raw["baseline"]["source_path"] = "outputs/source"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_pose_sensitivity_config(path, tmp_path)
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
            variant.perturbation_metadata_path,
            perturbation_metadata(mean_rotation=0.1 + index),
        )

    rows = collect_summary_rows(config)
    json_path, csv_path = write_summary_outputs(rows, tmp_path / "summary")
    plot_paths = write_pose_sensitivity_plots(rows, tmp_path / "summary")
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(rows) == 6
    assert rows[0]["variant_id"] == "baseline"
    assert rows[0]["psnr_drop"] == 0.0
    assert rows[1]["psnr_drop"] == 1.0
    assert rows[1]["lpips_increase"] == pytest.approx(0.01)
    assert "psnr_drop" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert summary[1]["mean_rotation_angle_degrees"] == 0.1
    assert {path.name for path in plot_paths} == {
        "psnr_vs_rotation.png",
        "psnr_drop_vs_rotation.png",
        "lpips_increase_vs_translation.png",
    }
    assert all(path.is_file() for path in plot_paths)
