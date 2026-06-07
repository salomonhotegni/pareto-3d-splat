from pathlib import Path

import pytest
import torch
import yaml

from pareto_splat import __version__
from pareto_splat.metrics import MetricInputError, matched_image_paths, psnr, ssim


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_baseline_config_contains_core_metrics() -> None:
    config_path = ROOT_DIR / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["experiment"]["seed"] == 0
    assert config["dataset"]["name"] == "nerf_synthetic_lego"
    assert config["dataset"]["white_background"] is True
    assert config["training"]["eval_split"] is True
    assert config["training"]["checkpoint_iterations"] == [
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
    ]
    assert config["training"]["checkpoint_retention"] == 2
    assert set(config["metrics"]["quality"]) == {"psnr", "ssim", "lpips"}
    assert set(config["metrics"]["efficiency"]) == {
        "fps",
        "model_size_mb",
        "gaussian_count",
        "peak_gpu_memory_mb",
    }


def test_baseline_is_pinned_to_a_commit() -> None:
    settings_path = ROOT_DIR / "scripts" / "baseline.env"
    settings = dict(
        line.split("=", maxsplit=1)
        for line in settings_path.read_text(encoding="utf-8").splitlines()
        if line
    )

    commit = settings["BASELINE_COMMIT"]
    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)


def test_training_wrapper_has_resume_and_checkpoint_retention() -> None:
    script = (ROOT_DIR / "scripts" / "train_baseline.sh").read_text(
        encoding="utf-8"
    )

    assert "--start_checkpoint" in script
    assert "--checkpoint_iterations" in script
    assert "CHECKPOINT_KEEP_COUNT=2" in script
    assert "prune_checkpoints" in script
    assert "run_graphdeco.py" in script


def test_comparison_video_has_expected_layout() -> None:
    script = (ROOT_DIR / "scripts" / "create_comparison_video.sh").read_text(
        encoding="utf-8"
    )

    assert "FRAME_COUNT=200" in script
    assert "FRAME_RATE=30" in script
    assert "Ground Truth" in script
    assert "3DGS Render" in script
    assert "hstack=inputs=2" in script
    assert "libx264" in script
    assert "resolve_media_tool" in script


def test_identical_image_quality_metrics() -> None:
    image = torch.rand(1, 3, 24, 24)

    assert torch.isinf(psnr(image, image)).all()
    assert ssim(image, image).item() == pytest.approx(1.0, abs=1e-6)


def test_metric_image_pairs_require_matching_names(tmp_path: Path) -> None:
    render_dir = tmp_path / "renders"
    ground_truth_dir = tmp_path / "gt"
    render_dir.mkdir()
    ground_truth_dir.mkdir()
    (render_dir / "00000.png").touch()
    (ground_truth_dir / "00001.png").touch()

    with pytest.raises(MetricInputError, match="missing renders"):
        matched_image_paths(render_dir, ground_truth_dir)


def test_evaluation_workflow_uses_all_test_views_and_lpips_vgg() -> None:
    evaluator = (ROOT_DIR / "scripts" / "evaluate_baseline.py").read_text(
        encoding="utf-8"
    )
    wrapper = (ROOT_DIR / "scripts" / "evaluate_baseline.sh").read_text(
        encoding="utf-8"
    )

    assert 'net="vgg"' in evaluator
    assert 'version="0.1"' in evaluator
    assert "normalize=True" in evaluator
    assert "--expected-count 200" in wrapper
    assert "--device auto" in wrapper
