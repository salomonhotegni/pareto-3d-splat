from pathlib import Path

import pytest
import torch
import yaml

from pareto_splat import __version__
from pareto_splat.config import ConfigError, load_experiment_config
from pareto_splat.metrics import MetricInputError, matched_image_paths, psnr, ssim
from pareto_splat.profiling import read_ply_vertex_count, summarize_latencies


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_baseline_config_contains_core_metrics() -> None:
    config_path = ROOT_DIR / "configs" / "baseline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["experiment"]["seed"] == 0
    assert config["dataset"]["name"] == "nerf_synthetic_lego"
    assert config["dataset"]["train_views"] == 100
    assert config["dataset"]["validation_views"] == 100
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
    runner = (ROOT_DIR / "scripts" / "run_experiment.py").read_text(
        encoding="utf-8"
    )

    assert "--config" in script
    assert "run_experiment.py" in script
    assert "--start_checkpoint" in runner
    assert "--checkpoint_iterations" in runner
    assert "checkpoint_retention" in runner
    assert "prune_checkpoints" in runner
    assert "run_graphdeco.py" in runner


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
    runner = (ROOT_DIR / "scripts" / "run_experiment.py").read_text(
        encoding="utf-8"
    )

    assert 'net="vgg"' in evaluator
    assert 'version="0.1"' in evaluator
    assert "normalize=True" in evaluator
    assert "config.test_views" in runner
    assert "config.evaluation_device" in runner


def test_ply_vertex_count_is_read_from_header(tmp_path: Path) -> None:
    point_cloud = tmp_path / "point_cloud.ply"
    point_cloud.write_bytes(
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 299799\n"
        b"property float x\n"
        b"end_header\n"
    )

    assert read_ply_vertex_count(point_cloud) == 299_799


def test_latency_summary_reports_fps_and_interpolated_p95() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["mean_ms"] == pytest.approx(3.0)
    assert summary["median_ms"] == pytest.approx(3.0)
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["frames_per_second"] == pytest.approx(1000.0 / 3.0)


def test_profiling_workflow_excludes_io_and_gpu_reference_images() -> None:
    profiler = (ROOT_DIR / "scripts" / "profile_baseline.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT_DIR / "scripts" / "run_experiment.py").read_text(
        encoding="utf-8"
    )

    assert "data_device=arguments.data_device" in profiler
    assert "torch.cuda.Event" in profiler
    assert "torch.cuda.reset_peak_memory_stats()" in profiler
    assert "save_image" not in profiler
    assert "config.test_views" in runner
    assert "config.profiling_warmup_views" in runner
    assert "config.profiling_repetitions" in runner


def test_experiment_config_drives_paths_and_workflow_settings() -> None:
    config = load_experiment_config(
        ROOT_DIR / "configs" / "baseline.yaml",
        ROOT_DIR,
    )

    assert config.source_path == ROOT_DIR / "data" / "nerf_synthetic" / "lego"
    assert config.model_path == ROOT_DIR / "results" / "baseline" / "lego" / "seed_0"
    assert config.iterations == 30_000
    assert config.render_iteration == 30_000
    assert config.train_views == 100
    assert config.validation_views == 100
    assert config.test_views == 200
    assert config.profiling_warmup_views == 10
    assert config.profiling_repetitions == 3


def test_drums_config_selects_second_scene() -> None:
    config = load_experiment_config(
        ROOT_DIR / "configs" / "drums.yaml",
        ROOT_DIR,
    )

    assert config.dataset_name == "nerf_synthetic_drums"
    assert config.source_path == (
        ROOT_DIR / "data" / "nerf_synthetic" / "drums"
    )
    assert config.model_path == (
        ROOT_DIR / "results" / "baseline" / "drums" / "seed_0"
    )
    assert config.iterations == 30_000
    assert config.train_views == 100
    assert config.validation_views == 100
    assert config.test_views == 200


def test_experiment_config_rejects_rendering_unsaved_iteration(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (ROOT_DIR / "configs" / "baseline.yaml").read_text(encoding="utf-8")
    )
    raw["rendering"]["iteration"] = 29_999
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match="rendering.iteration must be present",
    ):
        load_experiment_config(config_path, ROOT_DIR)
