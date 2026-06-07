from pathlib import Path

import yaml

from pareto_splat import __version__


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
