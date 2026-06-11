from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pareto_splat.config import load_experiment_config


ROOT_DIR = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "run_experiment",
    ROOT_DIR / "scripts" / "run_experiment.py",
)
assert SPEC is not None and SPEC.loader is not None
RUN_EXPERIMENT = module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_EXPERIMENT)


def test_training_command_is_generated_from_configuration() -> None:
    config = load_experiment_config(
        ROOT_DIR / "configs" / "baseline.yaml",
        ROOT_DIR,
    )

    command = RUN_EXPERIMENT.training_command(config, resume=None)

    assert command[command.index("--source_path") + 1] == str(
        config.source_path
    )
    assert command[command.index("--model_path") + 1] == str(config.model_path)
    assert command[command.index("--iterations") + 1] == "30000"
    assert "--eval" in command
    assert "--white_background" in command
    assert "--disable_viewer" in command


def test_checkpoint_pruning_uses_configured_retention(tmp_path: Path) -> None:
    for iteration in (5_000, 10_000, 15_000):
        (tmp_path / f"chkpnt{iteration}.pth").touch()
    log_path = tmp_path / "retention.log"

    RUN_EXPERIMENT.prune_checkpoints(tmp_path, 2, log_path)

    assert sorted(path.name for path in tmp_path.glob("chkpnt*.pth")) == [
        "chkpnt10000.pth",
        "chkpnt15000.pth",
    ]
    assert "removed chkpnt5000.pth" in log_path.read_text(encoding="utf-8")
