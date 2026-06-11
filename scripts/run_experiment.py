#!/usr/bin/env python3
"""Run configuration-driven baseline training and evaluation stages."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.config import ConfigError, ExperimentConfig, load_experiment_config


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "baseline.yaml",
        help="experiment YAML (default: configs/baseline.yaml)",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    train_parser = subparsers.add_parser("train", help="train the experiment")
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="resume from a retained GraphDeCo chkpnt*.pth file",
    )
    subparsers.add_parser("render", help="render held-out test views")
    subparsers.add_parser("evaluate", help="compute quality metrics")
    subparsers.add_parser("profile", help="profile rendering efficiency")
    return parser.parse_args()


def graphdeco_command(
    config: ExperimentConfig,
    entry_point: str,
    *arguments: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_graphdeco.py"),
        "--baseline-root",
        str(config.baseline_root),
        entry_point,
        *arguments,
    ]


def optional_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Command:", shlex.join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            raise
        finally:
            process.stdout.close()
        exit_code = process.wait()
    if exit_code:
        raise subprocess.CalledProcessError(exit_code, command)


def write_command(path: Path, command: list[str]) -> None:
    path.write_text(
        f"#!/usr/bin/env bash\ncd {shlex.quote(str(ROOT_DIR))}\n"
        f"{shlex.join(command)}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def validate_dataset(config: ExperimentConfig) -> None:
    if config.dataset_format != "nerf_synthetic":
        raise ConfigError(
            f"unsupported dataset format: {config.dataset_format}"
        )
    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "validate_nerf_synthetic.py"),
            str(config.source_path),
        ],
        cwd=ROOT_DIR,
        check=True,
    )


def training_command(
    config: ExperimentConfig,
    resume: Path | None,
) -> list[str]:
    command = graphdeco_command(
        config,
        "train.py",
        "--source_path",
        str(config.source_path),
        "--model_path",
        str(config.model_path),
        "--resolution",
        str(config.resolution),
        "--iterations",
        str(config.iterations),
        "--test_iterations",
        *(str(value) for value in config.test_iterations),
        "--save_iterations",
        *(str(value) for value in config.save_iterations),
        "--checkpoint_iterations",
        *(str(value) for value in config.checkpoint_iterations),
    )
    optional_flag(command, config.eval_split, "--eval")
    optional_flag(command, config.white_background, "--white_background")
    optional_flag(command, config.disable_viewer, "--disable_viewer")
    if resume is not None:
        command.extend(("--start_checkpoint", str(resume)))
    return command


def checkpoint_paths(model_path: Path) -> list[Path]:
    def iteration(path: Path) -> int:
        return int(path.stem.removeprefix("chkpnt"))

    return sorted(model_path.glob("chkpnt[0-9]*.pth"), key=iteration)


def prune_checkpoints(
    model_path: Path,
    keep_count: int,
    retention_log: Path,
) -> None:
    checkpoints = checkpoint_paths(model_path)
    removed = checkpoints[:-keep_count]
    if not removed:
        return
    with retention_log.open("a", encoding="utf-8") as log_file:
        for checkpoint in removed:
            checkpoint.unlink()
            log_file.write(f"{utc_now()} removed {checkpoint.name}\n")


def write_training_metadata(
    config: ExperimentConfig,
    attempt_dir: Path,
) -> None:
    shutil.copy2(config.config_path, attempt_dir / config.config_path.name)
    with (attempt_dir / "dataset_checksums.sha256").open(
        "w", encoding="utf-8"
    ) as checksum_file:
        for name in (
            "transforms_train.json",
            "transforms_val.json",
            "transforms_test.json",
        ):
            path = config.source_path / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksum_file.write(f"{digest}  {path}\n")

    def command_output(command: list[str]) -> str:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()

    dirty = bool(command_output(["git", "status", "--porcelain"]))
    system_lines = [
        f"start_utc={utc_now()}",
        f"hostname={command_output(['hostname'])}",
        f"project_commit={command_output(['git', 'rev-parse', 'HEAD'])}",
        f"project_dirty={str(dirty).lower()}",
        (
            "baseline_commit="
            + command_output(
                ["git", "-C", str(config.baseline_root), "rev-parse", "HEAD"]
            )
        ),
        f"python_executable={sys.executable}",
        command_output([sys.executable, "--version"]),
        command_output(
            [
                sys.executable,
                "-c",
                (
                    "import torch; print(f'torch={torch.__version__}'); "
                    "print(f'torch_cuda={torch.version.cuda}')"
                ),
            ]
        ),
        command_output(["uname", "-a"]),
        command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    ]
    (attempt_dir / "system.txt").write_text(
        "\n".join(system_lines) + "\n",
        encoding="utf-8",
    )
    if shutil.which("conda"):
        explicit = command_output(["conda", "list", "--explicit"])
        (attempt_dir / "conda-explicit.txt").write_text(
            explicit + "\n",
            encoding="utf-8",
        )


def run_train(config: ExperimentConfig, resume: Path | None) -> None:
    if not (config.baseline_root / "train.py").is_file():
        raise FileNotFoundError(
            f"baseline entry point not found: {config.baseline_root / 'train.py'}"
        )
    validate_dataset(config)

    if resume is None:
        if config.model_path.exists():
            raise FileExistsError(
                f"output path already exists: {config.model_path}; "
                "use --resume or preserve the existing run"
            )
    else:
        resume = resume.resolve()
        if not resume.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume}")
        if resume.parent != config.model_path:
            raise ValueError(
                f"resume checkpoint must be inside {config.model_path}"
            )
        if not (
            resume.name.startswith("chkpnt") and resume.name.endswith(".pth")
        ):
            raise ValueError(f"unexpected checkpoint name: {resume.name}")

    config.model_path.mkdir(parents=True, exist_ok=True)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt_dir = config.model_path / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    write_training_metadata(config, attempt_dir)
    command = training_command(config, resume)
    write_command(attempt_dir / "command.sh", command)

    start_time = time.monotonic()
    start_utc = utc_now()
    status = "failed"
    exit_code = 1
    monitor_stop = threading.Event()

    def monitor_checkpoints() -> None:
        while not monitor_stop.wait(config.checkpoint_poll_seconds):
            prune_checkpoints(
                config.model_path,
                config.checkpoint_retention,
                attempt_dir / "checkpoint_retention.log",
            )

    monitor = threading.Thread(target=monitor_checkpoints, daemon=True)
    try:
        monitor.start()
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print("Command:", shlex.join(command), flush=True)
        print(f"Attempt artifacts: {attempt_dir}", flush=True)
        assert process.stdout is not None
        with (attempt_dir / "train.log").open(
            "w", encoding="utf-8"
        ) as log_file:
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT)
                raise
            finally:
                process.stdout.close()
            exit_code = process.wait()
        if exit_code:
            raise subprocess.CalledProcessError(exit_code, command)
        status = "completed"
    finally:
        monitor_stop.set()
        monitor.join()
        prune_checkpoints(
            config.model_path,
            config.checkpoint_retention,
            attempt_dir / "checkpoint_retention.log",
        )
        duration = int(time.monotonic() - start_time)
        (attempt_dir / "status.txt").write_text(
            "\n".join(
                (
                    f"status={status}",
                    f"exit_code={exit_code}",
                    f"start_utc={start_utc}",
                    f"end_utc={utc_now()}",
                    f"duration_seconds={duration}",
                )
            )
            + "\n",
            encoding="utf-8",
        )


def run_render(config: ExperimentConfig) -> None:
    validate_dataset(config)
    point_cloud = (
        config.model_path
        / "point_cloud"
        / f"iteration_{config.render_iteration}"
        / "point_cloud.ply"
    )
    if not point_cloud.is_file():
        raise FileNotFoundError(f"trained model not found: {point_cloud}")
    output_dir = config.model_path / "evaluation"
    command = graphdeco_command(
        config,
        "render.py",
        "--model_path",
        str(config.model_path),
        "--source_path",
        str(config.source_path),
        "--iteration",
        str(config.render_iteration),
        "--resolution",
        str(config.resolution),
    )
    optional_flag(command, config.eval_split, "--eval")
    optional_flag(command, config.white_background, "--white_background")
    optional_flag(command, config.skip_train, "--skip_train")
    run_logged(command, output_dir / "render.log")


def run_evaluate(config: ExperimentConfig) -> None:
    method = f"ours_{config.render_iteration}"
    test_dir = config.model_path / "test" / method
    output_dir = config.model_path / "metrics" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "evaluate_baseline.py"),
        "--renders",
        str(test_dir / "renders"),
        "--ground-truth",
        str(test_dir / "gt"),
        "--output-dir",
        str(output_dir),
        "--expected-count",
        str(config.test_views),
        "--device",
        config.evaluation_device,
    ]
    write_command(output_dir / "command.sh", command)
    run_logged(command, output_dir / "evaluation.log")


def run_profile(config: ExperimentConfig) -> None:
    method = f"ours_{config.render_iteration}"
    output_dir = config.model_path / "profile" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "profile_baseline.py"),
        "--baseline-root",
        str(config.baseline_root),
        "--model-path",
        str(config.model_path),
        "--source-path",
        str(config.source_path),
        "--output-dir",
        str(output_dir),
        "--iteration",
        str(config.render_iteration),
        "--expected-view-count",
        str(config.test_views),
        "--warmup-views",
        str(config.profiling_warmup_views),
        "--repetitions",
        str(config.profiling_repetitions),
        "--resolution",
        str(config.resolution),
        "--sh-degree",
        str(config.sh_degree),
        "--data-device",
        config.data_device,
        "--background",
        config.background_name,
    ]
    write_command(output_dir / "command.sh", command)
    run_logged(command, output_dir / "profiling.log")


def main() -> int:
    arguments = parse_arguments()
    try:
        config = load_experiment_config(arguments.config, ROOT_DIR)
        if arguments.stage == "train":
            run_train(config, arguments.resume)
        elif arguments.stage == "render":
            run_render(config)
        elif arguments.stage == "evaluate":
            run_evaluate(config)
        elif arguments.stage == "profile":
            run_profile(config)
        else:
            raise AssertionError(f"unexpected stage: {arguments.stage}")
    except (
        ConfigError,
        FileExistsError,
        FileNotFoundError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
