#!/usr/bin/env python3
"""Run and summarize a camera-pose sensitivity study."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.pose_sensitivity import (  # noqa: E402
    PoseSensitivityConfig,
    PoseSensitivityError,
    PoseVariantSpec,
    collect_summary_rows,
    load_pose_sensitivity_config,
    prepare_pose_variant_dataset,
    prepare_pose_variant_model,
    write_pose_sensitivity_plots,
    write_summary_outputs,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "pose_sensitivity_lego.yaml",
        help="pose-sensitivity study YAML",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="optional variant ID to restrict prepare/render/evaluate/profile stages",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "render", "evaluate", "profile", "summarize"):
        subparsers.add_parser(name)
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path) -> None:
    """Run a command from the repo root while teeing output to a log file."""

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
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
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


def selected_variants(
    config: PoseSensitivityConfig,
    variant_ids: list[str] | None,
) -> tuple[PoseVariantSpec, ...]:
    if not variant_ids:
        return config.variants
    requested = set(variant_ids)
    variants = tuple(
        variant for variant in config.variants if variant.variant_id in requested
    )
    found = {variant.variant_id for variant in variants}
    missing = sorted(requested - found)
    if missing:
        raise PoseSensitivityError(f"unknown variant ID(s): {', '.join(missing)}")
    return variants


def require_prepared_variant(variant: PoseVariantSpec) -> None:
    if not variant.perturbation_metadata_path.is_file():
        raise PoseSensitivityError(
            f"perturbed dataset is not prepared: {variant.dataset_path}"
        )
    if not variant.point_cloud_path.is_file():
        raise PoseSensitivityError(
            f"pose-sensitivity model is not prepared: {variant.model_path}"
        )


def run_prepare(
    config: PoseSensitivityConfig,
    variants: tuple[PoseVariantSpec, ...],
) -> None:
    for variant in variants:
        dataset_metadata = prepare_pose_variant_dataset(config, variant)
        model_metadata = prepare_pose_variant_model(config, variant)
        print(f"{variant.variant_id}: dataset metadata {dataset_metadata}")
        print(f"{variant.variant_id}: model metadata {model_metadata}")


def render_command(config: PoseSensitivityConfig, variant: PoseVariantSpec) -> list[str]:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_graphdeco.py"),
        "--baseline-root",
        str(config.baseline_root),
        "render.py",
        "--model_path",
        str(variant.model_path),
        "--source_path",
        str(variant.dataset_path),
        "--iteration",
        str(config.iteration),
        "--resolution",
        str(config.resolution),
    ]
    if config.white_background:
        command.append("--white_background")
    if config.skip_train:
        command.append("--skip_train")
    command.append("--eval")
    return command


def run_render(
    config: PoseSensitivityConfig,
    variants: tuple[PoseVariantSpec, ...],
) -> None:
    for variant in variants:
        require_prepared_variant(variant)
        command = render_command(config, variant)
        log_path = variant.model_path / "evaluation" / "render.log"
        run_logged(command, log_path)


def evaluate_command(
    config: PoseSensitivityConfig,
    variant: PoseVariantSpec,
) -> list[str]:
    test_dir = variant.model_path / "test" / config.method_name
    output_dir = variant.model_path / "metrics" / config.method_name
    return [
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


def run_evaluate(
    config: PoseSensitivityConfig,
    variants: tuple[PoseVariantSpec, ...],
) -> None:
    for variant in variants:
        require_prepared_variant(variant)
        output_dir = variant.model_path / "metrics" / config.method_name
        output_dir.mkdir(parents=True, exist_ok=True)
        command = evaluate_command(config, variant)
        write_command(output_dir / "command.sh", command)
        run_logged(command, output_dir / "evaluation.log")


def profile_command(
    config: PoseSensitivityConfig,
    variant: PoseVariantSpec,
) -> list[str]:
    output_dir = variant.model_path / "profile" / config.method_name
    return [
        sys.executable,
        str(ROOT_DIR / "scripts" / "profile_baseline.py"),
        "--baseline-root",
        str(config.baseline_root),
        "--model-path",
        str(variant.model_path),
        "--source-path",
        str(variant.dataset_path),
        "--output-dir",
        str(output_dir),
        "--iteration",
        str(config.iteration),
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


def run_profile(
    config: PoseSensitivityConfig,
    variants: tuple[PoseVariantSpec, ...],
) -> None:
    for variant in variants:
        require_prepared_variant(variant)
        output_dir = variant.model_path / "profile" / config.method_name
        output_dir.mkdir(parents=True, exist_ok=True)
        command = profile_command(config, variant)
        write_command(output_dir / "command.sh", command)
        run_logged(command, output_dir / "profiling.log")


def run_summarize(config: PoseSensitivityConfig) -> None:
    rows = collect_summary_rows(config)
    summary_dir = config.output_root / "summary"
    json_path, csv_path = write_summary_outputs(rows, summary_dir)
    plot_paths = write_pose_sensitivity_plots(rows, summary_dir)
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    for path in plot_paths:
        print(f"Plot: {path}")


def main() -> int:
    arguments = parse_arguments()
    try:
        config = load_pose_sensitivity_config(arguments.config, ROOT_DIR)
        variants = selected_variants(config, arguments.variant)
        if arguments.stage == "prepare":
            run_prepare(config, variants)
        elif arguments.stage == "render":
            run_render(config, variants)
        elif arguments.stage == "evaluate":
            run_evaluate(config, variants)
        elif arguments.stage == "profile":
            run_profile(config, variants)
        elif arguments.stage == "summarize":
            run_summarize(config)
        else:
            raise AssertionError(f"unexpected stage: {arguments.stage}")
    except (
        PoseSensitivityError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
