#!/usr/bin/env python3
"""Run and summarize a training-input sensitivity study."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.input_sensitivity import (  # noqa: E402
    InputSensitivityConfig,
    InputSensitivityError,
    InputVariantSpec,
    collect_summary_rows,
    load_input_sensitivity_config,
    prepare_input_variant_dataset,
    write_input_sensitivity_plots,
    write_summary_outputs,
    write_variant_experiment_config,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "input_sensitivity_lego.yaml",
        help="input-sensitivity study YAML",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="optional variant ID to restrict prepare/train/render/evaluate/profile",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "train", "render", "evaluate", "profile", "summarize"):
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


def selected_variants(
    config: InputSensitivityConfig,
    variant_ids: list[str] | None,
) -> tuple[InputVariantSpec, ...]:
    if not variant_ids:
        return config.variants
    requested = set(variant_ids)
    variants = tuple(
        variant for variant in config.variants if variant.variant_id in requested
    )
    found = {variant.variant_id for variant in variants}
    missing = sorted(requested - found)
    if missing:
        raise InputSensitivityError(f"unknown variant ID(s): {', '.join(missing)}")
    return variants


def require_prepared_variant(variant: InputVariantSpec) -> None:
    if not variant.degradation_metadata_path.is_file():
        raise InputSensitivityError(
            f"degraded dataset is not prepared: {variant.dataset_path}"
        )
    if not variant.config_path.is_file():
        raise InputSensitivityError(
            f"variant experiment config is not prepared: {variant.config_path}"
        )


def run_prepare(
    config: InputSensitivityConfig,
    variants: tuple[InputVariantSpec, ...],
) -> None:
    for variant in variants:
        dataset_metadata = prepare_input_variant_dataset(config, variant)
        variant_config = write_variant_experiment_config(config, variant)
        print(f"{variant.variant_id}: dataset metadata {dataset_metadata}")
        print(f"{variant.variant_id}: experiment config {variant_config}")


def experiment_command(variant: InputVariantSpec, stage: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_experiment.py"),
        "--config",
        str(variant.config_path),
        stage,
    ]


def run_experiment_stage(
    config: InputSensitivityConfig,
    variants: tuple[InputVariantSpec, ...],
    stage: str,
) -> None:
    for variant in variants:
        require_prepared_variant(variant)
        log_path = (
            config.output_root / "_logs" / variant.variant_id / "train.log"
            if stage == "train"
            else variant.model_path / "input_sensitivity" / f"{stage}.log"
        )
        run_logged(experiment_command(variant, stage), log_path)


def run_summarize(config: InputSensitivityConfig) -> None:
    rows = collect_summary_rows(config)
    summary_dir = config.output_root / "summary"
    json_path, csv_path = write_summary_outputs(rows, summary_dir)
    plot_paths = write_input_sensitivity_plots(rows, summary_dir)
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    for path in plot_paths:
        print(f"Plot: {path}")


def main() -> int:
    arguments = parse_arguments()
    try:
        config = load_input_sensitivity_config(arguments.config, ROOT_DIR)
        variants = selected_variants(config, arguments.variant)
        if arguments.stage == "prepare":
            run_prepare(config, variants)
        elif arguments.stage in {"train", "render", "evaluate", "profile"}:
            run_experiment_stage(config, variants, arguments.stage)
        elif arguments.stage == "summarize":
            run_summarize(config)
        else:
            raise AssertionError(f"unexpected stage: {arguments.stage}")
    except (
        InputSensitivityError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
