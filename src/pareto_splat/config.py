"""Load and validate Pareto-Splat experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, expected: type) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required setting: {key}")
    value = mapping[key]
    if expected is int and isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    if not isinstance(value, expected):
        raise ConfigError(f"{key} must be a {expected.__name__}")
    return value


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = _required(mapping, key, int)
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _positive_int_list(mapping: dict[str, Any], key: str) -> tuple[int, ...]:
    value = _required(mapping, key, list)
    if not value or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise ConfigError(f"{key} must be a non-empty list of positive integers")
    if value != sorted(set(value)):
        raise ConfigError(f"{key} must be sorted and contain no duplicates")
    return tuple(value)


def _project_path(root_dir: Path, value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ConfigError(f"{key} must be relative to the project root")
    resolved = (root_dir / path).resolve()
    try:
        resolved.relative_to(root_dir.resolve())
    except ValueError as error:
        raise ConfigError(f"{key} must stay within the project root") from error
    return resolved


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated settings needed by the baseline workflow."""

    config_path: Path
    project_root: Path
    name: str
    seed: int
    baseline_root: Path
    source_path: Path
    model_path: Path
    dataset_name: str
    dataset_format: str
    train_views: int
    validation_views: int
    test_views: int
    white_background: bool
    resolution: int
    sh_degree: int
    data_device: str
    iterations: int
    eval_split: bool
    disable_viewer: bool
    save_iterations: tuple[int, ...]
    test_iterations: tuple[int, ...]
    checkpoint_iterations: tuple[int, ...]
    checkpoint_retention: int
    checkpoint_poll_seconds: int
    render_iteration: int
    skip_train: bool
    evaluation_device: str
    profiling_warmup_views: int
    profiling_repetitions: int

    @property
    def background_name(self) -> str:
        return "white" if self.white_background else "black"


def load_experiment_config(
    config_path: Path,
    project_root: Path,
) -> ExperimentConfig:
    """Load a YAML experiment configuration and validate workflow settings."""

    config_path = config_path.resolve()
    project_root = project_root.resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    experiment = _mapping(root.get("experiment"), "experiment")
    paths = _mapping(root.get("paths"), "paths")
    dataset = _mapping(root.get("dataset"), "dataset")
    training = _mapping(root.get("training"), "training")
    rendering = _mapping(root.get("rendering"), "rendering")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    profiling = _mapping(root.get("profiling"), "profiling")

    iterations = _positive_int(training, "iterations")
    save_iterations = _positive_int_list(training, "save_iterations")
    test_iterations = _positive_int_list(training, "test_iterations")
    checkpoint_iterations = _positive_int_list(
        training, "checkpoint_iterations"
    )
    for key, values in (
        ("training.save_iterations", save_iterations),
        ("training.test_iterations", test_iterations),
        ("training.checkpoint_iterations", checkpoint_iterations),
    ):
        if values[-1] > iterations:
            raise ConfigError(f"{key} cannot exceed training.iterations")

    render_iteration = _positive_int(rendering, "iteration")
    if render_iteration not in save_iterations:
        raise ConfigError(
            "rendering.iteration must be present in training.save_iterations"
        )

    evaluation_device = _required(evaluation, "device", str)
    if evaluation_device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("evaluation.device must be auto, cpu, or cuda")

    data_device = _required(dataset, "data_device", str)
    if data_device not in {"cpu", "cuda"}:
        raise ConfigError("dataset.data_device must be cpu or cuda")

    return ExperimentConfig(
        config_path=config_path,
        project_root=project_root,
        name=_required(experiment, "name", str),
        seed=_required(experiment, "seed", int),
        baseline_root=_project_path(
            project_root,
            _required(paths, "baseline_root", str),
            "paths.baseline_root",
        ),
        source_path=_project_path(
            project_root,
            _required(paths, "source_path", str),
            "paths.source_path",
        ),
        model_path=_project_path(
            project_root,
            _required(paths, "model_path", str),
            "paths.model_path",
        ),
        dataset_name=_required(dataset, "name", str),
        dataset_format=_required(dataset, "format", str),
        train_views=_positive_int(dataset, "train_views"),
        validation_views=_positive_int(dataset, "validation_views"),
        test_views=_positive_int(dataset, "test_views"),
        white_background=_required(dataset, "white_background", bool),
        resolution=_positive_int(dataset, "resolution"),
        sh_degree=_positive_int(dataset, "sh_degree"),
        data_device=data_device,
        iterations=iterations,
        eval_split=_required(training, "eval_split", bool),
        disable_viewer=_required(training, "disable_viewer", bool),
        save_iterations=save_iterations,
        test_iterations=test_iterations,
        checkpoint_iterations=checkpoint_iterations,
        checkpoint_retention=_positive_int(training, "checkpoint_retention"),
        checkpoint_poll_seconds=_positive_int(
            training, "checkpoint_poll_seconds"
        ),
        render_iteration=render_iteration,
        skip_train=_required(rendering, "skip_train", bool),
        evaluation_device=evaluation_device,
        profiling_warmup_views=_positive_int(profiling, "warmup_views"),
        profiling_repetitions=_positive_int(profiling, "repetitions"),
    )
