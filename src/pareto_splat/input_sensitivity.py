"""Configuration and summaries for training-input sensitivity studies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pareto_splat.input_degradation import (
    InputDegradationError,
    InputDegradationSettings,
    prepare_degraded_nerf_synthetic_dataset,
)


class InputSensitivityError(ValueError):
    """Raised when input-sensitivity study settings are invalid."""


VARIANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class InputVariantSpec:
    """One training-input robustness variant."""

    variant_id: str
    degradation: str
    seed: int
    dataset_path: Path
    model_path: Path
    config_path: Path
    train_view_count: int | None = None
    noise_std: float | None = None
    blur_radius: float | None = None
    brightness_factor: float | None = None

    @property
    def settings(self) -> InputDegradationSettings:
        return InputDegradationSettings(
            degradation=self.degradation,  # type: ignore[arg-type]
            seed=self.seed,
            train_view_count=self.train_view_count,
            noise_std=self.noise_std,
            blur_radius=self.blur_radius,
            brightness_factor=self.brightness_factor,
        )

    @property
    def degradation_metadata_path(self) -> Path:
        return self.dataset_path / "input_degradation.json"


@dataclass(frozen=True)
class InputSensitivityConfig:
    """Validated settings for a training-input sensitivity study."""

    config_path: Path
    project_root: Path
    name: str
    scene: str
    output_root: Path
    dataset_root: Path
    variant_config_root: Path
    baseline_config: Path
    baseline_model_path: Path
    source_path: Path
    image_policy: str
    test_views: int
    validation_views: int
    variants: tuple[InputVariantSpec, ...]

    @property
    def method_name(self) -> str:
        baseline = yaml.safe_load(self.baseline_config.read_text(encoding="utf-8"))
        return f"ours_{baseline['rendering']['iteration']}"

    @property
    def baseline_train_views(self) -> int:
        baseline = yaml.safe_load(self.baseline_config.read_text(encoding="utf-8"))
        return int(baseline["dataset"]["train_views"])


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputSensitivityError(f"{key} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, expected: type) -> Any:
    if key not in mapping:
        raise InputSensitivityError(f"missing required setting: {key}")
    value = mapping[key]
    if expected is int and isinstance(value, bool):
        raise InputSensitivityError(f"{key} must be an integer")
    if not isinstance(value, expected):
        raise InputSensitivityError(f"{key} must be a {expected.__name__}")
    return value


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = _required(mapping, key, int)
    if value <= 0:
        raise InputSensitivityError(f"{key} must be positive")
    return value


def _nonnegative_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputSensitivityError(f"{key} must be a number")
    number = float(value)
    if number < 0.0:
        raise InputSensitivityError(f"{key} must be non-negative")
    return number


def _optional_positive_int(mapping: dict[str, Any], key: str) -> int | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputSensitivityError(f"{key} must be a positive integer")
    return value


def _project_path(root_dir: Path, value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise InputSensitivityError(f"{key} must be relative to the project root")
    resolved = (root_dir / path).resolve()
    try:
        resolved.relative_to(root_dir.resolve())
    except ValueError as error:
        raise InputSensitivityError(f"{key} must stay within the project root") from error
    return resolved


def _relative_to_project(project_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _variant_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InputSensitivityError("variant id must be a non-empty string")
    if not VARIANT_ID_PATTERN.match(value):
        raise InputSensitivityError(
            "variant id may only contain letters, numbers, '.', '_', and '-'"
        )
    return value


def _optional_variant_float(
    variant_config: dict[str, Any],
    key: str,
) -> float | None:
    if key not in variant_config or variant_config[key] is None:
        return None
    return _nonnegative_float(variant_config, key)


def build_input_variants(
    output_root: Path,
    dataset_root: Path,
    variant_config_root: Path,
    input_config: dict[str, Any],
) -> tuple[InputVariantSpec, ...]:
    """Build deterministic variant specifications from study config."""

    raw_variants = _required(input_config, "variants", list)
    if not raw_variants:
        raise InputSensitivityError("input_degradation.variants must not be empty")

    variants: list[InputVariantSpec] = []
    for index, raw_variant in enumerate(raw_variants):
        variant_config = _mapping(raw_variant, f"input_degradation.variants[{index}]")
        variant_id = _variant_id(_required(variant_config, "id", str))
        variant = InputVariantSpec(
            variant_id=variant_id,
            degradation=_required(variant_config, "degradation", str),
            seed=_required(variant_config, "seed", int),
            train_view_count=_optional_positive_int(
                variant_config,
                "train_view_count",
            ),
            noise_std=_optional_variant_float(variant_config, "noise_std"),
            blur_radius=_optional_variant_float(variant_config, "blur_radius"),
            brightness_factor=_optional_variant_float(
                variant_config,
                "brightness_factor",
            ),
            dataset_path=dataset_root / variant_id,
            model_path=output_root / variant_id,
            config_path=variant_config_root / f"{variant_id}.yaml",
        )
        try:
            variant.settings
        except InputDegradationError as error:
            raise InputSensitivityError(str(error)) from error
        variants.append(variant)

    variant_ids = [variant.variant_id for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise InputSensitivityError("variant IDs must be unique")
    return tuple(variants)


def load_input_sensitivity_config(
    config_path: Path,
    project_root: Path,
) -> InputSensitivityConfig:
    """Load and validate an input-sensitivity YAML file."""

    config_path = config_path.resolve()
    project_root = project_root.resolve()
    if not config_path.is_file():
        raise InputSensitivityError(f"configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    study = _mapping(root.get("study"), "study")
    baseline = _mapping(root.get("baseline"), "baseline")
    dataset = _mapping(root.get("dataset"), "dataset")
    input_degradation = _mapping(
        root.get("input_degradation"),
        "input_degradation",
    )

    output_root = _project_path(
        project_root,
        _required(study, "output_root", str),
        "study.output_root",
    )
    dataset_root = _project_path(
        project_root,
        _required(study, "dataset_root", str),
        "study.dataset_root",
    )
    variant_config_root = output_root / "_configs"
    image_policy = _required(dataset, "image_policy", str)
    if image_policy not in {"symlink", "copy"}:
        raise InputSensitivityError("dataset.image_policy must be symlink or copy")

    return InputSensitivityConfig(
        config_path=config_path,
        project_root=project_root,
        name=_required(study, "name", str),
        scene=_required(study, "scene", str),
        output_root=output_root,
        dataset_root=dataset_root,
        variant_config_root=variant_config_root,
        baseline_config=_project_path(
            project_root,
            _required(baseline, "config", str),
            "baseline.config",
        ),
        baseline_model_path=_project_path(
            project_root,
            _required(baseline, "model_path", str),
            "baseline.model_path",
        ),
        source_path=_project_path(
            project_root,
            _required(baseline, "source_path", str),
            "baseline.source_path",
        ),
        image_policy=image_policy,
        test_views=_positive_int(dataset, "test_views"),
        validation_views=_positive_int(dataset, "validation_views"),
        variants=build_input_variants(
            output_root,
            dataset_root,
            variant_config_root,
            input_degradation,
        ),
    )


def write_variant_experiment_config(
    config: InputSensitivityConfig,
    variant: InputVariantSpec,
) -> Path:
    """Write a baseline-workflow config specialized to one variant."""

    raw = yaml.safe_load(config.baseline_config.read_text(encoding="utf-8"))
    raw["experiment"]["name"] = f"{config.name}_{variant.variant_id}"
    raw["experiment"]["seed"] = variant.seed
    raw["paths"]["source_path"] = _relative_to_project(
        config.project_root,
        variant.dataset_path,
    )
    raw["paths"]["model_path"] = _relative_to_project(
        config.project_root,
        variant.model_path,
    )
    raw["dataset"]["name"] = f"{config.scene}_{variant.variant_id}"
    raw["dataset"]["train_views"] = (
        variant.train_view_count
        if variant.train_view_count is not None
        else raw["dataset"]["train_views"]
    )
    raw["dataset"]["validation_views"] = config.validation_views
    raw["dataset"]["test_views"] = config.test_views

    variant.config_path.parent.mkdir(parents=True, exist_ok=True)
    variant.config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return variant.config_path


def prepare_input_variant_dataset(
    config: InputSensitivityConfig,
    variant: InputVariantSpec,
) -> Path:
    """Prepare the degraded source dataset for one variant."""

    if variant.degradation_metadata_path.is_file():
        return variant.degradation_metadata_path
    result = prepare_degraded_nerf_synthetic_dataset(
        config.source_path,
        variant.dataset_path,
        variant.settings,
        image_policy=config.image_policy,
    )
    return result.metadata_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputSensitivityError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_means(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "psnr": float(metrics["metrics"]["psnr"]["mean"]),
        "ssim": float(metrics["metrics"]["ssim"]["mean"]),
        "lpips_vgg": float(metrics["metrics"]["lpips_vgg"]["mean"]),
    }


def collect_summary_rows(config: InputSensitivityConfig) -> list[dict[str, Any]]:
    """Collect baseline and degraded-input quality rows."""

    method = config.method_name
    baseline_metrics = _metric_means(
        _read_json(config.baseline_model_path / "metrics" / method / "metrics.json")
    )
    rows: list[dict[str, Any]] = [
        {
            "scene": config.scene,
            "variant_id": "baseline",
            "degradation": "clean",
            "model_path": str(config.baseline_model_path),
            "dataset_path": str(config.source_path),
            "train_view_count": config.baseline_train_views,
            "noise_std": None,
            "blur_radius": None,
            "brightness_factor": None,
            **baseline_metrics,
            "psnr_drop": 0.0,
            "ssim_drop": 0.0,
            "lpips_increase": 0.0,
        }
    ]

    for variant in config.variants:
        metrics = _metric_means(
            _read_json(variant.model_path / "metrics" / method / "metrics.json")
        )
        metadata = _read_json(variant.degradation_metadata_path)
        train_summary = metadata["split_summaries"]["train"]
        rows.append(
            {
                "scene": config.scene,
                "variant_id": variant.variant_id,
                "degradation": variant.degradation,
                "model_path": str(variant.model_path),
                "dataset_path": str(variant.dataset_path),
                "train_view_count": train_summary["output_frame_count"],
                "noise_std": variant.noise_std,
                "blur_radius": variant.blur_radius,
                "brightness_factor": variant.brightness_factor,
                **metrics,
                "psnr_drop": baseline_metrics["psnr"] - metrics["psnr"],
                "ssim_drop": baseline_metrics["ssim"] - metrics["ssim"],
                "lpips_increase": metrics["lpips_vgg"] - baseline_metrics["lpips_vgg"],
            }
        )
    return rows


def write_summary_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write input-sensitivity summary JSON and CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def write_input_sensitivity_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write compact quality-drop plots for degraded-input results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    perturbed = frame[frame["variant_id"] != "baseline"]
    paths: list[Path] = []
    plot_specs = (
        ("psnr", "PSNR (dB)", "psnr_by_variant.png"),
        ("psnr_drop", "PSNR Drop (dB)", "psnr_drop_by_variant.png"),
        ("lpips_increase", "LPIPS Increase", "lpips_increase_by_variant.png"),
    )
    for y_name, y_label, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(perturbed["variant_id"], perturbed[y_name])
        axis.set_ylabel(y_label)
        axis.set_xlabel("variant")
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)
