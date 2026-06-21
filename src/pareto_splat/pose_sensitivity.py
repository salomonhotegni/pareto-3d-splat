"""Configuration and summaries for camera-pose sensitivity studies."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pareto_splat.pose_perturbation import (
    PosePerturbationError,
    PosePerturbationSettings,
    prepare_perturbed_nerf_synthetic_dataset,
)


class PoseSensitivityError(ValueError):
    """Raised when pose-sensitivity study inputs are invalid."""


GRAPHDECO_METADATA_FILES = ("cfg_args", "cameras.json", "exposure.json")
VARIANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class PoseVariantSpec:
    """One perturbed camera-pose operating point."""

    variant_id: str
    rotation_degrees: float
    translation_std: float
    seed: int
    dataset_path: Path
    model_path: Path
    iteration: int
    splits: tuple[str, ...] = ("test",)

    @property
    def settings(self) -> PosePerturbationSettings:
        return PosePerturbationSettings(
            rotation_degrees=self.rotation_degrees,
            translation_std=self.translation_std,
            seed=self.seed,
            splits=self.splits,
        )

    @property
    def point_cloud_path(self) -> Path:
        return (
            self.model_path
            / "point_cloud"
            / f"iteration_{self.iteration}"
            / "point_cloud.ply"
        )

    @property
    def perturbation_metadata_path(self) -> Path:
        return self.dataset_path / "pose_perturbation.json"


@dataclass(frozen=True)
class PoseSensitivityConfig:
    """Validated settings for a camera-pose sensitivity study."""

    config_path: Path
    project_root: Path
    name: str
    scene: str
    iteration: int
    output_root: Path
    dataset_root: Path
    baseline_model_path: Path
    source_path: Path
    baseline_root: Path
    test_views: int
    resolution: int
    white_background: bool
    sh_degree: int
    data_device: str
    image_policy: str
    skip_train: bool
    evaluation_device: str
    profiling_warmup_views: int
    profiling_repetitions: int
    variants: tuple[PoseVariantSpec, ...]

    @property
    def method_name(self) -> str:
        return f"ours_{self.iteration}"

    @property
    def source_point_cloud_path(self) -> Path:
        return (
            self.baseline_model_path
            / "point_cloud"
            / f"iteration_{self.iteration}"
            / "point_cloud.ply"
        )

    @property
    def background_name(self) -> str:
        return "white" if self.white_background else "black"


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PoseSensitivityError(f"{key} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, expected: type) -> Any:
    if key not in mapping:
        raise PoseSensitivityError(f"missing required setting: {key}")
    value = mapping[key]
    if expected is int and isinstance(value, bool):
        raise PoseSensitivityError(f"{key} must be an integer")
    if not isinstance(value, expected):
        raise PoseSensitivityError(f"{key} must be a {expected.__name__}")
    return value


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = _required(mapping, key, int)
    if value <= 0:
        raise PoseSensitivityError(f"{key} must be positive")
    return value


def _nonnegative_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseSensitivityError(f"{key} must be a number")
    number = float(value)
    if number < 0.0:
        raise PoseSensitivityError(f"{key} must be non-negative")
    return number


def _project_path(root_dir: Path, value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise PoseSensitivityError(f"{key} must be relative to the project root")
    resolved = (root_dir / path).resolve()
    try:
        resolved.relative_to(root_dir.resolve())
    except ValueError as error:
        raise PoseSensitivityError(f"{key} must stay within the project root") from error
    return resolved


def _split_tuple(values: object, key: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise PoseSensitivityError(f"{key} must be a non-empty list")
    return tuple(str(value) for value in values)


def _variant_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PoseSensitivityError("variant id must be a non-empty string")
    if not VARIANT_ID_PATTERN.match(value):
        raise PoseSensitivityError(
            "variant id may only contain letters, numbers, '.', '_', and '-'"
        )
    return value


def build_pose_variants(
    output_root: Path,
    dataset_root: Path,
    perturbation_config: dict[str, Any],
    iteration: int,
) -> tuple[PoseVariantSpec, ...]:
    """Build deterministic variant specifications from study config."""

    raw_variants = _required(perturbation_config, "variants", list)
    if not raw_variants:
        raise PoseSensitivityError("pose_perturbation.variants must not be empty")
    default_splits = _split_tuple(
        perturbation_config.get("splits", ["test"]),
        "pose_perturbation.splits",
    )

    variants: list[PoseVariantSpec] = []
    for index, raw_variant in enumerate(raw_variants):
        variant_config = _mapping(raw_variant, f"pose_perturbation.variants[{index}]")
        variant_id = _variant_id(_required(variant_config, "id", str))
        splits = (
            _split_tuple(variant_config["splits"], f"{variant_id}.splits")
            if "splits" in variant_config
            else default_splits
        )
        variant = PoseVariantSpec(
            variant_id=variant_id,
            rotation_degrees=_nonnegative_float(
                variant_config,
                "rotation_degrees",
            ),
            translation_std=_nonnegative_float(variant_config, "translation_std"),
            seed=_required(variant_config, "seed", int),
            dataset_path=dataset_root / variant_id,
            model_path=output_root / variant_id,
            iteration=iteration,
            splits=splits,
        )
        try:
            variant.settings
        except PosePerturbationError as error:
            raise PoseSensitivityError(str(error)) from error
        variants.append(variant)

    variant_ids = [variant.variant_id for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise PoseSensitivityError("variant IDs must be unique")
    return tuple(variants)


def load_pose_sensitivity_config(
    config_path: Path,
    project_root: Path,
) -> PoseSensitivityConfig:
    """Load and validate a pose-sensitivity YAML file."""

    config_path = config_path.resolve()
    project_root = project_root.resolve()
    if not config_path.is_file():
        raise PoseSensitivityError(f"configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    study = _mapping(root.get("study"), "study")
    baseline = _mapping(root.get("baseline"), "baseline")
    dataset = _mapping(root.get("dataset"), "dataset")
    rendering = _mapping(root.get("rendering"), "rendering")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    profiling = _mapping(root.get("profiling"), "profiling")
    perturbation = _mapping(root.get("pose_perturbation"), "pose_perturbation")

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
    evaluation_device = _required(evaluation, "device", str)
    if evaluation_device not in {"auto", "cpu", "cuda"}:
        raise PoseSensitivityError("evaluation.device must be auto, cpu, or cuda")
    data_device = _required(dataset, "data_device", str)
    if data_device not in {"cpu", "cuda"}:
        raise PoseSensitivityError("dataset.data_device must be cpu or cuda")
    image_policy = _required(dataset, "image_policy", str)
    if image_policy not in {"symlink", "copy"}:
        raise PoseSensitivityError("dataset.image_policy must be symlink or copy")
    iteration = _positive_int(study, "iteration")

    return PoseSensitivityConfig(
        config_path=config_path,
        project_root=project_root,
        name=_required(study, "name", str),
        scene=_required(study, "scene", str),
        iteration=iteration,
        output_root=output_root,
        dataset_root=dataset_root,
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
        baseline_root=_project_path(
            project_root,
            _required(baseline, "baseline_root", str),
            "baseline.baseline_root",
        ),
        test_views=_positive_int(dataset, "test_views"),
        resolution=_positive_int(dataset, "resolution"),
        white_background=_required(dataset, "white_background", bool),
        sh_degree=_positive_int(dataset, "sh_degree"),
        data_device=data_device,
        image_policy=image_policy,
        skip_train=_required(rendering, "skip_train", bool),
        evaluation_device=evaluation_device,
        profiling_warmup_views=_positive_int(profiling, "warmup_views"),
        profiling_repetitions=_positive_int(profiling, "repetitions"),
        variants=build_pose_variants(
            output_root,
            dataset_root,
            perturbation,
            iteration,
        ),
    )


def prepare_pose_variant_dataset(
    config: PoseSensitivityConfig,
    variant: PoseVariantSpec,
) -> Path:
    """Prepare the perturbed source dataset for one variant."""

    if variant.perturbation_metadata_path.is_file():
        return variant.perturbation_metadata_path
    result = prepare_perturbed_nerf_synthetic_dataset(
        config.source_path,
        variant.dataset_path,
        variant.settings,
        image_policy=config.image_policy,
    )
    return result.metadata_path


def prepare_pose_variant_model(
    config: PoseSensitivityConfig,
    variant: PoseVariantSpec,
) -> Path:
    """Stage a GraphDeCo model directory that points at the baseline PLY."""

    source_model_path = config.baseline_model_path.resolve()
    output_model_path = variant.model_path.resolve()
    source_point_cloud_path = config.source_point_cloud_path.resolve()
    if not source_point_cloud_path.is_file():
        raise PoseSensitivityError(
            f"baseline point cloud not found: {source_point_cloud_path}"
        )

    output_model_path.mkdir(parents=True, exist_ok=True)
    for name in GRAPHDECO_METADATA_FILES:
        source = source_model_path / name
        if source.is_file():
            shutil.copy2(source, output_model_path / name)

    output_point_cloud_path = variant.point_cloud_path
    output_point_cloud_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_point_cloud_path.exists() and not output_point_cloud_path.is_symlink():
        output_point_cloud_path.symlink_to(source_point_cloud_path)

    metadata = {
        "source_model_path": str(source_model_path),
        "source_point_cloud_path": str(source_point_cloud_path),
        "output_model_path": str(output_model_path),
        "output_point_cloud_path": str(output_point_cloud_path.resolve(strict=False)),
        "iteration": config.iteration,
    }
    metadata_path = output_point_cloud_path.parent / "pose_sensitivity_model.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PoseSensitivityError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_means(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "psnr": float(metrics["metrics"]["psnr"]["mean"]),
        "ssim": float(metrics["metrics"]["ssim"]["mean"]),
        "lpips_vgg": float(metrics["metrics"]["lpips_vgg"]["mean"]),
    }


def _test_split_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    summaries = metadata.get("split_summaries", {})
    if not isinstance(summaries, dict) or "test" not in summaries:
        return {}
    test_summary = summaries["test"]
    return test_summary if isinstance(test_summary, dict) else {}


def collect_summary_rows(config: PoseSensitivityConfig) -> list[dict[str, Any]]:
    """Collect baseline and perturbed-pose metric rows."""

    baseline_metrics_path = (
        config.baseline_model_path / "metrics" / config.method_name / "metrics.json"
    )
    baseline_metrics = _metric_means(_read_json(baseline_metrics_path))
    rows: list[dict[str, Any]] = [
        {
            "scene": config.scene,
            "variant_id": "baseline",
            "model_path": str(config.baseline_model_path),
            "dataset_path": str(config.source_path),
            "rotation_degrees_std": 0.0,
            "translation_std": 0.0,
            "seed": None,
            "mean_rotation_angle_degrees": 0.0,
            "max_rotation_angle_degrees": 0.0,
            "mean_translation_norm": 0.0,
            "max_translation_norm": 0.0,
            **baseline_metrics,
            "psnr_drop": 0.0,
            "ssim_drop": 0.0,
            "lpips_increase": 0.0,
        }
    ]

    for variant in config.variants:
        metrics_path = (
            variant.model_path / "metrics" / config.method_name / "metrics.json"
        )
        metrics = _metric_means(_read_json(metrics_path))
        metadata = _test_split_metadata(_read_json(variant.perturbation_metadata_path))
        rows.append(
            {
                "scene": config.scene,
                "variant_id": variant.variant_id,
                "model_path": str(variant.model_path),
                "dataset_path": str(variant.dataset_path),
                "rotation_degrees_std": variant.rotation_degrees,
                "translation_std": variant.translation_std,
                "seed": variant.seed,
                "mean_rotation_angle_degrees": float(
                    metadata.get("mean_rotation_angle_degrees", 0.0)
                ),
                "max_rotation_angle_degrees": float(
                    metadata.get("max_rotation_angle_degrees", 0.0)
                ),
                "mean_translation_norm": float(
                    metadata.get("mean_translation_norm", 0.0)
                ),
                "max_translation_norm": float(metadata.get("max_translation_norm", 0.0)),
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
    """Write pose-sensitivity summary JSON and CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def write_pose_sensitivity_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write simple metric-drop plots for perturbed-pose results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    perturbed = frame[frame["variant_id"] != "baseline"]
    paths: list[Path] = []

    plot_specs = (
        (
            "mean_rotation_angle_degrees",
            "psnr",
            "Mean Rotation Perturbation (deg)",
            "PSNR (dB)",
            "psnr_vs_rotation.png",
        ),
        (
            "mean_rotation_angle_degrees",
            "psnr_drop",
            "Mean Rotation Perturbation (deg)",
            "PSNR Drop (dB)",
            "psnr_drop_vs_rotation.png",
        ),
        (
            "mean_translation_norm",
            "lpips_increase",
            "Mean Translation Perturbation",
            "LPIPS Increase",
            "lpips_increase_vs_translation.png",
        ),
    )
    for x_name, y_name, x_label, y_label, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(7, 5))
        for translation_std, group in perturbed.groupby("translation_std"):
            axis.scatter(group[x_name], group[y_name], label=f"t_std={translation_std}")
            for _, row in group.iterrows():
                axis.annotate(
                    row["variant_id"],
                    (row[x_name], row[y_name]),
                    fontsize=7,
                    alpha=0.8,
                )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)
