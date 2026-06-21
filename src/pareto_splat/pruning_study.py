"""Configuration, aggregation, and plotting for pruning studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pareto_splat.pareto import QUALITY_EFFICIENCY_OBJECTIVES, annotate_pareto_ranks


class PruningStudyError(ValueError):
    """Raised when pruning-study inputs are invalid."""


PARETO_RANK_KEY = "pareto_rank"


def _mapping(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PruningStudyError(f"{key} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, expected: type) -> Any:
    if key not in mapping:
        raise PruningStudyError(f"missing required setting: {key}")
    value = mapping[key]
    if expected is int and isinstance(value, bool):
        raise PruningStudyError(f"{key} must be an integer")
    if not isinstance(value, expected):
        raise PruningStudyError(f"{key} must be a {expected.__name__}")
    return value


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = _required(mapping, key, int)
    if value <= 0:
        raise PruningStudyError(f"{key} must be positive")
    return value


def _fraction_list(mapping: dict[str, Any], key: str) -> tuple[float, ...]:
    values = _required(mapping, key, list)
    if not values:
        raise PruningStudyError(f"{key} must not be empty")
    fractions = tuple(float(value) for value in values)
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise PruningStudyError(f"{key} values must be in the interval (0, 1]")
    return fractions


def _threshold_list(mapping: dict[str, Any], key: str) -> tuple[float, ...]:
    values = _required(mapping, key, list)
    if not values:
        raise PruningStudyError(f"{key} must not be empty")
    thresholds = tuple(float(value) for value in values)
    if any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise PruningStudyError(f"{key} values must be in the interval (0, 1)")
    return thresholds


def _project_path(root_dir: Path, value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise PruningStudyError(f"{key} must be relative to the project root")
    resolved = (root_dir / path).resolve()
    try:
        resolved.relative_to(root_dir.resolve())
    except ValueError as error:
        raise PruningStudyError(f"{key} must stay within the project root") from error
    return resolved


def fraction_label(value: float) -> str:
    """Convert 0.5 to 050 for stable path labels."""

    return f"{int(round(value * 100)):03d}"


@dataclass(frozen=True)
class VariantSpec:
    """One pruned model variant to create and evaluate."""

    variant_id: str
    strategy: str
    model_path: Path
    iteration: int
    keep_fraction: float | None = None
    opacity_threshold: float | None = None
    seed: int | None = None

    @property
    def point_cloud_path(self) -> Path:
        return (
            self.model_path
            / "point_cloud"
            / f"iteration_{self.iteration}"
            / "point_cloud.ply"
        )


@dataclass(frozen=True)
class PruningStudyConfig:
    """Validated settings for a pruning study."""

    config_path: Path
    project_root: Path
    name: str
    scene: str
    iteration: int
    output_root: Path
    baseline_config: Path
    baseline_model_path: Path
    source_path: Path
    baseline_root: Path
    test_views: int
    resolution: int
    white_background: bool
    sh_degree: int
    data_device: str
    skip_train: bool
    evaluation_device: str
    profiling_warmup_views: int
    profiling_repetitions: int
    variants: tuple[VariantSpec, ...]

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


def build_variants(
    output_root: Path,
    pruning: dict[str, Any],
    iteration: int,
) -> tuple[VariantSpec, ...]:
    """Build deterministic variant specifications from pruning config."""

    variants: list[VariantSpec] = []

    if "random" in pruning:
        random_config = _mapping(pruning["random"], "pruning.random")
        keep_fractions = _fraction_list(random_config, "keep_fractions")
        seeds = tuple(int(seed) for seed in _required(random_config, "seeds", list))
        if not seeds:
            raise PruningStudyError("pruning.random.seeds must not be empty")
        for keep_fraction in keep_fractions:
            for seed in seeds:
                variant_id = (
                    f"random_keep_{fraction_label(keep_fraction)}_seed_{seed}"
                )
                variants.append(
                    VariantSpec(
                        variant_id=variant_id,
                        strategy="random",
                        model_path=output_root / variant_id,
                        iteration=iteration,
                        keep_fraction=keep_fraction,
                        seed=seed,
                    )
                )

    if "top-k" in pruning:
        top_k_config = _mapping(pruning["top-k"], "pruning.top-k")
        for keep_fraction in _fraction_list(top_k_config, "keep_fractions"):
            variant_id = f"top_k_keep_{fraction_label(keep_fraction)}"
            variants.append(
                VariantSpec(
                    variant_id=variant_id,
                    strategy="top-k",
                    model_path=output_root / variant_id,
                    iteration=iteration,
                    keep_fraction=keep_fraction,
                )
            )

    if "visibility-top-k" in pruning:
        visibility_top_k_config = _mapping(
            pruning["visibility-top-k"],
            "pruning.visibility-top-k",
        )
        for keep_fraction in _fraction_list(
            visibility_top_k_config,
            "keep_fractions",
        ):
            variant_id = f"visibility_top_k_keep_{fraction_label(keep_fraction)}"
            variants.append(
                VariantSpec(
                    variant_id=variant_id,
                    strategy="visibility-top-k",
                    model_path=output_root / variant_id,
                    iteration=iteration,
                    keep_fraction=keep_fraction,
                )
            )

    if "opacity-threshold" in pruning:
        threshold_config = _mapping(
            pruning["opacity-threshold"],
            "pruning.opacity-threshold",
        )
        for threshold in _threshold_list(threshold_config, "thresholds"):
            variant_id = f"opacity_threshold_{fraction_label(threshold)}"
            variants.append(
                VariantSpec(
                    variant_id=variant_id,
                    strategy="opacity-threshold",
                    model_path=output_root / variant_id,
                    iteration=iteration,
                    opacity_threshold=threshold,
                )
            )

    if not variants:
        raise PruningStudyError("at least one pruning strategy must be configured")
    variant_ids = [variant.variant_id for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise PruningStudyError("variant IDs must be unique")
    return tuple(variants)


def load_pruning_study_config(
    config_path: Path,
    project_root: Path,
) -> PruningStudyConfig:
    """Load and validate a pruning-study YAML file."""

    config_path = config_path.resolve()
    project_root = project_root.resolve()
    if not config_path.is_file():
        raise PruningStudyError(f"configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    study = _mapping(root.get("study"), "study")
    baseline = _mapping(root.get("baseline"), "baseline")
    dataset = _mapping(root.get("dataset"), "dataset")
    rendering = _mapping(root.get("rendering"), "rendering")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    profiling = _mapping(root.get("profiling"), "profiling")
    pruning = _mapping(root.get("pruning"), "pruning")

    output_root = _project_path(
        project_root,
        _required(study, "output_root", str),
        "study.output_root",
    )
    evaluation_device = _required(evaluation, "device", str)
    if evaluation_device not in {"auto", "cpu", "cuda"}:
        raise PruningStudyError("evaluation.device must be auto, cpu, or cuda")
    data_device = _required(dataset, "data_device", str)
    if data_device not in {"cpu", "cuda"}:
        raise PruningStudyError("dataset.data_device must be cpu or cuda")
    iteration = _positive_int(study, "iteration")

    return PruningStudyConfig(
        config_path=config_path,
        project_root=project_root,
        name=_required(study, "name", str),
        scene=_required(study, "scene", str),
        iteration=iteration,
        output_root=output_root,
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
        skip_train=_required(rendering, "skip_train", bool),
        evaluation_device=evaluation_device,
        profiling_warmup_views=_positive_int(profiling, "warmup_views"),
        profiling_repetitions=_positive_int(profiling, "repetitions"),
        variants=build_variants(output_root, pruning, iteration),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PruningStudyError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_artifacts(
    *,
    scene: str,
    variant_id: str,
    strategy: str,
    model_path: Path,
    method_name: str,
    metrics_path: Path,
    profile_path: Path,
    pruning_metadata_path: Path | None,
) -> dict[str, Any]:
    """Create one summary row from metric/profile/pruning artifacts."""

    metrics = _read_json(metrics_path)
    profile = _read_json(profile_path)
    pruning = _read_json(pruning_metadata_path) if pruning_metadata_path else None

    row: dict[str, Any] = {
        "scene": scene,
        "variant_id": variant_id,
        "strategy": strategy,
        "model_path": str(model_path),
        "method": method_name,
        "psnr": metrics["metrics"]["psnr"]["mean"],
        "ssim": metrics["metrics"]["ssim"]["mean"],
        "lpips_vgg": metrics["metrics"]["lpips_vgg"]["mean"],
        "fps": profile["rendering"]["frames_per_second"],
        "mean_latency_ms": profile["rendering"]["mean_ms"],
        "p95_latency_ms": profile["rendering"]["p95_ms"],
        "gaussian_count": profile["model"]["gaussian_count"],
        "serialized_mib": profile["model"]["serialized_mib"],
        "peak_allocated_mib": profile["gpu_memory"]["peak_allocated_mib"],
    }
    if pruning is None:
        row.update(
            {
                "input_count": profile["model"]["gaussian_count"],
                "output_count": profile["model"]["gaussian_count"],
                "keep_fraction": 1.0,
                "pruned_fraction": 0.0,
            }
        )
    else:
        row.update(
            {
                "input_count": pruning["input_count"],
                "output_count": pruning["output_count"],
                "keep_fraction": pruning["keep_fraction"],
                "pruned_fraction": pruning["pruned_fraction"],
            }
        )
    return row


def collect_summary_rows(config: PruningStudyConfig) -> list[dict[str, Any]]:
    """Collect baseline and variant rows from completed study artifacts."""

    rows = [
        row_from_artifacts(
            scene=config.scene,
            variant_id="baseline",
            strategy="baseline",
            model_path=config.baseline_model_path,
            method_name=config.method_name,
            metrics_path=(
                config.baseline_model_path
                / "metrics"
                / config.method_name
                / "metrics.json"
            ),
            profile_path=(
                config.baseline_model_path
                / "profile"
                / config.method_name
                / "profile.json"
            ),
            pruning_metadata_path=None,
        )
    ]

    for variant in config.variants:
        rows.append(
            row_from_artifacts(
                scene=config.scene,
                variant_id=variant.variant_id,
                strategy=variant.strategy,
                model_path=variant.model_path,
                method_name=config.method_name,
                metrics_path=(
                    variant.model_path / "metrics" / config.method_name / "metrics.json"
                ),
                profile_path=(
                    variant.model_path / "profile" / config.method_name / "profile.json"
                ),
                pruning_metadata_path=(
                    variant.point_cloud_path.parent / "pruning_metadata.json"
                ),
            )
        )
    return rows


def write_summary_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write pruning-study summary JSON and CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    annotated_rows = annotate_quality_efficiency_ranks(rows)
    json_path.write_text(
        json.dumps(annotated_rows, indent=2) + "\n",
        encoding="utf-8",
    )

    import pandas as pd

    frame = pd.DataFrame(annotated_rows)
    frame.to_csv(csv_path, index=False)
    return json_path, csv_path


def annotate_quality_efficiency_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add Pareto ranks for the default quality-efficiency objective set."""

    return annotate_pareto_ranks(
        rows,
        QUALITY_EFFICIENCY_OBJECTIVES,
        rank_key=PARETO_RANK_KEY,
    )


def write_tradeoff_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write quality-efficiency scatter plots and Pareto front projections."""

    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(annotate_quality_efficiency_ranks(rows))
    plot_specs = (
        ("gaussian_count", "psnr", "PSNR vs Gaussian Count", "psnr_vs_gaussians.png"),
        ("fps", "psnr", "PSNR vs FPS", "psnr_vs_fps.png"),
        ("serialized_mib", "lpips_vgg", "LPIPS vs Model Size", "lpips_vs_size.png"),
    )
    paths: list[Path] = []
    for x_name, y_name, title, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(7, 5))
        for strategy, group in frame.groupby("strategy"):
            axis.scatter(group[x_name], group[y_name], label=strategy)
            for _, row in group.iterrows():
                axis.annotate(
                    row["variant_id"],
                    (row[x_name], row[y_name]),
                    fontsize=7,
                    alpha=0.8,
                )
        axis.set_title(title)
        axis.set_xlabel(x_name)
        axis.set_ylabel(y_name)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    paths.extend(write_pareto_plots(rows, output_dir))
    return tuple(paths)


def _plot_strategy_scatter(axis: Any, frame: Any, x_name: str, y_name: str) -> None:
    for strategy, group in frame.groupby("strategy"):
        axis.scatter(group[x_name], group[y_name], label=strategy)
        for _, row in group.iterrows():
            axis.annotate(
                row["variant_id"],
                (row[x_name], row[y_name]),
                fontsize=7,
                alpha=0.8,
            )


def _plot_rank_zero_projection(axis: Any, frame: Any, x_name: str, y_name: str) -> None:
    front = frame[frame[PARETO_RANK_KEY] == 0].sort_values(x_name)
    axis.plot(
        front[x_name],
        front[y_name],
        color="black",
        linewidth=1.5,
        label="Pareto rank 0",
    )
    axis.scatter(
        front[x_name],
        front[y_name],
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        s=90,
    )


def write_pareto_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write 2D and 3D Pareto-front plots for quality-efficiency objectives."""

    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(annotate_quality_efficiency_ranks(rows))
    paths: list[Path] = []
    plot_specs = (
        (
            "fps",
            "psnr",
            "Pareto Front: PSNR vs FPS",
            "pareto_psnr_vs_fps.png",
            "FPS",
            "PSNR",
        ),
        (
            "serialized_mib",
            "psnr",
            "Pareto Front: PSNR vs Model Size",
            "pareto_psnr_vs_size.png",
            "Serialized model size (MiB)",
            "PSNR",
        ),
    )
    for x_name, y_name, title, filename, x_label, y_label in plot_specs:
        figure, axis = plt.subplots(figsize=(7, 5))
        _plot_strategy_scatter(axis, frame, x_name, y_name)
        _plot_rank_zero_projection(axis, frame, x_name, y_name)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    for strategy, group in frame.groupby("strategy"):
        axis.scatter(
            group["fps"],
            group["serialized_mib"],
            group["psnr"],
            label=strategy,
        )
    front = frame[frame[PARETO_RANK_KEY] == 0].sort_values("fps")
    axis.plot(
        front["fps"],
        front["serialized_mib"],
        front["psnr"],
        color="black",
        linewidth=1.5,
        label="Pareto rank 0",
    )
    axis.set_title("3D Pareto Front: PSNR, FPS, and Model Size")
    axis.set_xlabel("FPS")
    axis.set_ylabel("Serialized model size (MiB)")
    axis.set_zlabel("PSNR")
    axis.view_init(elev=22, azim=-55)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "pareto_psnr_fps_size_3d.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return tuple(paths)
