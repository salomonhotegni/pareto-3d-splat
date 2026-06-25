"""Portfolio asset generation from completed Pareto-Splat experiments."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PORTFOLIO_SCHEMA_VERSION = 1
DEFAULT_FRAMES = (0, 100)
CANVAS_BACKGROUND = (248, 250, 252)
PANEL_BACKGROUND = (255, 255, 255)
TEXT_COLOR = (17, 24, 39)
MUTED_TEXT_COLOR = (71, 85, 105)
BORDER_COLOR = (203, 213, 225)


class PortfolioError(ValueError):
    """Raised when portfolio assets cannot be generated."""


@dataclass(frozen=True)
class PanelItem:
    """One labeled image in a comparison panel."""

    label: str
    image_path: Path


@dataclass(frozen=True)
class AssetRecord:
    """A generated or copied portfolio artifact."""

    kind: str
    label: str
    path: Path
    source_path: Path | None = None


def build_portfolio_assets(
    project_root: Path,
    output_root: Path,
    *,
    frames: Sequence[int] = DEFAULT_FRAMES,
) -> dict[str, Any]:
    """Build the default Session 20 portfolio asset bundle."""

    project_root = project_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(exist_ok=True)
    (output_root / "plots").mkdir(exist_ok=True)
    (output_root / "videos").mkdir(exist_ok=True)

    records: list[AssetRecord] = []
    pruning_rows = _summary_by_variant(
        project_root
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "summary.json"
    )

    for frame_index in frames:
        records.append(
            AssetRecord(
                kind="image",
                label=f"Lego ground truth vs baseline render, frame {frame_index:05d}",
                path=write_image_panel(
                    (
                        PanelItem(
                            "Ground Truth",
                            _baseline_frame_path(project_root, "gt", frame_index),
                        ),
                        PanelItem(
                            _variant_label(pruning_rows, "baseline", "Baseline"),
                            _baseline_frame_path(
                                project_root,
                                "renders",
                                frame_index,
                            ),
                        ),
                    ),
                    (
                        output_root
                        / "images"
                        / f"lego_gt_vs_baseline_{frame_index:05d}.png"
                    ),
                    title="Lego: Ground Truth vs 3DGS Baseline",
                ),
            )
        )
        records.append(
            AssetRecord(
                kind="image",
                label=f"Lego pruning operating points, frame {frame_index:05d}",
                path=write_image_panel(
                    (
                        PanelItem(
                            "Ground Truth",
                            _baseline_frame_path(project_root, "gt", frame_index),
                        ),
                        PanelItem(
                            _variant_label(pruning_rows, "baseline", "Baseline"),
                            _baseline_frame_path(
                                project_root,
                                "renders",
                                frame_index,
                            ),
                        ),
                        PanelItem(
                            _variant_label(
                                pruning_rows,
                                "top_k_keep_075",
                                "75% top-k",
                            ),
                            _pruning_frame_path(
                                project_root,
                                "top_k_keep_075",
                                frame_index,
                            ),
                        ),
                        PanelItem(
                            _variant_label(
                                pruning_rows,
                                "top_k_keep_050",
                                "50% top-k",
                            ),
                            _pruning_frame_path(
                                project_root,
                                "top_k_keep_050",
                                frame_index,
                            ),
                        ),
                        PanelItem(
                            _variant_label(
                                pruning_rows,
                                "top_k_keep_025",
                                "25% top-k",
                            ),
                            _pruning_frame_path(
                                project_root,
                                "top_k_keep_025",
                                frame_index,
                            ),
                        ),
                    ),
                    (
                        output_root
                        / "images"
                        / f"lego_pruning_operating_points_{frame_index:05d}.png"
                    ),
                    title="Lego: Quality-Speed-Size Operating Points",
                ),
            )
        )

    for label, source, destination_name in _default_plot_specs(project_root):
        destination = output_root / "plots" / destination_name
        copy_existing_asset(source, destination)
        records.append(
            AssetRecord(
                kind="plot",
                label=label,
                path=destination,
                source_path=source,
            )
        )

    video_source = (
        project_root
        / "results"
        / "baseline"
        / "lego"
        / "seed_0"
        / "videos"
        / "lego_ground_truth_vs_3dgs.mp4"
    )
    video_destination = output_root / "videos" / "lego_ground_truth_vs_3dgs.mp4"
    copy_existing_asset(video_source, video_destination)
    records.append(
        AssetRecord(
            kind="video",
            label="Lego ground-truth versus 3DGS orbit video",
            path=video_destination,
            source_path=video_source,
        )
    )

    manifest = build_manifest(project_root, output_root, records)
    write_manifest(manifest, output_root / "manifest.json")
    write_portfolio_index(manifest, output_root / "index.md")
    return manifest


def write_image_panel(
    items: Sequence[PanelItem],
    output_path: Path,
    *,
    title: str,
    max_image_width: int = 320,
    padding: int = 20,
) -> Path:
    """Write a horizontal labeled comparison panel."""

    if not items:
        raise PortfolioError("at least one panel item is required")

    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    processed: list[tuple[PanelItem, Image.Image]] = []
    for item in items:
        image = _load_rgb_image(item.image_path)
        processed.append((item, _resize_to_width(image, max_image_width)))

    label_height = max(_text_height(font, item.label) for item, _ in processed) + 16
    title_height = _text_height(title_font, title) + 2 * padding
    image_height = max(image.height for _, image in processed)
    column_width = max(max_image_width, max(image.width for _, image in processed))
    width = padding + len(processed) * (column_width + padding)
    height = title_height + image_height + label_height + 2 * padding

    canvas = Image.new("RGB", (width, height), CANVAS_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, padding), title, fill=TEXT_COLOR, font=title_font)

    y_image = title_height
    y_label = y_image + image_height + 10
    for index, (item, image) in enumerate(processed):
        x = padding + index * (column_width + padding)
        panel = Image.new("RGB", (column_width, image_height), PANEL_BACKGROUND)
        panel.paste(image, ((column_width - image.width) // 2, 0))
        canvas.paste(panel, (x, y_image))
        draw.rectangle(
            (x, y_image, x + column_width - 1, y_image + image_height - 1),
            outline=BORDER_COLOR,
            width=1,
        )
        draw.multiline_text(
            (x, y_label),
            item.label,
            fill=MUTED_TEXT_COLOR,
            font=font,
            spacing=3,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def copy_existing_asset(source_path: Path, destination_path: Path) -> Path:
    """Copy an existing result artifact into the portfolio bundle."""

    if not source_path.is_file():
        raise PortfolioError(f"missing source asset: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def build_manifest(
    project_root: Path,
    output_root: Path,
    records: Sequence[AssetRecord],
) -> dict[str, Any]:
    """Build a manifest for generated portfolio assets."""

    grouped: dict[str, list[dict[str, str]]] = {
        "image": [],
        "plot": [],
        "video": [],
    }
    for record in records:
        grouped.setdefault(record.kind, []).append(
            {
                "label": record.label,
                "path": _relative_path(record.path, output_root),
                "source_path": (
                    _relative_path(record.source_path, project_root)
                    if record.source_path is not None
                    else ""
                ),
            }
        )

    return {
        "schema_version": PORTFOLIO_SCHEMA_VERSION,
        "objective": {
            "description": "maximize PSNR and FPS, minimize serialized model size",
            "vector": ["psnr", "fps", "-serialized_mib"],
        },
        "output_root": _relative_path(output_root, project_root),
        "assets": {
            "images": grouped.get("image", []),
            "plots": grouped.get("plot", []),
            "videos": grouped.get("video", []),
        },
    }


def write_manifest(manifest: Mapping[str, Any], output_path: Path) -> Path:
    """Write the asset manifest JSON."""

    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_portfolio_index(manifest: Mapping[str, Any], output_path: Path) -> Path:
    """Write a Markdown index for the local portfolio bundle."""

    assets = manifest.get("assets", {})
    lines = [
        "# Pareto-Splat Portfolio Assets",
        "",
        "This directory is generated by `make portfolio-assets`.",
        "",
        "Default objective vector:",
        "",
        "```math",
        "f(x) = [\\mathrm{PSNR}(x), \\mathrm{FPS}(x), -\\mathrm{SizeMiB}(x)]",
        "```",
        "",
    ]
    for title, key in (
        ("Comparison Images", "images"),
        ("Plots", "plots"),
        ("Videos", "videos"),
    ):
        lines.extend([f"## {title}", ""])
        records = assets.get(key, [])
        if not records:
            lines.extend(["No assets generated.", ""])
            continue
        for record in records:
            label = record.get("label", "asset")
            path = record.get("path", "")
            lines.append(f"- [{label}]({path})")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _baseline_frame_path(project_root: Path, split: str, frame_index: int) -> Path:
    return (
        project_root
        / "results"
        / "baseline"
        / "lego"
        / "seed_0"
        / "test"
        / "ours_30000"
        / split
        / f"{frame_index:05d}.png"
    )


def _pruning_frame_path(project_root: Path, variant_id: str, frame_index: int) -> Path:
    return (
        project_root
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / variant_id
        / "test"
        / "ours_30000"
        / "renders"
        / f"{frame_index:05d}.png"
    )


def _default_plot_specs(project_root: Path) -> tuple[tuple[str, Path, str], ...]:
    pruning_summary = (
        project_root
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
    )
    importance_summary = (
        project_root
        / "results"
        / "importance_ablation"
        / "lego"
        / "study_30000"
        / "summary"
    )
    pose_summary = (
        project_root
        / "results"
        / "pose_sensitivity"
        / "lego"
        / "study_30000"
        / "summary"
    )
    input_summary = (
        project_root
        / "results"
        / "input_sensitivity"
        / "lego"
        / "study_30000"
        / "summary"
    )
    return (
        (
            "Pruning Pareto front: PSNR vs FPS",
            pruning_summary / "pareto_psnr_vs_fps.png",
            "pruning_pareto_psnr_vs_fps.png",
        ),
        (
            "Pruning Pareto front: PSNR vs model size",
            pruning_summary / "pareto_psnr_vs_size.png",
            "pruning_pareto_psnr_vs_size.png",
        ),
        (
            "Pruning 3D Pareto front",
            pruning_summary / "pareto_psnr_fps_size_3d.png",
            "pruning_pareto_psnr_fps_size_3d.png",
        ),
        (
            "Importance ablation Pareto front: PSNR vs FPS",
            importance_summary / "pareto_psnr_vs_fps.png",
            "importance_pareto_psnr_vs_fps.png",
        ),
        (
            "Pose sensitivity: PSNR drop vs rotation",
            pose_summary / "psnr_drop_vs_rotation.png",
            "pose_psnr_drop_vs_rotation.png",
        ),
        (
            "Input sensitivity: PSNR drop by variant",
            input_summary / "psnr_drop_by_variant.png",
            "input_psnr_drop_by_variant.png",
        ),
    )


def _summary_by_variant(summary_path: Path) -> dict[str, dict[str, Any]]:
    if not summary_path.is_file():
        raise PortfolioError(f"missing summary file: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise PortfolioError(f"summary must be a list: {summary_path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        variant_id = row.get("variant_id")
        if isinstance(variant_id, str):
            rows[variant_id] = dict(row)
    return rows


def _variant_label(
    rows_by_variant: Mapping[str, Mapping[str, Any]],
    variant_id: str,
    fallback: str,
) -> str:
    row = rows_by_variant.get(variant_id)
    if row is None:
        return fallback
    parts = [fallback]
    psnr = _number(row.get("psnr"))
    fps = _number(row.get("fps"))
    size = _number(row.get("serialized_mib"))
    if psnr is not None and fps is not None:
        parts.append(f"{psnr:.2f} dB | {fps:.0f} FPS")
    if size is not None:
        parts.append(f"{size:.1f} MiB")
    return "\n".join(parts)


def _load_rgb_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise PortfolioError(f"missing image: {path}")
    try:
        image = Image.open(path)
    except UnidentifiedImageError as error:
        raise PortfolioError(f"invalid image: {path}") from error
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image.copy()
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _text_height(font: ImageFont.ImageFont, text: str) -> int:
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    return box[3] - box[1]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_path(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
