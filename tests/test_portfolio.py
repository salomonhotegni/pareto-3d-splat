import json
from pathlib import Path

import pytest
from PIL import Image

from pareto_splat.portfolio import (
    PanelItem,
    PortfolioError,
    build_portfolio_assets,
    write_image_panel,
)


def write_png(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (40, 30), color).save(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_minimal_project_outputs(root: Path) -> None:
    frame_name = "00000.png"
    baseline_root = root / "results" / "baseline" / "lego" / "seed_0"
    write_png(
        baseline_root / "test" / "ours_30000" / "gt" / frame_name,
        (255, 255, 255, 255),
    )
    write_png(
        baseline_root / "test" / "ours_30000" / "renders" / frame_name,
        (220, 20, 60, 255),
    )
    for variant_id, color in (
        ("top_k_keep_075", (0, 128, 255, 255)),
        ("top_k_keep_050", (0, 180, 120, 255)),
        ("top_k_keep_025", (255, 180, 0, 255)),
    ):
        write_png(
            root
            / "results"
            / "pruning"
            / "lego"
            / "study_30000"
            / variant_id
            / "test"
            / "ours_30000"
            / "renders"
            / frame_name,
            color,
        )

    write_json(
        root
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "summary.json",
        [
            {
                "variant_id": "baseline",
                "psnr": 35.9,
                "fps": 283.0,
                "serialized_mib": 70.9,
            },
            {
                "variant_id": "top_k_keep_075",
                "psnr": 34.5,
                "fps": 456.0,
                "serialized_mib": 53.2,
            },
            {
                "variant_id": "top_k_keep_050",
                "psnr": 29.2,
                "fps": 635.0,
                "serialized_mib": 35.5,
            },
            {
                "variant_id": "top_k_keep_025",
                "psnr": 22.4,
                "fps": 896.0,
                "serialized_mib": 17.7,
            },
        ],
    )

    plot_paths = (
        root
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "pareto_psnr_vs_fps.png",
        root
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "pareto_psnr_vs_size.png",
        root
        / "results"
        / "pruning"
        / "lego"
        / "study_30000"
        / "summary"
        / "pareto_psnr_fps_size_3d.png",
        root
        / "results"
        / "importance_ablation"
        / "lego"
        / "study_30000"
        / "summary"
        / "pareto_psnr_vs_fps.png",
        root
        / "results"
        / "pose_sensitivity"
        / "lego"
        / "study_30000"
        / "summary"
        / "psnr_drop_vs_rotation.png",
        root
        / "results"
        / "input_sensitivity"
        / "lego"
        / "study_30000"
        / "summary"
        / "psnr_drop_by_variant.png",
    )
    for index, path in enumerate(plot_paths):
        write_png(path, (index * 20, 100, 180, 255))

    video_path = baseline_root / "videos" / "lego_ground_truth_vs_3dgs.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake mp4")


def test_write_image_panel_creates_labeled_png(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    write_png(first, (255, 0, 0, 255))
    write_png(second, (0, 0, 255, 255))

    output = write_image_panel(
        (
            PanelItem("First", first),
            PanelItem("Second\nwith subtitle", second),
        ),
        tmp_path / "panel.png",
        title="Comparison",
        max_image_width=30,
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.width > 60
        assert image.height > 30


def test_build_portfolio_assets_writes_manifest_and_index(tmp_path: Path) -> None:
    create_minimal_project_outputs(tmp_path)

    manifest = build_portfolio_assets(
        tmp_path,
        tmp_path / "results" / "portfolio",
        frames=(0,),
    )

    output_root = tmp_path / "results" / "portfolio"
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "index.md").is_file()
    assert len(manifest["assets"]["images"]) == 2
    assert len(manifest["assets"]["plots"]) == 6
    assert len(manifest["assets"]["videos"]) == 1
    assert (output_root / manifest["assets"]["images"][0]["path"]).is_file()
    assert manifest["objective"]["vector"] == ["psnr", "fps", "-serialized_mib"]


def test_write_image_panel_rejects_missing_image(tmp_path: Path) -> None:
    with pytest.raises(PortfolioError, match="missing image"):
        write_image_panel(
            (PanelItem("missing", tmp_path / "missing.png"),),
            tmp_path / "panel.png",
            title="Missing",
        )
