import json
from pathlib import Path

import numpy as np
import pytest

from pareto_splat.visibility import (
    Camera,
    VisibilityError,
    frustum_mask,
    load_cameras_json,
    project_points,
    visibility_aware_importance,
    visibility_weights,
)


def make_camera() -> Camera:
    return Camera(
        camera_id=0,
        image_name="synthetic",
        width=100,
        height=100,
        position=np.array([0.0, 0.0, 0.0]),
        rotation=np.eye(3),
        fx=100.0,
        fy=100.0,
    )


def make_vertices() -> np.ndarray:
    vertices = np.zeros(
        3,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("opacity", "f4"),
        ],
    )
    vertices["x"] = [0.0, 0.0, 10.0]
    vertices["y"] = [0.0, 0.0, 0.0]
    vertices["z"] = [1.0, 2.0, 2.0]
    vertices["opacity"] = [0.0, 0.0, 10.0]
    return vertices


def test_load_cameras_json_reads_graphdeco_export(tmp_path: Path) -> None:
    path = tmp_path / "cameras.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "img_name": "r_0",
                    "width": 800,
                    "height": 600,
                    "position": [1.0, 2.0, 3.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "fx": 500.0,
                    "fy": 510.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    (camera,) = load_cameras_json(path)

    assert camera.camera_id == 7
    assert camera.image_name == "r_0"
    assert camera.width == 800
    assert camera.height == 600
    assert camera.position.tolist() == [1.0, 2.0, 3.0]


def test_load_cameras_json_rejects_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "cameras.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(VisibilityError, match="non-empty"):
        load_cameras_json(path)


def test_project_points_uses_graphdeco_camera_to_world_rotation() -> None:
    camera = Camera(
        camera_id=0,
        image_name="rotated",
        width=100,
        height=100,
        position=np.array([0.0, 0.0, 0.0]),
        rotation=np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        fx=100.0,
        fy=100.0,
    )

    pixels, depth = project_points(np.array([[-2.0, 0.0, 0.0]]), camera)

    assert depth[0] == pytest.approx(2.0)
    assert pixels[0].tolist() == pytest.approx([50.0, 50.0])


def test_frustum_mask_rejects_offscreen_and_behind_camera_points() -> None:
    camera = make_camera()
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ]
    )

    mask, depth = frustum_mask(points, camera)

    assert mask.tolist() == [True, False, False]
    assert depth.tolist() == pytest.approx([2.0, 2.0, -2.0])


def test_visibility_weights_accumulate_inverse_depth_for_visible_points() -> None:
    camera = make_camera()
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [10.0, 0.0, 2.0],
        ]
    )

    visibility, counts = visibility_weights(
        points,
        [camera],
        depth_epsilon=1e-6,
    )

    assert counts.tolist() == [1, 1, 0]
    assert visibility[0] == pytest.approx(1.0 / (1.0 + 1e-6))
    assert visibility[1] == pytest.approx(1.0 / (4.0 + 1e-6))
    assert visibility[2] == 0.0


def test_visibility_aware_importance_combines_opacity_and_visibility() -> None:
    scores = visibility_aware_importance(
        make_vertices(),
        [make_camera()],
        depth_epsilon=1e-6,
    )

    assert scores.visibility_count.tolist() == [1, 1, 0]
    assert scores.opacity[0] == pytest.approx(0.5)
    assert scores.importance[0] > scores.importance[1]
    assert scores.importance[2] == 0.0
