"""Visibility-aware Gaussian importance scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class VisibilityError(ValueError):
    """Raised when visibility-score inputs are invalid."""


@dataclass(frozen=True)
class Camera:
    """GraphDeCo-exported pinhole camera parameters."""

    camera_id: int
    image_name: str
    width: int
    height: int
    position: np.ndarray
    rotation: np.ndarray
    fx: float
    fy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise VisibilityError("camera width and height must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise VisibilityError("camera focal lengths must be positive")
        position = np.asarray(self.position, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if position.shape != (3,):
            raise VisibilityError("camera position must have shape (3,)")
        if rotation.shape != (3, 3):
            raise VisibilityError("camera rotation must have shape (3, 3)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise VisibilityError("camera pose contains non-finite values")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)


@dataclass(frozen=True)
class ImportanceScores:
    """Per-Gaussian visibility and importance arrays."""

    opacity: np.ndarray
    visibility: np.ndarray
    visibility_count: np.ndarray
    importance: np.ndarray


def _required(mapping: dict[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise VisibilityError(f"camera entry is missing key: {key}") from error


def camera_from_json(entry: dict[str, Any]) -> Camera:
    """Create a camera from one GraphDeCo `cameras.json` entry."""

    if not isinstance(entry, dict):
        raise VisibilityError("camera entry must be a mapping")
    return Camera(
        camera_id=int(_required(entry, "id")),
        image_name=str(_required(entry, "img_name")),
        width=int(_required(entry, "width")),
        height=int(_required(entry, "height")),
        position=np.asarray(_required(entry, "position"), dtype=np.float64),
        rotation=np.asarray(_required(entry, "rotation"), dtype=np.float64),
        fx=float(_required(entry, "fx")),
        fy=float(_required(entry, "fy")),
    )


def load_cameras_json(path: Path) -> tuple[Camera, ...]:
    """Load GraphDeCo camera metadata from `cameras.json`."""

    if not path.is_file():
        raise VisibilityError(f"camera file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise VisibilityError("camera file must contain a non-empty list")
    return tuple(camera_from_json(entry) for entry in raw)


def positions_from_vertices(vertices: np.ndarray) -> np.ndarray:
    """Extract Gaussian centers from a structured GraphDeCo PLY vertex table."""

    names = vertices.dtype.names
    if names is None:
        raise VisibilityError("vertices must be a structured array")
    missing = {"x", "y", "z"} - set(names)
    if missing:
        raise VisibilityError(f"vertices are missing position field(s): {missing}")
    points = np.column_stack(
        [
            np.asarray(vertices["x"], dtype=np.float64),
            np.asarray(vertices["y"], dtype=np.float64),
            np.asarray(vertices["z"], dtype=np.float64),
        ]
    )
    return validate_points(points)


def opacity_from_vertices(vertices: np.ndarray) -> np.ndarray:
    """Return activated opacity values from raw GraphDeCo opacity logits."""

    names = vertices.dtype.names
    if names is None or "opacity" not in names:
        raise VisibilityError("vertices must contain an opacity field")
    logits = np.asarray(vertices["opacity"], dtype=np.float64)
    if not np.all(np.isfinite(logits)):
        raise VisibilityError("opacity field contains non-finite values")

    positive = logits >= 0.0
    opacity = np.empty_like(logits, dtype=np.float64)
    opacity[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    opacity[~positive] = exp_values / (1.0 + exp_values)
    return opacity


def validate_points(points: np.ndarray) -> np.ndarray:
    """Validate and normalize an `(N, 3)` point array."""

    normalized = np.asarray(points, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 3:
        raise VisibilityError("points must have shape (N, 3)")
    if len(normalized) == 0:
        raise VisibilityError("points must not be empty")
    if not np.all(np.isfinite(normalized)):
        raise VisibilityError("points contain non-finite values")
    return normalized


def world_to_camera(points: np.ndarray, camera: Camera) -> np.ndarray:
    """Transform world-space points into a GraphDeCo camera coordinate frame."""

    points = validate_points(points)
    # cameras.json stores camera-to-world rotation and camera center. With row
    # vectors, right-multiplying by R is equivalent to R.T @ (p - C).
    return (points - camera.position) @ camera.rotation


def project_points(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points to image coordinates for one camera."""

    camera_points = world_to_camera(points, camera)
    z = camera_points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera.fx * (camera_points[:, 0] / z) + camera.width / 2.0
        v = camera.fy * (camera_points[:, 1] / z) + camera.height / 2.0
    pixels = np.column_stack([u, v])
    return pixels, z


def frustum_mask(
    points: np.ndarray,
    camera: Camera,
    *,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return points that project inside the camera image bounds."""

    if min_depth <= 0.0:
        raise VisibilityError("min_depth must be positive")
    pixels, depth = project_points(points, camera)
    mask = (
        (depth > min_depth)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < camera.height)
    )
    return mask, depth


def visibility_weights(
    points: np.ndarray,
    cameras: Sequence[Camera],
    *,
    min_depth: float = 1e-6,
    depth_epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute depth-weighted camera visibility for each point."""

    points = validate_points(points)
    if not cameras:
        raise VisibilityError("at least one camera is required")
    if depth_epsilon <= 0.0:
        raise VisibilityError("depth_epsilon must be positive")

    visibility = np.zeros(len(points), dtype=np.float64)
    visibility_count = np.zeros(len(points), dtype=np.int32)
    for camera in cameras:
        mask, depth = frustum_mask(points, camera, min_depth=min_depth)
        visibility_count[mask] += 1
        visibility[mask] += 1.0 / (np.square(depth[mask]) + depth_epsilon)
    return visibility, visibility_count


def visibility_aware_importance(
    vertices: np.ndarray,
    cameras: Sequence[Camera],
    *,
    min_depth: float = 1e-6,
    depth_epsilon: float = 1e-6,
) -> ImportanceScores:
    """Compute opacity-weighted visibility importance for PLY vertices.

    The score is `sigmoid(opacity_logit) * log(1 + visibility)`, where
    visibility sums inverse-depth weights over cameras whose image bounds
    contain the Gaussian center.
    """

    points = positions_from_vertices(vertices)
    opacity = opacity_from_vertices(vertices)
    visibility, visibility_count = visibility_weights(
        points,
        cameras,
        min_depth=min_depth,
        depth_epsilon=depth_epsilon,
    )
    importance = opacity * np.log1p(visibility)
    return ImportanceScores(
        opacity=opacity,
        visibility=visibility,
        visibility_count=visibility_count,
        importance=importance,
    )
