"""Camera-pose perturbations for NeRF synthetic transform files."""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


POSE_PERTURBATION_SCHEMA_VERSION = 1
NERF_SYNTHETIC_TRANSFORM_FILES = {
    "train": "transforms_train.json",
    "val": "transforms_val.json",
    "test": "transforms_test.json",
}


class PosePerturbationError(ValueError):
    """Raised when pose-perturbation inputs are invalid."""


@dataclass(frozen=True)
class PosePerturbationSettings:
    """Noise settings for deterministic camera-pose perturbations."""

    rotation_degrees: float
    translation_std: float
    seed: int = 0
    splits: tuple[str, ...] = ("test",)

    def __post_init__(self) -> None:
        if self.rotation_degrees < 0.0:
            raise PosePerturbationError("rotation_degrees must be non-negative")
        if self.translation_std < 0.0:
            raise PosePerturbationError("translation_std must be non-negative")
        unknown = sorted(set(self.splits) - set(NERF_SYNTHETIC_TRANSFORM_FILES))
        if unknown:
            raise PosePerturbationError(
                f"unknown split(s): {', '.join(unknown)}"
            )
        if not self.splits:
            raise PosePerturbationError("at least one split must be selected")

    @property
    def rotation_radians(self) -> float:
        """Return the standard deviation of rotation noise in radians."""

        return math.radians(self.rotation_degrees)


@dataclass(frozen=True)
class PerturbedDatasetResult:
    """Summary of one prepared perturbed NeRF synthetic dataset."""

    source_path: Path
    output_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return the matrix ``[v]_x`` such that ``[v]_x u = v x u``."""

    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (3,):
        raise PosePerturbationError("skew_symmetric expects a 3-vector")
    x, y, z = values
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert an axis-angle vector to a 3x3 rotation matrix."""

    omega = np.asarray(axis_angle, dtype=np.float64)
    if omega.shape != (3,):
        raise PosePerturbationError("axis-angle rotation must be a 3-vector")
    theta = float(np.linalg.norm(omega))
    if theta == 0.0:
        return np.eye(3, dtype=np.float64)

    axis = omega / theta
    cross = skew_symmetric(axis)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * cross
        + (1.0 - math.cos(theta)) * (cross @ cross)
    )


def validate_transform_matrix(matrix: object) -> np.ndarray:
    """Validate and return a homogeneous 4x4 transform matrix."""

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise PosePerturbationError("transform_matrix must be 4x4")
    if not np.all(np.isfinite(transform)):
        raise PosePerturbationError("transform_matrix contains non-finite values")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise PosePerturbationError(
            "transform_matrix must have homogeneous last row [0, 0, 0, 1]"
        )
    return transform.copy()


def perturb_transform_matrix(
    matrix: object,
    rng: np.random.Generator,
    settings: PosePerturbationSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one random SE(3) perturbation to a camera-to-world matrix."""

    transform = validate_transform_matrix(matrix)
    rotation_noise = (
        rng.normal(0.0, settings.rotation_radians, size=3)
        if settings.rotation_radians > 0.0
        else np.zeros(3, dtype=np.float64)
    )
    translation_noise = (
        rng.normal(0.0, settings.translation_std, size=3)
        if settings.translation_std > 0.0
        else np.zeros(3, dtype=np.float64)
    )

    perturbed = transform.copy()
    delta_rotation = axis_angle_to_rotation_matrix(rotation_noise)
    perturbed[:3, :3] = delta_rotation @ transform[:3, :3]
    perturbed[:3, 3] = transform[:3, 3] + translation_noise

    metadata = {
        "rotation_axis_angle_radians": rotation_noise.tolist(),
        "rotation_angle_degrees": math.degrees(float(np.linalg.norm(rotation_noise))),
        "translation_offset": translation_noise.tolist(),
        "translation_norm": float(np.linalg.norm(translation_noise)),
    }
    return perturbed, metadata


def perturb_transforms_payload(
    payload: dict[str, Any],
    settings: PosePerturbationSettings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perturb every frame transform in a NeRF synthetic JSON payload."""

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise PosePerturbationError("transform payload must contain frames")

    rng = np.random.default_rng(settings.seed)
    output = copy.deepcopy(payload)
    frame_metadata: list[dict[str, Any]] = []
    for index, frame in enumerate(output["frames"]):
        if not isinstance(frame, dict):
            raise PosePerturbationError("each frame must be a mapping")
        if "transform_matrix" not in frame:
            raise PosePerturbationError("frame is missing transform_matrix")
        matrix, metadata = perturb_transform_matrix(
            frame["transform_matrix"],
            rng,
            settings,
        )
        frame["transform_matrix"] = matrix.tolist()
        frame_metadata.append(
            {
                "index": index,
                "file_path": frame.get("file_path"),
                **metadata,
            }
        )

    rotation_angles = [
        float(frame["rotation_angle_degrees"]) for frame in frame_metadata
    ]
    translation_norms = [float(frame["translation_norm"]) for frame in frame_metadata]
    metadata = {
        "frame_count": len(frame_metadata),
        "rotation_degrees_std": settings.rotation_degrees,
        "translation_std": settings.translation_std,
        "seed": settings.seed,
        "mean_rotation_angle_degrees": float(np.mean(rotation_angles)),
        "max_rotation_angle_degrees": float(np.max(rotation_angles)),
        "mean_translation_norm": float(np.mean(translation_norms)),
        "max_translation_norm": float(np.max(translation_norms)),
        "frames": frame_metadata,
    }
    return output, metadata


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PosePerturbationError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise PosePerturbationError(f"JSON payload must be a mapping: {path}")
    return payload


def _copy_or_link_entry(source: Path, output: Path, image_policy: str) -> None:
    if image_policy == "symlink":
        output.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif image_policy == "copy":
        if source.is_dir():
            shutil.copytree(source, output)
        else:
            shutil.copy2(source, output)
    else:
        raise PosePerturbationError("image_policy must be symlink or copy")


def _copy_dataset_assets(
    source_path: Path,
    output_path: Path,
    image_policy: str,
) -> None:
    transform_names = set(NERF_SYNTHETIC_TRANSFORM_FILES.values())
    for source in source_path.iterdir():
        if source.name in transform_names:
            continue
        output = output_path / source.name
        if output.exists() or output.is_symlink():
            continue
        _copy_or_link_entry(source, output, image_policy)


def prepare_perturbed_nerf_synthetic_dataset(
    source_path: Path,
    output_path: Path,
    settings: PosePerturbationSettings,
    *,
    image_policy: str = "symlink",
    overwrite: bool = False,
) -> PerturbedDatasetResult:
    """Create a NeRF synthetic dataset copy with perturbed camera poses."""

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if not source_path.is_dir():
        raise PosePerturbationError(f"source dataset not found: {source_path}")
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise PosePerturbationError(f"output dataset already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    _copy_dataset_assets(source_path, output_path, image_policy)

    split_metadata: dict[str, Any] = {}
    for split, filename in NERF_SYNTHETIC_TRANSFORM_FILES.items():
        source_json = source_path / filename
        if not source_json.is_file():
            continue
        output_json = output_path / filename
        payload = _load_json(source_json)
        if split in settings.splits:
            perturbed_payload, metadata = perturb_transforms_payload(
                payload,
                settings,
            )
            output_json.write_text(
                json.dumps(perturbed_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            split_metadata[split] = metadata
        else:
            shutil.copy2(source_json, output_json)

    if not split_metadata:
        selected = ", ".join(settings.splits)
        raise PosePerturbationError(
            f"none of the selected split transform files exist: {selected}"
        )

    metadata = {
        "schema_version": POSE_PERTURBATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "image_policy": image_policy,
        "splits": list(settings.splits),
        "rotation_degrees_std": settings.rotation_degrees,
        "translation_std": settings.translation_std,
        "seed": settings.seed,
        "split_summaries": split_metadata,
    }
    metadata_path = output_path / "pose_perturbation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return PerturbedDatasetResult(
        source_path=source_path,
        output_path=output_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def split_tuple(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize split names from config or CLI inputs."""

    splits = tuple(str(value) for value in values)
    PosePerturbationSettings(
        rotation_degrees=0.0,
        translation_std=0.0,
        splits=splits,
    )
    return splits
