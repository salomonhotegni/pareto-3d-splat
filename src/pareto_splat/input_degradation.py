"""Training-input degradation utilities for NeRF Synthetic datasets."""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageFilter


INPUT_DEGRADATION_SCHEMA_VERSION = 1
NERF_SYNTHETIC_TRANSFORM_FILES = {
    "train": "transforms_train.json",
    "val": "transforms_val.json",
    "test": "transforms_test.json",
}

InputDegradationKind = Literal[
    "clean",
    "gaussian-noise",
    "gaussian-blur",
    "brightness",
]


class InputDegradationError(ValueError):
    """Raised when input-degradation settings or datasets are invalid."""


@dataclass(frozen=True)
class InputDegradationSettings:
    """Settings for one training-input robustness variant."""

    degradation: InputDegradationKind
    seed: int = 0
    train_view_count: int | None = None
    noise_std: float | None = None
    blur_radius: float | None = None
    brightness_factor: float | None = None

    def __post_init__(self) -> None:
        if self.degradation not in {
            "clean",
            "gaussian-noise",
            "gaussian-blur",
            "brightness",
        }:
            raise InputDegradationError(
                f"unsupported degradation: {self.degradation}"
            )
        if self.train_view_count is not None and self.train_view_count <= 0:
            raise InputDegradationError("train_view_count must be positive")
        if self.noise_std is not None and self.noise_std < 0.0:
            raise InputDegradationError("noise_std must be non-negative")
        if self.blur_radius is not None and self.blur_radius < 0.0:
            raise InputDegradationError("blur_radius must be non-negative")
        if self.brightness_factor is not None and self.brightness_factor < 0.0:
            raise InputDegradationError("brightness_factor must be non-negative")

        expected = {
            "clean": (),
            "gaussian-noise": ("noise_std",),
            "gaussian-blur": ("blur_radius",),
            "brightness": ("brightness_factor",),
        }[self.degradation]
        provided = {
            "noise_std": self.noise_std,
            "blur_radius": self.blur_radius,
            "brightness_factor": self.brightness_factor,
        }
        for key, value in provided.items():
            if key in expected and value is None:
                raise InputDegradationError(f"{key} is required")
            if key not in expected and value is not None:
                raise InputDegradationError(
                    f"{key} is not valid for {self.degradation}"
                )


@dataclass(frozen=True)
class DegradedDatasetResult:
    """Summary of one prepared degraded training-input dataset."""

    source_path: Path
    output_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InputDegradationError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise InputDegradationError(f"JSON payload must be a mapping: {path}")
    return payload


def _frames(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise InputDegradationError(f"{path}: frames must be a non-empty list")
    if not all(isinstance(frame, dict) for frame in frames):
        raise InputDegradationError(f"{path}: every frame must be a mapping")
    return frames


def _frame_image_path(dataset_path: Path, frame: dict[str, Any]) -> Path:
    frame_path = frame.get("file_path")
    if not isinstance(frame_path, str) or not frame_path:
        raise InputDegradationError("frame file_path must be a non-empty string")
    relative_path = Path(frame_path)
    if relative_path.is_absolute():
        raise InputDegradationError("frame file_path must be relative")
    image_path = (dataset_path / f"{frame_path}.png").resolve()
    try:
        image_path.relative_to(dataset_path.resolve())
    except ValueError as error:
        raise InputDegradationError(
            "frame file_path escapes the dataset directory"
        ) from error
    return image_path


def evenly_spaced_indices(total_count: int, selected_count: int) -> tuple[int, ...]:
    """Select deterministic indices with broad camera coverage."""

    if total_count <= 0:
        raise InputDegradationError("total_count must be positive")
    if selected_count <= 0 or selected_count > total_count:
        raise InputDegradationError(
            f"selected_count must be in [1, {total_count}]"
        )
    if selected_count == total_count:
        return tuple(range(total_count))
    indices = np.linspace(0, total_count - 1, selected_count)
    selected = tuple(int(round(index)) for index in indices)
    if len(set(selected)) != selected_count:
        selected = tuple(int(index) for index in np.linspace(0, total_count, selected_count, endpoint=False))
    if len(set(selected)) != selected_count:
        raise InputDegradationError("could not build unique train-view subset")
    return selected


def select_train_frames(
    frames: list[dict[str, Any]],
    train_view_count: int | None,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Return a deterministic train-frame subset and selected indices."""

    count = len(frames) if train_view_count is None else train_view_count
    indices = evenly_spaced_indices(len(frames), count)
    return [copy.deepcopy(frames[index]) for index in indices], indices


def _rgb_alpha_arrays(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None, str]:
    mode = image.mode
    if mode == "RGBA":
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        return rgba[..., :3], rgba[..., 3:4], "RGBA"
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return rgb, None, "RGB"


def _compose_image(rgb: np.ndarray, alpha: np.ndarray | None, mode: str) -> Image.Image:
    rgb_uint8 = np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)
    if mode == "RGBA" and alpha is not None:
        alpha_uint8 = np.asarray(np.clip(alpha, 0.0, 1.0) * 255.0, dtype=np.uint8)
        return Image.fromarray(np.concatenate([rgb_uint8, alpha_uint8], axis=2), "RGBA")
    return Image.fromarray(rgb_uint8, "RGB")


def degrade_image(
    image: Image.Image,
    settings: InputDegradationSettings,
    rng: np.random.Generator,
) -> Image.Image:
    """Apply a training-image degradation while preserving alpha when present."""

    rgb, alpha, mode = _rgb_alpha_arrays(image)
    if settings.degradation == "clean":
        return image.copy()
    if settings.degradation == "gaussian-noise":
        assert settings.noise_std is not None
        noise = rng.normal(0.0, settings.noise_std, size=rgb.shape)
        return _compose_image(rgb + noise, alpha, mode)
    if settings.degradation == "brightness":
        assert settings.brightness_factor is not None
        return _compose_image(rgb * settings.brightness_factor, alpha, mode)
    if settings.degradation == "gaussian-blur":
        assert settings.blur_radius is not None
        rgb_image = _compose_image(rgb, None, "RGB")
        blurred = rgb_image.filter(ImageFilter.GaussianBlur(settings.blur_radius))
        blurred_rgb = np.asarray(blurred, dtype=np.float32) / 255.0
        return _compose_image(blurred_rgb, alpha, mode)
    raise AssertionError(f"unexpected degradation: {settings.degradation}")


def _copy_or_link(source: Path, output: Path, image_policy: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        return
    if image_policy == "symlink":
        output.symlink_to(source.resolve())
    elif image_policy == "copy":
        shutil.copy2(source, output)
    else:
        raise InputDegradationError("image_policy must be symlink or copy")


def _write_degraded_image(
    source: Path,
    output: Path,
    settings: InputDegradationSettings,
    rng: np.random.Generator,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        degraded = degrade_image(image, settings, rng)
        degraded.save(output)


def _copy_top_level_files(source_path: Path, output_path: Path) -> None:
    transform_names = set(NERF_SYNTHETIC_TRANSFORM_FILES.values())
    for source in source_path.iterdir():
        if source.name in transform_names or source.is_dir():
            continue
        output = output_path / source.name
        if output.exists() or output.is_symlink():
            continue
        shutil.copy2(source, output)


def prepare_degraded_nerf_synthetic_dataset(
    source_path: Path,
    output_path: Path,
    settings: InputDegradationSettings,
    *,
    image_policy: str = "symlink",
    overwrite: bool = False,
) -> DegradedDatasetResult:
    """Create a NeRF Synthetic dataset with degraded training inputs."""

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if not source_path.is_dir():
        raise InputDegradationError(f"source dataset not found: {source_path}")
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise InputDegradationError(f"output dataset already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    _copy_top_level_files(source_path, output_path)

    rng = np.random.default_rng(settings.seed)
    split_summaries: dict[str, Any] = {}
    for split, filename in NERF_SYNTHETIC_TRANSFORM_FILES.items():
        source_json = source_path / filename
        if not source_json.is_file():
            continue
        payload = _load_json(source_json)
        frames = _frames(payload, source_json)
        output_payload = copy.deepcopy(payload)

        if split == "train":
            selected_frames, selected_indices = select_train_frames(
                frames,
                settings.train_view_count,
            )
            output_payload["frames"] = selected_frames
            for frame in selected_frames:
                source_image = _frame_image_path(source_path, frame)
                output_image = _frame_image_path(output_path, frame)
                if settings.degradation == "clean":
                    _copy_or_link(source_image, output_image, image_policy)
                else:
                    _write_degraded_image(source_image, output_image, settings, rng)
            split_summaries[split] = {
                "input_frame_count": len(frames),
                "output_frame_count": len(selected_frames),
                "selected_indices": list(selected_indices),
                "degradation": settings.degradation,
            }
        else:
            output_payload["frames"] = copy.deepcopy(frames)
            for frame in frames:
                source_image = _frame_image_path(source_path, frame)
                output_image = _frame_image_path(output_path, frame)
                _copy_or_link(source_image, output_image, image_policy)
            split_summaries[split] = {
                "input_frame_count": len(frames),
                "output_frame_count": len(frames),
                "degradation": "clean",
            }

        (output_path / filename).write_text(
            json.dumps(output_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    if "train" not in split_summaries:
        raise InputDegradationError("source dataset has no train transform file")

    metadata = {
        "schema_version": INPUT_DEGRADATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "image_policy": image_policy,
        "degradation": settings.degradation,
        "seed": settings.seed,
        "train_view_count": settings.train_view_count,
        "noise_std": settings.noise_std,
        "blur_radius": settings.blur_radius,
        "brightness_factor": settings.brightness_factor,
        "split_summaries": split_summaries,
    }
    metadata_path = output_path / "input_degradation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return DegradedDatasetResult(
        source_path=source_path,
        output_path=output_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def variant_label(value: float) -> str:
    """Convert numeric settings to stable path-safe labels."""

    if not math.isfinite(value):
        raise InputDegradationError("variant label value must be finite")
    return f"{value:.3f}".replace(".", "p").rstrip("0").rstrip("p")
