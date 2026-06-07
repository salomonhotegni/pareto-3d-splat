"""Quality metrics and rendered-image pair validation."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image, UnidentifiedImageError


EXPECTED_IMAGE_SIZE = (800, 800)


class MetricInputError(ValueError):
    """Raised when rendered images do not satisfy the evaluation contract."""


def matched_image_paths(
    render_dir: Path,
    ground_truth_dir: Path,
    *,
    expected_count: int | None = None,
) -> tuple[tuple[Path, Path], ...]:
    """Return deterministically ordered render and ground-truth PNG pairs."""

    render_dir = render_dir.resolve()
    ground_truth_dir = ground_truth_dir.resolve()
    if not render_dir.is_dir():
        raise MetricInputError(f"render directory does not exist: {render_dir}")
    if not ground_truth_dir.is_dir():
        raise MetricInputError(
            f"ground-truth directory does not exist: {ground_truth_dir}"
        )

    render_paths = {path.name: path for path in render_dir.glob("*.png")}
    ground_truth_paths = {
        path.name: path for path in ground_truth_dir.glob("*.png")
    }
    if render_paths.keys() != ground_truth_paths.keys():
        missing_renders = sorted(ground_truth_paths.keys() - render_paths.keys())
        missing_ground_truth = sorted(
            render_paths.keys() - ground_truth_paths.keys()
        )
        details = []
        if missing_renders:
            details.append(f"missing renders: {', '.join(missing_renders)}")
        if missing_ground_truth:
            details.append(
                f"missing ground truth: {', '.join(missing_ground_truth)}"
            )
        raise MetricInputError("; ".join(details))

    names = sorted(render_paths)
    if expected_count is not None and len(names) != expected_count:
        raise MetricInputError(
            f"expected {expected_count} image pairs, found {len(names)}"
        )
    if not names:
        raise MetricInputError("no PNG image pairs found")

    return tuple(
        (render_paths[name], ground_truth_paths[name]) for name in names
    )


def load_rgb_tensor(path: Path) -> torch.Tensor:
    """Load one RGB PNG as a float tensor in NCHW layout and the [0, 1] range."""

    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise MetricInputError(
                    f"expected RGB image at {path}, found {image.mode}"
                )
            if image.size != EXPECTED_IMAGE_SIZE:
                raise MetricInputError(
                    f"expected {EXPECTED_IMAGE_SIZE} image at {path}, "
                    f"found {image.size}"
                )
            image_bytes = bytearray(image.tobytes())
    except (OSError, UnidentifiedImageError) as error:
        raise MetricInputError(f"could not decode image: {path}") from error

    tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)
    tensor = tensor.reshape(EXPECTED_IMAGE_SIZE[1], EXPECTED_IMAGE_SIZE[0], 3)
    return tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def psnr(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute GraphDeCo's RGB PSNR for a batch of images in [0, 1]."""

    mean_squared_error = (
        (image - reference).square().reshape(image.shape[0], -1).mean(dim=1)
    )
    return 20.0 * torch.log10(1.0 / torch.sqrt(mean_squared_error))


def _gaussian_window(
    *,
    size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype)
    coordinates -= size // 2
    gaussian = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    gaussian /= gaussian.sum()
    window_2d = gaussian[:, None] @ gaussian[None, :]
    return (
        window_2d.unsqueeze(0)
        .unsqueeze(0)
        .expand(channels, 1, size, size)
        .contiguous()
    )


def ssim(
    image: torch.Tensor,
    reference: torch.Tensor,
    *,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute GraphDeCo's RGB SSIM, including its zero-padded image boundary."""

    if image.shape != reference.shape:
        raise ValueError(
            f"metric tensor shapes differ: {image.shape} and {reference.shape}"
        )
    if image.ndim != 4:
        raise ValueError(f"expected NCHW tensors, found shape {image.shape}")

    channels = image.shape[1]
    window = _gaussian_window(
        size=window_size,
        sigma=1.5,
        channels=channels,
        device=image.device,
        dtype=image.dtype,
    )
    padding = window_size // 2

    mean_image = functional.conv2d(
        image, window, padding=padding, groups=channels
    )
    mean_reference = functional.conv2d(
        reference, window, padding=padding, groups=channels
    )
    mean_image_squared = mean_image.square()
    mean_reference_squared = mean_reference.square()
    mean_product = mean_image * mean_reference

    variance_image = functional.conv2d(
        image.square(), window, padding=padding, groups=channels
    ) - mean_image_squared
    variance_reference = functional.conv2d(
        reference.square(), window, padding=padding, groups=channels
    ) - mean_reference_squared
    covariance = functional.conv2d(
        image * reference, window, padding=padding, groups=channels
    ) - mean_product

    constant_1 = 0.01**2
    constant_2 = 0.03**2
    ssim_map = (
        (2.0 * mean_product + constant_1)
        * (2.0 * covariance + constant_2)
    ) / (
        (mean_image_squared + mean_reference_squared + constant_1)
        * (variance_image + variance_reference + constant_2)
    )
    return ssim_map.mean()


def finite_metric(value: torch.Tensor, name: str) -> float:
    """Convert a scalar metric tensor to a finite Python float."""

    result = value.detach().item()
    if not math.isfinite(result):
        raise ValueError(f"{name} produced a non-finite value")
    return float(result)
