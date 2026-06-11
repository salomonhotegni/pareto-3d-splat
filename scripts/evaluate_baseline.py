#!/usr/bin/env python3
"""Evaluate rendered baseline test views with PSNR, SSIM, and LPIPS."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import lpips
import torch
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.metrics import (  # noqa: E402
    finite_metric,
    load_rgb_tensor,
    matched_image_paths,
    psnr,
    ssim,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto selects CUDA when available and otherwise uses CPU",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    pairs = matched_image_paths(
        arguments.renders,
        arguments.ground_truth,
        expected_count=arguments.expected_count,
    )
    sample_image = load_rgb_tensor(pairs[0][0])
    image_size = [int(sample_image.shape[3]), int(sample_image.shape[2])]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(pairs)} test-view pairs on {device}")
    print("LPIPS network: VGG, version 0.1")
    lpips_model = lpips.LPIPS(
        net="vgg",
        version="0.1",
        verbose=False,
    ).to(device)
    lpips_model.eval()

    per_view: dict[str, dict[str, float]] = {}
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    with torch.inference_mode():
        for render_path, ground_truth_path in tqdm(
            pairs,
            desc="Metric evaluation",
            unit="view",
        ):
            render = load_rgb_tensor(render_path).to(device)
            ground_truth = load_rgb_tensor(ground_truth_path).to(device)

            psnr_value = finite_metric(
                psnr(render, ground_truth).mean(), "PSNR"
            )
            ssim_value = finite_metric(ssim(render, ground_truth), "SSIM")
            lpips_value = finite_metric(
                lpips_model(render, ground_truth, normalize=True).mean(),
                "LPIPS",
            )

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)
            per_view[render_path.name] = {
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips_vgg": lpips_value,
            }

    aggregate = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_pair_count": len(pairs),
        "image_size": image_size,
        "device": str(device),
        "metrics": {
            "psnr": summarize(psnr_values),
            "ssim": summarize(ssim_values),
            "lpips_vgg": summarize(lpips_values),
        },
        "implementation": {
            "psnr": "GraphDeCo RGB formula, data range [0, 1]",
            "ssim": (
                "GraphDeCo RGB formula, 11x11 Gaussian window, sigma 1.5, "
                "zero-padded boundary"
            ),
            "lpips": "lpips 0.1.4, VGG trunk, version 0.1",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "lpips": version("lpips"),
        },
    }

    aggregate_path = arguments.output_dir / "metrics.json"
    per_view_path = arguments.output_dir / "per_view.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    per_view_path.write_text(
        json.dumps(per_view, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"PSNR: {aggregate['metrics']['psnr']['mean']:.7f} dB")
    print(f"SSIM: {aggregate['metrics']['ssim']['mean']:.7f}")
    print(f"LPIPS-VGG: {aggregate['metrics']['lpips_vgg']['mean']:.7f}")
    print(f"Aggregate metrics: {aggregate_path}")
    print(f"Per-view metrics: {per_view_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
