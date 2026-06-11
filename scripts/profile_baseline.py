#!/usr/bin/env python3
"""Profile GraphDeCo rendering speed, model size, and GPU memory."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.graphdeco_compat import (  # noqa: E402
    install_nerf_synthetic_compositing_patch,
)
from pareto_splat.profiling import (  # noqa: E402
    read_ply_vertex_count,
    summarize_latencies,
)


MODEL_TENSOR_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--expected-view-count", type=int, default=200)
    parser.add_argument("--warmup-views", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="white",
    )
    return parser.parse_args()


def bytes_to_megabytes(byte_count: int) -> float:
    return byte_count / 1_000_000


def bytes_to_mebibytes(byte_count: int) -> float:
    return byte_count / (1024**2)


def model_parameter_bytes(gaussians: object) -> int:
    total = 0
    for name in MODEL_TENSOR_NAMES:
        tensor = getattr(gaussians, name)
        total += tensor.numel() * tensor.element_size()
    return total


def main() -> int:
    arguments = parse_arguments()
    baseline_root = arguments.baseline_root.resolve()
    if not baseline_root.is_dir():
        raise FileNotFoundError(
            f"baseline directory not found: {baseline_root}"
        )
    sys.path.insert(0, str(baseline_root))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GraphDeCo rendering profiling")
    if arguments.warmup_views <= 0:
        raise ValueError("--warmup-views must be positive")
    if arguments.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    model_path = arguments.model_path.resolve()
    source_path = arguments.source_path.resolve()
    output_dir = arguments.output_dir.resolve()
    point_cloud_path = (
        model_path
        / "point_cloud"
        / f"iteration_{arguments.iteration}"
        / "point_cloud.ply"
    )
    if not point_cloud_path.is_file():
        raise FileNotFoundError(f"trained point cloud not found: {point_cloud_path}")
    if not source_path.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {source_path}")

    install_nerf_synthetic_compositing_patch()

    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    except ImportError:
        separate_sh = False
    else:
        separate_sh = True

    dataset = SimpleNamespace(
        sh_degree=arguments.sh_degree,
        source_path=str(source_path),
        model_path=str(model_path),
        images="images",
        depths="",
        resolution=arguments.resolution,
        white_background=arguments.background == "white",
        train_test_exp=False,
        data_device=arguments.data_device,
        eval=True,
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pre_load_allocated_bytes = torch.cuda.memory_allocated()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=arguments.iteration,
        shuffle=False,
    )
    test_views = scene.getTestCameras()
    if len(test_views) != arguments.expected_view_count:
        raise ValueError(
            f"expected {arguments.expected_view_count} test views, "
            f"found {len(test_views)}"
        )
    if arguments.warmup_views > len(test_views):
        raise ValueError(
            f"warm-up count {arguments.warmup_views} exceeds "
            f"test-view count {len(test_views)}"
        )

    background = torch.tensor(
        [1.0, 1.0, 1.0]
        if arguments.background == "white"
        else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    torch.cuda.synchronize()
    loaded_scene_allocated_bytes = torch.cuda.memory_allocated()

    with torch.no_grad():
        for view in test_views[: arguments.warmup_views]:
            rendering = render(
                view,
                gaussians,
                pipeline,
                background,
                separate_sh=separate_sh,
            )
            del rendering
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    measurement_baseline_allocated_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    timing_records: list[dict[str, int | float | str]] = []
    timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with torch.no_grad():
        for repetition in range(arguments.repetitions):
            for view_index, view in enumerate(
                tqdm(
                    test_views,
                    desc=f"Profiling repetition {repetition + 1}",
                    unit="view",
                )
            ):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                rendering = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    separate_sh=separate_sh,
                )
                end.record()
                del rendering
                timing_events.append((start, end))
                timing_records.append(
                    {
                        "repetition": repetition,
                        "view_index": view_index,
                        "image_name": view.image_name,
                    }
                )

    torch.cuda.synchronize()
    latencies_ms = [
        start.elapsed_time(end) for start, end in timing_events
    ]
    for record, latency_ms in zip(timing_records, latencies_ms):
        record["latency_ms"] = latency_ms

    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    incremental_peak_allocated_bytes = (
        peak_allocated_bytes - measurement_baseline_allocated_bytes
    )
    gaussian_count = read_ply_vertex_count(point_cloud_path)
    loaded_gaussian_count = int(gaussians.get_xyz.shape[0])
    if gaussian_count != loaded_gaussian_count:
        raise ValueError(
            f"PLY contains {gaussian_count} Gaussians, "
            f"but the loaded model contains {loaded_gaussian_count}"
        )

    serialized_model_bytes = point_cloud_path.stat().st_size
    parameter_bytes = model_parameter_bytes(gaussians)
    device_properties = torch.cuda.get_device_properties(0)
    latency_summary = summarize_latencies(latencies_ms)

    profile = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": arguments.iteration,
        "test_view_count": len(test_views),
        "warmup_view_count": arguments.warmup_views,
        "repetitions": arguments.repetitions,
        "measured_frame_count": len(latencies_ms),
        "image_size": [
            int(test_views[0].image_width),
            int(test_views[0].image_height),
        ],
        "rendering": latency_summary,
        "model": {
            "gaussian_count": gaussian_count,
            "serialized_path": str(point_cloud_path),
            "serialized_bytes": serialized_model_bytes,
            "serialized_mb": bytes_to_megabytes(serialized_model_bytes),
            "serialized_mib": bytes_to_mebibytes(serialized_model_bytes),
            "parameter_bytes": parameter_bytes,
            "parameter_mb": bytes_to_megabytes(parameter_bytes),
            "parameter_mib": bytes_to_mebibytes(parameter_bytes),
        },
        "gpu_memory": {
            "pre_load_allocated_bytes": pre_load_allocated_bytes,
            "loaded_scene_allocated_bytes": loaded_scene_allocated_bytes,
            "measurement_baseline_allocated_bytes": (
                measurement_baseline_allocated_bytes
            ),
            "peak_allocated_bytes": peak_allocated_bytes,
            "incremental_render_peak_allocated_bytes": (
                incremental_peak_allocated_bytes
            ),
            "peak_reserved_bytes": peak_reserved_bytes,
            "peak_allocated_mib": bytes_to_mebibytes(peak_allocated_bytes),
            "incremental_render_peak_allocated_mib": bytes_to_mebibytes(
                incremental_peak_allocated_bytes
            ),
            "peak_reserved_mib": bytes_to_mebibytes(peak_reserved_bytes),
        },
        "methodology": {
            "timing": (
                "CUDA events around gaussian_renderer.render only; excludes "
                "scene loading, image encoding, and disk writes"
            ),
            "memory": (
                "PyTorch CUDA allocator statistics; camera ground-truth "
                "images are stored on CPU"
            ),
            "latency_percentile": "linear interpolation over all measured frames",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": device_properties.name,
            "gpu_total_memory_bytes": device_properties.total_memory,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.json"
    latency_path = output_dir / "latencies.json"
    profile_path.write_text(
        json.dumps(profile, indent=2) + "\n",
        encoding="utf-8",
    )
    latency_path.write_text(
        json.dumps(timing_records, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Gaussian count: {gaussian_count}")
    print(
        "Serialized model: "
        f"{profile['model']['serialized_mib']:.2f} MiB"
    )
    print(f"Mean latency: {latency_summary['mean_ms']:.3f} ms")
    print(f"P95 latency: {latency_summary['p95_ms']:.3f} ms")
    print(f"FPS: {latency_summary['frames_per_second']:.2f}")
    print(
        "Peak allocated GPU memory: "
        f"{profile['gpu_memory']['peak_allocated_mib']:.2f} MiB"
    )
    print(
        "Incremental render peak: "
        f"{profile['gpu_memory']['incremental_render_peak_allocated_mib']:.2f} MiB"
    )
    print(f"Profile: {profile_path}")
    print(f"Per-frame latencies: {latency_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
