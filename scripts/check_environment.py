#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


class CheckResults:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[ OK ] {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")


def load_baseline_settings() -> dict[str, str]:
    settings: dict[str, str] = {}
    path = ROOT_DIR / "scripts" / "baseline.env"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        settings[key] = value
    return settings


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{result.stdout}\n{result.stderr}".strip()


def check_python(results: CheckResults) -> None:
    version = sys.version_info
    if version[:2] == (3, 10):
        results.ok(f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        results.error(
            "Python 3.10 is required; "
            f"found {version.major}.{version.minor}.{version.micro}"
        )


def check_packages(results: CheckResults) -> object | None:
    required = {
        "pareto_splat": "0.1.0",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "numpy": "1.26.4",
        "lpips": "0.1.4",
        "joblib": "1.4.2",
        "yaml": None,
        "plyfile": None,
        "cv2": None,
    }

    torch_module: object | None = None
    for module_name, expected_version in required.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            results.error(f"Python package {module_name} is unavailable: {exc}")
            continue

        if module_name == "torch":
            torch_module = module

        package_names = {
            "pareto_splat": "pareto-3d-splat",
            "yaml": "pyyaml",
        }
        package_name = package_names.get(module_name, module_name)
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            actual_version = getattr(module, "__version__", "unknown")

        if expected_version and actual_version != expected_version:
            results.error(
                f"{module_name} {expected_version} is required; found {actual_version}"
            )
        else:
            results.ok(f"{module_name} {actual_version}")

    return torch_module


def check_cuda(results: CheckResults, torch_module: object | None) -> None:
    nvcc_output = command_output(["nvcc", "--version"])
    nvcc_major: str | None = None
    if nvcc_output is None:
        results.error("nvcc is unavailable")
    else:
        match = re.search(r"release\s+(\d+)\.(\d+)", nvcc_output)
        if match:
            nvcc_major = match.group(1)
            results.ok(f"nvcc CUDA {match.group(1)}.{match.group(2)}")
        else:
            results.warn("nvcc exists, but its CUDA version could not be parsed")

    if torch_module is None:
        return

    torch_cuda_version = getattr(getattr(torch_module, "version"), "cuda")
    if torch_cuda_version is None:
        results.error("PyTorch is a CPU-only build")
        return

    results.ok(f"PyTorch CUDA runtime {torch_cuda_version}")
    if nvcc_major and torch_cuda_version.split(".", maxsplit=1)[0] != nvcc_major:
        results.error(
            "nvcc and PyTorch CUDA runtime have different major versions: "
            f"{nvcc_major} vs {torch_cuda_version}"
        )

    cuda = getattr(torch_module, "cuda")
    if not cuda.is_available():
        results.error("PyTorch cannot access a CUDA GPU")
        return

    device_name = cuda.get_device_name(0)
    capability = cuda.get_device_capability(0)
    results.ok(f"CUDA device 0: {device_name}, compute capability {capability[0]}.{capability[1]}")
    if capability < (7, 0):
        results.error("The baseline requires compute capability 7.0 or newer")


def check_commands(results: CheckResults) -> None:
    if shutil.which("ffmpeg"):
        results.ok("ffmpeg is available")
    else:
        results.error("ffmpeg is unavailable")

    if shutil.which("colmap"):
        results.ok("COLMAP is available")
    else:
        results.warn("COLMAP is unavailable; it is optional until using custom images")


def check_baseline(results: CheckResults) -> None:
    settings = load_baseline_settings()
    baseline_dir = ROOT_DIR / settings["BASELINE_RELATIVE_DIR"]
    if not (baseline_dir / ".git").is_dir():
        results.error("Pinned baseline checkout is unavailable")
        return

    actual_commit = command_output(["git", "-C", str(baseline_dir), "rev-parse", "HEAD"])
    expected_commit = settings["BASELINE_COMMIT"]
    if actual_commit == expected_commit:
        results.ok(f"Baseline commit {actual_commit}")
    else:
        results.error(f"Expected baseline commit {expected_commit}; found {actual_commit}")

    for module_name in ("diff_gaussian_rasterization", "simple_knn._C", "fused_ssim"):
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            results.error(f"Baseline extension {module_name} is unavailable: {exc}")
        else:
            results.ok(f"Baseline extension {module_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Pareto-Splat environment.")
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="also require the pinned baseline checkout and compiled extensions",
    )
    args = parser.parse_args()

    results = CheckResults()
    check_python(results)
    torch_module = check_packages(results)
    check_cuda(results, torch_module)
    check_commands(results)
    if args.require_baseline:
        check_baseline(results)

    print()
    print(f"Environment check completed with {len(results.errors)} error(s) "
          f"and {len(results.warnings)} warning(s).")
    return 1 if results.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
