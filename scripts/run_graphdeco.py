#!/usr/bin/env python3
"""Run a pinned GraphDeCo entry point with project compatibility fixes."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT_DIR / "third_party" / "gaussian-splatting"
ALLOWED_ENTRY_POINTS = {"render.py", "train.py"}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ALLOWED_ENTRY_POINTS:
        choices = ", ".join(sorted(ALLOWED_ENTRY_POINTS))
        print(
            f"usage: {Path(sys.argv[0]).name} <{choices}> [arguments...]",
            file=sys.stderr,
        )
        return 2

    entry_point = sys.argv[1]
    entry_path = BASELINE_DIR / entry_point
    if not entry_path.is_file():
        print(f"error: baseline entry point not found: {entry_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT_DIR / "src"))
    sys.path.insert(0, str(BASELINE_DIR))

    from pareto_splat.graphdeco_compat import (
        install_nerf_synthetic_compositing_patch,
    )

    install_nerf_synthetic_compositing_patch()
    sys.argv = [str(entry_path), *sys.argv[2:]]
    runpy.run_path(str(entry_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
