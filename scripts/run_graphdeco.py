#!/usr/bin/env python3
"""Run a pinned GraphDeCo entry point with project compatibility fixes."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ALLOWED_ENTRY_POINTS = {"render.py", "train.py"}


def main() -> int:
    arguments = sys.argv[1:]
    baseline_dir = ROOT_DIR / "third_party" / "gaussian-splatting"
    if len(arguments) >= 2 and arguments[0] == "--baseline-root":
        baseline_dir = Path(arguments[1]).resolve()
        arguments = arguments[2:]

    if not arguments or arguments[0] not in ALLOWED_ENTRY_POINTS:
        choices = ", ".join(sorted(ALLOWED_ENTRY_POINTS))
        print(
            f"usage: {Path(sys.argv[0]).name} "
            f"[--baseline-root PATH] <{choices}> [arguments...]",
            file=sys.stderr,
        )
        return 2

    entry_point = arguments[0]
    entry_path = baseline_dir / entry_point
    if not entry_path.is_file():
        print(f"error: baseline entry point not found: {entry_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT_DIR / "src"))
    sys.path.insert(0, str(baseline_dir))

    from pareto_splat.graphdeco_compat import (
        install_nerf_synthetic_compositing_patch,
    )

    install_nerf_synthetic_compositing_patch()
    sys.argv = [str(entry_path), *arguments[1:]]
    runpy.run_path(str(entry_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
