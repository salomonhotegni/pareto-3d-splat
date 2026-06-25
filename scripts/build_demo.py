#!/usr/bin/env python
"""Build the static Pareto-Splat demo page."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.demo import build_demo_site, discover_summary_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "results" / "demo" / "index.html",
        help="HTML file to write",
    )
    parser.add_argument(
        "--summary",
        dest="summaries",
        type=Path,
        action="append",
        default=None,
        help=(
            "summary JSON to include; repeat to provide an explicit list. "
            "By default, all results/**/summary/summary.json files are used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = (
        tuple(
            path if path.is_absolute() else ROOT_DIR / path
            for path in args.summaries
        )
        if args.summaries
        else None
    )
    if summaries is None:
        summaries = discover_summary_paths(ROOT_DIR)

    output_path = build_demo_site(
        ROOT_DIR,
        args.output,
        summary_paths=summaries,
    )
    print(f"Demo written to {output_path}")
    print(f"Included summaries: {len(summaries)}")


if __name__ == "__main__":
    main()
