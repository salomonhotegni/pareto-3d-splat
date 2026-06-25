#!/usr/bin/env python
"""Build curated portfolio assets from completed experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pareto_splat.portfolio import build_portfolio_assets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "results" / "portfolio",
        help="directory where portfolio assets are written",
    )
    parser.add_argument(
        "--frame",
        dest="frames",
        type=int,
        action="append",
        default=None,
        help="test frame index to include; repeat for multiple frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT_DIR / args.output
    manifest = build_portfolio_assets(
        ROOT_DIR,
        output,
        frames=tuple(args.frames) if args.frames else (0, 100),
    )
    asset_count = sum(len(records) for records in manifest["assets"].values())
    print(f"Portfolio assets written to {output}")
    print(f"Assets: {asset_count}")
    print(f"Manifest: {output / 'manifest.json'}")
    print(f"Index: {output / 'index.md'}")


if __name__ == "__main__":
    main()
