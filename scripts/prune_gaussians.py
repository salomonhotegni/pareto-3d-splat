#!/usr/bin/env python3
"""Prune a trained GraphDeCo Gaussian point-cloud PLY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="source point_cloud.ply",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="destination pruned point_cloud.ply",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("random", "opacity-threshold", "top-k", "visibility-top-k"),
    )
    parser.add_argument(
        "--keep-count",
        type=int,
        help="number of Gaussians to keep for random/top-k pruning",
    )
    parser.add_argument(
        "--keep-fraction",
        type=float,
        help="fraction of Gaussians to keep for fixed-budget pruning",
    )
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        help="activated opacity threshold for opacity-threshold pruning",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed used by random pruning",
    )
    parser.add_argument(
        "--importance-mode",
        choices=("opacity_visibility", "visibility", "opacity_count"),
        default="opacity_visibility",
        help="visibility-top-k importance score mode",
    )
    parser.add_argument(
        "--source-model-path",
        type=Path,
        help="optional original GraphDeCo model directory for metadata copy",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        help="optional pruned GraphDeCo model directory for metadata copy",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        from pareto_splat.pruning import PruningError, prune_ply
    except ImportError as error:
        print(
            "error: pruning dependencies are unavailable; run this command "
            "inside the project environment",
            file=sys.stderr,
        )
        print(f"details: {error}", file=sys.stderr)
        return 1

    try:
        result = prune_ply(
            arguments.input,
            arguments.output,
            arguments.strategy,
            keep_count=arguments.keep_count,
            keep_fraction=arguments.keep_fraction,
            opacity_threshold=arguments.opacity_threshold,
            seed=arguments.seed,
            importance_mode=arguments.importance_mode,
            source_model_path=arguments.source_model_path,
            output_model_path=arguments.output_model_path,
        )
    except PruningError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Strategy: {result.strategy}")
    print(f"Input Gaussians: {result.input_count}")
    print(f"Output Gaussians: {result.output_count}")
    print(f"Keep fraction: {result.keep_fraction:.6f}")
    print(f"Pruned fraction: {result.pruned_fraction:.6f}")
    print(f"Pruned PLY: {result.output_path}")
    print(f"Metadata: {result.output_path.parent / 'pruning_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
