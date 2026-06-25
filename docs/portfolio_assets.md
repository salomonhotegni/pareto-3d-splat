# Portfolio Assets

Session 20 adds a reproducible asset builder for portfolio-ready images,
plots, videos, and local asset indexes. The builder collects completed
experiment outputs and writes a compact bundle under:

```text
results/portfolio/
```

Generated portfolio files remain local by default because `results/` is
ignored by Git. The committed project state is the generator, tests, and
documentation.

## Build

Generate the default bundle with:

```bash
make portfolio-assets
```

The underlying script is:

```bash
python scripts/build_portfolio_assets.py --output results/portfolio
```

The default bundle uses Lego test frames `00000` and `00100`. To choose other
frames:

```bash
python scripts/build_portfolio_assets.py \
  --frame 0 \
  --frame 75 \
  --frame 150 \
  --output results/portfolio
```

## Generated Files

The generated bundle contains:

```text
results/portfolio/
  images/
    lego_gt_vs_baseline_00000.png
    lego_gt_vs_baseline_00100.png
    lego_pruning_operating_points_00000.png
    lego_pruning_operating_points_00100.png
  plots/
    pruning_pareto_psnr_vs_fps.png
    pruning_pareto_psnr_vs_size.png
    pruning_pareto_psnr_fps_size_3d.png
    importance_pareto_psnr_vs_fps.png
    pose_psnr_drop_vs_rotation.png
    input_psnr_drop_by_variant.png
  videos/
    lego_ground_truth_vs_3dgs.mp4
  manifest.json
  index.md
```

The comparison images are newly generated from local render frames. The plots
and video are copied from existing experiment outputs so a viewer gets one
small, curated directory instead of having to browse the full `results/` tree.

## Operating-Point Objective

The pruning panels and Pareto plots are interpreted with the same default
quality-efficiency objective used by the summary workflow:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

The negative size term converts a minimization objective into a maximization
coordinate:

```math
\arg\max_x f_3(x)
= \arg\max_x \left[-\mathrm{SizeMiB}(x)\right]
= \arg\min_x \mathrm{SizeMiB}(x).
```

The generated pruning comparison panels show how visible quality changes along
this trade-off: baseline quality is highest, 75% top-k preserves most visual
detail while improving speed and size, and 25% top-k is much faster/smaller but
visibly degraded.

## Source Artifacts

The default builder expects these source artifacts to exist:

- Lego baseline held-out ground-truth and render frames;
- Lego pruning-study top-k render frames;
- pruning, importance-ablation, pose-sensitivity, and input-sensitivity summary
  plots;
- the Lego ground-truth versus 3DGS comparison video.

If one is missing, the builder fails fast with a `PortfolioError` so the asset
bundle is not silently incomplete.

## Pipeline Diagram

The project pipeline diagram is documented in [docs/pipeline.md](pipeline.md).
It is written as Mermaid Markdown so it can render directly on GitHub and stay
version controlled without requiring an external diagram tool.
