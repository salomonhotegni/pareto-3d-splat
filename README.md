# Pareto-Splat

[![CI](https://github.com/salomonhotegni/pareto-3d-splat/actions/workflows/main.yml/badge.svg)](https://github.com/salomonhotegni/pareto-3d-splat/actions/workflows/main.yml)

Pareto-Splat is a reproducible research pipeline for studying quality, speed,
and model-size trade-offs in
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
It wraps a pinned GraphDeCo baseline with held-out evaluation, CUDA profiling,
post-training Gaussian pruning, Pareto-front analysis, controlled robustness
studies, and presentation tooling.

The 24-session project roadmap is complete. For a method-first introduction,
read the [detailed project article](docs/article.md).

## Highlights

- Configuration-driven training, rendering, evaluation, and profiling.
- PSNR, SSIM, and LPIPS-VGG over all 200 held-out test views.
- Random, opacity-threshold, opacity top-k, and visibility-aware pruning.
- Matched-budget comparisons and 2D/3D non-dominated fronts.
- Camera-pose, image degradation, brightness, and sparse-view studies.
- Reproducible run metadata, dataset checksums, resumable checkpoints, and
  72 automated tests.
- Static result explorer and generated portfolio asset bundle.

## Key Result

On NeRF Synthetic Lego with seed 0, retaining the 75% most opaque Gaussians
produced the strongest demonstrated compression point:

| Metric | Baseline | 75% opacity top-k | Change |
| --- | ---: | ---: | ---: |
| Gaussian count | 299,799 | 224,849 | -25.0% |
| Serialized size | 70.91 MiB | 53.18 MiB | -25.0% |
| Renderer throughput | 282.98 FPS | 456.45 FPS | +61.3% |
| PSNR | 35.917 dB | 34.496 dB | -1.421 dB |
| SSIM | 0.983729 | 0.980445 | -0.003284 |
| LPIPS-VGG | 0.019002 | 0.021636 | +0.002634 |

Quality metrics use 200 held-out 800 x 800 views. Throughput is
renderer-only, measured with CUDA events on an NVIDIA A100-SXM4-40GB; it
excludes image encoding, storage, viewer, and network overhead.

The clean workflow was also validated on NeRF Synthetic Drums. Pruning,
ablation, and robustness conclusions remain primarily Lego-only and use one
seed, so they are point estimates rather than statistically significant or
scene-general results.

## Method

Each model variant \(x\) is compared with the default objective vector:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

A variant is Pareto-optimal within the measured set when no other variant is
at least as good in every objective and strictly better in one. The project
uses this analysis to expose deployment choices rather than collapse quality,
speed, and size into one arbitrary weighted score.

Post-training pruning selects an unchanged subset of the trained Gaussian PLY.
The strongest tested rule ranks each Gaussian by activated opacity:

```math
\alpha_i = \sigma(o_i),
\qquad
G' = \text{TopK}(G, \alpha, k).
```

See the [technical report](docs/technical_report.md) or
[detailed article](docs/article.md) for the complete method and results.

## Documentation

| Goal | Document |
| --- | --- |
| Understand the complete method | [Detailed article](docs/article.md) |
| Read the formal project report | [Technical report](docs/technical_report.md) |
| Reproduce the experiments | [Reproducibility protocol](docs/reproducibility.md) |
| Review claim boundaries | [Claims and evidence](docs/claims_and_evidence.md) |
| Inspect baseline results | [Lego](docs/baseline_results.md) and [Drums](docs/drums_baseline_results.md) |
| Understand pruning and Pareto sorting | [Pruning](docs/pruning.md) and [Pareto analysis](docs/pareto.md) |
| Review ablations and failures | [Importance ablation](docs/importance_ablation.md) and [limitations](docs/limitations.md) |
| Review robustness studies | [Pose sensitivity](docs/pose_sensitivity.md) and [input sensitivity](docs/input_sensitivity.md) |
| Build the demo and visual assets | [Demo](docs/demo.md) and [portfolio assets](docs/portfolio_assets.md) |
| Use presentation materials | [Portfolio summary](docs/portfolio_summary.md), [blog post](docs/blog_post.md), and [interview guide](docs/interview_guide.md) |

## Target System

- Linux
- NVIDIA A100-SXM4 40 GB
- NVIDIA driver 580.126.16
- CUDA 12.x compiler toolkit
- Conda

The pinned Python environment uses Python 3.10, PyTorch 2.5.1, and the PyTorch
CUDA 12.1 runtime. Newer NVIDIA drivers are backward compatible with that
runtime. A CUDA 12.x `nvcc` compiler is required to build the baseline's custom
CUDA extensions.

## Quick Start

Create the Conda environment, install the pinned GraphDeCo baseline, validate
the GPU stack, and run the tests:

```bash
make env
make install
make check
make test
```

Download and validate Lego, then run the clean baseline:

```bash
make dataset-lego
make check-data-lego
make train-baseline
make render-baseline
make evaluate-baseline
make profile-baseline
```

Run `make help` for the command index. Follow the
[reproducibility protocol](docs/reproducibility.md) for Drums, pruning,
importance ablations, robustness studies, and artifact generation.

Environment troubleshooting is documented in [setup.md](docs/setup.md).
Dataset provenance is documented in [dataset_notes.md](docs/dataset_notes.md).

## Repository Layout

```text
configs/       Reproducible experiment configurations
data/          Local datasets; generated/downloaded data is ignored by Git
docs/          Project plan, roadmap, setup notes, and reports
results/       Local experiment artifacts; generated results are ignored by Git
scripts/       Environment, data, training, and evaluation entry points
src/           Pareto-Splat evaluation, compression, and Pareto-analysis code
third_party/   Downloaded pinned baseline; generated and ignored by Git
```

## Baseline

The baseline is the official GraphDeco-INRIA implementation pinned to commit
`54c035f7834b564019656c3e3fcc3646292f727d`. The pin prevents upstream changes
from silently changing experiment results.

## License 

Pareto-Splat is licensed under the MIT License. The downloaded 3D Gaussian
Splatting baseline retains its own license and attribution.
