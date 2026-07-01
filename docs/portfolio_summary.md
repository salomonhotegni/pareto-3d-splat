# Pareto-Splat Portfolio Summary

## Project

**Pareto-Splat: Quality-Efficiency Trade-offs in 3D Gaussian Splatting**

Pareto-Splat is an experimental system for measuring and improving the
deployment efficiency of 3D Gaussian Splatting (3DGS). It turns the official
GraphDeCo implementation into a reproducible workflow for training, rendering,
quality evaluation, GPU profiling, post-training pruning, robustness testing,
and Pareto-front analysis.

## Problem

A high-quality 3DGS scene may contain hundreds of thousands of Gaussians.
Keeping every Gaussian can preserve detail, but increases model size and render
cost. There is no single best compressed model because deployment decisions
balance competing objectives:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

The project asks: **How much rendering efficiency can be gained before
novel-view quality degrades unacceptably?**

## What I Built

- A configuration-driven wrapper around a pinned GraphDeCo 3DGS baseline.
- Dataset validation and corrected RGBA compositing for NeRF Synthetic scenes.
- Held-out PSNR, SSIM, and LPIPS-VGG evaluation over 200 test views.
- CUDA-event profiling for renderer latency, FPS, GPU memory, Gaussian count,
  and serialized model size.
- Random, opacity-threshold, opacity top-k, and visibility-aware post-training
  pruning.
- Pareto dominance, non-dominated sorting, and 2D/3D front visualization.
- Camera-pose, image degradation, brightness, and sparse-view sensitivity
  studies.
- A static result explorer, portfolio asset builder, and tested clean-room
  reproduction protocol.

## Headline Result

On NeRF Synthetic Lego, seed 0, the strongest demonstrated compression point
retains 75% of the Gaussians using opacity top-k pruning:

| Metric | Baseline | 75% opacity top-k | Change |
| --- | ---: | ---: | ---: |
| Gaussians | 299,799 | 224,849 | -25.0% |
| Model size | 70.91 MiB | 53.18 MiB | -25.0% |
| Renderer throughput | 282.98 FPS | 456.45 FPS | +61.3% |
| PSNR | 35.917 dB | 34.496 dB | -1.421 dB |
| SSIM | 0.983729 | 0.980445 | -0.003284 |

The FPS result is renderer-only throughput measured on an NVIDIA
A100-SXM4-40GB. It excludes encoding, disk, viewer, and network overhead.

Opacity top-k also outperformed random pruning at matched 25%, 50%, and 75%
retention budgets in both quality and measured renderer throughput.

## What the Failures Revealed

The project did not stop at the strongest result:

- Raw frustum visibility was a poor standalone importance score. At 75%
  retention it achieved only 15.259 dB, versus 34.496 dB for opacity top-k.
- A mean camera rotation perturbation of 0.3994 degrees reduced PSNR by
  16.8185 dB.
- Training-image brightness shifts reduced clean-test PSNR by 11-13 dB.
- Retaining only 25% of Gaussians raised throughput to 895.98 FPS but cost
  13.540 dB PSNR.

These failures define the method's operating envelope: opacity is essential
for the tested pruning score, calibration and appearance consistency matter,
and aggressive compression is not free.

## Engineering Approach

The repository emphasizes reproducibility as part of the result:

- pinned baseline revision and package versions;
- YAML experiment contracts and deterministic output paths;
- per-attempt command, environment, checksum, log, and completion metadata;
- machine-readable aggregate and per-view results;
- 72 automated tests, including local documentation-link validation;
- generated data and heavy result artifacts kept outside source control.

## Tools

Python, PyTorch, CUDA, GraphDeCo 3DGS, NumPy, SciPy, LPIPS, Matplotlib, Pandas,
OpenCV, YAML, pytest, Make, Bash, FFmpeg, and Conda.

## Explore

- [Technical report](technical_report.md)
- [Claims and evidence](claims_and_evidence.md)
- [Reproducibility protocol](reproducibility.md)
- [Pareto analysis](pareto.md)
- [Failure cases and limitations](limitations.md)
- [Portfolio asset builder](portfolio_assets.md)
- [Static demo](demo.md)

## Scope

The pruning, ablation, and robustness conclusions are primarily from NeRF
Synthetic Lego with one seed. Drums validates the clean baseline workflow on a
second scene. Results should not be presented as statistically significant,
hardware-independent, or representative of real-world scenes.
