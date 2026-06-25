# Pareto-Splat Technical Report Draft

## Abstract

Pareto-Splat studies quality-speed-memory trade-offs in 3D Gaussian Splatting
(3DGS). Starting from the official GraphDeCo 3DGS implementation, the project
builds a reproducible experiment workflow for NeRF Synthetic scenes, evaluates
visual quality and renderer efficiency, applies post-training Gaussian pruning,
and analyzes the resulting operating points with Pareto dominance.

The main experimental result is that opacity-aware post-training pruning
creates useful deployment trade-offs on NeRF Synthetic Lego. The uncompressed
Lego baseline reaches 35.9166 dB PSNR at 282.98 FPS with 299,799 Gaussians and
a 70.91 MiB serialized model. A 75% opacity top-k model keeps 224,849
Gaussians, preserves 34.496 dB PSNR, improves renderer throughput to
456.45 FPS, and reduces serialized size to 53.18 MiB. More aggressive pruning
continues to improve speed and size, but the quality loss becomes visible and
large. The 25% top-k model renders at 895.98 FPS and is 17.73 MiB, but drops to
22.377 dB PSNR.

Robustness studies show that the current pipeline depends strongly on accurate
camera poses and consistent train-test appearance. Sub-degree camera rotation
perturbations reduce Lego PSNR by more than 16 dB. Brightness shifts in the
training images reduce clean-test PSNR by 11-13 dB. Importance-score ablations
also show that raw frustum visibility is not a reliable standalone pruning
signal; opacity remains the essential stabilizing term.

## 1. Motivation

3D Gaussian Splatting can render high-quality novel views at real-time rates,
but practical deployment rarely has one universal best operating point. A
larger scene representation may preserve more visual detail, while a smaller
one may render faster and fit tighter memory budgets. This project treats
3DGS deployment as a multi-objective problem rather than a single-metric
optimization.

The central question is:

```math
\text{How can we reduce 3DGS model cost while preserving novel-view quality?}
```

In this project, "cost" is measured through renderer throughput, latency,
Gaussian count, serialized model size, and GPU memory. "Quality" is measured
with PSNR, SSIM, LPIPS-VGG, and qualitative rendered comparisons.

## 2. Problem Formulation

Given multi-view images and calibrated camera poses for a static scene, the
baseline training process learns a set of 3D Gaussians:

```math
G = \{g_i\}_{i=1}^{N}.
```

Each Gaussian has parameters such as position, scale, rotation, opacity, and
view-dependent color coefficients. A renderer maps the Gaussian set and a
camera pose \(T_c\) to an image:

```math
\hat{I}_c = R(G, T_c).
```

The project evaluates each trained or pruned model variant \(x\) with a
quality-efficiency objective vector:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

PSNR and FPS are maximized. Serialized model size is minimized by negating it.
For two variants \(a\) and \(b\), \(a\) Pareto-dominates \(b\) if:

```math
\forall j,\; f_j(a) \ge f_j(b)
\quad \text{and} \quad
\exists j,\; f_j(a) > f_j(b).
```

This gives a set of non-dominated operating points instead of forcing a single
weighted score. The interpretation is practical: each rank-0 point represents
a model that cannot be improved in one tracked objective without losing in at
least one other tracked objective.

## 3. Reproducible Workflow

The repository wraps the pinned GraphDeCo baseline with project-level scripts,
configuration files, and tests. Each experiment stage can be run through Make
targets or direct Python entry points:

- dataset download and validation;
- baseline training;
- held-out rendering;
- PSNR, SSIM, and LPIPS-VGG evaluation;
- renderer-only profiling;
- post-training pruning;
- study summarization and Pareto-front plotting;
- static demo and portfolio asset generation.

The main configuration contract is documented in
[experiment_workflow.md](experiment_workflow.md). Generated artifacts are kept
under `results/`, while source code, configs, tests, and reports are version
controlled.

NeRF Synthetic images contain alpha channels. The compatibility launcher
composites RGBA images onto the configured white background before they are
passed into the GraphDeCo camera pipeline. This matters because evaluating
against uncomposited RGB channels would produce invalid ground truth for these
scenes.

## 4. Baseline Results

The first complete baseline uses NeRF Synthetic Lego with 100 training views,
100 validation views, and 200 held-out test views at 800 x 800 resolution. It
trains for 30,000 iterations with seed 0 on an NVIDIA A100-SXM4-40GB.

| Metric | Lego baseline |
| --- | ---: |
| PSNR | 35.9166 dB |
| SSIM | 0.983729 |
| LPIPS-VGG | 0.019002 |
| Gaussian count | 299,799 |
| Serialized model size | 70.91 MiB |
| Mean latency | 3.534 ms |
| Renderer throughput | 282.98 FPS |
| Peak allocated GPU memory | 282.92 MiB |

The second clean baseline uses NeRF Synthetic Drums. It validates the same
configuration-driven workflow on another scene:

| Metric | Lego | Drums |
| --- | ---: | ---: |
| PSNR | 35.9166 dB | 26.1724 dB |
| SSIM | 0.983729 | 0.955651 |
| LPIPS-VGG | 0.019002 | 0.043743 |
| Gaussian count | 299,799 | 318,647 |
| Serialized model size | 70.91 MiB | 75.37 MiB |
| Mean latency | 3.534 ms | 3.284 ms |
| Renderer throughput | 282.98 FPS | 304.49 FPS |

The quality gap indicates that Drums is a harder scene for this setup.
Cross-scene FPS is not a controlled hardware comparison because the two runs
used different A100 memory variants.

## 5. Pruning Methods

The pruning module applies post-training pruning to an existing `point_cloud.ply`
without retraining. The study includes:

- random fixed-budget pruning;
- opacity-threshold pruning;
- activated-opacity top-k pruning;
- visibility-aware top-k pruning.

For a target retention fraction \(r\), fixed-budget variants keep:

```math
k = \operatorname{round}(rN)
```

Gaussians from the source model. Opacity top-k ranks Gaussians by activated
opacity:

```math
\alpha_i = \sigma(o_i),
\qquad
s_i = \alpha_i.
```

Visibility-aware variants add a camera-frustum proxy. For Gaussian \(i\) and
camera \(c\), define:

```math
m_{ic} =
\mathbf{1}
\left[
z_{ic} > 0,\;
0 \le u_{ic} < W_c,\;
0 \le v_{ic} < H_c
\right].
```

The depth-weighted visibility proxy is:

```math
V_i = \sum_c \frac{m_{ic}}{z_{ic}^2 + \epsilon}.
```

The default visibility-aware score tested before the ablation is:

```math
s_i = \alpha_i \log(1 + V_i).
```

## 6. Quality-Efficiency Results

The Lego pruning study shows a clear speed-size-quality trade-off:

| Variant | Keep | Gaussians | PSNR | SSIM | LPIPS-VGG | FPS | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100% | 299,799 | 35.917 | 0.983729 | 0.019002 | 282.98 | 70.91 MiB |
| top_k_keep_075 | 75% | 224,849 | 34.496 | 0.980445 | 0.021636 | 456.45 | 53.18 MiB |
| top_k_keep_050 | 50% | 149,900 | 29.169 | 0.957937 | 0.041038 | 635.44 | 35.45 MiB |
| top_k_keep_025 | 25% | 74,950 | 22.377 | 0.877730 | 0.102224 | 895.98 | 17.73 MiB |
| random_keep_050_seed_0 | 50% | 149,900 | 27.143 | 0.927109 | 0.071232 | 459.79 | 35.45 MiB |

Opacity top-k dominates random pruning at matched 25%, 50%, and 75% retention:
it gives both higher quality and higher measured renderer throughput.

The 75% top-k point is the most useful demonstrated compression point in the
current Lego study. It loses only:

```math
35.917 - 34.496 = 1.421 \text{ dB}
```

while improving throughput from 282.98 FPS to 456.45 FPS and reducing model
size from 70.91 MiB to 53.18 MiB. The 25% point is useful for exposing the
Pareto frontier but is not a high-fidelity operating point:

```math
35.917 - 22.377 = 13.540 \text{ dB}.
```

## 7. Pareto Analysis

The summary workflow computes Pareto ranks over PSNR, FPS, and serialized
model size. In the initial pruning study, the baseline plus top-k and
opacity-threshold variants lie on the first front, while random variants are
dominated.

This result depends on the objective set. A variant may be rank 0 under
PSNR/FPS/size and no longer rank 0 if LPIPS, GPU memory, training time, or
worst-view quality are added. The report therefore treats Pareto rank as a
decision aid, not an absolute quality label.

## 8. Importance-Score Ablations

The importance ablation keeps the pruning budget fixed and changes only the
ranking score. It compares:

| Mode | Score |
| --- | --- |
| opacity top-k control | \(s_i = \alpha_i\) |
| `opacity_visibility` | \(s_i = \alpha_i \log(1 + V_i)\) |
| `visibility` | \(s_i = \log(1 + V_i)\) |
| `opacity_count` | \(s_i = \alpha_i \log(1 + C_i)\) |

The key finding is that raw visibility alone fails:

| Keep | Visibility-only PSNR | Opacity top-k PSNR |
| ---: | ---: | ---: |
| 25% | 10.363 | 22.377 |
| 50% | 11.824 | 29.169 |
| 75% | 15.259 | 34.496 |

The visibility proxy only checks whether projected Gaussian centers fall inside
camera frusta and applies a simple depth weighting. It does not measure
screen-space footprint, occlusion, compositing contribution, color residual, or
gradient importance. The practical conclusion is that opacity should remain in
the pruning score. Visibility may be useful as a secondary reweighting signal,
but not as the only selection criterion.

## 9. Robustness Studies

### Camera-Pose Sensitivity

The pose study keeps the trained Gaussian model fixed and perturbs held-out
test camera poses:

```math
T_i =
\begin{bmatrix}
R_i & t_i \\
0 & 1
\end{bmatrix},
\qquad
T'_i =
\begin{bmatrix}
\Delta R_i R_i & t_i + \epsilon_i \\
0 & 1
\end{bmatrix}.
```

Lego quality is highly sensitive to small pose errors:

| Variant | Mean rotation | Mean translation norm | PSNR | PSNR drop |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0000 deg | 0.000000 | 35.9166 | 0.0000 |
| rot_0p25deg | 0.3994 deg | 0.000000 | 19.0981 | 16.8185 |
| trans_0p005 | 0.0000 deg | 0.007987 | 24.5416 | 11.3750 |
| rot_0p50deg_trans_0p010 | 0.7876 deg | 0.015463 | 16.5437 | 19.3728 |

The current pipeline should therefore not be described as robust to camera
miscalibration.

### Training-Input Sensitivity

The input-sensitivity study retrains models after degrading training images or
reducing the number of training views, then evaluates on the clean held-out
test split:

```math
D'_{\mathrm{train}} = \{(I'_i, T_i)\}_{i \in S}.
```

Observed Lego drops:

| Variant | Training input | Train views | PSNR | PSNR drop |
| --- | --- | ---: | ---: | ---: |
| baseline | clean | 100 | 35.92 | 0.00 |
| noise_std_0p02 | Gaussian noise, sigma 0.02 | 100 | 35.33 | 0.59 |
| blur_radius_1p0 | Gaussian blur, radius 1.0 | 100 | 32.30 | 3.62 |
| brightness_0p75 | brightness scale 0.75 | 100 | 22.85 | 13.07 |
| brightness_1p25 | brightness scale 1.25 | 100 | 24.83 | 11.08 |
| train_views_050 | clean subset | 50 | 34.24 | 1.67 |
| train_views_025 | clean subset | 25 | 29.90 | 6.02 |

Mild zero-mean noise is mostly absorbed. Blur and sparse views matter more.
Global brightness mismatch is the most damaging tested input shift because the
model learns biased appearance while the evaluation target remains clean.

## 10. Demo and Portfolio Outputs

The project includes two presentation layers:

- a static browser demo generated by `make demo`, written to
  `results/demo/index.html`;
- curated portfolio assets generated by `make portfolio-assets`, written to
  `results/portfolio/`.

The demo loads available `summary.json` files and lets the viewer select a
study and operating point. It is intentionally dependency-light: a generated
HTML file with embedded JSON, CSS, and JavaScript.

The portfolio builder creates:

- ground-truth versus baseline comparison images;
- pruning operating-point panels;
- copied Pareto and robustness plots;
- the Lego comparison video;
- a `manifest.json` and local Markdown index.

The project pipeline diagram is documented in [pipeline.md](pipeline.md).

## 11. Limitations

The current evidence base is strongest for NeRF Synthetic Lego. Drums is used
as a second clean baseline, but the pruning, robustness, and ablation studies
are mostly Lego-only. The scenes are synthetic, static, object-centric, and
cleanly segmented.

The strongest measured limitations are:

- camera-pose perturbations can dominate quality;
- train-test brightness mismatch causes large quality drops;
- raw visibility is a weak standalone pruning signal;
- aggressive pruning creates visible artifacts even when FPS improves;
- Pareto ranks depend on the selected objectives;
- FPS is renderer-only and excludes PNG encoding, disk writes, viewer overhead,
  and network transport.

The report should not claim scene-general robustness or real-world deployment
readiness yet.

## 12. Future Work

The most useful next steps are:

- repeat pruning, ablation, and robustness studies on additional scenes;
- add real-world datasets with imperfect calibration and exposure variation;
- evaluate contribution-aware importance scores using rasterized opacity,
  screen-space footprint, residual error, or gradient statistics;
- test pruning followed by fine-tuning;
- add multi-seed summaries and confidence intervals;
- include qualitative failure panels for pose error, brightness shift,
  sparse views, and aggressive pruning;
- compare Pareto fronts under expanded objective sets that include LPIPS,
  memory, latency, and worst-view quality.

## 13. Summary of Claims

The current defensible claims are:

1. The project implements a reproducible 3DGS experiment workflow for NeRF
   Synthetic scenes, including training, rendering, evaluation, profiling,
   pruning, summarization, demo generation, and portfolio asset generation.
2. On Lego, opacity-aware pruning gives better matched-budget quality and FPS
   than random pruning.
3. On Lego, 75% opacity top-k pruning is the strongest demonstrated compression
   point so far, preserving most baseline quality while improving speed and
   size.
4. Raw frustum visibility alone is not a reliable Gaussian importance score in
   this setup.
5. The current pipeline is highly sensitive to pose perturbations and
   train-test appearance shifts, so robustness claims must remain narrow.

## References Within This Repository

- [Project plan](project_plan.md)
- [Lego baseline results](baseline_results.md)
- [Drums baseline results](drums_baseline_results.md)
- [Pruning study](pruning_study.md)
- [Pareto dominance](pareto.md)
- [Visibility importance](visibility_importance.md)
- [Importance ablations](importance_ablation.md)
- [Pose sensitivity](pose_sensitivity.md)
- [Input sensitivity](input_sensitivity.md)
- [Limitations](limitations.md)
- [Demo](demo.md)
- [Portfolio assets](portfolio_assets.md)
- [Pipeline diagram](pipeline.md)
