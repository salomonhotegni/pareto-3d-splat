# Failure Cases and Practical Limitations

Session 18 consolidates the main ways the current Pareto-Splat workflow can
fail or produce misleading conclusions. The goal is not to weaken the project
claims, but to state their operating envelope precisely.

The current evidence base is strongest for the NeRF Synthetic Lego scene, with
an additional clean Drums baseline. Most robustness and pruning conclusions
come from Lego, so they should be treated as scene-specific until repeated on
more datasets.

All experiments use seed 0. If a reported effect is written as

```math
\Delta m = m_{\mathrm{variant}} - m_{\mathrm{reference}},
```

then \(\Delta m\) is an observed point estimate. One run per condition cannot
measure run-to-run variance, confidence intervals, or statistical
significance. Small metric differences should therefore not be interpreted as
stable expected improvements without multi-seed replication.

## 1. Camera-Pose Accuracy Is a Hard Requirement

The pose-sensitivity study keeps the trained Gaussian model fixed and perturbs
the held-out test camera transform:

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

Even small perturbations severely reduce clean-test quality:

| Variant | Mean rotation | Mean translation norm | PSNR | PSNR drop |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0000 deg | 0.000000 | 35.9166 | 0.0000 |
| `rot_0p25deg` | 0.3994 deg | 0.000000 | 19.0981 | 16.8185 |
| `trans_0p005` | 0.0000 deg | 0.007987 | 24.5416 | 11.3750 |
| `rot_0p50deg_trans_0p010` | 0.7876 deg | 0.015463 | 16.5437 | 19.3728 |

This means the present pipeline should not be described as robust to
miscalibration. Its high baseline quality assumes camera poses are accurate.
Sub-degree orientation errors are already large enough to dominate the quality
budget.

## 2. Training-Input Distribution Shift Can Dominate Quality

The training-input study replaces the clean training set

```math
D_{\mathrm{train}} = \{(I_i, T_i)\}_{i=1}^{N}
```

with degraded or reduced variants:

```math
D'_{\mathrm{train}} = \{(I'_i, T_i)\}_{i \in S}.
```

The model is still evaluated on clean held-out images. If the training images
have a different appearance distribution from the clean test images, the
learned radiance field becomes biased:

```math
p_{\mathrm{train}}(I') \ne p_{\mathrm{test}}(I).
```

Observed Lego results:

| Variant | Training input | Train views | PSNR | PSNR drop |
| --- | --- | ---: | ---: | ---: |
| baseline | clean | 100 | 35.92 | 0.00 |
| `noise_std_0p02` | Gaussian noise, sigma 0.02 | 100 | 35.33 | 0.59 |
| `blur_radius_1p0` | Gaussian blur, radius 1.0 | 100 | 32.30 | 3.62 |
| `brightness_0p75` | brightness scale 0.75 | 100 | 22.85 | 13.07 |
| `brightness_1p25` | brightness scale 1.25 | 100 | 24.83 | 11.08 |
| `train_views_050` | clean subset | 50 | 34.24 | 1.67 |
| `train_views_025` | clean subset | 25 | 29.90 | 6.02 |

Mild zero-mean noise is absorbed reasonably well. Global brightness changes,
blur, and sparse-view coverage are much more damaging. The practical
limitation is that the current workflow does not compensate for exposure
changes, lost high-frequency supervision, or sparse view geometry.

## 3. Aggressive Pruning Has a Clear Quality Floor

For a retention fraction \(r\), fixed-budget pruning keeps

```math
k = \operatorname{round}(rN)
```

Gaussians from the baseline model. On Lego, opacity top-k pruning gives a good
speed-quality trade-off at 75% retention, but the quality cost becomes large at
lower budgets:

| Variant | Keep | Gaussians | PSNR | FPS | Size |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 100% | 299,799 | 35.917 | 282.98 | 70.91 MiB |
| `top_k_keep_075` | 75% | 224,849 | 34.496 | 456.45 | 53.18 MiB |
| `top_k_keep_050` | 50% | 149,900 | 29.169 | 635.44 | 35.45 MiB |
| `top_k_keep_025` | 25% | 74,950 | 22.377 | 895.98 | 17.73 MiB |

The 25% model is much faster and smaller, but loses 13.540 dB relative to the
baseline:

```math
\Delta \mathrm{PSNR}
= 35.917 - 22.377
= 13.540 \text{ dB}.
```

So pruning should be presented as a controllable trade-off, not a free
compression step. Random pruning is weaker than opacity-aware pruning at
matched budgets and should mainly remain a sanity-check baseline.

## 4. Visibility Alone Is Not a Reliable Importance Signal

Session 17 tested score ablations for Gaussian \(i\), including opacity

```math
\alpha_i = \sigma(o_i),
```

a depth-weighted visibility proxy

```math
V_i = \sum_c \frac{m_{ic}}{z_{ic}^2 + \epsilon},
```

and an unweighted visibility count

```math
C_i = \sum_c m_{ic}.
```

The tested scores were:

| Mode | Score |
| --- | --- |
| opacity top-k control | \(s_i = \alpha_i\) |
| `opacity_visibility` | \(s_i = \alpha_i \log(1 + V_i)\) |
| `visibility` | \(s_i = \log(1 + V_i)\) |
| `opacity_count` | \(s_i = \alpha_i \log(1 + C_i)\) |

The raw `visibility` mode fails badly:

| Keep | `visibility` PSNR | Opacity top-k PSNR |
| ---: | ---: | ---: |
| 25% | 10.363 | 22.377 |
| 50% | 11.824 | 29.169 |
| 75% | 15.259 | 34.496 |

The visibility proxy only checks whether projected Gaussian centers fall in
camera frusta and applies a simple depth weighting. It does not measure actual
render contribution, occlusion, screen-space footprint, color error, or
gradient importance. In this project, opacity remains the essential stabilizer
for pruning scores.

## 5. Pareto Ranks Depend on the Objective Set

A variant \(a\) dominates variant \(b\) only with respect to the selected
objectives. For the current summaries, the main objective vector is:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

Variant \(a\) dominates \(b\) if:

```math
\forall j,\; f_j(a) \ge f_j(b)
\quad \text{and} \quad
\exists j,\; f_j(a) > f_j(b).
```

Changing the objective set can change the front. For example, adding LPIPS,
peak GPU memory, training time, or per-scene worst-case quality may demote
variants that are rank 0 under PSNR/FPS/size alone. Pareto rank is therefore a
decision aid, not an absolute quality label.

## 6. Metric and Systems Scope

Quality metrics are useful but incomplete:

- PSNR strongly penalizes pixel misalignment, which explains the large
  pose-sensitivity drops.
- SSIM and LPIPS-VGG add perceptual signal, but they still do not certify
  geometric correctness, temporal stability, or downstream task usefulness.
- Aggregate metrics can hide bad individual views unless per-view results are
  inspected.

Efficiency numbers also have a fixed measurement scope. Profiling uses CUDA
events over renderer calls and excludes PNG encoding, disk writes, data loading,
viewer overhead, and network transport. Reported FPS is therefore a
renderer-only measurement on the tested GPU, not an end-to-end application FPS
guarantee. Lego and Drums were also profiled on different A100 memory variants,
so their FPS values are not a controlled cross-scene hardware comparison.

Post-training pruning is evaluated without fine-tuning. The results isolate
the effect of selecting and removing Gaussians, but they do not establish how
much quality could be recovered by optimizing the retained representation.

## Practical Guidance

- Use opacity top-k or opacity-weighted visibility scores for current pruning
  experiments. Do not use raw visibility-only pruning as a serious method.
- Treat 75% opacity top-k as the safest demonstrated compression point on Lego.
  Lower budgets are useful for exploring the Pareto curve, but they visibly
  sacrifice quality.
- Do not claim robustness to camera calibration errors. The pose study shows
  that small rotation errors can dominate all other effects.
- Report the objective definition whenever reporting Pareto ranks.
- Report hardware and profiling scope whenever reporting FPS or latency.
- Repeat pruning and robustness studies on more scenes before making
  scene-general claims.
- Treat reported differences as single-seed point estimates until multi-seed
  experiments quantify uncertainty.

## Future Work

The most useful next limitations work would be:

- repeat pruning, score ablations, and sensitivity studies on Drums and more
  scenes;
- add real-world datasets with imperfect calibration and exposure variation;
- evaluate contribution-aware scores that include rasterized opacity,
  screen-space footprint, color residuals, or gradient statistics;
- add multi-seed summaries and confidence intervals;
- include qualitative failure panels for pose error, brightness shift, sparse
  views, and aggressive pruning.
