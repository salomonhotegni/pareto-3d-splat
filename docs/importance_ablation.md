# Importance-Score Ablations

Session 17 tests which terms in the visibility-aware Gaussian importance score
actually matter. The study keeps the pruning budget fixed and changes only the
score used to rank Gaussians.

## Scores

For Gaussian \(i\), let \(o_i\) be the raw opacity logit and

```math
\alpha_i = \sigma(o_i)
```

be the activated opacity. For camera \(c\), the Gaussian center projects to
camera depth \(z_{ic}\) and image coordinates \((u_{ic}, v_{ic})\). Define the
inside-frustum indicator

```math
m_{ic} =
\mathbf{1}
\left[
z_{ic} > 0,\;
0 \leq u_{ic} < W_c,\;
0 \leq v_{ic} < H_c
\right].
```

The depth-weighted visibility proxy is

```math
V_i = \sum_c \frac{m_{ic}}{z_{ic}^2 + \epsilon},
```

and the unweighted visibility count is

```math
C_i = \sum_c m_{ic}.
```

Session 17 compares four ranking scores at matched keep fractions:

| Mode | Score |
| --- | --- |
| opacity top-k control | \(s_i = \alpha_i\) |
| `opacity_visibility` | \(s_i = \alpha_i \log(1 + V_i)\) |
| `visibility` | \(s_i = \log(1 + V_i)\) |
| `opacity_count` | \(s_i = \alpha_i \log(1 + C_i)\) |

For a keep fraction \(r\), each fixed-budget variant keeps

```math
k = \operatorname{round}(rN)
```

Gaussians from the Lego baseline, where \(N = 299{,}799\).

## Workflow

The default Lego ablation is configured in:

```text
configs/importance_ablation_lego.yaml
```

It writes outputs under:

```text
results/importance_ablation/lego/study_30000/
```

Run the full study with:

```bash
make pruning-study-prune PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-render PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-evaluate PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-profile PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-summarize PRUNING_CONFIG=configs/importance_ablation_lego.yaml
```

Run a single variant while iterating:

```bash
make pruning-study-render \
  PRUNING_CONFIG=configs/importance_ablation_lego.yaml \
  PRUNING_VARIANT=visibility_top_k_keep_050_opacity_visibility
```

## Summary Artifacts

The completed study writes:

```text
results/importance_ablation/lego/study_30000/summary/summary.json
results/importance_ablation/lego/study_30000/summary/summary.csv
results/importance_ablation/lego/study_30000/summary/psnr_vs_gaussians.png
results/importance_ablation/lego/study_30000/summary/psnr_vs_fps.png
results/importance_ablation/lego/study_30000/summary/lpips_vs_size.png
results/importance_ablation/lego/study_30000/summary/pareto_psnr_vs_fps.png
results/importance_ablation/lego/study_30000/summary/pareto_psnr_vs_size.png
results/importance_ablation/lego/study_30000/summary/pareto_psnr_fps_size_3d.png
```

## Session 17 Results

All variants were pruned, rendered on the 200-view Lego test split, evaluated,
profiled, and summarized. Quality metrics use the clean held-out test images.
Profiling uses three repetitions over all 200 test views and excludes image
encoding and disk writes.

| Keep | Score mode | Gaussians | PSNR | SSIM | LPIPS-VGG | FPS | Size | Pareto rank |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | baseline | 299,799 | 35.917 | 0.983729 | 0.019002 | 282.98 | 70.91 MiB | 0 |
| 25% | opacity top-k | 74,950 | 22.377 | 0.877730 | 0.102224 | 899.68 | 17.73 MiB | 1 |
| 25% | `opacity_visibility` | 74,950 | 22.156 | 0.875899 | 0.103076 | 911.47 | 17.73 MiB | 0 |
| 25% | `visibility` | 74,950 | 10.363 | 0.801398 | 0.222682 | 520.16 | 17.73 MiB | 2 |
| 25% | `opacity_count` | 74,950 | 22.377 | 0.877730 | 0.102224 | 904.13 | 17.73 MiB | 0 |
| 50% | opacity top-k | 149,900 | 29.169 | 0.957937 | 0.041038 | 640.21 | 35.45 MiB | 0 |
| 50% | `opacity_visibility` | 149,900 | 28.939 | 0.956881 | 0.041970 | 639.00 | 35.45 MiB | 1 |
| 50% | `visibility` | 149,900 | 11.824 | 0.849170 | 0.163927 | 420.53 | 35.45 MiB | 2 |
| 50% | `opacity_count` | 149,900 | 29.169 | 0.957937 | 0.041038 | 633.61 | 35.45 MiB | 1 |
| 75% | opacity top-k | 224,849 | 34.496 | 0.980445 | 0.021636 | 453.87 | 53.18 MiB | 1 |
| 75% | `opacity_visibility` | 224,849 | 34.488 | 0.980438 | 0.021677 | 453.37 | 53.18 MiB | 2 |
| 75% | `visibility` | 224,849 | 15.259 | 0.908455 | 0.093578 | 358.24 | 53.18 MiB | 3 |
| 75% | `opacity_count` | 224,849 | 34.496 | 0.980445 | 0.021636 | 454.95 | 53.18 MiB | 0 |

The main ablation pattern is:

- Raw visibility alone is not a useful pruning signal in this setup. Even at
  75% retained Gaussians, `visibility` reaches only 15.259 dB PSNR, far below
  the 34.496 dB opacity top-k control.
- Opacity is the essential stabilizing term. Both opacity-weighted modes stay
  near the opacity top-k controls, while removing opacity collapses quality.
- `opacity_count` matches the opacity top-k quality metrics exactly in this
  run and has similar profiling behavior. The count term did not improve the
  matched-budget operating points.
- `opacity_visibility` is close to opacity top-k but slightly lower quality:
  -0.221 dB at 25%, -0.230 dB at 50%, and -0.008 dB at 75%.
- Visibility-only selection is also slower at the same Gaussian count. At 50%
  keep, it renders at 420.53 FPS versus roughly 640 FPS for the opacity-based
  alternatives.

For the current Lego baseline, the practical conclusion is that the opacity
term should remain in the importance score. Visibility can be used as a small
tie-breaking or reweighting signal, but raw frustum visibility is too weak on
its own.
