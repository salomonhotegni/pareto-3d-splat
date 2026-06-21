# Pruning Study Workflow and Results

Session 10 evaluates pruned 3DGS operating points and summarizes the
quality-efficiency trade-offs.

The default study is configured in `configs/pruning_lego.yaml`. It uses the
corrected Lego baseline as the source model and creates variants for:

- random pruning at 25%, 50%, and 75% retained Gaussians;
- opacity top-k pruning at 25%, 50%, and 75% retained Gaussians;
- visibility-aware top-k pruning at 25%, 50%, and 75% retained Gaussians;
- opacity-threshold pruning at activated opacity thresholds 0.1, 0.3, and 0.5.

## Stages

Create all pruned models:

```bash
make pruning-study-prune
```

Render, evaluate, and profile a specific variant:

```bash
make pruning-study-render PRUNING_VARIANT=top_k_keep_050

make pruning-study-evaluate PRUNING_VARIANT=top_k_keep_050

make pruning-study-profile PRUNING_VARIANT=top_k_keep_050
```

Leave `PRUNING_VARIANT` unset to run a stage for every configured pruning
variant.

After every configured variant has metrics and profile artifacts, summarize the
study:

```bash
make pruning-study-summarize
```

The equivalent Python entry point is:

```bash
python scripts/run_pruning_study.py \
  --config configs/pruning_lego.yaml \
  --variant top_k_keep_050 \
  render
```

The summary stage writes:

```text
results/pruning/lego/study_30000/summary/summary.json
results/pruning/lego/study_30000/summary/summary.csv
results/pruning/lego/study_30000/summary/psnr_vs_gaussians.png
results/pruning/lego/study_30000/summary/psnr_vs_fps.png
results/pruning/lego/study_30000/summary/lpips_vs_size.png
results/pruning/lego/study_30000/summary/pareto_psnr_vs_fps.png
results/pruning/lego/study_30000/summary/pareto_psnr_vs_size.png
results/pruning/lego/study_30000/summary/pareto_psnr_fps_size_3d.png
```

## Session 10 Results

The completed Lego study includes the baseline plus nine pruned operating
points. Quality metrics use all 200 held-out test views. Renderer throughput
uses three profiling repetitions over 200 test views and excludes image
encoding and disk writes.

| Variant | Keep | Gaussians | PSNR | SSIM | LPIPS-VGG | FPS | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 299,799 | 35.917 | 0.983729 | 0.019002 | 282.98 | 70.91 MiB |
| top_k_keep_075 | 0.750 | 224,849 | 34.496 | 0.980445 | 0.021636 | 456.45 | 53.18 MiB |
| opacity_threshold_010 | 0.737 | 220,887 | 34.331 | 0.979975 | 0.021994 | 459.38 | 52.24 MiB |
| top_k_keep_050 | 0.500 | 149,900 | 29.169 | 0.957937 | 0.041038 | 635.44 | 35.45 MiB |
| opacity_threshold_030 | 0.487 | 146,041 | 28.826 | 0.955719 | 0.042964 | 636.38 | 34.54 MiB |
| top_k_keep_025 | 0.250 | 74,950 | 22.377 | 0.877730 | 0.102224 | 895.98 | 17.73 MiB |
| opacity_threshold_050 | 0.342 | 102,433 | 25.126 | 0.918014 | 0.071806 | 792.71 | 24.23 MiB |
| random_keep_075_seed_0 | 0.750 | 224,849 | 31.857 | 0.966304 | 0.034873 | 360.79 | 53.18 MiB |
| random_keep_050_seed_0 | 0.500 | 149,900 | 27.143 | 0.927109 | 0.071232 | 459.79 | 35.45 MiB |
| random_keep_025_seed_0 | 0.250 | 74,950 | 20.995 | 0.843278 | 0.136132 | 685.22 | 17.73 MiB |

At matched keep fractions, top-k opacity pruning dominates random pruning:
it gives higher quality and higher FPS at 25%, 50%, and 75% retained
Gaussians. Opacity-threshold pruning produces similar behavior to top-k at
nearby Gaussian counts, with `opacity_threshold_010` preserving 34.331 dB PSNR
while improving throughput from 282.98 FPS to 459.38 FPS.

Session 11 formalizes this comparison with Pareto dominance and
non-dominated sorting. Using PSNR, FPS, and serialized model size as the first
quality-efficiency objective set, the baseline plus top-k and
opacity-threshold variants are rank 0, while all random variants are rank 1.
See [docs/pareto.md](pareto.md) for the objective definitions and ranking API.

Session 12 extends the summary step with Pareto-front outputs. The JSON and
CSV summaries now include a `pareto_rank` column computed from PSNR, FPS, and
serialized model size. The rank-0 variants are highlighted in 2D projections
for PSNR-vs-FPS and PSNR-vs-size, and in a 3D PSNR/FPS/size plot.

Session 14 adds `visibility-top-k` variants to the configured grid so random,
opacity top-k, and visibility-aware top-k methods can be compared at matched
25%, 50%, and 75% Gaussian budgets. These variants use the source model's
`cameras.json` to compute the Session 13 visibility-aware importance score.

## Session 14 Matched-Budget Results

The matched-budget comparison evaluates random pruning, opacity top-k, and
visibility-aware top-k at the same retained Gaussian fractions. For a retention
fraction `r`, each fixed-budget method keeps:

```text
k = round(rN)
```

where `N = 299,799` for the Lego baseline.

| Budget | Method | Gaussians | PSNR | SSIM | LPIPS-VGG | FPS | Pareto rank |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | random | 74,950 | 20.995 | 0.843278 | 0.136132 | 685.22 | 1 |
| 25% | opacity top-k | 74,950 | 22.377 | 0.877730 | 0.102224 | 895.98 | 0 |
| 25% | visibility top-k | 74,950 | 22.156 | 0.875899 | 0.103076 | 905.41 | 0 |
| 50% | random | 149,900 | 27.143 | 0.927109 | 0.071232 | 459.79 | 1 |
| 50% | opacity top-k | 149,900 | 29.169 | 0.957937 | 0.041038 | 635.44 | 0 |
| 50% | visibility top-k | 149,900 | 28.939 | 0.956881 | 0.041970 | 642.69 | 0 |
| 75% | random | 224,849 | 31.857 | 0.966304 | 0.034873 | 360.79 | 2 |
| 75% | opacity top-k | 224,849 | 34.496 | 0.980445 | 0.021636 | 456.45 | 0 |
| 75% | visibility top-k | 224,849 | 34.488 | 0.980438 | 0.021677 | 453.83 | 1 |

Opacity top-k remains the strongest matched-budget quality method. The
visibility-aware score is close in quality and gives slightly higher FPS at
25% and 50%, but the 75% visibility-aware variant is dominated by opacity
top-k because it has slightly lower PSNR and FPS at the same model size.

## Notes

Rendering and profiling require CUDA. The summary includes the baseline row
plus each pruning variant, with PSNR, SSIM, LPIPS-VGG, FPS, latency, Gaussian
count, serialized model size, peak allocated GPU memory, and Pareto rank.
