# Training-Input Sensitivity

Session 16 evaluates how robust the 3DGS training workflow is to degraded
training observations and fewer input views. Unlike Session 15, the camera
poses remain unchanged. The training images or the number of training frames
are changed, each variant is retrained, and the resulting model is evaluated on
the clean held-out test split.

## Model

The clean training set is

```math
D_{\mathrm{train}} = \{(I_i, T_i)\}_{i=1}^{N},
```

where \(I_i\) is an RGBA training image and \(T_i\) is its camera-to-world
pose. Session 16 creates derived training sets

```math
D'_{\mathrm{train}} = \{(I'_i, T_i)\}_{i \in S},
```

where \(S\) is either the full training index set or a deterministic
evenly-spaced subset.

The default image degradations are:

```math
I'_i = \operatorname{clip}(I_i + \epsilon_i),
\qquad
\epsilon_i \sim \mathcal{N}(0, \sigma^2),
```

```math
I'_i = \operatorname{GaussianBlur}(I_i, r),
```

```math
I'_i = \operatorname{clip}(\alpha I_i).
```

The RGB channels are degraded while the alpha channel is preserved. Validation
and test images remain clean.

For fewer-view variants, the selected training index set is

```math
S_k = \left\{
\operatorname{round}\left(\frac{j(N-1)}{k-1}\right)
\;|\; j = 0, \dots, k-1
\right\},
```

with \(k < N\), so the subset keeps broad camera coverage.

## Workflow

The default Lego study is configured in
`configs/input_sensitivity_lego.yaml`. It writes derived datasets under:

```text
data/nerf_synthetic/lego_input_sensitivity/
```

and model outputs under:

```text
results/input_sensitivity/lego/study_30000/
```

Run all stages with:

```bash
make input-sensitivity-prepare
make input-sensitivity-train
make input-sensitivity-render
make input-sensitivity-evaluate
make input-sensitivity-summarize
```

Run one variant while iterating:

```bash
make input-sensitivity-prepare INPUT_VARIANT=noise_std_0p02
make input-sensitivity-train INPUT_VARIANT=noise_std_0p02
make input-sensitivity-render INPUT_VARIANT=noise_std_0p02
make input-sensitivity-evaluate INPUT_VARIANT=noise_std_0p02
```

Profiling is available if a variant changes model size or speed enough to
matter:

```bash
make input-sensitivity-profile INPUT_VARIANT=train_views_050
```

## Summary Metrics

The summary stage writes:

```text
results/input_sensitivity/lego/study_30000/summary/summary.json
results/input_sensitivity/lego/study_30000/summary/summary.csv
results/input_sensitivity/lego/study_30000/summary/psnr_by_variant.png
results/input_sensitivity/lego/study_30000/summary/psnr_drop_by_variant.png
results/input_sensitivity/lego/study_30000/summary/lpips_increase_by_variant.png
```

The table includes the clean baseline row and computes degradation as:

```math
\Delta \mathrm{PSNR}
= \mathrm{PSNR}_{\mathrm{baseline}}
- \mathrm{PSNR}_{\mathrm{variant}},
```

```math
\Delta \mathrm{SSIM}
= \mathrm{SSIM}_{\mathrm{baseline}}
- \mathrm{SSIM}_{\mathrm{variant}},
```

```math
\Delta \mathrm{LPIPS}
= \mathrm{LPIPS}_{\mathrm{variant}}
- \mathrm{LPIPS}_{\mathrm{baseline}}.
```

Positive values mean worse clean-test quality after training on degraded or
reduced inputs.

## Session 16 Results

All configured Lego variants were prepared, trained for 30,000 iterations,
rendered on the clean 200-view test split, evaluated, profiled, and summarized.
The summary artifacts are:

```text
results/input_sensitivity/lego/study_30000/summary/summary.json
results/input_sensitivity/lego/study_30000/summary/summary.csv
results/input_sensitivity/lego/study_30000/summary/psnr_by_variant.png
results/input_sensitivity/lego/study_30000/summary/psnr_drop_by_variant.png
results/input_sensitivity/lego/study_30000/summary/lpips_increase_by_variant.png
```

Quality is measured against the clean held-out test split. Drops are relative
to the clean Lego baseline:

| Variant | Training input | Train views | PSNR | Delta PSNR | SSIM | Delta SSIM | LPIPS-VGG | Delta LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | clean | 100 | 35.92 | 0.00 | 0.9837 | 0.0000 | 0.0190 | 0.0000 |
| `noise_std_0p02` | Gaussian noise, sigma 0.02 | 100 | 35.33 | 0.59 | 0.9782 | 0.0056 | 0.0258 | 0.0068 |
| `blur_radius_1p0` | Gaussian blur, radius 1.0 | 100 | 32.30 | 3.62 | 0.9640 | 0.0197 | 0.0593 | 0.0403 |
| `brightness_0p75` | brightness scale 0.75 | 100 | 22.85 | 13.07 | 0.9607 | 0.0230 | 0.0295 | 0.0105 |
| `brightness_1p25` | brightness scale 1.25 | 100 | 24.83 | 11.08 | 0.9646 | 0.0191 | 0.0351 | 0.0161 |
| `train_views_050` | clean subset | 50 | 34.24 | 1.67 | 0.9754 | 0.0083 | 0.0263 | 0.0073 |
| `train_views_025` | clean subset | 25 | 29.90 | 6.02 | 0.9506 | 0.0332 | 0.0506 | 0.0316 |

The matching profile results are:

| Variant | Gaussians | Serialized model | Mean latency | P95 latency | FPS | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 299,799 | 70.91 MiB | 3.534 ms | 4.352 ms | 282.98 | 282.92 MiB |
| `noise_std_0p02` | 284,778 | 67.35 MiB | 3.337 ms | 3.726 ms | 299.64 | 275.78 MiB |
| `blur_radius_1p0` | 201,525 | 47.66 MiB | 2.690 ms | 3.064 ms | 371.80 | 208.69 MiB |
| `brightness_0p75` | 283,207 | 66.98 MiB | 3.578 ms | 4.339 ms | 279.45 | 283.17 MiB |
| `brightness_1p25` | 304,192 | 71.95 MiB | 3.259 ms | 3.699 ms | 306.83 | 280.45 MiB |
| `train_views_050` | 289,840 | 68.55 MiB | 3.466 ms | 4.199 ms | 288.49 | 281.67 MiB |
| `train_views_025` | 285,477 | 67.52 MiB | 3.585 ms | 4.372 ms | 278.90 | 285.18 MiB |

The main robustness pattern is:

- Mild Gaussian noise is mostly absorbed by training, with only a
  \(0.59\) dB PSNR drop.
- Blur removes high-frequency supervision, causing a larger
  \(3.62\) dB PSNR drop, but it also learns a smaller model and renders faster.
- Brightness mismatches are the most damaging variants. The train images are
  globally darker or brighter while the test split remains clean, so the learned
  appearance is systematically biased at evaluation time.
- Reducing from 100 to 50 clean views is still usable, with a
  \(1.67\) dB PSNR drop. Reducing to 25 views produces a much larger
  \(6.02\) dB drop, showing that sparse-view coverage is a stronger bottleneck.
- Efficiency is not monotonic with quality. Fewer views produce slightly fewer
  Gaussians, but do not improve latency or memory in this run; blur is the only
  variant that clearly improves both model size and speed.
