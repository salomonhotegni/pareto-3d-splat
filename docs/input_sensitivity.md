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

Pending. The tooling is staged; the train/render/evaluate/summarize runs still
need to be executed.
