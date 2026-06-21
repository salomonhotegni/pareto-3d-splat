# Camera-Pose Sensitivity

Session 15 evaluates how much the trained 3DGS model depends on accurate
held-out camera poses. The Gaussian model is not changed. Instead, the NeRF
Synthetic test camera transforms are perturbed, rendered, and compared against
the original test images.

## Model

Each NeRF Synthetic frame stores a camera-to-world matrix

```math
T_i =
\begin{bmatrix}
R_i & t_i \\
0 & 1
\end{bmatrix}.
```

For a pose-sensitivity variant, the test pose is replaced by

```math
T'_i =
\begin{bmatrix}
\Delta R_i R_i & t_i + \epsilon_i \\
0 & 1
\end{bmatrix},
```

where

```math
\omega_i \sim \mathcal{N}(0, \sigma_R^2 I),
\qquad
\Delta R_i = \exp([\omega_i]_\times),
\qquad
\epsilon_i \sim \mathcal{N}(0, \sigma_t^2 I).
```

Here, `rotation_degrees` configures \(\sigma_R\) after conversion to radians,
and `translation_std` configures \(\sigma_t\) in the dataset coordinate system.
The perturbations are deterministic for a fixed seed.

## Workflow

The default Lego study is configured in
`configs/pose_sensitivity_lego.yaml`. It prepares perturbed copies of the test
transform JSON files under:

```text
data/nerf_synthetic/lego_pose_sensitivity/
```

Each variant also gets a separate GraphDeCo-compatible model output directory
under:

```text
results/pose_sensitivity/lego/study_30000/
```

The variant model directories symlink the baseline point cloud so the baseline
model remains unchanged while renders and metrics are written to separate
locations.

Run the stages with:

```bash
make pose-sensitivity-prepare
make pose-sensitivity-render
make pose-sensitivity-evaluate
make pose-sensitivity-summarize
```

Profiling is available for consistency with the pruning workflow:

```bash
make pose-sensitivity-profile
```

While iterating, restrict a stage to one variant:

```bash
make pose-sensitivity-render POSE_VARIANT=rot_0p25deg
make pose-sensitivity-evaluate POSE_VARIANT=rot_0p25deg
```

## Summary Metrics

The completed Lego study writes:

```text
results/pose_sensitivity/lego/study_30000/summary/summary.json
results/pose_sensitivity/lego/study_30000/summary/summary.csv
results/pose_sensitivity/lego/study_30000/summary/psnr_vs_rotation.png
results/pose_sensitivity/lego/study_30000/summary/psnr_drop_vs_rotation.png
results/pose_sensitivity/lego/study_30000/summary/lpips_increase_vs_translation.png
```

The table includes the baseline row and computes degradation as:

```math
\Delta \mathrm{PSNR}
= \mathrm{PSNR}_{\mathrm{baseline}}
- \mathrm{PSNR}_{\mathrm{perturbed}},
```

```math
\Delta \mathrm{SSIM}
= \mathrm{SSIM}_{\mathrm{baseline}}
- \mathrm{SSIM}_{\mathrm{perturbed}},
```

```math
\Delta \mathrm{LPIPS}
= \mathrm{LPIPS}_{\mathrm{perturbed}}
- \mathrm{LPIPS}_{\mathrm{baseline}}.
```

Positive values mean worse quality under pose perturbation.

## Session 15 Results

The completed study uses the corrected Lego baseline as the reference:

| Variant | Mean rotation angle | Mean translation norm | PSNR | PSNR drop | SSIM | SSIM drop | LPIPS-VGG | LPIPS increase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0000 deg | 0.000000 | 35.9166 | 0.0000 | 0.983729 | 0.000000 | 0.019002 | 0.000000 |
| rot_0p25deg | 0.3994 deg | 0.000000 | 19.0981 | 16.8185 | 0.774357 | 0.209373 | 0.118881 | 0.099879 |
| rot_0p50deg | 0.7987 deg | 0.000000 | 16.6587 | 19.2579 | 0.745250 | 0.238479 | 0.168238 | 0.149236 |
| trans_0p005 | 0.0000 deg | 0.007987 | 24.5416 | 11.3750 | 0.874998 | 0.108731 | 0.053999 | 0.034997 |
| rot_0p25deg_trans_0p005 | 0.3938 deg | 0.007731 | 18.9473 | 16.9693 | 0.771919 | 0.211810 | 0.120891 | 0.101889 |
| rot_0p50deg_trans_0p010 | 0.7876 deg | 0.015463 | 16.5437 | 19.3728 | 0.744766 | 0.238963 | 0.170081 | 0.151078 |

The largest single-axis effect in this batch is angular error. The
`rot_0p25deg` variant loses 16.8185 dB PSNR, while the translation-only
`trans_0p005` variant loses 11.3750 dB. Combining small translation noise with
rotation noise only slightly worsens the corresponding rotation-only variants:

```math
\Delta\mathrm{PSNR}_{\mathrm{rot0.25+trans0.005}}
- \Delta\mathrm{PSNR}_{\mathrm{rot0.25}}
= 16.9693 - 16.8185
= 0.1508 \text{ dB},
```

```math
\Delta\mathrm{PSNR}_{\mathrm{rot0.50+trans0.010}}
- \Delta\mathrm{PSNR}_{\mathrm{rot0.50}}
= 19.3728 - 19.2579
= 0.1149 \text{ dB}.
```

For this trained Lego model, the held-out-view rendering quality is therefore
highly sensitive to sub-degree camera orientation perturbations. Small
translation perturbations are also damaging, but less severe than the tested
rotation perturbations.
