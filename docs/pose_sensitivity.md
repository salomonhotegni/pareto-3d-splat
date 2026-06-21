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

The summary stage writes:

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
