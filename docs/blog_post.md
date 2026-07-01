# Pareto-Splat: Finding Useful 3D Gaussian Splatting Operating Points

3D Gaussian Splatting can produce compelling novel views at real-time render
rates, but a trained scene is not automatically deployment-efficient. A model
may contain hundreds of thousands of Gaussians, and the highest-quality model
is not always the right choice for a constrained device or latency target.

I built Pareto-Splat to study that decision as a reproducible multi-objective
experiment: preserve image quality while reducing representation size and
improving rendering throughput.

## From One Score to Competing Objectives

A 3DGS model is a set of learned Gaussians:

```math
G = \{g_i\}_{i=1}^{N}.
```

Each Gaussian stores position, scale, rotation, opacity, and view-dependent
color features. Given camera pose \(T_c\), the renderer produces:

```math
\hat{I}_c = R(G, T_c).
```

Compression changes several outcomes at once. Fewer Gaussians can lower model
size and render cost, but may remove geometry or appearance needed by held-out
views. Instead of collapsing those outcomes into an arbitrary weighted sum, I
used the objective vector:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

PSNR and FPS are maximized, while negating size converts minimization into the
same orientation. Model \(a\) dominates model \(b\) when:

```math
\forall j,\; f_j(a) \ge f_j(b)
\quad \text{and} \quad
\exists j,\; f_j(a) > f_j(b).
```

The non-dominated models form a Pareto front. Each one remains relevant because
no measured alternative improves one objective without sacrificing another.

## Building a Reproducible Baseline

The project begins with the official GraphDeCo implementation pinned to a
specific commit. I wrapped it in a configuration-driven workflow that controls
the dataset, seed, paths, training schedule, saved iterations, evaluation
split, and profiling settings.

The workflow performs:

1. dataset download and structural validation;
2. 30,000-iteration baseline training;
3. rendering over 200 held-out test views;
4. PSNR, SSIM, and LPIPS-VGG evaluation;
5. CUDA-event renderer profiling;
6. pruning, re-rendering, and re-evaluation;
7. Pareto sorting and plot generation.

Each training attempt records its configuration, exact command, environment
metadata, dataset checksums, log, duration, and completion status. This made
failed and resumed runs traceable rather than mysterious.

NeRF Synthetic also exposed a subtle compatibility issue: source images are
RGBA. The project composites them onto the configured white background before
the upstream camera pipeline uses them. Otherwise the evaluated ground truth
would not match the images used by the model.

## Establishing the Reference Point

The NeRF Synthetic Lego baseline reached:

| Metric | Baseline |
| --- | ---: |
| PSNR | 35.9166 dB |
| SSIM | 0.983729 |
| LPIPS-VGG | 0.019002 |
| Gaussians | 299,799 |
| Serialized size | 70.91 MiB |
| Mean renderer latency | 3.534 ms |
| Renderer throughput | 282.98 FPS |

The quality metrics cover all 200 test views. Throughput was measured with 600
CUDA-event samples on an NVIDIA A100-SXM4-40GB and excludes image encoding and
disk writes.

I repeated the clean workflow on Drums to verify that it was not hard-coded to
one scene. Drums reached 26.1724 dB PSNR with 318,647 Gaussians. Because the
two scenes were profiled on different A100 memory variants, their FPS values
are not treated as a controlled cross-scene comparison.

## Pruning Without Retraining

The core compression experiments remove Gaussians from the trained model
without fine-tuning. For retention fraction \(r\), a fixed-budget method keeps:

```math
k = \operatorname{round}(rN)
```

Gaussians. Random pruning provides a sanity-check baseline. Opacity top-k ranks
each Gaussian by its activated opacity:

```math
\alpha_i = \sigma(o_i),
\qquad
s_i = \alpha_i.
```

This simple ranking was surprisingly effective:

| Variant | Gaussians | PSNR | FPS | Size |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 299,799 | 35.917 | 282.98 | 70.91 MiB |
| 75% top-k | 224,849 | 34.496 | 456.45 | 53.18 MiB |
| 50% top-k | 149,900 | 29.169 | 635.44 | 35.45 MiB |
| 25% top-k | 74,950 | 22.377 | 895.98 | 17.73 MiB |

At 75% retention, model size fell by 25%, renderer throughput increased by
61.3%, and PSNR decreased by 1.421 dB. That is the strongest demonstrated
compression point among the tested Lego variants. More aggressive pruning
continued to improve efficiency, but quality dropped quickly.

Opacity top-k also beat random pruning at the same 25%, 50%, and 75% budgets.
That comparison matters: fewer Gaussians alone do not explain the result;
which Gaussians survive is crucial.

## A Reasonable Idea That Failed

I expected camera visibility to improve Gaussian importance. For Gaussian
\(i\), I counted whether its projected center fell inside each training-camera
frustum and weighted visible observations by inverse squared depth:

```math
V_i = \sum_c \frac{m_{ic}}{z_{ic}^2 + \epsilon}.
```

I tested raw visibility and opacity-weighted variants. Raw visibility failed
badly: at 75% retention it produced 15.259 dB, compared with 34.496 dB for
opacity top-k.

The proxy was too crude. Frustum membership does not measure screen-space
footprint, occlusion, alpha-compositing contribution, color residual, or
gradient importance. A Gaussian can be visible to many cameras and still
contribute little to the final image. The useful conclusion was not
"visibility never matters," but that opacity must remain a stabilizing signal
for this proxy.

## Stress-Testing the Pipeline

Compression is only one deployment risk, so I added controlled sensitivity
studies.

Small camera-pose errors were severe. A perturbation with mean rotation
0.3994 degrees reduced Lego PSNR from 35.9166 to 19.0981 dB. A tested
translation perturbation reduced it to 24.5416 dB.

Training-input changes produced a broader range:

- Gaussian noise with standard deviation 0.02 cost 0.59 dB;
- Gaussian blur with radius 1.0 cost 3.62 dB;
- reducing training views from 100 to 25 cost 6.02 dB;
- global brightness changes cost 11-13 dB.

These results make the boundary clear: this pipeline assumes accurate
calibration and reasonably consistent train-test appearance.

## What I Learned

**Reproducibility is system design.** Configuration snapshots, checksums,
metadata, deterministic paths, and tests were not administrative extras. They
made expensive GPU results auditable and recoverable.

**Simple baselines deserve respect.** Activated opacity outperformed a more
elaborate visibility proxy because it was more closely tied to compositing
importance.

**A Pareto front is conditional.** A model can be non-dominated under
PSNR/FPS/size and become dominated after adding LPIPS, peak memory, or
worst-view quality. The objective vector must always accompany the rank.

**Failure cases are part of the result.** Pose and brightness sensitivity say
where the system should not yet be trusted. That is more useful than claiming
generic robustness.

## Limits and Next Steps

The pruning, ablation, and robustness studies are primarily Lego-only and use
seed 0. The measured differences are point estimates without confidence
intervals. The scenes are synthetic and object-centric, and FPS is a
renderer-only hardware-specific measurement.

The next scientific steps would be multi-seed experiments, additional
synthetic and real scenes, contribution-aware scores, and pruning followed by
fine-tuning. A stronger importance signal could combine opacity with
screen-space footprint, accumulated compositing weight, residual error, or
gradient statistics.

The repository includes the complete
[technical report](technical_report.md), [evidence audit](claims_and_evidence.md),
and [reproducibility protocol](reproducibility.md).
