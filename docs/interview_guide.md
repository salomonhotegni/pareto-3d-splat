# Pareto-Splat Interview Guide

## 30-Second Explanation

Pareto-Splat is a reproducible research system for studying quality,
rendering-speed, and model-size trade-offs in 3D Gaussian Splatting. I wrapped
the official GraphDeCo implementation with configuration-driven training,
evaluation, CUDA profiling, pruning, robustness studies, and Pareto analysis.
On NeRF Synthetic Lego, opacity top-k pruning removed 25% of Gaussians and
increased renderer throughput from 283 to 456 FPS while costing 1.42 dB PSNR.
I also tested failure modes and found strong sensitivity to camera-pose and
brightness mismatch.

## Two-Minute Explanation

The problem was that a trained 3D Gaussian Splatting model gives one operating
point, but deployment usually has competing quality, latency, and memory
requirements.

I first made the baseline reproducible: I pinned the upstream implementation,
used YAML experiment contracts, validated the datasets, evaluated all 200 test
views with PSNR, SSIM, and LPIPS, and profiled renderer-only latency and GPU
memory with CUDA events. Every run records its command, configuration,
environment, checksums, logs, and status.

Then I implemented post-training pruning. Random pruning was the control, and
opacity top-k retained the Gaussians with the highest activated opacity. I
evaluated fixed 25%, 50%, and 75% retention budgets and computed Pareto fronts
over PSNR, FPS, and model size. The strongest tested Lego compression point
kept 75% of the Gaussians: size fell from 70.91 to 53.18 MiB, renderer
throughput rose from 282.98 to 456.45 FPS, and PSNR fell from 35.917 to
34.496 dB.

I also investigated where the method fails. A raw visibility score performed
far worse than opacity because frustum membership is not the same as actual
render contribution. Small pose errors and brightness mismatch caused large
quality losses. The main lesson was to present compression as an explicit
Pareto trade-off and make the evidence boundary part of the result.

## Five-Minute Technical Structure

Use this order for a whiteboard or deep technical discussion:

1. **Representation:** A scene is \(G=\{g_i\}_{i=1}^{N}\), with each Gaussian
   storing geometry, opacity, and appearance.
2. **Reference model:** Train for 30,000 iterations and evaluate 200 held-out
   views.
3. **Compression:** For retention \(r\), keep
   \(k=\operatorname{round}(rN)\) Gaussians ranked by
   \(\alpha_i=\sigma(o_i)\).
4. **Measurement:** Evaluate PSNR, SSIM, LPIPS-VGG, model size, Gaussian count,
   latency, FPS, and peak GPU memory.
5. **Selection:** Compare operating points with
   \(f(x)=[\mathrm{PSNR},\mathrm{FPS},-\mathrm{SizeMiB}]\).
6. **Result:** At 75% retention, gain 61.3% renderer throughput and reduce size
   25% for a 1.421 dB PSNR cost.
7. **Ablation:** Raw visibility is insufficient because it ignores occlusion,
   footprint, and compositing contribution.
8. **Robustness:** Pose and brightness experiments reveal major deployment
   assumptions.
9. **Scope:** Lego, seed 0, post-training pruning, renderer-only A100 profiling.

## Likely Questions

### Why use Pareto dominance instead of a weighted score?

A weighted score such as

```math
J(x) =
\lambda_q Q(x) +
\lambda_s S(x) -
\lambda_m M(x)
```

requires choosing weights before knowing the deployment preference, and its
scale depends on metric normalization. Pareto sorting keeps all non-dominated
options visible. A deployment owner can later select from the front using an
actual latency, quality, or memory constraint.

### Why did opacity top-k work?

In alpha compositing, low-opacity Gaussians often have limited influence on the
final pixel color. Activated opacity is therefore a simple proxy for potential
render contribution. It is imperfect because contribution also depends on
coverage, depth ordering, and appearance, but it preserved quality much better
than random selection at matched budgets.

### Why did visibility-only pruning fail?

The tested proxy only asked whether a Gaussian center projected inside a
training-camera frustum and weighted it by depth:

```math
V_i = \sum_c \frac{m_{ic}}{z_{ic}^2+\epsilon}.
```

It ignored screen-space covariance, occlusion, accumulated alpha, color
residual, and gradient information. Frequently visible does not necessarily
mean visually important.

### How was FPS measured?

CUDA events measured renderer calls after warm-up over repeated held-out
views. The reported value is:

```math
\mathrm{FPS} =
\frac{1000}{\operatorname{mean\ latency\ in\ ms}}.
```

It excludes PNG encoding, disk access, viewer overhead, and networking, so I
call it renderer throughput rather than application FPS.

### Why does pruning sometimes improve FPS nonlinearly?

Reducing Gaussian count affects projection, sorting, tile overlap, and
compositing work. Runtime is not determined by file size or Gaussian count
alone; spatial distribution and screen-space coverage matter. GPU timing also
contains system variability, which is why I report the profiling protocol and
avoid treating tiny differences as universal.

### What was the hardest engineering issue?

Keeping the upstream CUDA baseline reproducible while adding project logic
around it. I separated the pinned third-party checkout from configuration,
compatibility, evaluation, and study code. Run snapshots and checksums made
resumes and expensive experiments auditable. Correct RGBA compositing was also
critical because a silent background mismatch would invalidate image metrics.

### What would you change with more compute?

I would repeat every key point across seeds and scenes, then add real-world
data. Methodologically, I would test contribution-aware importance using
screen-space footprint, accumulated compositing weight, residuals, or
gradients, followed by short fine-tuning after pruning. I would report
confidence intervals and fronts that include LPIPS, memory, and worst-view
quality.

### What does the project not prove?

It does not prove real-world generalization, calibration robustness,
hardware-independent speedup, geometric accuracy, or statistical
significance. Most pruning and robustness evidence is Lego-only and all runs
use seed 0.

## STAR Version

**Situation:** A standard 3DGS training run produced a high-quality model but
did not expose deployment choices across quality, speed, and size.

**Task:** Build a reproducible way to compress the representation, measure the
trade-offs, and identify defensible operating points.

**Action:** I pinned and wrapped the baseline, implemented evaluation and CUDA
profiling, added matched-budget pruning methods, formalized Pareto dominance,
and ran importance and robustness ablations with automated tests and artifact
tracking.

**Result:** The best demonstrated Lego compression point removed 25% of the
Gaussians, reduced model size 25%, and increased renderer throughput 61.3% at
a 1.421 dB PSNR cost. The failure studies also showed that raw visibility,
camera-pose error, and brightness mismatch require careful treatment.

## Evidence Links

- [Portfolio summary](portfolio_summary.md)
- [Technical report](technical_report.md)
- [Claims and evidence](claims_and_evidence.md)
- [Reproducibility protocol](reproducibility.md)
- [Limitations](limitations.md)
