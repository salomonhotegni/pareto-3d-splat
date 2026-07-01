# Claims and Evidence

This document freezes the experimental evidence available at the end of
Session 22 and defines the scope of claims made by Pareto-Splat. A claim is
considered supported only within the scenes, seeds, hardware, metrics, and
perturbations that were actually evaluated.

## Evidence Scope

The evidence consists of:

- clean NeRF Synthetic Lego and Drums baselines;
- post-training pruning, Pareto analysis, and importance-score ablations on
  Lego;
- camera-pose and training-input sensitivity studies on Lego;
- renderer-only profiling on NVIDIA A100 GPUs.

All reported experiments use seed 0. For an observed metric difference,

```math
\Delta m = m_{\mathrm{variant}} - m_{\mathrm{reference}},
```

the project reports a point estimate. A single seed does not provide enough
samples to estimate run-to-run variance, a standard error, or a confidence
interval.

## Supported Claims

| Claim | Evidence | Scope and qualification |
| --- | --- | --- |
| The repository provides a reproducible 3DGS workflow. | Complete configuration-driven training, rendering, evaluation, profiling, pruning, summarization, demo, and portfolio workflows; clean baselines on Lego and Drums. | NeRF Synthetic scenes with the pinned GraphDeCo baseline and documented environment. |
| Opacity top-k pruning is better than random pruning at matched budgets. | At 25%, 50%, and 75% retention, opacity top-k has higher PSNR and measured FPS than the corresponding random variant. | Lego, seed 0, post-training pruning without fine-tuning. |
| The 75% opacity top-k variant is the strongest demonstrated compression point. | 34.496 dB PSNR, 456.45 FPS, and 53.18 MiB versus 35.917 dB, 282.98 FPS, and 70.91 MiB for the baseline. | A practical judgment among tested Lego variants, not a universal optimum. |
| Raw frustum visibility is not a reliable standalone importance score. | Visibility-only pruning reaches 10.363-15.259 dB across tested budgets, substantially below opacity top-k. | The tested center-projection and inverse-depth proxy on Lego. |
| The evaluated pipeline is sensitive to camera-pose error. | Mean rotation error of 0.3994 degrees reduces PSNR by 16.8185 dB; the tested translation perturbation reduces it by 11.3750 dB. | Held-out Lego views and the documented perturbation distributions. |
| Train-test appearance mismatch can dominate reconstruction quality. | Training brightness scales of 0.75 and 1.25 reduce clean-test PSNR by 13.07 and 11.08 dB. | Lego, seed 0, global synthetic brightness shifts. |

## Preliminary Observations

These observations are useful hypotheses but are not broad conclusions:

- visibility may help as a secondary opacity reweighting signal;
- 75% retention may be a useful default for other object-centric synthetic
  scenes;
- mild zero-mean image noise may be less damaging than blur, brightness
  mismatch, or sparse views;
- Pareto rank-0 variants may be useful deployment candidates when PSNR, FPS,
  and serialized size are the relevant objectives.

They require additional scenes, seeds, or scoring methods before being promoted
to general claims.

## Claims Not Supported

The current experiments do not support claims of:

- generalization to real-world or unbounded scenes;
- robustness to camera miscalibration or exposure changes;
- universal optimality of opacity top-k or 75% retention;
- geometric accuracy, temporal stability, or downstream-task performance;
- end-to-end application FPS;
- statistically significant differences across random seeds;
- hardware-independent speedups.

## Interpretation Rules

Pareto dominance is always relative to the declared objective vector:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

Changing the objectives can change the non-dominated set. Likewise, measured
speedup

```math
S = \frac{\mathrm{FPS}_{\mathrm{variant}}}
         {\mathrm{FPS}_{\mathrm{baseline}}}
```

describes renderer-only throughput on the tested hardware. It excludes image
encoding, storage, viewer, and network costs.

Future reports should preserve these qualifications unless new experiments
directly expand the evidence.
