# Pareto-Splat Resume Bullets

These bullets are scoped to the completed experiments. Choose one primary
bullet and, where space allows, one supporting bullet.

## Recommended

- Built a reproducible 3D Gaussian Splatting evaluation and pruning pipeline;
  reduced a Lego scene model by 25% and increased A100 renderer throughput by
  61% (283 to 456 FPS) with a 1.42 dB PSNR trade-off using opacity top-k
  pruning.

## Machine Learning / Computer Vision

- Implemented post-training random, opacity, and visibility-aware pruning for
  3D Gaussian Splatting, then used PSNR, SSIM, LPIPS, FPS, and model size to
  identify non-dominated deployment operating points.
- Designed pose and input-sensitivity studies for a 3DGS pipeline, quantifying
  11-19 dB PSNR losses from tested calibration and brightness shifts and
  documenting the system's operating limits.

## Research Engineering

- Built a configuration-driven 3DGS experimentation system with pinned
  dependencies, dataset checksums, run metadata, resumable checkpoints,
  per-view metrics, CUDA profiling, Pareto analysis, and 72 automated tests.
- Evaluated matched-budget Gaussian pruning on NeRF Synthetic Lego and found
  opacity top-k consistently outperformed random pruning at 25%, 50%, and 75%
  retention in quality and measured renderer throughput.

## Software / Systems

- Engineered an end-to-end GPU experiment workflow spanning data validation,
  training, rendering, metric evaluation, CUDA-event profiling, result
  summarization, static visualization, and clean-room reproduction.
- Created tested CLI and Make workflows around a pinned third-party CUDA
  baseline, preserving exact configuration, environment, checksums, logs, and
  completion metadata for each experiment.

## Compact Variants

- Improved 3DGS renderer throughput 61% while shrinking a tested scene model
  25%, using opacity-aware pruning and Pareto analysis to quantify the quality
  trade-off.
- Built a reproducible PyTorch/CUDA pipeline for 3DGS compression, profiling,
  robustness testing, and multi-objective model selection.

## Accuracy Notes

Keep these qualifications available during interviews:

- The headline result is NeRF Synthetic Lego, seed 0.
- FPS is renderer-only on an NVIDIA A100-SXM4-40GB.
- The 75% model loses 1.421 dB PSNR; do not describe the compression as
  lossless.
- Pruning is post-training and does not include fine-tuning.
- One seed does not establish statistical significance.

The full supporting evidence is in
[claims_and_evidence.md](claims_and_evidence.md).
