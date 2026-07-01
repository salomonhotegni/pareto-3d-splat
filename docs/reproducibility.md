# Reproducibility Protocol

This is the canonical protocol for reproducing Pareto-Splat from a clean
checkout. Commands are run from the repository root. GPU stages require a
CUDA-capable Linux machine; published measurements were collected on NVIDIA
A100 GPUs.

## 1. Record the Source Revision

Before running an experiment, record the exact repository state:

```bash
git rev-parse HEAD
git status --short
```

Use a clean worktree for a published run. Generated datasets, the pinned
GraphDeCo checkout, and experiment artifacts are intentionally ignored by Git.

## 2. Create and Validate the Environment

```bash
make env
make install
make check
make test
```

`make install` downloads the GraphDeCo baseline at the commit recorded in
`scripts/baseline.env` and builds its CUDA extensions. `make check` verifies
the Python and CUDA environment, GPU access, required commands, baseline
revision, and compiled extensions. `make test` runs the project test suite.

To inspect available workflow commands:

```bash
make help
```

The default Conda environment is `pareto3dsplat`. Override it consistently
with, for example, `make test CONDA_ENV=myenv`.

## 3. Download and Validate Data

```bash
make dataset-lego
make check-data-lego
make dataset-drums
make check-data-drums
```

The download scripts validate split sizes, camera metadata, image dimensions,
and RGBA format. Dataset provenance and checksums are documented in
[dataset_notes.md](dataset_notes.md) and [drums_dataset.md](drums_dataset.md).

## 4. Reproduce Clean Baselines

Run the four stages in order for Lego:

```bash
make train-baseline CONFIG=configs/baseline.yaml
make render-baseline CONFIG=configs/baseline.yaml
make evaluate-baseline CONFIG=configs/baseline.yaml
make profile-baseline CONFIG=configs/baseline.yaml
```

Repeat with `CONFIG=configs/drums.yaml` for Drums. The primary output roots are:

```text
results/baseline/lego/seed_0/
results/baseline/drums/seed_0/
```

Training writes a configuration snapshot, exact command, dataset checksums,
environment metadata, logs, checkpoints, and completion state. Evaluation and
profiling write machine-readable JSON beneath the same model directory.

Compare reproduced values with [baseline_results.md](baseline_results.md) and
[drums_baseline_results.md](drums_baseline_results.md). Quality metrics should
use all 200 held-out test views. FPS and latency are renderer-only measurements
and should be compared only when hardware and profiling settings match.

## 5. Reproduce Pruning and Pareto Results

The Lego baseline must exist before these stages:

```bash
make pruning-study-prune
make pruning-study-render
make pruning-study-evaluate
make pruning-study-profile
make pruning-study-summarize
```

Outputs are written under:

```text
results/pruning/lego/study_30000/
```

The summary stage collects each operating point, computes Pareto ranks over

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right],
```

and writes summary JSON plus 2D and 3D plots. Reproduce the importance-score
ablation by overriding the study configuration:

```bash
make pruning-study-prune PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-render PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-evaluate PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-profile PRUNING_CONFIG=configs/importance_ablation_lego.yaml
make pruning-study-summarize PRUNING_CONFIG=configs/importance_ablation_lego.yaml
```

## 6. Reproduce Robustness Studies

Camera-pose sensitivity reuses the trained Lego model:

```bash
make pose-sensitivity-prepare
make pose-sensitivity-render
make pose-sensitivity-evaluate
make pose-sensitivity-profile
make pose-sensitivity-summarize
```

Training-input sensitivity creates degraded datasets and retrains each variant:

```bash
make input-sensitivity-prepare
make input-sensitivity-train
make input-sensitivity-render
make input-sensitivity-evaluate
make input-sensitivity-profile
make input-sensitivity-summarize
```

These outputs are written under `results/pose_sensitivity/lego/study_30000/`
and `results/input_sensitivity/lego/study_30000/`.

## 7. Build Presentation Artifacts

After study summaries exist:

```bash
make demo
make portfolio-assets
```

The generated outputs are `results/demo/index.html` and
`results/portfolio/`. They are local artifacts and are not committed.

## 8. Reproduction Levels

Use the smallest level that supports the intended claim:

| Level | Required work | What it validates |
| --- | --- | --- |
| Source | `make test` | Deterministic project logic and documentation integrity |
| Environment | `make check` | GPU software stack and pinned baseline |
| Baseline | Sections 3-4 | Training, rendering, metrics, and profiling |
| Study | Sections 5-6 | Pruning, Pareto, ablation, and robustness findings |
| Presentation | Section 7 | Demo and portfolio generation from available results |

## 9. Reporting Checklist

Every new result should report:

- repository revision and experiment configuration;
- dataset scene, split, resolution, and random seed;
- GPU model, CUDA/PyTorch versions, and profiling scope;
- Gaussian count, serialized size, and quality metrics;
- warm-up count, repetitions, latency statistic, and FPS definition;
- whether pruning was followed by fine-tuning;
- the objective vector used for Pareto ranking.

Current experiments use one seed, so metric differences are point estimates
without confidence intervals. The final scope of defensible conclusions is
recorded in [claims_and_evidence.md](claims_and_evidence.md).
