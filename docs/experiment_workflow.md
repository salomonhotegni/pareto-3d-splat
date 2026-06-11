# Configuration-Driven Experiment Workflow

The training, rendering, quality evaluation, and efficiency profiling stages
load experiment settings from one YAML file. The default configuration is
`configs/baseline.yaml`.

## Run Stages

From the repository root:

```bash
make train-baseline CONFIG=configs/baseline.yaml
make render-baseline CONFIG=configs/baseline.yaml
make evaluate-baseline CONFIG=configs/baseline.yaml
make profile-baseline CONFIG=configs/baseline.yaml
```

The shell wrappers remain available for direct use:

```bash
bash scripts/train_baseline.sh --config configs/baseline.yaml
bash scripts/render_baseline.sh --config configs/baseline.yaml
bash scripts/evaluate_baseline.sh --config configs/baseline.yaml
bash scripts/profile_baseline.sh --config configs/baseline.yaml
```

Resume training with:

```bash
bash scripts/train_baseline.sh \
  --config configs/baseline.yaml \
  --resume results/baseline/lego/seed_0/chkpnt25000.pth
```

## Configuration Contract

The configuration defines:

- experiment name and random seed;
- baseline, dataset, and output paths;
- dataset format, view counts, resolution, background, and device placement;
- training iterations, evaluation/save/checkpoint schedules, and retention;
- render iteration and train-view rendering policy;
- quality-evaluation device;
- profiling warm-up count and repetitions.

Paths must be relative to the project root and cannot escape it. Iteration
lists must be sorted, unique, positive, and no larger than the configured
training duration. The render iteration must be one of the saved iterations.

Each training attempt snapshots the selected YAML alongside the exact command,
dataset checksums, environment metadata, logs, and completion status. Render,
metric, and profile outputs continue to use the configured model directory.
