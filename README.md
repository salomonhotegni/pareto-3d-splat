# Pareto-Splat

Pareto-Splat studies quality-efficiency trade-offs in 3D Gaussian Splatting by
optimizing reconstructed scenes for visual quality, rendering speed, and memory
footprint.

The project starts from the official
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
implementation and will add reproducible evaluation, Gaussian pruning methods,
and Pareto-front analysis.

## Project Status

**Sessions 4-7 are complete.** The first corrected NeRF Synthetic Lego baseline
has been trained, rendered, evaluated, and profiled, and the workflow is now
configuration-driven. The next milestone is **Session 8: run the clean baseline
workflow on a second scene**. See
[the baseline report](docs/baseline_results.md) for methodology and detailed
results, and [the roadmap](docs/roadmap.md) for the complete 12-week plan.

## Baseline Results

| Quality / efficiency metric | Result |
| --- | ---: |
| PSNR | 35.9166 dB |
| SSIM | 0.983729 |
| LPIPS-VGG | 0.019002 |
| Renderer throughput | 282.98 FPS |
| Mean render latency | 3.534 ms |
| Gaussian count | 299,799 |
| Serialized model size | 70.91 MiB |
| Peak allocated GPU memory | 282.92 MiB |

Quality metrics use all 200 held-out 800 x 800 test views. Renderer throughput
uses 600 CUDA-event measurements on an NVIDIA A100-SXM4-40GB and excludes
image encoding and disk writes.

## Target System

- Linux
- NVIDIA A100-SXM4 40 GB
- NVIDIA driver 580.126.16
- CUDA 12.x compiler toolkit
- Conda

The pinned Python environment uses Python 3.10, PyTorch 2.5.1, and the PyTorch
CUDA 12.1 runtime. Newer NVIDIA drivers are backward compatible with that
runtime. A CUDA 12.x `nvcc` compiler is required to build the baseline's custom
CUDA extensions.

## Quick Start

Create or update the project environment:

```bash
conda env update --name pareto3dsplat --file environment.yml --prune
conda activate pareto3dsplat
```

Download the pinned baseline and compile its CUDA extensions:

```bash
bash scripts/bootstrap_baseline.sh
bash scripts/install_baseline.sh
```

Validate the complete setup:

```bash
python scripts/check_environment.py --require-baseline
```

The equivalent Make targets are:

```bash
make env
make install
make check
```

Download and validate the first dataset:

```bash
make dataset
make check-data
```

Train the baseline:

```bash
make train-baseline
```

All experiment-specific settings are loaded from `configs/baseline.yaml`.
Select another configuration without editing scripts:

```bash
make train-baseline CONFIG=configs/another_scene.yaml
make render-baseline CONFIG=configs/another_scene.yaml
make evaluate-baseline CONFIG=configs/another_scene.yaml
make profile-baseline CONFIG=configs/another_scene.yaml
```

After training, render the held-out test views with corrected NeRF Synthetic
alpha compositing:

```bash
make render-baseline
```

Evaluate all 200 held-out views with PSNR, SSIM, and LPIPS-VGG:

```bash
make evaluate-baseline
```

Profile renderer-only FPS, latency, GPU memory, Gaussian count, and model size:

```bash
make profile-baseline
```

Create a labeled side-by-side orbit video:

```bash
make comparison-video
```

See [docs/setup.md](docs/setup.md) for troubleshooting and machine-specific
details, and [docs/dataset_notes.md](docs/dataset_notes.md) for dataset
provenance and format checks.

## Repository Layout

```text
configs/       Reproducible experiment configurations
data/          Local datasets; generated/downloaded data is ignored by Git
docs/          Project plan, roadmap, setup notes, and reports
results/       Local experiment artifacts; generated results are ignored by Git
scripts/       Environment, data, training, and evaluation entry points
src/           Pareto-Splat evaluation, compression, and Pareto-analysis code
third_party/   Downloaded pinned baseline; generated and ignored by Git
```

## Baseline

The baseline is the official GraphDeco-INRIA implementation pinned to commit
`54c035f7834b564019656c3e3fcc3646292f727d`. The pin prevents upstream changes
from silently changing experiment results.

## License

Pareto-Splat is licensed under the MIT License. The downloaded 3D Gaussian
Splatting baseline retains its own license and attribution.
