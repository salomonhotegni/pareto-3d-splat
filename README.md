# Pareto-Splat

Pareto-Splat studies quality-efficiency trade-offs in 3D Gaussian Splatting by
optimizing reconstructed scenes for visual quality, rendering speed, and memory
footprint.

The project starts from the official
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
implementation and will add reproducible evaluation, Gaussian pruning methods,
and Pareto-front analysis.

## Project Status

**Week 2, Session 3 is complete.** The NeRF Synthetic Lego scene now has a
checksum-pinned download and validation workflow. The next milestone is
**Session 4: train and render the first baseline**. See
[the roadmap](docs/roadmap.md) for the complete 12-week plan.

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

The prepared baseline training entry point for Session 4 is:

```bash
make train-baseline
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
