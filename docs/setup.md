# Environment and Baseline Setup

## Supported Target

The initial supported system is:

| Component | Target |
|---|---|
| GPU | NVIDIA A100-SXM4 40 GB |
| Driver | 580.126.16 or compatible |
| Host CUDA compiler | CUDA 12.x |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| PyTorch CUDA runtime | 12.1 |
| OS | Linux |

The NVIDIA driver may advertise CUDA 13.0 while the project uses a CUDA 12.1
PyTorch runtime. This is expected: the driver supports older CUDA runtimes.
The host `nvcc` compiler and PyTorch CUDA runtime must have the same major
version when compiling the baseline extensions.

## Why This Baseline

Pareto-Splat uses the official GraphDeco-INRIA 3D Gaussian Splatting
implementation. It provides the reference optimizer, differentiable
rasterizer, rendering pipeline, and standard quality metrics needed for the
baseline experiments.

The bootstrap script pins the baseline to:

```text
repository: https://github.com/graphdeco-inria/gaussian-splatting.git
commit:     54c035f7834b564019656c3e3fcc3646292f727d
```

The downloaded repository lives under `third_party/` and is ignored by Git.
Our project code and experiment configurations remain separate from the
upstream implementation.

## Installation

From the repository root:

```bash
conda env update --name pareto3dsplat --file environment.yml --prune
conda activate pareto3dsplat
bash scripts/bootstrap_baseline.sh
bash scripts/install_baseline.sh
python scripts/check_environment.py --require-baseline
```

The extension installer defaults to `TORCH_CUDA_ARCH_LIST=8.0`, the compute
capability of the A100. Override it before installation when targeting another
GPU:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6" bash scripts/install_baseline.sh
```

## COLMAP

COLMAP is optional for the first public-dataset baseline because that dataset
will already include camera poses. It becomes required when processing a new
set of images or a phone-captured scene.

Check whether it is available with:

```bash
colmap -h
```

## Common Problems

### PyTorch cannot see the GPU

Run:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` works but PyTorch returns `False`, confirm that the shell is on
the GPU machine and that the activated environment contains `pytorch-cuda`.

### CUDA extension build reports a version mismatch

Compare:

```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

The minor versions may differ, but both should be CUDA 12.x for this setup.

### Rebuild the baseline extensions

Remove pip's installed extension packages and reinstall:

```bash
python -m pip uninstall -y diff-gaussian-rasterization simple-knn fused-ssim
bash scripts/install_baseline.sh
```
