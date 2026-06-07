# NeRF Synthetic Lego Baseline Results

This report records the first complete Pareto-Splat baseline, trained and
evaluated on June 7, 2026. It establishes the uncompressed reference point for
later pruning and Pareto-front experiments.

## Experimental Setup

| Setting | Value |
| --- | --- |
| Dataset | NeRF Synthetic Lego |
| Views | 100 train, 100 validation, 200 test |
| Image resolution | 800 x 800 |
| Background | White |
| Training iterations | 30,000 |
| Random seed | 0 |
| Baseline | GraphDeCo 3D Gaussian Splatting |
| Baseline commit | `54c035f7834b564019656c3e3fcc3646292f727d` |
| GPU | NVIDIA A100-SXM4-40GB |
| PyTorch / CUDA runtime | 2.5.1 / 12.1 |

NeRF Synthetic images contain transparency. The project compatibility launcher
composites each RGBA image onto the configured white background before the
camera object is created. This correction is applied consistently during
training and rendering; evaluating against uncomposited RGB channels would
produce invalid ground truth.

## Training

The corrected 30,000-iteration run completed in **401 seconds (6 minutes,
41 seconds)**. The run retained the final two resumable checkpoints and saved
the deployable point cloud at iteration 30,000.

| Artifact | Value |
| --- | --- |
| Final Gaussian count | 299,799 |
| Serialized PLY size | 74,351,683 bytes (70.91 MiB) |
| In-memory Gaussian parameters | 70,752,564 bytes (67.47 MiB) |
| PLY SHA-256 | `f1d4bc7ed81ea1da8669649984a1272610e2a046d8793c81ca6a9701dd54a0b4` |

The training attempt metadata, exact command, environment snapshot, dataset
checksums, log, and completion status are stored under:

```text
results/baseline/lego/seed_0/attempts/20260607T215231Z/
```

## Novel-View Quality

Metrics were computed over all 200 held-out test views using corrected
white-background ground truth.

| Metric | Mean |
| --- | ---: |
| PSNR | 35.9166 dB |
| SSIM | 0.983729 |
| LPIPS-VGG | 0.019002 |

PSNR and SSIM use the pinned GraphDeCo formulas. LPIPS uses the VGG trunk,
version 0.1, from `lpips==0.1.4`. Aggregate and per-view records are stored
under:

```text
results/baseline/lego/seed_0/metrics/ours_30000/
```

## Rendering Efficiency

Renderer timing used CUDA events around `gaussian_renderer.render` only. It
excluded scene loading, PNG encoding, and disk writes. Ten views were used for
warm-up, followed by three repetitions of all 200 test cameras, for 600 timed
frames.

| Measurement | Value |
| --- | ---: |
| Mean latency | 3.534 ms |
| Median latency | 3.441 ms |
| P95 latency | 4.352 ms |
| Mean throughput | 282.98 FPS |
| Peak allocated GPU memory | 282.92 MiB |
| Incremental rendering peak | 205.76 MiB |

Ground-truth camera images remained in CPU memory during profiling, while
camera transforms and model parameters remained on CUDA. Memory values are
PyTorch CUDA allocator measurements, not total process or driver memory.
Detailed per-frame measurements are stored under:

```text
results/baseline/lego/seed_0/profile/ours_30000/
```

## Qualitative Output

The corrected model produced 200 held-out renders and a labeled side-by-side
ground-truth comparison video:

```text
results/baseline/lego/seed_0/test/ours_30000/
results/baseline/lego/seed_0/videos/lego_ground_truth_vs_3dgs.mp4
```

The H.264 video contains 200 frames at 30 FPS, has a resolution of 1600 x 800,
and lasts 6.67 seconds.

## Reproduction

After creating the environment, downloading the dataset, and installing the
pinned baseline, reproduce the complete workflow with:

```bash
make train-baseline
make render-baseline
make evaluate-baseline
make profile-baseline
make comparison-video
```

Generated experiment artifacts remain ignored by Git. The reusable scripts,
configuration, tests, and this summary are version controlled.

## Scope And Limitations

These measurements describe one synthetic scene, one random seed, and one GPU.
Rendering speed is hardware-dependent, and PyTorch allocator statistics do not
represent total system GPU usage. Later experiments should repeat the workflow
on additional scenes and compare compressed models under the same metric and
profiling definitions.
