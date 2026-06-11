# NeRF Synthetic Drums Baseline Results

This report records the second complete Pareto-Splat baseline, trained and
evaluated on June 11, 2026. It verifies that the configuration-driven workflow
reproduces the complete experiment lifecycle on another scene.

## Experimental Setup

| Setting | Value |
| --- | --- |
| Dataset | NeRF Synthetic Drums |
| Views | 100 train, 100 validation, 200 test |
| Image resolution | 800 x 800 |
| Background | White |
| Training iterations | 30,000 |
| Random seed | 0 |
| Project commit | `8f56b5b8fdcd4dcc0a0b58e826f8d9fd9c1e25a5` |
| Baseline commit | `54c035f7834b564019656c3e3fcc3646292f727d` |
| GPU | NVIDIA A100-SXM4-80GB |
| PyTorch / CUDA runtime | 2.5.1 / 12.1 |

The run used `configs/drums.yaml`. The same corrected white-background alpha
compositing used for Lego was applied during training and rendering.

## Training

The 30,000-iteration run completed in **360 seconds (6 minutes)**. GraphDeCo's
iteration-30,000 test evaluation reported 26.2348 dB PSNR.

| Artifact | Value |
| --- | --- |
| Final Gaussian count | 318,647 |
| Serialized PLY size | 79,025,987 bytes (75.37 MiB) |
| PLY SHA-256 | `46cb59e160742105663af89ea855ce42eab218a0ef41f87b78df2559a78b8ff9` |

The workflow retained the iteration-25,000 and iteration-30,000 checkpoints.
The exact command, configuration snapshot, environment metadata, dataset
checksums, log, and completion status are stored under:

```text
results/baseline/drums/seed_0/attempts/20260611T211708Z/
```

## Novel-View Quality

Metrics were computed over all 200 held-out test views:

| Metric | Mean |
| --- | ---: |
| PSNR | 26.1724 dB |
| SSIM | 0.955651 |
| LPIPS-VGG | 0.043743 |

Aggregate and per-view records are stored under:

```text
results/baseline/drums/seed_0/metrics/ours_30000/
```

## Rendering Efficiency

Renderer timing used ten warm-up views followed by three repetitions of all
200 test cameras, for 600 CUDA-event measurements.

| Measurement | Value |
| --- | ---: |
| Mean latency | 3.284 ms |
| Median latency | 3.244 ms |
| P95 latency | 3.762 ms |
| Mean throughput | 304.49 FPS |
| Peak allocated GPU memory | 266.46 MiB |
| Incremental rendering peak | 185.02 MiB |

Detailed profile and per-frame latency records are stored under:

```text
results/baseline/drums/seed_0/profile/ours_30000/
```

## Cross-Scene Baseline

| Metric | Lego | Drums |
| --- | ---: | ---: |
| PSNR | 35.9166 dB | 26.1724 dB |
| SSIM | 0.983729 | 0.955651 |
| LPIPS-VGG | 0.019002 | 0.043743 |
| Gaussian count | 299,799 | 318,647 |
| Serialized model size | 70.91 MiB | 75.37 MiB |
| Mean render latency | 3.534 ms | 3.284 ms |
| Renderer throughput | 282.98 FPS | 304.49 FPS |

The quality difference confirms that Drums is a materially harder scene for
the baseline. Cross-scene speed values were measured on different A100 memory
variants, so they should not be interpreted as a controlled hardware
comparison.

## Scope and Limitations

These results describe one seed and one GPU run. The 3DGS test PSNR printed
during training differs slightly from the standalone full evaluation because
the two paths use different evaluation implementations. Later compression
experiments should use the standalone metrics as the consistent comparison
surface.
