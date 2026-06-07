# Dataset Notes: NeRF Synthetic Lego

## Selection

The first Pareto-Splat baseline uses the Lego scene from the NeRF Synthetic
dataset introduced with NeRF. It is a useful first scene because it is small,
public, has exact camera poses, includes transparent image backgrounds, and is
supported directly by the pinned GraphDeco 3D Gaussian Splatting loader.

The source is the official NeRF example archive referenced by the authors'
repository:

- Archive:
  `https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip`
- Archive size: 370,385,516 bytes (353 MiB)
- SHA-256:
  `ce4e94e031c099a19ef04cfb6c71f1e47225d97d365be610b476e379a386c25f`
- Local scene: `data/nerf_synthetic/lego`
- Installed scene size: approximately 163 MiB

Run `make dataset` to download, verify, extract, and validate the scene. The
verified source archive is cached under `data/.downloads/` so the scene can be
recreated without another download.

## Format

The scene follows the NeRF Synthetic Blender convention:

```text
data/nerf_synthetic/lego/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/   # 100 RGBA PNG images
├── val/     # 100 RGBA PNG images
└── test/    # 200 RGBA PNG images
```

Each transform file contains a horizontal camera field of view and a list of
4x4 camera-to-world matrices. Images are 800x800 RGBA PNGs. The baseline appends
`.png` to each frame's relative `file_path`.

`scripts/validate_nerf_synthetic.py` checks:

- The known 100/100/200 camera counts.
- Valid and consistent camera fields of view.
- Finite 4x4 transforms with valid homogeneous rows.
- Relative, non-overlapping image references.
- The existence and decodability of every referenced image.
- Consistent 800x800 RGBA PNG image properties.

## Baseline Use

The pinned GraphDeco loader uses `transforms_train.json` for optimization and
`transforms_test.json` for held-out evaluation when `--eval` is enabled. It
does not consume the validation split. The split is retained to preserve the
official dataset and support later ablations.

The PNG alpha channel must be composited onto white. The pinned baseline
contains this compositing in its transform reader, but its camera loader
reopens the original RGBA path and bypasses the converted image. Pareto-Splat
therefore launches `train.py` and `render.py` through
`scripts/run_graphdeco.py`, which applies the intended compositing at runtime
without modifying the pinned checkout. Using the raw loader or the default
black background changes the ground-truth appearance and invalidates metrics.

The prepared Session 4 training command is:

```bash
conda activate pareto3dsplat
make train-baseline
```

This runs 30,000 iterations at native resolution, evaluates and saves at 7,000
and 30,000 iterations, disables the network viewer, and writes the run to:

```text
results/baseline/lego/seed_0/
├── cfg_args
├── events.out.tfevents.*
├── input.ply
├── cameras.json
├── chkpnt25000.pth
├── chkpnt30000.pth
├── attempts/
│   └── <UTC timestamp>/
│       ├── baseline.yaml
│       ├── command.sh
│       ├── conda-explicit.txt
│       ├── dataset_checksums.sha256
│       ├── system.txt
│       ├── status.txt
│       └── train.log
└── point_cloud/
    ├── iteration_7000/point_cloud.ply
    └── iteration_30000/point_cloud.ply
```

Resumable checkpoints are created every 5,000 iterations. A background
retention task keeps only the newest two checkpoint files to bound disk usage.
Resume an interrupted run from one of those files with:

```bash
bash scripts/train_baseline.sh \
    --resume results/baseline/lego/seed_0/chkpnt15000.pth
```

The upstream checkpoint captures model and optimizer state, but not every
random-number-generator state. Resuming is suitable for recovery, although it
is not guaranteed to be bit-for-bit identical to uninterrupted training.

The official baseline initializes Python, NumPy, and PyTorch with seed `0`;
`configs/baseline.yaml` records that implementation-defined seed.

## License and Attribution

The dataset is distributed by the NeRF authors. Cite the original NeRF work in
reports and derived results:

> Ben Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance Fields
> for View Synthesis." ECCV 2020.

The project scripts and checksum do not change the dataset's original terms.
