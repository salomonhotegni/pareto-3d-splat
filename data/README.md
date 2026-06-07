# Data

Datasets are stored locally in this directory and are not committed to Git.

The first baseline experiment uses the Lego scene from the official NeRF
Synthetic dataset. Download and validate it with:

```bash
conda activate pareto3dsplat
make dataset
make check-data
```

The installed scene is stored at `data/nerf_synthetic/lego`. The downloader
verifies the official archive checksum and extracts only camera transforms and
RGBA color images; bundled Fern data and auxiliary depth/normal images are not
installed.

See `docs/dataset_notes.md` for provenance, format details, and the baseline
train/test split.
