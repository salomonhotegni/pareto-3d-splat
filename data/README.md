# Data

Datasets are stored locally in this directory and are not committed to Git.

The baseline experiments use scenes from the official NeRF Synthetic dataset.
Download and validate Lego with:

```bash
conda activate pareto3dsplat
make dataset
make check-data
```

Download and validate the second Drums scene with:

```bash
make dataset-drums
make check-data-drums
```

Installed scenes are stored under `data/nerf_synthetic/`. The downloaders
verify official archive checksums and extract only camera transforms and RGBA
color images; auxiliary depth and normal images are not installed.

See `docs/dataset_notes.md` and `docs/drums_dataset.md` for provenance, format,
license, and split details.
