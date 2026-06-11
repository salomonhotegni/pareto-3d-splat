# NeRF Synthetic Drums Dataset

## Selection

Session 8 uses Drums as the second baseline scene. It preserves the exact
NeRF Synthetic camera, image, and split format used by Lego while adding thin
geometry, strong highlights, and reflective materials.

The source is the full synthetic archive linked by the official NeRF
repository:

- Official data folder:
  `https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4`
- Archive file: `nerf_synthetic.zip`
- Google Drive file ID: `1OsiBs2udl32-1CqTXCitmov4NQCYdA9g`
- Archive size: 1,266,782,178 bytes
- SHA-256:
  `554fed5d4884028f95e0da422f99f5fc071fffefc9f2fc72c5f9361801df2599`
- Local scene: `data/nerf_synthetic/drums`

The archive contains all eight synthetic scenes. The project caches the
verified archive under `data/.downloads/` and extracts only Drums poses and
RGBA color images.

## Format

Drums follows the official NeRF Synthetic contract:

```text
data/nerf_synthetic/drums/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/   # 100 RGBA PNG images
├── val/     # 100 RGBA PNG images
└── test/    # 200 RGBA PNG images
```

Images are 800 x 800. The downloader excludes bundled depth and normal maps,
then runs the same strict validation used for Lego.

## Commands

Prepare the scene:

```bash
make dataset-drums
make check-data-drums
```

Run the complete baseline workflow:

```bash
make train-baseline CONFIG=configs/drums.yaml
make render-baseline CONFIG=configs/drums.yaml
make evaluate-baseline CONFIG=configs/drums.yaml
make profile-baseline CONFIG=configs/drums.yaml
```

Outputs are stored under:

```text
results/baseline/drums/seed_0/
```

## License and Attribution

The archive README attributes the Drums Blender model to Bryan Jones under
CC-BY and identifies the rendered dataset as part of the NeRF paper:

Ben Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance Fields
for View Synthesis." ECCV 2020.
