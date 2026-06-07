# Results

Generated checkpoints, renders, videos, logs, and metric files are stored
locally in this directory and are not committed to Git by default.

Small final tables and selected portfolio assets may be committed later in
dedicated report or asset directories.

The first baseline run uses:

```text
results/baseline/lego/seed_0/
```

The directory contains the baseline configuration snapshot, TensorBoard event
file, training log, input point cloud, camera metadata, and saved Gaussian
point clouds. Later rendering and evaluation steps will add held-out renders
and metric files beneath the same run directory.
