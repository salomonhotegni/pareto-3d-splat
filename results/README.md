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
file, input point cloud, camera metadata, saved Gaussian point clouds, and the
two newest resumable checkpoints. Each training or resume attempt has a
timestamped `attempts/` directory containing its exact command, environment
metadata, dataset checksums, log, duration, and exit status. Later rendering
and evaluation steps will add held-out renders and metric files beneath the
same run directory.
