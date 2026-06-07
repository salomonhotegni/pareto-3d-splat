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

After rendering the held-out test orbit, create a labeled ground-truth versus
3DGS comparison video with:

```bash
make comparison-video
```

The generated MP4 and its `ffprobe` metadata are stored under
`results/baseline/lego/seed_0/videos/`.

Evaluate all 200 rendered test views with:

```bash
make evaluate-baseline
```

Aggregate metrics, per-view metrics, the exact command, and the evaluation log
are stored under `results/baseline/lego/seed_0/metrics/ours_30000/`.
