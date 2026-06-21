# Visibility-Aware Importance

Session 13 adds a CPU-only visibility-aware Gaussian importance score in
`src/pareto_splat/visibility.py`. The score is designed as a practical proxy
before adding a full renderer-based importance metric.

## Camera Convention

GraphDeCo writes `cameras.json` with each camera center `C_c`, camera-to-world
rotation `R_c`, image size, and focal lengths. For Gaussian center `x_i`, the
camera-space coordinate is:

```text
p_ic = R_c^T (x_i - C_c)
```

With row-vector NumPy arrays this is implemented as:

```text
p_ic = (x_i - C_c) R_c
```

Projection uses the exported pinhole intrinsics:

```text
u_ic = fx_c * p_ic.x / p_ic.z + W_c / 2
v_ic = fy_c * p_ic.y / p_ic.z + H_c / 2
```

A Gaussian center is counted as visible in camera `c` when:

```text
p_ic.z > 0
0 <= u_ic < W_c
0 <= v_ic < H_c
```

## Score

The depth-weighted visibility proxy is:

```text
V_i = sum_c 1[i is inside camera c] / (z_ic^2 + epsilon)
```

The final importance score combines this proxy with activated opacity:

```text
alpha_i = sigmoid(o_i)
I_i = alpha_i * log(1 + V_i)
```

This favors Gaussians that are both opaque and repeatedly visible from the
available camera set, while avoiding an expensive render pass.

## Python API

```python
from plyfile import PlyData

from pareto_splat.visibility import (
    load_cameras_json,
    visibility_aware_importance,
)

vertices = PlyData.read(point_cloud_path)["vertex"].data
cameras = load_cameras_json(model_path / "cameras.json")
scores = visibility_aware_importance(vertices, cameras)

importance = scores.importance
visibility = scores.visibility
visibility_count = scores.visibility_count
opacity = scores.opacity
```

The returned arrays have one value per Gaussian vertex. Session 14 uses
`importance` through the `visibility-top-k` pruning strategy for matched
Gaussian-budget comparisons.
