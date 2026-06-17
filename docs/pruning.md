# Gaussian Pruning

Session 9 implements post-training pruning for GraphDeCo-compatible Gaussian
PLY files. The pruner reads a trained `point_cloud.ply`, selects a subset of
Gaussian rows, writes a smaller PLY with the same vertex properties, and saves
JSON metadata next to the output.

## Model

A trained scene contains `N` Gaussians:

```text
G = {g_i}_{i=1}^N
g_i = (x_i, c_i, o_i, s_i, q_i)
```

where `x_i` is position, `c_i` are spherical-harmonic color features, `o_i` is
the raw opacity logit, `s_i` is scale, and `q_i` is rotation. The activated
opacity is:

```text
alpha_i = sigmoid(o_i) = 1 / (1 + exp(-o_i))
```

Pruning defines a binary mask:

```text
m_i in {0, 1}
G' = {g_i : m_i = 1}
N' = sum_i m_i
keep_fraction = N' / N
```

All retained Gaussian attributes are copied unchanged.

## Strategies

Random pruning keeps a uniformly selected fixed-size subset:

```text
k = round(rN)
S ~ Uniform({S subset {1,...,N} : |S| = k})
m_i = 1[i in S]
```

Opacity-threshold pruning keeps Gaussians above an activated opacity threshold:

```text
m_i = 1[alpha_i >= tau]
```

Since PLY opacity is stored as a logit, this is equivalent to:

```text
m_i = 1[o_i >= logit(tau)]
logit(tau) = ln(tau / (1 - tau))
```

Top-k pruning keeps the `k` largest activated opacities:

```text
score_i = alpha_i
m_i = 1[i in top_k(score)]
```

## CLI

Create a pruned Lego model keeping 50% of Gaussians by opacity:

```bash
python scripts/prune_gaussians.py \
  --input results/baseline/lego/seed_0/point_cloud/iteration_30000/point_cloud.ply \
  --output results/pruning/lego/top_k/keep_050/point_cloud/iteration_30000/point_cloud.ply \
  --strategy top-k \
  --keep-fraction 0.5 \
  --source-model-path results/baseline/lego/seed_0 \
  --output-model-path results/pruning/lego/top_k/keep_050
```

The output model can be rendered by pointing the experiment workflow at the
pruned model directory while keeping the original scene dataset path.

For the configured Session 10 grid over multiple pruning levels, use the
workflow and results in [docs/pruning_study.md](pruning_study.md).
