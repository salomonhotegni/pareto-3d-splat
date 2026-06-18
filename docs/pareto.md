# Pareto Dominance

Session 11 formalizes Pareto objectives and implements non-dominated sorting.
The utilities live in `src/pareto_splat/pareto.py`.

## Objectives

Each objective has a name and an optimization direction. For objective `j`,
define:

```text
s_j = +1  if objective j is maximized
s_j = -1  if objective j is minimized
```

For a point `x`, with raw scalar objective value `f_j(x)`, the oriented value
is:

```text
u_j(x) = s_j f_j(x)
```

After this transformation, larger is always better.

The default objective groups are:

```text
quality:
  maximize psnr
  maximize ssim
  minimize lpips_vgg

efficiency:
  maximize fps
  minimize mean_latency_ms
  minimize serialized_mib
  minimize peak_allocated_mib

quality-efficiency trade-off:
  maximize psnr
  maximize fps
  minimize serialized_mib
```

## Dominance

Point `x` Pareto-dominates point `y` when `x` is no worse than `y` for every
objective and strictly better for at least one objective:

```text
x dominates y
iff
for all j: u_j(x) >= u_j(y)
and
exists j: u_j(x) > u_j(y)
```

Equal points do not dominate each other because neither is strictly better.

## Non-Dominated Sorting

The first Pareto front contains points that are not dominated by any other
point in the set:

```text
F_0 = {x in P : no y in P dominates x}
```

Later fronts are extracted after removing earlier fronts:

```text
F_1 = {x in P \\ F_0 : no y in P \\ F_0 dominates x}
F_2 = ...
```

The implementation returns zero-based ranks:

```text
rank(x) = 0  for points in F_0
rank(x) = 1  for points in F_1
...
```

## Python API

```python
from pareto_splat.pareto import (
    Objective,
    annotate_pareto_ranks,
    non_dominated_sort,
)

objectives = (
    Objective("psnr", "maximize"),
    Objective("fps", "maximize"),
    Objective("serialized_mib", "minimize"),
)

fronts = non_dominated_sort(rows, objectives)
annotated = annotate_pareto_ranks(rows, objectives)
```

For the Session 10 pruning summary, `QUALITY_EFFICIENCY_OBJECTIVES` compares
visual quality, renderer throughput, and serialized model size. This keeps
the first Pareto analysis simple before Session 12 adds front visualizations.
