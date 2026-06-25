# Static Pareto Demo

Session 19 adds a lightweight static demo for selecting experiment studies and
Pareto operating points. The demo is generated from existing summary JSON
files, so it does not rerun training, rendering, evaluation, or profiling.

## What It Shows

The generated page lets a viewer:

- choose an experiment study, such as the Lego pruning study or importance
  ablation;
- select an operating point, such as `baseline`, `top_k_keep_075`, or
  `visibility_top_k_keep_050_opacity_visibility`;
- inspect PSNR, FPS, serialized model size, Gaussian count, keep fraction, and
  PSNR drop when those fields exist in the source summary;
- view rank-0 Pareto points in a PSNR-vs-FPS scatter plot when the study
  includes efficiency metrics;
- inspect robustness-only studies, such as pose and input sensitivity, even
  when they do not define FPS/model-size objectives.

The default Pareto objective vector is:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

The first two objectives are maximized. The negative size term means smaller
serialized models are preferred:

```math
\mathrm{SizeMiB}(a) < \mathrm{SizeMiB}(b)
\quad \Longleftrightarrow \quad
-\mathrm{SizeMiB}(a) > -\mathrm{SizeMiB}(b).
```

The demo does not recompute Pareto ranks in the browser. It displays the
`pareto_rank` fields already written by the pruning-study summary workflow.

## Generate the Demo

Build the static page with:

```bash
make demo
```

By default, this writes:

```text
results/demo/index.html
```

The output is a self-contained HTML file with embedded JSON, CSS, and
JavaScript. Open it in a browser from the local filesystem. The `results/`
directory is ignored by Git, so regenerate the page after new summaries are
created.

To write somewhere else:

```bash
make demo DEMO_OUTPUT=/tmp/pareto_splat_demo.html
```

The underlying script is:

```bash
python scripts/build_demo.py --output results/demo/index.html
```

By default, it includes every matching summary:

```text
results/**/summary/summary.json
```

For a smaller demo, pass explicit summaries:

```bash
python scripts/build_demo.py \
  --summary results/pruning/lego/study_30000/summary/summary.json \
  --summary results/importance_ablation/lego/study_30000/summary/summary.json \
  --output results/demo/index.html
```

## Implementation Notes

The reusable implementation lives in `src/pareto_splat/demo.py`. It has three
small responsibilities:

1. discover available `summary.json` files;
2. normalize scalar row fields into a browser-friendly payload;
3. write a standalone HTML document.

The generated page intentionally avoids Streamlit, Gradio, or a local server.
That keeps the demo portable for portfolio review: the only committed code is
the generator, and the produced HTML can be rebuilt from local experiment
artifacts.
