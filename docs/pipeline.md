# Pipeline Diagram

This diagram summarizes the current Pareto-Splat workflow from dataset setup
through portfolio assets.

```mermaid
flowchart LR
    dataset["NeRF Synthetic scene<br/>images + camera poses"]
    config["YAML experiment config"]
    train["Train 3D Gaussian Splatting<br/>30k iterations"]
    render["Render held-out test views"]
    quality["Evaluate quality<br/>PSNR / SSIM / LPIPS"]
    profile["Profile efficiency<br/>FPS / latency / memory / size"]
    prune["Post-training pruning<br/>random / threshold / top-k / visibility"]
    sensitivity["Robustness studies<br/>pose + input sensitivity"]
    summarize["Summary tables + Pareto ranks"]
    demo["Static demo<br/>study + operating-point selector"]
    portfolio["Portfolio assets<br/>images / plots / video"]
    docs["Reports and roadmap"]

    dataset --> train
    config --> train
    train --> render
    render --> quality
    train --> profile
    train --> prune
    prune --> render
    prune --> profile
    dataset --> sensitivity
    sensitivity --> summarize
    quality --> summarize
    profile --> summarize
    summarize --> demo
    summarize --> portfolio
    render --> portfolio
    docs --> portfolio
```

The main quality-efficiency objective used by the Pareto summaries is:

```math
f(x) =
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

For two operating points \(a\) and \(b\), \(a\) dominates \(b\) when:

```math
\forall j,\; f_j(a) \ge f_j(b)
\quad \text{and} \quad
\exists j,\; f_j(a) > f_j(b).
```

The portfolio assets are the presentation layer for this pipeline. They do not
change the experiments; they package already-computed renders, plots, and
videos into a smaller local bundle for review.
