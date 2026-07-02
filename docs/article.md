# Pareto-Splat: A Detailed Guide to Quality-Efficiency Trade-offs in 3D Gaussian Splatting

## Who This Article Is For

This article explains Pareto-Splat to a reader who understands standard
machine learning ideas such as training and test splits, gradient-based
optimization, image reconstruction losses, and evaluation metrics, but who may
not yet know how 3D Gaussian Splatting works.

The project is not a new 3D reconstruction model trained from scratch. It uses
the official GraphDeCo 3D Gaussian Splatting implementation as a pinned
baseline, then builds an experimental system around it. The original work in
this repository is the reproducible workflow, model measurement, post-training
Gaussian pruning, visibility-based importance experiments, Pareto analysis,
robustness studies, and presentation tooling.

The main question is:

```math
\text{How can a trained 3DGS scene be made cheaper to render and store}
```

```math
\text{without losing an unacceptable amount of novel-view quality?}
```

That wording matters. Compression is not treated as automatically beneficial.
It creates several competing outcomes, so the project measures the full
quality-efficiency trade-off.

---

## 1. A Mental Model of 3D Gaussian Splatting

### 1.1 From pixels to a continuous 3D representation

Suppose we have photographs of a static scene from known camera poses. A
traditional point cloud stores 3D points, but points alone have no spatial
extent and do not directly explain how color should be rendered into a new
camera.

3D Gaussian Splatting, abbreviated 3DGS, represents the scene with a set of
anisotropic 3D Gaussians:

```math
G = \{g_i\}_{i=1}^{N}.
```

Each Gaussian \(g_i\) contains approximately the following learned
information:

```math
g_i =
\left(
\boldsymbol{\mu}_i,
\boldsymbol{\Sigma}_i,
\alpha_i,
\mathbf{c}_i
\right),
```

where:

- $\boldsymbol{\mu}_i \in ℝ^3$ is the 3D center;
- $\boldsymbol{\Sigma}_i \in ℝ^{3 \times 3}$ is the 3D covariance,
  which controls size, orientation, and anisotropic shape;
- $\alpha_i \in (0,1)$ is opacity;
- $\mathbf{c}_i$ contains view-dependent color coefficients, represented by
  spherical harmonics in the GraphDeCo model.

In the serialized GraphDeCo PLY, the covariance is represented through learned
scale and rotation fields rather than stored as one unrestricted matrix. At a
conceptual level, it can be written as:

```math
\boldsymbol{\Sigma}_i
=
\mathbf{R}_i
\mathbf{S}_i
\mathbf{S}_i^{\mathsf{T}}
\mathbf{R}_i^{\mathsf{T}},
```

where $\mathbf{R}_i$ is derived from a learned rotation and
$\mathbf{S}_i$ is a scale matrix. This parameterization keeps the covariance
valid while allowing elongated, rotated Gaussians.

The PLY stores opacity as a raw logit $o_i$. Its activated opacity is:

```math
\alpha_i
=
\sigma(o_i)
=
\frac{1}{1+\exp(-o_i)}.
```

This distinction becomes important during pruning because a threshold such as
$\alpha_i \ge 0.1$ must be interpreted in probability space, even though
the file contains logits.

### 1.2 Why "splatting" is fast

For a new camera, each 3D Gaussian is projected into image space, producing an
elliptical 2D footprint. The rasterizer determines which pixels overlap that
footprint, orders contributions by depth, and alpha-composites their colors.

For pixel $p$, a simplified front-to-back compositing equation is:

```math
\mathbf{C}(p)
=
\sum_{i=1}^{M(p)}
T_i(p)\,
a_i(p)\,
\mathbf{c}_i(p),
```

with transmittance:

```math
T_i(p)
=
\prod_{j=1}^{i-1}
\left(1-a_j(p)\right).
```

Here:

- $M(p)$ is the number of projected Gaussians affecting pixel $p$;
- $a_i(p)$ combines learned opacity with the value of the projected Gaussian
  footprint at the pixel;
- $\mathbf{c}_i(p)$ is the view-dependent color;
- $T_i(p)$ describes how much light remains after earlier Gaussians.

This differentiable rasterization process is efficient on a GPU because it
uses explicit primitives rather than evaluating a large neural network at
many samples along every ray.

### 1.3 What training does

The pinned GraphDeCo implementation optimizes Gaussian attributes so rendered
training views match observed images. It also manages the representation over
time by creating, splitting, and removing Gaussians according to its standard
training procedure.

Pareto-Splat does not replace this optimizer. It invokes the pinned
implementation with a controlled configuration:

- 30,000 training iterations;
- spherical-harmonic degree 3;
- white-background NeRF Synthetic data;
- explicit evaluation, save, and checkpoint iterations;
- seed 0;
- viewer disabled for non-interactive runs.

The result of training is a GraphDeCo-compatible model whose central artifact
is:

```text
point_cloud/iteration_30000/point_cloud.ply
```

Pareto-Splat begins its compression analysis after this model has already been
trained.

---

## 2. Why This Is a Multi-Objective Problem

For a trained or pruned model variant $x$, the project observes several
quantities:

- reconstruction quality;
- renderer latency and throughput;
- number of Gaussians;
- serialized model size;
- allocated GPU memory.

These objectives conflict. Keeping more Gaussians often preserves fine detail,
but increases the amount of projection, sorting, and compositing work. Removing
Gaussians may improve speed and size while introducing holes, blur, or missing
view-dependent effects.

A scalar objective could combine everything:

```math
J(x)
=
\lambda_q Q(x)
+ \lambda_s S(x)
+ \lambda_m M(x),
```

but the weights $\lambda_q,\lambda_s,\lambda_m$ encode a deployment decision
before the experiments have even shown the available trade-offs. They also
depend on metric scale. A one-unit change in FPS is not directly comparable to
a one-unit change in PSNR or MiB.

Pareto-Splat therefore uses a vector objective:

```math
f(x)
=
\left[
\mathrm{PSNR}(x),
\mathrm{FPS}(x),
-\mathrm{SizeMiB}(x)
\right].
```

PSNR and FPS are maximized. Size is minimized, so it is negated in this
all-maximization notation.

This formulation does not declare one universal winner. It identifies models
that remain rational choices under different deployment constraints.

---

## 3. The End-to-End Experimental System

The project organizes the work as a sequence of stages:

```text
dataset
  -> validate
  -> train
  -> render held-out views
  -> evaluate image quality
  -> profile renderer efficiency
  -> prune the trained model
  -> render/evaluate/profile each pruned model
  -> summarize and compute Pareto fronts
  -> run ablations and robustness studies
  -> generate demo and portfolio artifacts
```

Each stage has a narrow responsibility. This separation is useful for both
scientific validity and practical recovery:

- training does not silently include evaluation assumptions;
- rendering can be rerun without retraining;
- metrics can be recomputed from saved images;
- profiling excludes image encoding and storage;
- pruning can be compared at fixed budgets;
- summaries are derived from machine-readable artifacts.

The default Make commands are wrappers around configuration-driven Python and
Bash entry points. For example:

```bash
make train-baseline CONFIG=configs/baseline.yaml
make render-baseline CONFIG=configs/baseline.yaml
make evaluate-baseline CONFIG=configs/baseline.yaml
make profile-baseline CONFIG=configs/baseline.yaml
```

The same workflow is reused for Lego and Drums by changing the YAML file rather
than editing scripts.

---

## 4. Dataset Handling and a Crucial Alpha-Compositing Fix

### 4.1 Dataset splits

The experiments use NeRF Synthetic scenes. Each scene contains:

- 100 training cameras;
- 100 validation cameras;
- 200 test cameras;
- 800 by 800 RGBA PNG images;
- camera-to-world transforms;
- a shared horizontal field of view.

The test split is held out from training and is used for the final novel-view
quality measurements.

Dataset scripts check the split sizes, metadata shape, image dimensions, and
RGBA format before expensive GPU work starts. This catches failures such as an
incomplete extraction or a malformed transform file.

### 4.2 Why RGBA must be treated carefully

NeRF Synthetic foreground objects are stored with an alpha channel. The RGB
values in transparent pixels are not, by themselves, the final image that
should be compared against a white-background render.

For foreground color $\mathbf{F}(p)$, alpha $A(p)$, and background color
$\mathbf{B}$, the correct composite is:

```math
\mathbf{I}(p)
=
A(p)\mathbf{F}(p)
+
\left(1-A(p)\right)\mathbf{B}.
```

With the configured white background:

```math
\mathbf{B} = [1,1,1].
```

The pinned upstream loader required a compatibility patch so NeRF Synthetic
RGBA images were composited consistently before entering the GraphDeCo camera
pipeline. Without this correction, training or evaluation could compare
different background conventions and produce misleading metrics.

The compatibility layer is intentionally narrow. It patches only the
NeRF Synthetic loading path and delegates non-synthetic inputs to the original
loader.

---

## 5. Reproducible Training and Provenance

Expensive GPU experiments are difficult to trust if a result cannot be traced
to its code, configuration, data, and machine. Pareto-Splat treats provenance
as part of the method.

### 5.1 Configuration contract

The YAML configuration defines:

- experiment name and seed;
- pinned baseline path;
- source dataset and model output paths;
- expected split sizes and image resolution;
- background convention and spherical-harmonic degree;
- training duration;
- evaluation, save, and checkpoint schedules;
- checkpoint retention;
- render iteration;
- metric device;
- profiling warm-up and repetition counts.

Paths are resolved relative to the project root and validated to prevent
accidental escape into unrelated directories. Iteration lists must be sorted,
unique, positive, and within the configured training duration. Rendering must
request an iteration that training actually saved.

### 5.2 Per-attempt metadata

Before training, the runner validates the dataset and creates a timestamped
attempt directory. It records:

- a copy of the selected YAML;
- the exact executable command;
- SHA-256 checksums of train, validation, and test transform files;
- project Git commit and dirty-state flag;
- pinned GraphDeCo commit;
- Python and PyTorch versions;
- CUDA runtime, GPU model, driver, and memory;
- host information;
- explicit Conda package state when available;
- training log;
- start time, end time, duration, exit code, and completion status.

This design answers practical questions later:

- Was the run produced by committed code?
- Did two runs use the same camera metadata?
- Which CUDA and GPU environment produced the profile?
- Was training completed or interrupted?
- What exact command can reproduce the attempt?

### 5.3 Checkpoint recovery without unbounded storage

Training saves checkpoints at regular intervals. A monitor retains only the
newest configured number of resumable checkpoints and logs removals. This
balances failure recovery against large checkpoint storage costs.

A resume must point to a correctly named checkpoint inside the experiment's
model directory. This prevents accidentally continuing from an unrelated
scene or output tree.

---

## 6. Measuring Image Quality

After training or pruning, the model renders all 200 held-out test cameras.
Render and ground-truth PNGs must have identical filenames, counts, dimensions,
and RGB format. The evaluator rejects incomplete or mismatched sets rather than
quietly averaging whatever files happen to exist.

### 6.1 PSNR

For a rendered image $\hat{\mathbf{I}}$ and reference image
$\mathbf{I}$, both scaled to $[0,1]$, mean squared error is:

```math
\mathrm{MSE}
=
\frac{1}{3HW}
\sum_{c=1}^{3}
\sum_{u=1}^{W}
\sum_{v=1}^{H}
\left(
\hat{I}_{cuv} - I_{cuv}
\right)^2.
```

Peak signal-to-noise ratio is:

```math
\mathrm{PSNR}
=
10\log_{10}
\left(
\frac{1}{\mathrm{MSE}}
\right)
=
20\log_{10}
\left(
\frac{1}{\sqrt{\mathrm{MSE}}}
\right).
```

Higher is better. Because PSNR compares corresponding pixels, it is highly
sensitive to spatial misalignment. That sensitivity becomes important in the
camera-pose study.

### 6.2 SSIM

Structural similarity compares local luminance, contrast, and covariance. In
simplified form:

```math
\mathrm{SSIM}(x,y)
=
\frac{
\left(2\mu_x\mu_y+C_1\right)
\left(2\sigma_{xy}+C_2\right)
}{
\left(\mu_x^2+\mu_y^2+C_1\right)
\left(\sigma_x^2+\sigma_y^2+C_2\right)
}.
```

The implementation follows GraphDeCo's RGB convention using:

- an 11 by 11 Gaussian window;
- Gaussian standard deviation 1.5;
- zero-padded image boundaries;
- $C_1=0.01^2$;
- $C_2=0.03^2$.

Higher SSIM is better.

### 6.3 LPIPS-VGG

LPIPS compares deep feature activations rather than only raw pixels. The
project uses:

- LPIPS version 0.1.4;
- VGG feature trunk;
- LPIPS model version 0.1;
- normalized RGB inputs.

Lower LPIPS is better. It provides a perceptual complement to PSNR and SSIM,
but it still does not prove geometric correctness or temporal stability.

### 6.4 Aggregation

The evaluator stores per-view values, then reports the mean, population
standard deviation, minimum, and maximum over all 200 views. Keeping per-view
data is important because an aggregate average can hide a small number of
severe failures.

---

## 7. Measuring Rendering Efficiency

Image quality and speed are measured in separate stages. The profiler loads
the scene, warms up the GPU, and times only the renderer call.

### 7.1 Timing protocol

The baseline configuration uses:

- 10 warm-up views;
- 200 held-out test views;
- 3 complete repetitions;
- 600 measured render calls.

CUDA events are recorded immediately before and after
`gaussian_renderer.render`. The profiler synchronizes the GPU before reading
elapsed times. This avoids the common mistake of measuring only asynchronous
CPU launch overhead.

For latency samples $\ell_1,\ldots,\ell_K$, mean latency is:

```math
\bar{\ell}
=
\frac{1}{K}
\sum_{k=1}^{K}\ell_k.
```

Renderer throughput is defined as:

```math
\mathrm{FPS}
=
\frac{1000}{\bar{\ell}},
```

when latency is measured in milliseconds.

The profiler also records median, linearly interpolated p95, population
standard deviation, minimum, and maximum latency.

### 7.2 What FPS includes and excludes

The measurement includes the GPU work inside the renderer call. It excludes:

- scene loading;
- PNG encoding;
- disk writes;
- data transfer for a networked viewer;
- user-interface overhead;
- application logic.

The result is therefore renderer throughput, not guaranteed end-to-end product
FPS.

### 7.3 Memory and model size

The profiler records:

- PLY vertex count;
- serialized PLY bytes and MiB;
- in-memory Gaussian parameter bytes;
- allocated memory before scene loading;
- allocated memory after loading;
- measurement baseline allocation;
- peak allocated and reserved memory;
- incremental peak caused during measured rendering.

The PLY Gaussian count is cross-checked against the number loaded by the
GraphDeCo model. A mismatch fails the profile because it would indicate that
the file and runtime representation are not the same experiment.

---

## 8. Post-Training Gaussian Pruning

### 8.1 What pruning changes

The pruner reads the structured vertex table from a trained
`point_cloud.ply`. It builds a binary selection mask:

```math
m_i \in \{0,1\},
```

and creates:

```math
G'
=
\{g_i \mid m_i=1\}.
```

The retained count and fraction are:

```math
N'
=
\sum_{i=1}^{N}m_i,
\qquad
r
=
\frac{N'}{N}.
```

All properties of retained Gaussians are copied unchanged. The project does
not modify their positions, scales, rotations, colors, or opacity values, and
does not fine-tune the pruned model. This isolates the effect of selection.

The output remains GraphDeCo-compatible:

- the complete PLY vertex schema is preserved;
- byte order and PLY metadata are preserved;
- lightweight model metadata is copied;
- pruning parameters and counts are written to JSON.

### 8.2 Random fixed-budget pruning

For target retention fraction \(r\):

```math
k
=
\operatorname{round}(rN).
```

Random pruning samples a size-\(k\) subset without replacement:

```math
S
\sim
\operatorname{Uniform}
\left(
\{S \subseteq \{1,\ldots,N\}: |S|=k\}
\right).
```

Then:

```math
m_i
=
\mathbf{1}[i \in S].
```

The NumPy random generator is seeded, so the subset is deterministic for a
given seed. Random pruning is not expected to be a strong method. It is a
control that asks whether an importance rule does better than merely reducing
the count.

### 8.3 Opacity-threshold pruning

Threshold pruning keeps every Gaussian whose activated opacity exceeds
\(\tau\):

```math
m_i
=
\mathbf{1}[\alpha_i \ge \tau].
```

Because the PLY stores logits:

```math
\alpha_i \ge \tau
\iff
o_i
\ge
\log
\left(
\frac{\tau}{1-\tau}
\right).
```

Thresholding is intuitive, but it does not guarantee a fixed model size.
Different scenes or checkpoints may have different opacity distributions.

### 8.4 Opacity top-k pruning

Opacity top-k scores each Gaussian by:

```math
s_i = \alpha_i.
```

It keeps exactly the \(k\) largest scores:

```math
m_i
=
\mathbf{1}
\left[
i \in \operatorname{TopK}(\{s_j\}_{j=1}^{N},k)
\right].
```

The implementation uses a stable sort. A stable ordering makes ties
deterministic relative to the original PLY order.

Since sigmoid is monotonic, ranking by activated opacity gives the same order
as ranking by raw logits. Activation is still useful conceptually and for
thresholds because it expresses opacity on the familiar \((0,1)\) scale.

Top-k is particularly suitable for matched-budget studies because random and
importance-based methods retain exactly the same number of Gaussians.

---

## 9. Visibility-Aware Importance

Opacity is local to a Gaussian. It does not explicitly say how many cameras
observe that Gaussian or how close it is to them. The project therefore tests
a CPU-only geometric proxy.

### 9.1 Camera projection

For camera \(c\), GraphDeCo exports:

- camera center \(\mathbf{C}_c\);
- camera-to-world rotation \(\mathbf{R}_c\);
- image width \(W_c\) and height \(H_c\);
- focal lengths \(f_{x,c}\) and \(f_{y,c}\).

For Gaussian center \(\mathbf{x}_i\), camera-space coordinates are:

```math
\mathbf{p}_{ic}
=
\mathbf{R}_c^{\mathsf{T}}
\left(
\mathbf{x}_i-\mathbf{C}_c
\right).
```

The implementation stores points as row vectors, so the equivalent operation
is:

```math
\mathbf{p}_{ic}^{\mathsf{row}}
=
\left(
\mathbf{x}_i-\mathbf{C}_c
\right)^{\mathsf{row}}
\mathbf{R}_c.
```

Perspective projection is:

```math
u_{ic}
=
f_{x,c}
\frac{p_{ic,x}}{p_{ic,z}}
+
\frac{W_c}{2},
```

```math
v_{ic}
=
f_{y,c}
\frac{p_{ic,y}}{p_{ic,z}}
+
\frac{H_c}{2}.
```

The center is considered inside the camera frustum when:

```math
m_{ic}
=
\mathbf{1}
\left[
p_{ic,z} > 10^{-6},
\;
0 \le u_{ic} < W_c,
\;
0 \le v_{ic} < H_c
\right].
```

### 9.2 Visibility proxies

The unweighted visibility count is:

```math
C_i
=
\sum_c m_{ic}.
```

The depth-weighted proxy is:

```math
V_i
=
\sum_c
\frac{m_{ic}}
{p_{ic,z}^2+\epsilon},
\qquad
\epsilon=10^{-6}.
```

Inverse squared depth gives more weight to cameras where the center is closer.
The logarithm later compresses large dynamic ranges.

### 9.3 Tested scores

The ablation compares:

```math
s_i^{\mathrm{opacity}}
=
\alpha_i,
```

```math
s_i^{\mathrm{opacity\_visibility}}
=
\alpha_i\log(1+V_i),
```

```math
s_i^{\mathrm{visibility}}
=
\log(1+V_i),
```

```math
s_i^{\mathrm{opacity\_count}}
=
\alpha_i\log(1+C_i).
```

Each mode is used only to rank Gaussians. At a given retention level, the final
count remains fixed.

### 9.4 What this proxy does not measure

The method checks only whether a Gaussian center projects into an image. It
does not evaluate:

- the projected ellipse's screen-space area;
- whether another Gaussian occludes it;
- its actual transmittance-weighted compositing contribution;
- view-dependent color importance;
- reconstruction residual around its footprint;
- gradient magnitude;
- whether its center is outside while part of its footprint overlaps the
  image.

This distinction explains why "visible to many cameras" can still be a weak
definition of importance.

---

## 10. Matched-Budget Experimental Design

A fair comparison between pruning scores needs the same resource budget. If
one method keeps 100,000 Gaussians and another keeps 200,000, quality
differences cannot be attributed only to ranking.

Pareto-Splat evaluates fixed retention fractions:

```math
r \in \{0.25, 0.50, 0.75\}.
```

For each fixed-budget variant:

1. load the same trained Lego PLY;
2. compute the strategy's mask;
3. write an unchanged subset of Gaussian rows;
4. render the same 200 held-out cameras;
5. evaluate against the same ground-truth images;
6. profile with the same warm-up and repetition settings;
7. collect quality, speed, memory, count, and size;
8. compute Pareto ranks.

This is a post-training study. No method gets an additional optimization stage
after pruning.

The study runner validates configuration names, paths, strategies, budgets,
and expected artifacts. It can execute all variants or one named variant,
which is useful when a long GPU stage needs to be resumed.

---

## 11. Pareto Dominance and Non-Dominated Sorting

### 11.1 Orienting objectives

The implementation represents each objective with a name and a direction. Let:

```math
d_j
=
\begin{cases}
+1, & \text{if objective }j\text{ is maximized},\\
-1, & \text{if objective }j\text{ is minimized}.
\end{cases}
```

For raw objective value \(f_j(x)\), define:

```math
u_j(x)
=
d_j f_j(x).
```

Now larger is always better.

The library supports objective groups beyond the default front:

- quality: maximize PSNR and SSIM, minimize LPIPS;
- efficiency: maximize FPS, minimize latency, size, and peak memory;
- default quality-efficiency: maximize PSNR and FPS, minimize serialized size.

### 11.2 Dominance

Variant \(a\) dominates variant \(b\) when:

```math
\forall j,\;
u_j(a) \ge u_j(b)
```

and:

```math
\exists j,\;
u_j(a) > u_j(b).
```

The strict condition means equal points do not dominate each other.

The implementation also accepts a non-negative numerical tolerance
\(\delta\). Under tolerance, \(a\) is considered no worse when:

```math
u_j(a)
\ge
u_j(b)-\delta,
```

and strictly better only when:

```math
u_j(a)
>
u_j(b)+\delta.
```

The reported studies use the declared comparison behavior without claiming
that negligible floating-point differences are scientifically meaningful.

### 11.3 Pareto fronts

The first front contains every point not dominated by another:

```math
F_0
=
\{
x \in P
\mid
\nexists y \in P:
y \succ x
\}.
```

Remove \(F_0\), then repeat:

```math
F_1
=
\{
x \in P\setminus F_0
\mid
\nexists y \in P\setminus F_0:
y \succ x
\}.
```

This produces zero-based ranks:

```math
\operatorname{rank}(x)=k
\quad \text{when} \quad
x \in F_k.
```

The repository uses a direct pairwise implementation. That is appropriate for
the small number of experiment variants and has the advantage of transparent,
testable behavior.

### 11.4 How to interpret rank 0

Rank 0 does not mean "best in every way." It means no observed alternative is
at least as good in every selected objective and strictly better in one.

The selected objective set is part of the claim. Adding LPIPS, peak memory,
training cost, or worst-view quality can change the front. A 2D projection can
also make one point appear dominated even though its third objective keeps it
non-dominated in 3D.

---

## 12. Camera-Pose Sensitivity

The clean test metrics assume accurate camera calibration. Session 15 tests
that assumption while keeping the trained Gaussian model fixed.

### 12.1 Pose perturbation model

Each test camera has camera-to-world transform:

```math
\mathbf{T}_i
=
\begin{bmatrix}
\mathbf{R}_i & \mathbf{t}_i\\
\mathbf{0}^{\mathsf{T}} & 1
\end{bmatrix}.
```

A perturbed pose is:

```math
\mathbf{T}'_i
=
\begin{bmatrix}
\Delta\mathbf{R}_i\mathbf{R}_i
&
\mathbf{t}_i+\boldsymbol{\epsilon}_i\\
\mathbf{0}^{\mathsf{T}} & 1
\end{bmatrix}.
```

Rotation noise is sampled as a three-dimensional rotation vector:

```math
\boldsymbol{\omega}_i
\sim
\mathcal{N}
\left(
\mathbf{0},
\sigma_R^2\mathbf{I}
\right).
```

It is converted to a valid rotation matrix through the exponential map:

```math
\Delta\mathbf{R}_i
=
\exp
\left(
[\boldsymbol{\omega}_i]_{\times}
\right),
```

where the skew-symmetric matrix is:

```math
[\boldsymbol{\omega}]_{\times}
=
\begin{bmatrix}
0 & -\omega_z & \omega_y\\
\omega_z & 0 & -\omega_x\\
-\omega_y & \omega_x & 0
\end{bmatrix}.
```

Translation noise is:

```math
\boldsymbol{\epsilon}_i
\sim
\mathcal{N}
\left(
\mathbf{0},
\sigma_t^2\mathbf{I}
\right).
```

Perturbations are deterministic for a fixed seed.

### 12.2 Controlled comparison

Only test transform files change. The same trained point cloud is linked into
separate output directories, so no retraining or model mutation can confound
the result.

The perturbed pose renders are compared against the original clean test
images. This asks:

```math
\text{How much does evaluation degrade when the renderer uses the wrong pose?}
```

Quality degradation is reported as:

```math
\Delta\mathrm{PSNR}
=
\mathrm{PSNR}_{\mathrm{baseline}}
-
\mathrm{PSNR}_{\mathrm{perturbed}},
```

```math
\Delta\mathrm{LPIPS}
=
\mathrm{LPIPS}_{\mathrm{perturbed}}
-
\mathrm{LPIPS}_{\mathrm{baseline}}.
```

Positive values indicate degradation.

---

## 13. Training-Input Sensitivity

Session 16 asks a different robustness question. Instead of perturbing test
poses for a fixed model, it changes the training observations and retrains each
variant.

The clean training set is:

```math
D_{\mathrm{train}}
=
\{
(\mathbf{I}_i,\mathbf{T}_i)
\}_{i=1}^{N}.
```

A modified training set is:

```math
D'_{\mathrm{train}}
=
\{
(\mathbf{I}'_i,\mathbf{T}_i)
\}_{i \in S}.
```

Camera poses remain unchanged.

### 13.1 Gaussian image noise

For RGB channels:

```math
\mathbf{I}'_i
=
\operatorname{clip}
\left(
\mathbf{I}_i+\boldsymbol{\eta}_i,
0,1
\right),
```

with:

```math
\boldsymbol{\eta}_i
\sim
\mathcal{N}
\left(
\mathbf{0},
\sigma^2\mathbf{I}
\right).
```

The alpha channel is preserved.

### 13.2 Blur

The RGB image is transformed with a Gaussian blur:

```math
\mathbf{I}'_i
=
\operatorname{GaussianBlur}
\left(
\mathbf{I}_i,r
\right).
```

This removes high-frequency supervision while preserving camera geometry and
alpha.

### 13.3 Brightness shift

The RGB image is scaled and clipped:

```math
\mathbf{I}'_i
=
\operatorname{clip}
\left(
\beta\mathbf{I}_i,
0,1
\right).
```

Training images become systematically darker or brighter, while validation
and test images remain clean. The experiment therefore measures train-test
appearance mismatch.

### 13.4 Fewer views

For \(k<N\), the method selects deterministic, approximately evenly spaced
indices:

```math
S_k
=
\left\{
\operatorname{round}
\left(
\frac{j(N-1)}{k-1}
\right)
\;\middle|\;
j=0,\ldots,k-1
\right\}.
```

This preserves broad trajectory coverage better than simply taking the first
\(k\) cameras.

Each variant is trained for the same 30,000 iterations and evaluated on the
same clean 200-view test split.

---

## 14. Main Experimental Results

### 14.1 Clean baselines

The Lego baseline is:

| Metric | Lego |
| --- | ---: |
| PSNR | 35.9166 dB |
| SSIM | 0.983729 |
| LPIPS-VGG | 0.019002 |
| Gaussians | 299,799 |
| Serialized model | 70.91 MiB |
| Mean renderer latency | 3.534 ms |
| Renderer throughput | 282.98 FPS |
| Peak allocated GPU memory | 282.92 MiB |

The Drums baseline is:

| Metric | Drums |
| --- | ---: |
| PSNR | 26.1724 dB |
| SSIM | 0.955651 |
| LPIPS-VGG | 0.043743 |
| Gaussians | 318,647 |
| Serialized model | 75.37 MiB |
| Mean renderer latency | 3.284 ms |
| Renderer throughput | 304.49 FPS |

Drums verifies the clean workflow on a second scene. Lego and Drums used
different A100 memory variants, so their FPS values are not a controlled
cross-scene speed comparison.

### 14.2 Opacity top-k trade-off

| Variant | Keep | Gaussians | PSNR | SSIM | LPIPS-VGG | FPS | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100% | 299,799 | 35.917 | 0.983729 | 0.019002 | 282.98 | 70.91 MiB |
| opacity top-k | 75% | 224,849 | 34.496 | 0.980445 | 0.021636 | 456.45 | 53.18 MiB |
| opacity top-k | 50% | 149,900 | 29.169 | 0.957937 | 0.041038 | 635.44 | 35.45 MiB |
| opacity top-k | 25% | 74,950 | 22.377 | 0.877730 | 0.102224 | 895.98 | 17.73 MiB |

The 75% point removes:

```math
299{,}799 - 224{,}849
=
74{,}950
```

Gaussians, a 25% reduction. Its model size also drops by approximately 25%:

```math
\frac{70.91-53.18}{70.91}
\times 100
\approx
25.0\%.
```

Renderer throughput increases by:

```math
\left(
\frac{456.45}{282.98}-1
\right)
\times 100
\approx
61.3\%.
```

The quality cost is:

```math
35.917-34.496
=
1.421\ \mathrm{dB}.
```

This makes 75% opacity top-k the strongest demonstrated compression point in
the current Lego study. "Strongest demonstrated" is deliberately narrower
than "universally optimal."

At 25% retention, throughput reaches 895.98 FPS and size falls to 17.73 MiB,
but:

```math
\Delta\mathrm{PSNR}
=
35.917-22.377
=
13.540\ \mathrm{dB}.
```

That is a major quality loss. The point helps show the frontier's shape but is
not a high-fidelity replacement for the baseline.

Opacity top-k also outperforms random pruning at matched 25%, 50%, and 75%
budgets in both quality and measured renderer throughput. This demonstrates
that selection matters, not only count reduction.

### 14.3 Visibility ablation

| Keep | Visibility-only PSNR | Opacity top-k PSNR |
| ---: | ---: | ---: |
| 25% | 10.363 dB | 22.377 dB |
| 50% | 11.824 dB | 29.169 dB |
| 75% | 15.259 dB | 34.496 dB |

Raw visibility fails dramatically. Opacity-weighted visibility remains close
to opacity top-k, but does not improve it in the tested setting.

The `opacity_count` score produces the same quality as opacity top-k in this
run. That suggests the count term does not materially change the ranking at
the tested budgets.

### 14.4 Pose sensitivity

| Variant | Mean rotation | Mean translation norm | PSNR | PSNR drop |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0000 deg | 0.000000 | 35.9166 | 0.0000 |
| rotation | 0.3994 deg | 0.000000 | 19.0981 | 16.8185 |
| translation | 0.0000 deg | 0.007987 | 24.5416 | 11.3750 |
| combined | 0.7876 deg | 0.015463 | 16.5437 | 19.3728 |

The large PSNR loss is consistent with a pixel-aligned metric comparing an
image rendered from a perturbed camera against a reference from the original
camera. It also has a practical meaning: accurate poses are a hard assumption
for the current pipeline.

### 14.5 Training-input sensitivity

| Variant | Training change | PSNR | PSNR drop |
| --- | --- | ---: | ---: |
| baseline | clean, 100 views | 35.92 | 0.00 |
| noise | standard deviation 0.02 | 35.33 | 0.59 |
| blur | radius 1.0 | 32.30 | 3.62 |
| darker | brightness 0.75 | 22.85 | 13.07 |
| brighter | brightness 1.25 | 24.83 | 11.08 |
| fewer views | 50 clean views | 34.24 | 1.67 |
| fewer views | 25 clean views | 29.90 | 6.02 |

Mild zero-mean noise is mostly absorbed. Blur removes useful high-frequency
supervision. Reducing views eventually creates a coverage bottleneck. Global
brightness mismatch is most damaging because the model learns a systematically
biased appearance and is then evaluated against clean images.

---

## 15. What the Results Mean

### 15.1 Opacity is a useful but incomplete importance proxy

Opacity directly participates in alpha compositing, so very low-opacity
Gaussians often have limited influence. That gives opacity top-k a meaningful
connection to rendering.

Opacity is still not a complete contribution measure. A highly opaque Gaussian
may cover few pixels, be occluded, or encode redundant appearance. A future
score could include:

- accumulated transmittance-weighted alpha;
- projected footprint;
- residual error;
- gradient magnitude;
- frequency of actual raster contribution.

### 15.2 Equal Gaussian counts do not imply equal speed

Visibility-only selection can be slower than opacity-based selection at the
same count. Runtime depends on where retained Gaussians project, how many tiles
they overlap, and how much sorting and compositing work they create. Gaussian
count is a useful resource indicator, but not a full runtime model.

### 15.3 Pareto rank is conditional

A rank-0 point is non-dominated only under the declared objectives and measured
environment. If a deployment values LPIPS, p95 latency, peak memory, or
worst-view quality, the front should be recomputed with those objectives.

### 15.4 Failure analysis improves the method

The visibility ablation prevents a plausible but weak proxy from being
presented as an improvement. Pose and brightness experiments show that strong
clean-test results do not imply robustness. These negative findings make the
project's claims more precise.

---

## 16. Reproducibility and Generated Artifacts

The complete experiment sequence is documented in
[reproducibility.md](reproducibility.md). The source repository commits code,
configuration, tests, and result summaries, while large generated artifacts
remain under `results/`.

Important generated artifacts include:

- baseline point clouds and checkpoints;
- held-out renders and ground-truth pairs;
- aggregate and per-view metric JSON;
- aggregate profiles and per-frame latency JSON;
- pruning metadata;
- study summary JSON and CSV;
- 2D and 3D Pareto plots;
- pose and input-sensitivity plots;
- a static browser demo;
- comparison panels and a video bundle.

The project has 72 automated tests covering:

- configuration contracts;
- dataset validation;
- RGBA compositing;
- metric behavior;
- profiling summaries;
- pruning masks and metadata;
- visibility geometry;
- Pareto dominance and sorting;
- pose perturbations;
- image degradations;
- study collection and plotting;
- demo and portfolio generation;
- local Markdown-link integrity.

Tests do not replace GPU reproduction, but they protect deterministic logic and
the contracts connecting expensive stages.

---

## 17. Limitations

The current evidence has a deliberately narrow scope:

1. Pruning, visibility ablation, and robustness conclusions are primarily from
   NeRF Synthetic Lego.
2. Drums validates the clean workflow but does not replicate the complete
   pruning and robustness grid.
3. All reported experiments use seed 0.
4. One run per condition gives point estimates, not confidence intervals or
   statistical significance.
5. The scenes are synthetic, static, object-centric, and cleanly segmented.
6. Pruned models are not fine-tuned.
7. Visibility is a center-projection proxy, not renderer-derived contribution.
8. FPS is renderer-only and hardware-specific.
9. PSNR, SSIM, and LPIPS do not measure geometric accuracy, temporal
   consistency, or downstream-task performance.
10. Pareto ranks depend on the chosen objectives.

Accordingly, the project does not claim:

- universal optimality of 75% retention;
- robustness to calibration or exposure errors;
- real-world scene generalization;
- hardware-independent speedups;
- end-to-end application FPS;
- statistically significant improvements.

The exact claim boundary is maintained in
[claims_and_evidence.md](claims_and_evidence.md).

---

## 18. The Project as a Scientific Loop

The most useful way to understand Pareto-Splat is as an iterative experimental
loop:

```text
define a measurable question
  -> lock data, code, and configuration
  -> run one controlled change
  -> measure quality and systems behavior
  -> compare at matched budgets
  -> inspect non-dominated choices
  -> test plausible explanations
  -> document failures and scope
```

The opacity study asks whether a simple learned parameter can identify
removable Gaussians. Random pruning supplies the control. Matched budgets
isolate ranking quality. Profiling measures the systems effect. Pareto sorting
avoids hiding trade-offs in one score. Visibility ablation tests a richer
hypothesis. Robustness studies test assumptions outside compression.

That complete loop is the method.

---

## 19. Final Takeaway

Pareto-Splat demonstrates that post-training opacity-aware pruning can create
useful 3DGS deployment operating points. On the tested Lego run, retaining 75%
of Gaussians reduces model size by 25% and increases renderer throughput by
61.3%, with a 1.421 dB PSNR cost.

More importantly, the project shows how to make that result interpretable:

- compare selection methods at equal Gaussian budgets;
- measure quality over every held-out view;
- profile only a clearly defined renderer boundary;
- preserve exact experiment provenance;
- use Pareto fronts instead of an arbitrary scalar score;
- ablate importance terms;
- stress camera and input assumptions;
- state what the evidence does not prove.

For implementation details and compact references, continue with:

- [Technical report](technical_report.md)
- [Pruning methods](pruning.md)
- [Pareto dominance](pareto.md)
- [Visibility importance](visibility_importance.md)
- [Importance ablation](importance_ablation.md)
- [Pose sensitivity](pose_sensitivity.md)
- [Training-input sensitivity](input_sensitivity.md)
- [Failure cases and limitations](limitations.md)
- [Reproducibility protocol](reproducibility.md)
