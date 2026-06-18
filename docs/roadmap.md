# Pareto-Splat Working Roadmap

The project follows a 12-week schedule with two focused sessions per week.
This file is the working checklist; the broader motivation and deliverables
are defined in `docs/project_plan.md`.

## Phase 1: Baseline Setup

- [x] **Session 1:** Define the project motivation, objectives, and deliverables.
- [x] **Session 2:** Create a reproducible environment, repository structure,
  pinned 3DGS baseline, and setup validation.
- [x] **Session 3:** Select, download, and document the first public dataset.
- [x] **Session 4:** Train the first baseline and save renders, logs, metrics,
  checkpoint, model size, and a camera-path video.

## Phase 2: Evaluation and Reproducibility

- [x] **Session 5:** Evaluate PSNR, SSIM, LPIPS, Gaussian count, and model size.
- [x] **Session 6:** Profile FPS, render latency, and GPU memory.
- [x] **Session 7:** Refactor training and evaluation into configuration-driven
  workflows.
- [x] **Session 8:** Run the clean baseline workflow on a second scene.

## Phase 3: Pruning and Pareto Analysis

- [x] **Session 9:** Implement random, opacity-threshold, and top-k pruning.
- [x] **Session 10:** Evaluate pruning levels and plot quality-efficiency
  trade-offs.
- [x] **Session 11:** Formalize objectives and implement Pareto dominance and
  non-dominated sorting.
- [ ] **Session 12:** Generate the first 2D and 3D Pareto fronts.
- [ ] **Session 13:** Implement a visibility-aware Gaussian importance score.
- [ ] **Session 14:** Compare all pruning methods at matched Gaussian budgets.

## Phase 4: Robustness and Ablations

- [ ] **Session 15:** Evaluate sensitivity to camera-pose perturbations.
- [ ] **Session 16:** Evaluate image noise, blur, brightness changes, and fewer
  input views.
- [ ] **Session 17:** Run importance-score ablations.
- [ ] **Session 18:** Document failure cases and practical limitations.

## Phase 5: Demo and Portfolio

- [ ] **Session 19:** Build a demo for selecting scenes and Pareto operating
  points.
- [ ] **Session 20:** Produce comparison images, videos, plots, and a pipeline
  diagram.
- [ ] **Session 21:** Write the technical report draft.
- [ ] **Session 22:** Finalize experiments, claims, and limitations.
- [ ] **Session 23:** Polish repository documentation and reproducibility.
- [ ] **Session 24:** Prepare the portfolio summary, blog post, resume bullet,
  and interview explanation.
