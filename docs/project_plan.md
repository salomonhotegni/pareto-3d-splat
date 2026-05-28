# Pareto-Splat: Multi-Objective 3D Gaussian Splatting for Quality–Speed–Memory Trade-offs

## 1. Project Motivation

Modern 3D computer vision systems increasingly need to reconstruct realistic 3D scenes from images and render them efficiently for downstream applications such as autonomous driving, robotics, AR/VR, digital twins, and simulation. Recent methods such as 3D Gaussian Splatting have made real-time novel-view synthesis possible with high visual quality. However, practical deployment still involves important trade-offs.

A model that produces very high-quality renderings may require many Gaussians, high memory consumption, and slower rendering. Conversely, a highly compressed model may render faster but lose important geometric or visual details. In real-world systems, there is rarely a single best model. Instead, practitioners often need to choose between different operating points depending on the available compute, memory budget, and quality requirements.

This project aims to explore 3D reconstruction as a multi-objective optimization problem. The goal is not only to train a 3D Gaussian Splatting model, but also to analyze and improve the trade-off between rendering quality, inference speed, and memory usage. This makes the project relevant for both 3D computer vision and efficient machine learning deployment.

## 2. Problem Definition

Given a set of multi-view images of a static scene, the task is to reconstruct a 3D representation that can render novel views of the scene. The base representation will be 3D Gaussian Splatting, where the scene is represented by a collection of learnable 3D Gaussians with parameters such as position, scale, opacity, color, and orientation.

The core problem is to obtain not just one reconstructed scene, but a set of models representing different trade-offs between:

* Visual quality, measured using metrics such as PSNR, SSIM, and LPIPS.
* Rendering speed, measured in frames per second or average render time.
* Memory efficiency, measured by the number of Gaussians and total model size.
* Optional geometric consistency, measured through qualitative inspection or depth/pose consistency when available.

Formally, the project studies the following question:

> How can we produce efficient 3D Gaussian Splatting representations that preserve high novel-view synthesis quality while reducing memory usage and improving rendering speed?

Instead of optimizing only for reconstruction quality, the project will investigate pruning, compression, and multi-objective analysis strategies to generate a Pareto front of 3D scene representations.

## 3. Objectives

The main objective is to build a complete 3D computer vision project that demonstrates practical understanding of 3D reconstruction, neural rendering, model compression, and multi-objective evaluation.

The specific objectives are:

1. **Implement or adapt a 3D Gaussian Splatting baseline**
   Set up a working pipeline that takes multi-view images as input and trains a 3D Gaussian Splatting model for novel-view synthesis.

2. **Evaluate reconstruction quality**
   Measure the visual quality of rendered novel views using standard metrics such as PSNR, SSIM, and LPIPS, together with qualitative visual comparisons.

3. **Analyze efficiency metrics**
   Measure rendering speed, number of Gaussians, model size, and memory consumption to understand the computational cost of the baseline model.

4. **Introduce efficiency-oriented modifications**
   Implement pruning or compression strategies that remove less important Gaussians while attempting to preserve visual quality.

5. **Generate quality–speed–memory trade-offs**
   Produce multiple model variants with different compression levels and evaluate each variant across quality, speed, and memory metrics.

6. **Construct and analyze a Pareto front**
   Identify non-dominated models that represent strong trade-offs between visual quality, rendering speed, and memory usage.

7. **Build a clear portfolio-ready demo**
   Create visual outputs, comparison plots, and an interactive or recorded demo showing novel-view rendering and trade-off behavior.

8. **Document the project professionally**
   Prepare a GitHub repository with clear instructions, reproducible experiments, results, limitations, and future work.

## 4. Expected Deliverables

By the end of the project, the expected deliverables are:

### 4.1 GitHub Repository

A clean and well-structured GitHub repository containing:

* Source code for training and evaluating the 3D Gaussian Splatting baseline.
* Scripts for pruning or compressing Gaussian representations.
* Evaluation scripts for quality, speed, and memory metrics.
* Configuration files for reproducible experiments.
* Clear installation and usage instructions.
* A polished `README.md` explaining the project, motivation, method, and results.

### 4.2 Baseline 3D Reconstruction Pipeline

A working pipeline that:

* Takes a set of multi-view images as input.
* Trains a 3D Gaussian Splatting model.
* Renders novel views from unseen camera poses.
* Saves rendered images and videos for qualitative evaluation.

### 4.3 Efficiency and Compression Module

A module that performs at least one efficiency-improving strategy, such as:

* Opacity-based Gaussian pruning.
* Size- or contribution-based pruning.
* Progressive Gaussian reduction.
* Post-training compression.
* Comparison between different pruning thresholds.

The goal is to create multiple model variants with different quality–efficiency trade-offs.

### 4.4 Evaluation Report

A technical report containing:

* Project motivation and problem formulation.
* Description of the baseline method.
* Description of the pruning/compression strategy.
* Experimental setup.
* Quantitative results.
* Pareto-front analysis.
* Qualitative visual comparisons.
* Limitations and possible extensions.

### 4.5 Pareto-Front Analysis

A set of plots and tables showing the trade-offs between:

* PSNR vs. FPS.
* PSNR vs. model size.
* LPIPS vs. number of Gaussians.
* Rendering quality vs. memory usage.
* Quality–speed–memory Pareto front.

This will highlight which compressed models are non-dominated and therefore represent good deployment candidates.

### 4.6 Demo Video or Interactive Viewer

A short demo showing:

* The reconstructed 3D scene.
* Novel-view rendering results.
* Comparison between the baseline and compressed models.
* Visual effects of different pruning levels.
* The final Pareto-front trade-off plot.

### 4.7 Portfolio Summary

A concise project summary.
