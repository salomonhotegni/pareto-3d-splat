SHELL := /bin/bash

CONDA_ENV ?= pareto3dsplat
CONFIG ?= configs/baseline.yaml
PRUNING_CONFIG ?= configs/pruning_lego.yaml
PRUNING_VARIANT ?=
PRUNING_VARIANT_ARG := $(if $(PRUNING_VARIANT),--variant $(PRUNING_VARIANT),)
POSE_SENSITIVITY_CONFIG ?= configs/pose_sensitivity_lego.yaml
POSE_VARIANT ?=
POSE_VARIANT_ARG := $(if $(POSE_VARIANT),--variant $(POSE_VARIANT),)
INPUT_SENSITIVITY_CONFIG ?= configs/input_sensitivity_lego.yaml
INPUT_VARIANT ?=
INPUT_VARIANT_ARG := $(if $(INPUT_VARIANT),--variant $(INPUT_VARIANT),)
DEMO_OUTPUT ?= results/demo/index.html
PORTFOLIO_OUTPUT ?= results/portfolio

.PHONY: help env baseline install dataset dataset-lego dataset-drums check-data check-data-lego check-data-drums train-baseline render-baseline evaluate-baseline profile-baseline comparison-video pruning-study-prune pruning-study-render pruning-study-evaluate pruning-study-profile pruning-study-summarize pose-sensitivity-prepare pose-sensitivity-render pose-sensitivity-evaluate pose-sensitivity-profile pose-sensitivity-summarize input-sensitivity-prepare input-sensitivity-train input-sensitivity-render input-sensitivity-evaluate input-sensitivity-profile input-sensitivity-summarize demo portfolio-assets check check-core test

help:
	@printf '%s\n' \
		'Pareto-Splat workflow targets' \
		'' \
		'Setup and validation:' \
		'  env, install, check, check-core, test' \
		'Datasets:' \
		'  dataset-lego, check-data-lego, dataset-drums, check-data-drums' \
		'Clean baseline:' \
		'  train-baseline, render-baseline, evaluate-baseline, profile-baseline' \
		'Pruning study:' \
		'  pruning-study-{prune,render,evaluate,profile,summarize}' \
		'Robustness studies:' \
		'  pose-sensitivity-{prepare,render,evaluate,profile,summarize}' \
		'  input-sensitivity-{prepare,train,render,evaluate,profile,summarize}' \
		'Presentation:' \
		'  comparison-video, demo, portfolio-assets' \
		'' \
		'Common overrides:' \
		'  CONDA_ENV, CONFIG, PRUNING_CONFIG, PRUNING_VARIANT' \
		'  POSE_SENSITIVITY_CONFIG, POSE_VARIANT' \
		'  INPUT_SENSITIVITY_CONFIG, INPUT_VARIANT' \
		'' \
		'See docs/reproducibility.md for the ordered end-to-end protocol.'

env:
	conda env update --name $(CONDA_ENV) --file environment.yml --prune

baseline:
	bash scripts/bootstrap_baseline.sh

install: baseline
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/install_baseline.sh

dataset: dataset-lego

dataset-lego:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/download_lego_dataset.sh

dataset-drums:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/download_drums_dataset.sh

check-data: check-data-lego

check-data-lego:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/validate_nerf_synthetic.py

check-data-drums:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/validate_nerf_synthetic.py data/nerf_synthetic/drums

train-baseline:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/train_baseline.sh --config $(CONFIG)

render-baseline:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/render_baseline.sh --config $(CONFIG)

evaluate-baseline:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/evaluate_baseline.sh --config $(CONFIG)

profile-baseline:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/profile_baseline.sh --config $(CONFIG)

comparison-video:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/create_comparison_video.sh

pruning-study-prune:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pruning_study.py --config $(PRUNING_CONFIG) $(PRUNING_VARIANT_ARG) prune

pruning-study-render:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pruning_study.py --config $(PRUNING_CONFIG) $(PRUNING_VARIANT_ARG) render

pruning-study-evaluate:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pruning_study.py --config $(PRUNING_CONFIG) $(PRUNING_VARIANT_ARG) evaluate

pruning-study-profile:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pruning_study.py --config $(PRUNING_CONFIG) $(PRUNING_VARIANT_ARG) profile

pruning-study-summarize:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pruning_study.py --config $(PRUNING_CONFIG) summarize

pose-sensitivity-prepare:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pose_sensitivity.py --config $(POSE_SENSITIVITY_CONFIG) $(POSE_VARIANT_ARG) prepare

pose-sensitivity-render:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pose_sensitivity.py --config $(POSE_SENSITIVITY_CONFIG) $(POSE_VARIANT_ARG) render

pose-sensitivity-evaluate:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pose_sensitivity.py --config $(POSE_SENSITIVITY_CONFIG) $(POSE_VARIANT_ARG) evaluate

pose-sensitivity-profile:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pose_sensitivity.py --config $(POSE_SENSITIVITY_CONFIG) $(POSE_VARIANT_ARG) profile

pose-sensitivity-summarize:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_pose_sensitivity.py --config $(POSE_SENSITIVITY_CONFIG) summarize

input-sensitivity-prepare:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_input_sensitivity.py --config $(INPUT_SENSITIVITY_CONFIG) $(INPUT_VARIANT_ARG) prepare

input-sensitivity-train:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_input_sensitivity.py --config $(INPUT_SENSITIVITY_CONFIG) $(INPUT_VARIANT_ARG) train

input-sensitivity-render:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_input_sensitivity.py --config $(INPUT_SENSITIVITY_CONFIG) $(INPUT_VARIANT_ARG) render

input-sensitivity-evaluate:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_input_sensitivity.py --config $(INPUT_SENSITIVITY_CONFIG) $(INPUT_VARIANT_ARG) evaluate

input-sensitivity-profile:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_input_sensitivity.py --config $(INPUT_SENSITIVITY_CONFIG) $(INPUT_VARIANT_ARG) profile

input-sensitivity-summarize:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/run_input_sensitivity.py --config $(INPUT_SENSITIVITY_CONFIG) summarize

demo:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/build_demo.py --output $(DEMO_OUTPUT)

portfolio-assets:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/build_portfolio_assets.py --output $(PORTFOLIO_OUTPUT)

check-core:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py

check:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py --require-baseline

test:
	conda run --no-capture-output -n $(CONDA_ENV) pytest
