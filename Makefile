SHELL := /bin/bash

CONDA_ENV ?= pareto3dsplat
CONFIG ?= configs/baseline.yaml
PRUNING_CONFIG ?= configs/pruning_lego.yaml
PRUNING_VARIANT ?=
PRUNING_VARIANT_ARG := $(if $(PRUNING_VARIANT),--variant $(PRUNING_VARIANT),)

.PHONY: env baseline install dataset dataset-lego dataset-drums check-data check-data-lego check-data-drums train-baseline render-baseline evaluate-baseline profile-baseline comparison-video pruning-study-prune pruning-study-render pruning-study-evaluate pruning-study-profile pruning-study-summarize check check-core test

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

check-core:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py

check:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py --require-baseline

test:
	conda run --no-capture-output -n $(CONDA_ENV) pytest
