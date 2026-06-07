SHELL := /bin/bash

CONDA_ENV ?= pareto3dsplat

.PHONY: env baseline install dataset check-data train-baseline render-baseline check check-core test

env:
	conda env update --name $(CONDA_ENV) --file environment.yml --prune

baseline:
	bash scripts/bootstrap_baseline.sh

install: baseline
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/install_baseline.sh

dataset:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/download_lego_dataset.sh

check-data:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/validate_nerf_synthetic.py

train-baseline:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/train_baseline.sh

render-baseline:
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/render_baseline.sh

check-core:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py

check:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py --require-baseline

test:
	conda run --no-capture-output -n $(CONDA_ENV) pytest
