SHELL := /bin/bash

CONDA_ENV ?= pareto3dsplat

.PHONY: env baseline install check check-core test

env:
	conda env update --name $(CONDA_ENV) --file environment.yml --prune

baseline:
	bash scripts/bootstrap_baseline.sh

install: baseline
	conda run --no-capture-output -n $(CONDA_ENV) bash scripts/install_baseline.sh

check-core:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py

check:
	conda run --no-capture-output -n $(CONDA_ENV) python scripts/check_environment.py --require-baseline

test:
	conda run --no-capture-output -n $(CONDA_ENV) pytest
