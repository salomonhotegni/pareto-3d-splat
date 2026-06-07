#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=baseline.env
source "${ROOT_DIR}/scripts/baseline.env"
# shellcheck source=dataset.env
source "${ROOT_DIR}/scripts/dataset.env"

PYTHON="${PYTHON:-python}"
BASELINE_DIR="${ROOT_DIR}/${BASELINE_RELATIVE_DIR}"
DATASET_DIR="${ROOT_DIR}/${LEGO_DATASET_RELATIVE_DIR}"
OUTPUT_DIR="${ROOT_DIR}/results/baseline/lego/seed_0"

if [[ ! -f "${BASELINE_DIR}/train.py" ]]; then
    echo "error: baseline missing; run scripts/bootstrap_baseline.sh first" >&2
    exit 1
fi

"${PYTHON}" "${ROOT_DIR}/scripts/validate_nerf_synthetic.py" "${DATASET_DIR}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "error: output path already exists: ${OUTPUT_DIR}" >&2
    echo "Choose how to preserve or remove that run before starting another." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

command=(
    "${PYTHON}"
    "${BASELINE_DIR}/train.py"
    --source_path "${DATASET_DIR}"
    --model_path "${OUTPUT_DIR}"
    --eval
    --white_background
    --resolution 1
    --iterations 30000
    --test_iterations 7000 30000
    --save_iterations 7000 30000
    --disable_viewer
)

printf 'Training command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"

