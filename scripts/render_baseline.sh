#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=dataset.env
source "${ROOT_DIR}/scripts/dataset.env"

PYTHON="${PYTHON:-python}"
DATASET_DIR="${ROOT_DIR}/${LEGO_DATASET_RELATIVE_DIR}"
OUTPUT_DIR="${ROOT_DIR}/results/baseline/lego/seed_0"
ITERATION=30000
MODEL_PATH="${OUTPUT_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
LOG_DIR="${OUTPUT_DIR}/evaluation"

"${PYTHON}" "${ROOT_DIR}/scripts/validate_nerf_synthetic.py" "${DATASET_DIR}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "error: trained model not found: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

command=(
    "${PYTHON}"
    "${ROOT_DIR}/scripts/run_graphdeco.py"
    render.py
    --model_path "${OUTPUT_DIR}"
    --source_path "${DATASET_DIR}"
    --iteration "${ITERATION}"
    --eval
    --white_background
    --resolution 1
    --skip_train
)

printf 'Rendering command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}" 2>&1 | tee "${LOG_DIR}/render.log"

