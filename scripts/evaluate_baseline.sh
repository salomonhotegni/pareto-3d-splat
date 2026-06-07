#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${ROOT_DIR}/results/baseline/lego/seed_0"
RENDER_DIR="${OUTPUT_DIR}/test/ours_30000/renders"
GROUND_TRUTH_DIR="${OUTPUT_DIR}/test/ours_30000/gt"
METRIC_DIR="${OUTPUT_DIR}/metrics/ours_30000"
LOG_PATH="${METRIC_DIR}/evaluation.log"
COMMAND_PATH="${METRIC_DIR}/command.sh"

mkdir -p "${METRIC_DIR}"

command=(
    "${PYTHON}"
    "${ROOT_DIR}/scripts/evaluate_baseline.py"
    --renders "${RENDER_DIR}"
    --ground-truth "${GROUND_TRUTH_DIR}"
    --output-dir "${METRIC_DIR}"
    --expected-count 200
    --device auto
)

{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "${ROOT_DIR}"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "${COMMAND_PATH}"
chmod +x "${COMMAND_PATH}"

printf 'Evaluation command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}" 2>&1 | tee "${LOG_PATH}"
