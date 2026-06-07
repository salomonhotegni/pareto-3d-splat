#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=dataset.env
source "${ROOT_DIR}/scripts/dataset.env"

PYTHON="${PYTHON:-python}"
MODEL_PATH="${ROOT_DIR}/results/baseline/lego/seed_0"
SOURCE_PATH="${ROOT_DIR}/${LEGO_DATASET_RELATIVE_DIR}"
PROFILE_DIR="${MODEL_PATH}/profile/ours_30000"
LOG_PATH="${PROFILE_DIR}/profiling.log"
COMMAND_PATH="${PROFILE_DIR}/command.sh"

mkdir -p "${PROFILE_DIR}"

command=(
    "${PYTHON}"
    "${ROOT_DIR}/scripts/profile_baseline.py"
    --model-path "${MODEL_PATH}"
    --source-path "${SOURCE_PATH}"
    --output-dir "${PROFILE_DIR}"
    --iteration 30000
    --expected-view-count 200
    --warmup-views 10
    --repetitions 3
)

{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "${ROOT_DIR}"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "${COMMAND_PATH}"
chmod +x "${COMMAND_PATH}"

printf 'Profiling command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}" 2>&1 | tee "${LOG_PATH}"
