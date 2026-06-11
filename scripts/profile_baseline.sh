#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

CONFIG="${ROOT_DIR}/configs/baseline.yaml"

if [[ $# -ge 2 && "$1" == "--config" ]]; then
    CONFIG="$2"
    shift 2
fi

exec "${PYTHON}" "${ROOT_DIR}/scripts/run_experiment.py" \
    --config "${CONFIG}" profile "$@"
