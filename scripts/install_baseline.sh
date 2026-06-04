#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=baseline.env
source "${ROOT_DIR}/scripts/baseline.env"

BASELINE_DIR="${ROOT_DIR}/${BASELINE_RELATIVE_DIR}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "${BASELINE_DIR}/.git" ]]; then
    echo "error: baseline missing; run scripts/bootstrap_baseline.sh first" >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "error: nvcc is required to compile the baseline CUDA extensions" >&2
    exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    export CUDA_HOME
fi

export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

"${PYTHON}" -m pip install --no-build-isolation --no-deps -e "${ROOT_DIR}"

for extension in diff-gaussian-rasterization simple-knn fused-ssim; do
    extension_path="${BASELINE_DIR}/submodules/${extension}"
    if [[ ! -d "${extension_path}" ]]; then
        echo "error: missing baseline extension ${extension_path}" >&2
        exit 1
    fi
    "${PYTHON}" -m pip install --no-build-isolation --no-deps "${extension_path}"
done

echo "Baseline extensions installed for CUDA architectures: ${TORCH_CUDA_ARCH_LIST}"
