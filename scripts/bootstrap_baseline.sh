#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=baseline.env
source "${ROOT_DIR}/scripts/baseline.env"

BASELINE_DIR="${ROOT_DIR}/${BASELINE_RELATIVE_DIR}"

if [[ -e "${BASELINE_DIR}" && ! -d "${BASELINE_DIR}/.git" ]]; then
    echo "error: ${BASELINE_DIR} exists but is not a Git checkout" >&2
    exit 1
fi

if [[ ! -d "${BASELINE_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${BASELINE_DIR}")"
    git clone --filter=blob:none "${BASELINE_REPOSITORY}" "${BASELINE_DIR}"
fi

remote_url="$(git -C "${BASELINE_DIR}" remote get-url origin)"
if [[ "${remote_url}" != "${BASELINE_REPOSITORY}" ]]; then
    echo "error: unexpected baseline origin: ${remote_url}" >&2
    exit 1
fi

if ! git -C "${BASELINE_DIR}" diff --quiet ||
   ! git -C "${BASELINE_DIR}" diff --cached --quiet; then
    echo "error: baseline checkout has local changes; refusing to replace them" >&2
    exit 1
fi

git -C "${BASELINE_DIR}" fetch origin "${BASELINE_COMMIT}" --depth 1
git -C "${BASELINE_DIR}" checkout --detach "${BASELINE_COMMIT}"
git -C "${BASELINE_DIR}" submodule sync --recursive
git -C "${BASELINE_DIR}" submodule update --init --recursive --depth 1

actual_commit="$(git -C "${BASELINE_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${BASELINE_COMMIT}" ]]; then
    echo "error: expected ${BASELINE_COMMIT}, found ${actual_commit}" >&2
    exit 1
fi

echo "Baseline ready at ${BASELINE_DIR}"
echo "Pinned commit: ${actual_commit}"
