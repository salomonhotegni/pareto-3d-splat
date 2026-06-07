#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=dataset.env
source "${ROOT_DIR}/scripts/dataset.env"

PYTHON="${PYTHON:-python}"
DATASET_DIR="${ROOT_DIR}/${LEGO_DATASET_RELATIVE_DIR}"
DOWNLOAD_DIR="${ROOT_DIR}/data/.downloads"
ARCHIVE_PATH="${DOWNLOAD_DIR}/nerf_example_data.zip"
PARTIAL_PATH="${ARCHIVE_PATH}.part"

if [[ -e "${DATASET_DIR}" ]]; then
    if [[ ! -d "${DATASET_DIR}" ]]; then
        echo "error: ${DATASET_DIR} exists but is not a directory" >&2
        exit 1
    fi
    "${PYTHON}" "${ROOT_DIR}/scripts/validate_nerf_synthetic.py" "${DATASET_DIR}"
    echo "Dataset is already ready at ${DATASET_DIR}"
    exit 0
fi

for command in curl sha256sum unzip; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: ${command} is required to download the dataset" >&2
        exit 1
    fi
done

mkdir -p "${DOWNLOAD_DIR}"

if [[ -f "${ARCHIVE_PATH}" ]]; then
    actual_checksum="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
    if [[ "${actual_checksum}" != "${NERF_EXAMPLE_DATA_SHA256}" ]]; then
        invalid_path="${ARCHIVE_PATH}.invalid.$(date +%s)"
        mv "${ARCHIVE_PATH}" "${invalid_path}"
        echo "warning: moved invalid cached archive to ${invalid_path}" >&2
    fi
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Downloading the official NeRF example archive (353 MiB)..."
    curl \
        --location \
        --fail \
        --retry 3 \
        --continue-at - \
        --output "${PARTIAL_PATH}" \
        "${NERF_EXAMPLE_DATA_URL}"

    actual_checksum="$(sha256sum "${PARTIAL_PATH}" | awk '{print $1}')"
    if [[ "${actual_checksum}" != "${NERF_EXAMPLE_DATA_SHA256}" ]]; then
        invalid_path="${PARTIAL_PATH}.invalid.$(date +%s)"
        mv "${PARTIAL_PATH}" "${invalid_path}"
        echo "error: dataset archive checksum mismatch" >&2
        echo "expected: ${NERF_EXAMPLE_DATA_SHA256}" >&2
        echo "actual:   ${actual_checksum}" >&2
        echo "invalid download retained at ${invalid_path}" >&2
        exit 1
    fi
    mv "${PARTIAL_PATH}" "${ARCHIVE_PATH}"
fi

staging_dir="$(mktemp -d "${ROOT_DIR}/data/.lego-extract.XXXXXX")"
trap 'rm -rf "${staging_dir}"' EXIT

echo "Extracting Lego poses and RGBA color images..."
unzip -q \
    "${ARCHIVE_PATH}" \
    'nerf_synthetic/lego/*' \
    -x '*_depth_0001.png' '*_normal_0001.png' \
    -d "${staging_dir}"

extracted_dir="${staging_dir}/nerf_synthetic/lego"
"${PYTHON}" "${ROOT_DIR}/scripts/validate_nerf_synthetic.py" "${extracted_dir}"

mkdir -p "$(dirname "${DATASET_DIR}")"
mv "${extracted_dir}" "${DATASET_DIR}"

echo "Dataset ready at ${DATASET_DIR}"
echo "Verified archive cached at ${ARCHIVE_PATH}"

