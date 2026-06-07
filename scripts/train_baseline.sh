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
CHECKPOINT_KEEP_COUNT=2
CHECKPOINT_POLL_SECONDS=30
CHECKPOINT_ITERATIONS=(5000 10000 15000 20000 25000 30000)
RESUME_CHECKPOINT=""
MONITOR_PID=""
ATTEMPT_DIR=""
START_EPOCH=""
START_UTC=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_baseline.sh
  bash scripts/train_baseline.sh --resume PATH_TO_CHECKPOINT

The initial command refuses to overwrite an existing run. The resume command
continues the existing run from a GraphDeCo chkpnt*.pth file in the model
directory. Only the two newest checkpoints are retained.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            if [[ $# -lt 2 ]]; then
                echo "error: --resume requires a checkpoint path" >&2
                usage >&2
                exit 2
            fi
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${BASELINE_DIR}/train.py" ]]; then
    echo "error: baseline missing; run scripts/bootstrap_baseline.sh first" >&2
    exit 1
fi

"${PYTHON}" "${ROOT_DIR}/scripts/validate_nerf_synthetic.py" "${DATASET_DIR}"

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
        echo "error: resume checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
        exit 1
    fi
    RESUME_CHECKPOINT="$(realpath "${RESUME_CHECKPOINT}")"
    if [[ "$(dirname "${RESUME_CHECKPOINT}")" != "${OUTPUT_DIR}" ]]; then
        echo "error: resume checkpoint must be inside ${OUTPUT_DIR}" >&2
        exit 1
    fi
    if [[ ! "$(basename "${RESUME_CHECKPOINT}")" =~ ^chkpnt[0-9]+\.pth$ ]]; then
        echo "error: unexpected checkpoint name: ${RESUME_CHECKPOINT}" >&2
        exit 1
    fi
    if [[ ! -d "${OUTPUT_DIR}" ]]; then
        echo "error: model directory does not exist: ${OUTPUT_DIR}" >&2
        exit 1
    fi
elif [[ -e "${OUTPUT_DIR}" ]]; then
    echo "error: output path already exists: ${OUTPUT_DIR}" >&2
    echo "Use --resume with a retained checkpoint or preserve the existing run." >&2
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
    --checkpoint_iterations "${CHECKPOINT_ITERATIONS[@]}"
    --disable_viewer
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    command+=(--start_checkpoint "${RESUME_CHECKPOINT}")
fi

prune_checkpoints() {
    local checkpoint
    local checkpoints=()

    mapfile -t checkpoints < <(
        find "${OUTPUT_DIR}" \
            -maxdepth 1 \
            -type f \
            -name 'chkpnt*.pth' \
            -printf '%f\n' |
            sort -V
    )

    while (( ${#checkpoints[@]} > CHECKPOINT_KEEP_COUNT )); do
        checkpoint="${checkpoints[0]}"
        rm -- "${OUTPUT_DIR}/${checkpoint}"
        printf '%s removed %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "${checkpoint}" \
            >> "${ATTEMPT_DIR}/checkpoint_retention.log"
        checkpoints=("${checkpoints[@]:1}")
    done
}

monitor_checkpoints() {
    while true; do
        prune_checkpoints
        sleep "${CHECKPOINT_POLL_SECONDS}"
    done
}

stop_checkpoint_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
    MONITOR_PID=""
}

finalize_attempt() {
    local exit_code=$?
    local end_epoch
    local end_utc
    local status

    stop_checkpoint_monitor
    if [[ -n "${ATTEMPT_DIR}" ]]; then
        prune_checkpoints || true
    fi

    end_epoch="$(date +%s)"
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    status="failed"
    if [[ "${exit_code}" -eq 0 ]]; then
        status="completed"
    fi

    if [[ -n "${ATTEMPT_DIR}" ]]; then
        {
            printf 'status=%s\n' "${status}"
            printf 'exit_code=%s\n' "${exit_code}"
            printf 'start_utc=%s\n' "${START_UTC}"
            printf 'end_utc=%s\n' "${end_utc}"
            printf 'duration_seconds=%s\n' "$((end_epoch - START_EPOCH))"
        } > "${ATTEMPT_DIR}/status.txt"
    fi
}

START_EPOCH="$(date +%s)"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
attempt_id="$(date -u +%Y%m%dT%H%M%SZ)"
ATTEMPT_DIR="${OUTPUT_DIR}/attempts/${attempt_id}"
mkdir -p "${ATTEMPT_DIR}"

cp "${ROOT_DIR}/configs/baseline.yaml" "${ATTEMPT_DIR}/baseline.yaml"
sha256sum \
    "${DATASET_DIR}/transforms_train.json" \
    "${DATASET_DIR}/transforms_val.json" \
    "${DATASET_DIR}/transforms_test.json" \
    > "${ATTEMPT_DIR}/dataset_checksums.sha256"

{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "${ROOT_DIR}"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "${ATTEMPT_DIR}/command.sh"
chmod +x "${ATTEMPT_DIR}/command.sh"

{
    printf 'start_utc=%s\n' "${START_UTC}"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'project_commit=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'project_dirty=%s\n' "$(
        if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
            printf 'true'
        else
            printf 'false'
        fi
    )"
    printf 'baseline_commit=%s\n' "$(git -C "${BASELINE_DIR}" rev-parse HEAD)"
    printf 'python_executable=%s\n' "$(command -v "${PYTHON}")"
    "${PYTHON}" --version
    "${PYTHON}" -c \
        'import torch; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
    uname -a
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
} > "${ATTEMPT_DIR}/system.txt" 2>&1

if command -v conda >/dev/null 2>&1; then
    conda list --explicit > "${ATTEMPT_DIR}/conda-explicit.txt"
fi

trap finalize_attempt EXIT

printf 'Training command:'
printf ' %q' "${command[@]}"
printf '\n'
printf 'Attempt artifacts: %s\n' "${ATTEMPT_DIR}"

monitor_checkpoints &
MONITOR_PID=$!

"${command[@]}" 2>&1 | tee "${ATTEMPT_DIR}/train.log"
