#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results/baseline/lego/seed_0"
RENDER_ROOT="${OUTPUT_DIR}/test/ours_30000"
GT_DIR="${RENDER_ROOT}/gt"
PREDICTION_DIR="${RENDER_ROOT}/renders"
VIDEO_DIR="${OUTPUT_DIR}/videos"
VIDEO_PATH="${VIDEO_DIR}/lego_ground_truth_vs_3dgs.mp4"
METADATA_PATH="${VIDEO_DIR}/lego_ground_truth_vs_3dgs.json"
LOG_PATH="${VIDEO_DIR}/comparison_video.log"
FRAME_COUNT=200
FRAME_RATE=30

resolve_media_tool() {
    local tool="$1"
    local candidate
    local conda_base

    candidate="$(command -v "${tool}" 2>/dev/null || true)"
    if [[ -n "${candidate}" ]] && "${candidate}" -version >/dev/null 2>&1; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
        candidate="${conda_base}/bin/${tool}"
        if [[ -x "${candidate}" ]] && "${candidate}" -version >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    echo "error: no working ${tool} binary was found" >&2
    return 1
}

FFMPEG="$(resolve_media_tool ffmpeg)"
FFPROBE="$(resolve_media_tool ffprobe)"

font_path=""
for candidate in \
    /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
    /usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf \
    /usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf; do
    if [[ -f "${candidate}" ]]; then
        font_path="${candidate}"
        break
    fi
done

if [[ -z "${font_path}" ]]; then
    echo "error: no supported bold TrueType font was found" >&2
    exit 1
fi

for ((index = 0; index < FRAME_COUNT; index++)); do
    printf -v frame_name '%05d.png' "${index}"
    if [[ ! -f "${GT_DIR}/${frame_name}" ]]; then
        echo "error: missing ground-truth frame ${frame_name}" >&2
        exit 1
    fi
    if [[ ! -f "${PREDICTION_DIR}/${frame_name}" ]]; then
        echo "error: missing prediction frame ${frame_name}" >&2
        exit 1
    fi
done

mkdir -p "${VIDEO_DIR}"

filter_complex="[0:v]drawtext=fontfile=${font_path}:text='Ground Truth':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.65:boxborderw=12:x=24:y=24[left];[1:v]drawtext=fontfile=${font_path}:text='3DGS Render':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.65:boxborderw=12:x=24:y=24[right];[left][right]hstack=inputs=2[comparison]"

command=(
    "${FFMPEG}"
    -hide_banner
    -nostdin
    -y
    -framerate "${FRAME_RATE}"
    -start_number 0
    -i "${GT_DIR}/%05d.png"
    -framerate "${FRAME_RATE}"
    -start_number 0
    -i "${PREDICTION_DIR}/%05d.png"
    -filter_complex "${filter_complex}"
    -map '[comparison]'
    -frames:v "${FRAME_COUNT}"
    -c:v libx264
    -preset medium
    -crf 18
    -pix_fmt yuv420p
    -movflags +faststart
    "${VIDEO_PATH}"
)

printf 'Video command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}" 2>&1 | tee "${LOG_PATH}"

"${FFPROBE}" \
    -v error \
    -show_entries \
    stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size \
    -of json \
    "${VIDEO_PATH}" \
    > "${METADATA_PATH}"

echo "Comparison video ready at ${VIDEO_PATH}"
echo "Media metadata written to ${METADATA_PATH}"
