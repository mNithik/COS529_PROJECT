#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

INPUT_MODE="${INPUT_MODE:-RGB+IR+MF}"
RUN_NAME="${RUN_NAME:-phase1_baseline}"
MODEL_CFG="${MODEL_CFG:-${ROOT_DIR}/yolov5/models/MyYolo.yaml}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_IMG_SIZE="${TRAIN_IMG_SIZE:-1024}"
TEST_IMG_SIZE="${TEST_IMG_SIZE:-512}"
DEVICE="${DEVICE:-0}"

if [[ -z "${VEDAI_DATA_ROOT:-}" ]]; then
  if [[ -d "${ROOT_DIR}/data/VEDAI/labels" ]]; then
    export VEDAI_DATA_ROOT="${ROOT_DIR}/data/VEDAI"
  elif [[ -d "${ROOT_DIR}/../MultiModalFusion/data/VEDAI/labels" ]]; then
    export VEDAI_DATA_ROOT="${ROOT_DIR}/../MultiModalFusion/data/VEDAI"
  fi
fi

if [[ -z "${VEDAI_DATA_ROOT:-}" || ! -d "${VEDAI_DATA_ROOT}/labels" ]]; then
  echo "Could not find a usable VEDAI label directory."
  echo "Checked:"
  echo "  ${ROOT_DIR}/data/VEDAI/labels"
  echo "  ${ROOT_DIR}/../MultiModalFusion/data/VEDAI/labels"
  echo
  echo "Set VEDAI_DATA_ROOT manually if your full dataset lives elsewhere."
  exit 1
fi

python "${ROOT_DIR}/scripts/prepare_vedai_manifests.py"
python "${ROOT_DIR}/scripts/prepare_phase1_dataset.py" \
  --source-root "${VEDAI_DATA_ROOT}" \
  --target-root "${ROOT_DIR}/data/phase1_vedai"

export VEDAI_DATA_ROOT="${ROOT_DIR}/data/phase1_vedai"

echo "Using VEDAI_DATA_ROOT=${VEDAI_DATA_ROOT}"
echo "Running ${RUN_NAME} with INPUT_MODE=${INPUT_MODE}"
echo "Using MODEL_CFG=${MODEL_CFG}"

python "${ROOT_DIR}/yolov5/train1.py" \
  --weights "${ROOT_DIR}/yolov5m.pt" \
  --cfg "${MODEL_CFG}" \
  --data "${ROOT_DIR}/yolov5/data/vedai_phase1.yaml" \
  --hyp "${ROOT_DIR}/yolov5/data/hyp.scratch.yaml" \
  --input_mode "${INPUT_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --train_img_size "${TRAIN_IMG_SIZE}" \
  --test_img_size "${TEST_IMG_SIZE}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --name "${RUN_NAME}"
