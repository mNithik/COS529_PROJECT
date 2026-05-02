#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export INPUT_MODE=RGB+IR+MF
export RUN_NAME=phase2_fem
export MODEL_CFG="${ROOT_DIR}/yolov5/models/MyYolo_FEM.yaml"

bash "${ROOT_DIR}/scripts/run_phase1_train.sh"
