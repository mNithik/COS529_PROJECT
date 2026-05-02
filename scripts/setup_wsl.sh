#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${ROOT_DIR}/yolov5/requirements.txt"
python -m pip install tensorboard timm numba xlsxwriter

python "${ROOT_DIR}/scripts/prepare_vedai_manifests.py"

echo
echo "Phase 1 environment is ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
