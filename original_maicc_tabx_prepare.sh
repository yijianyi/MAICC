#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 RAW_TABX_DIR CONVERTED_DIR BASELINE_DIR [TRAJS_PER_TASK]"
  exit 2
fi

RAW_TABX_DIR="$1"
CONVERTED_DIR="$2"
BASELINE_DIR="$3"
TRAJS_PER_TASK="${4:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INCLUDE_SHORT_TERMINAL="${INCLUDE_SHORT_TERMINAL:-0}"

"${PYTHON_BIN}" tools/scan_tabx_dataset.py "${RAW_TABX_DIR}"

"${PYTHON_BIN}" tools/convert_tabx_to_maicc.py \
  "${RAW_TABX_DIR}" \
  --output-dir "${CONVERTED_DIR}" \
  --sequence-length 64 \
  --stride 64 \
  --partial-policy drop \
  --max-agents 20 \
  --expected-action-dim 8

BUILD_ARGS=()
if [[ "${INCLUDE_SHORT_TERMINAL}" == "1" ]]; then
  BUILD_ARGS+=(--include-short-terminal)
fi

"${PYTHON_BIN}" tools/build_original_maicc_tabx_dataset.py \
  "${CONVERTED_DIR}/manifest.json" \
  --output-dir "${BASELINE_DIR}" \
  --trajs-per-task "${TRAJS_PER_TASK}" \
  --selection-seed 7 \
  --train-fraction 0.8 \
  --validation-fraction 0.1 \
  --test-fraction 0.1 \
  "${BUILD_ARGS[@]}"

echo "Original-MAICC TABX manifest:"
echo "${BASELINE_DIR}/train_manifest.json"
