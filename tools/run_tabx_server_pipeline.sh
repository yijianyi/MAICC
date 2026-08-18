#!/usr/bin/env bash
set -euo pipefail

# Usage: bash tools/run_tabx_server_pipeline.sh prepare|stage0|index|stage1|all
COMMAND="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_DATA="${RAW_DATA:-/mnt/cfs/shaopt/data/MARL/0803_overfit_7task_train_7000record_balance_winrate}"
DATA_OUT="${DATA_OUT:-/mnt/cfs/shaopt/data/MARL/maicc_tabx/0803_7task_l32}"
MAX_ALLIES="${MAX_ALLIES:-10}"
MAX_ENEMIES="${MAX_ENEMIES:-10}"
MAX_ZONES="${MAX_ZONES:-4}"
OBS_DIM="${OBS_DIM:-342}"
STAGE0_CONFIG="${STAGE0_CONFIG:-src/config/tabx/tabx_stage0.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-src/config/tabx/tabx_stage1.yaml}"

prepare() {
  mkdir -p "$DATA_OUT"
  "$PYTHON_BIN" tools/scan_tabx_dataset.py "$RAW_DATA" \
    --max-agents "$MAX_ALLIES" \
    --max-enemies "$MAX_ENEMIES" \
    --max-zones "$MAX_ZONES" \
    --expected-action-dim 8 \
    --expected-obs-dim "$OBS_DIM" \
    --output "$DATA_OUT/preflight.json"

  "$PYTHON_BIN" tools/convert_tabx_to_maicc.py "$RAW_DATA" \
    --output-dir "$DATA_OUT" \
    --sequence-length 32 \
    --stride 32 \
    --partial-policy drop \
    --compression lzf \
    --invalid-record-policy error \
    --max-agents "$MAX_ALLIES" \
    --expected-obs-dim "$OBS_DIM" \
    --expected-action-dim 8

  "$PYTHON_BIN" tools/compute_tabx_normalization.py \
    "$DATA_OUT/manifest.json" \
    --output "$DATA_OUT/obs_normalization.npz"
}

case "$COMMAND" in
  prepare)
    prepare
    ;;
  stage0)
    "$PYTHON_BIN" tools/train_tabx_stage0.py --config "$STAGE0_CONFIG"
    ;;
  index)
    "$PYTHON_BIN" tools/build_tabx_index.py --config "$STAGE1_CONFIG"
    ;;
  stage1)
    "$PYTHON_BIN" tools/train_tabx_stage1.py --config "$STAGE1_CONFIG"
    ;;
  all)
    prepare
    "$PYTHON_BIN" tools/train_tabx_stage0.py --config "$STAGE0_CONFIG"
    "$PYTHON_BIN" tools/build_tabx_index.py --config "$STAGE1_CONFIG"
    "$PYTHON_BIN" tools/train_tabx_stage1.py --config "$STAGE1_CONFIG"
    ;;
  *)
    echo "unknown command: $COMMAND" >&2
    exit 2
    ;;
esac
