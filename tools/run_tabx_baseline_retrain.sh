#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MAICC_DATA_DIR="${MAICC_DATA_DIR:-/mnt/cfs/shaopt/data/MARL/yxq/maicc_tabx/0811_16task_l100}"
export MAICC_SEED="${MAICC_SEED:-7}"
export MAICC_RUN_DIR="${MAICC_RUN_DIR:-/mnt/cfs/shaopt/data/MARL/yxq/maicc_results/0811_16task_l100_baseline_v2_seed${MAICC_SEED}}"
BASELINE_PYTHON="${BASELINE_PYTHON:-python}"
BASELINE_STAGE0=src/config/tabx/baseline_stage0.yaml
BASELINE_STAGE1=src/config/tabx/baseline_stage1.yaml
case "${1:-check}" in
  verify)
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$BASELINE_PYTHON" -m pytest tests/tabx -q
    ;;
  check)
    "$BASELINE_PYTHON" tools/check_tabx_baseline.py --config "$BASELINE_STAGE0"
    ;;
  smoke)
    "$BASELINE_PYTHON" tools/check_tabx_baseline.py --config "$BASELINE_STAGE0"
    "$BASELINE_PYTHON" tools/train_tabx_stage0.py --config "$BASELINE_STAGE0" \
      --steps 2 --max-batch-size 1 --output-dir "$MAICC_RUN_DIR/smoke_stage0"
    ;;
  stage0)
    "$BASELINE_PYTHON" tools/train_tabx_stage0.py --config "$BASELINE_STAGE0" "${@:2}"
    ;;
  index)
    "$BASELINE_PYTHON" tools/build_tabx_index.py --config "$BASELINE_STAGE1" "${@:2}"
    ;;
  stage1)
    "$BASELINE_PYTHON" tools/train_tabx_stage1.py --config "$BASELINE_STAGE1" "${@:2}"
    ;;
  all)
    bash "$0" check
    bash "$0" stage0
    bash "$0" index
    bash "$0" stage1
    ;;
  eval)
    "$BASELINE_PYTHON" tools/eval_tabx_online.py --config "$BASELINE_STAGE1" \
      --checkpoint "$MAICC_RUN_DIR/stage1/step_000500000" --episodes 100 \
      --output-dir "$MAICC_RUN_DIR/eval" "${@:2}"
    ;;
  *) echo "Usage: bash $0 {verify|check|smoke|stage0|index|stage1|all|eval} [extra arguments]" >&2; exit 2 ;;
esac
