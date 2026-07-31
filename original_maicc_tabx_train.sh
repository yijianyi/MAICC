#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage:"
  echo "  stage 0: $0 TRAIN_MANIFEST 0"
  echo "  stage 1: $0 TRAIN_MANIFEST 1 STAGE0_CHECKPOINT_ROOT"
  exit 2
fi

TRAIN_MANIFEST="$1"
STAGE="$2"
CHECKPOINT_ROOT="${3:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-7}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DATA_PARALLEL="${DATA_PARALLEL:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/original_maicc_tabx/seed_${SEED}}"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-}"

ARGS=(
  "${TRAIN_MANIFEST}"
  --stage "${STAGE}"
  --seed "${SEED}"
  --batch-size 32
  --results-dir "${RESULTS_DIR}"
  --device cuda
)

if [[ "${DATA_PARALLEL}" == "1" ]]; then
  if [[ "${MICRO_BATCH_SIZE}" != "0" ]]; then
    echo "DATA_PARALLEL=1 requires MICRO_BATCH_SIZE=0"
    exit 2
  fi
  ARGS+=(--data-parallel)
else
  ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE}")
  if [[ "${MICRO_BATCH_SIZE}" != "0" ]]; then
    ARGS+=(--gradient-checkpointing)
  fi
fi

if [[ -n "${EMBEDDING_CACHE}" ]]; then
  ARGS+=(--embedding-cache "${EMBEDDING_CACHE}")
fi

if [[ "${STAGE}" == "1" ]]; then
  if [[ -z "${CHECKPOINT_ROOT}" ]]; then
    echo "Stage 1 requires STAGE0_CHECKPOINT_ROOT"
    exit 2
  fi
  ARGS+=(--checkpoint-path "${CHECKPOINT_ROOT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON_BIN}" tools/run_original_maicc_tabx.py "${ARGS[@]}"
