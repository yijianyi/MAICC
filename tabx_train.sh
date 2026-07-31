#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-src/config/tabx/tabx_stage0.yaml}"

python tools/train_tabx_stage0.py \
  --config "${CONFIG}" \
  --no-train-cem
