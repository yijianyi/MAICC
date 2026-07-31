#!/usr/bin/env bash
set -euo pipefail

RAW="${1:-/mnt/cfs/shaopt/data/MARL/offline_sequence_samples/overfit16_1280rec_3000_clean_8gpu4proc_20260728_174515}"
OUT="${2:-/mnt/cfs/shaopt/data/MARL/maicc_tabx/overfit16_l64}"

python tools/scan_tabx_dataset.py "${RAW}" \
  --max-agents 20 \
  --max-enemies 20 \
  --max-zones 20 \
  --expected-action-dim 8 \
  --output "${OUT}/preflight.json"

python tools/convert_tabx_to_maicc.py "${RAW}" \
  --output-dir "${OUT}" \
  --sequence-length 64 \
  --stride 64 \
  --partial-policy keep \
  --compression lzf \
  --invalid-record-policy error \
  --max-agents 20 \
  --expected-action-dim 8

python tools/compute_tabx_normalization.py "${OUT}/manifest.json" \
  --output "${OUT}/obs_normalization.npz"

echo "TABX preparation complete: ${OUT}"
