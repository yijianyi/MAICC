# TABX training on the 16-scenario server dataset

This adapter trains MAICC from TABX without routing TABX through the original
Gymma/SMACv2 runner or `InContextReplayBuffer`.

The raw server dataset is expected at:

```text
/mnt/cfs/shaopt/data/MARL/offline_sequence_samples/
  overfit16_1280rec_3000_clean_8gpu4proc_20260728_174515
```

The converter supports the production layout:

```text
task_pkgs/task_*/
job_*/records/task_*/record-*
```

## 1. Environment

Use the original MAICC Python environment:

```bash
conda activate MAICC
pip install -r requirements.txt
```

Install the complete requirements, including FAISS: the original
`EpisodeBatch` module imports it even though stage-0 does not perform a
retrieval query. Run all commands from the repository root.

## 2. Preflight the raw dataset

```bash
RAW=/mnt/cfs/shaopt/data/MARL/offline_sequence_samples/overfit16_1280rec_3000_clean_8gpu4proc_20260728_174515
OUT=/mnt/cfs/shaopt/data/MARL/maicc_tabx/overfit16_l64

python tools/scan_tabx_dataset.py "$RAW" \
  --max-agents 20 \
  --max-enemies 20 \
  --max-zones 20 \
  --expected-action-dim 8 \
  --output "$OUT/preflight.json"
```

Do not continue until the report has `"ok": true`. If the report contains
incomplete records, review them first. `--allow-invalid-records` only changes
the scanner exit status; it never hides the invalid-record list.

Some production exports contain records but omit `task_pkgs/task.json`. This
is supported: the scanner infers active enemies from `unit_team.npy` and
`unit_max_health.npy`, and reads padded capacities from `meta.json`. Before
conversion, confirm that `global_schema` contains all of:

```text
max_active_allies
max_active_enemies
max_ally_capacity
max_enemy_capacity
max_unit_capacity
max_zone_capacity
```

`max_actual_zones` can remain `null` when task packages are absent; in that
case the stricter `max_zone_capacity` is checked against `--max-zones`.

## 3. Convert to task-sharded HDF5

Use an explicit sequence length. Do not use `auto` on the complete dataset.

```bash
python tools/convert_tabx_to_maicc.py "$RAW" \
  --output-dir "$OUT" \
  --sequence-length 64 \
  --stride 64 \
  --partial-policy keep \
  --compression lzf \
  --invalid-record-policy error \
  --max-agents 20 \
  --expected-action-dim 8
```

If preflight confirms that specific raw records are incomplete copies, rerun
with `--invalid-record-policy skip`. Every skipped record is retained in the
output manifest with its error.

The converter rejects cross-task observation or action dimension changes. The
model reads `obs_dim` from the manifest; it does not assume that the current
16-scenario dataset already uses the future 20/20/20 observation width.

## 4. Compute zero-preserving observation scaling

```bash
python tools/compute_tabx_normalization.py "$OUT/manifest.json" \
  --output "$OUT/obs_normalization.npz"
```

The default normalizer uses RMS scaling without mean centering, so zero-padded
entity and zone features remain zero.

## 5. Run a short stage-0 pipeline check

The checked-in config points at the server paths above:

```bash
python tools/train_tabx_stage0.py \
  --config src/config/tabx/tabx_stage0.yaml \
  --steps 100 \
  --max-batch-size 2 \
  --no-train-cem \
  --output-dir /mnt/cfs/shaopt/data/MARL/maicc_results/overfit16_smoke
```

This trains the decentralized embedding model only. Check that all losses and
gradient norms remain finite and that a checkpoint is written.

## 6. Stage-0 production training

```bash
python tools/train_tabx_stage0.py \
  --config src/config/tabx/tabx_stage0.yaml \
  --no-train-cem
```

The loader chooses a task uniformly, reads only the sampled HDF5 rows, and
reduces batch size according to `agents * sequence_length`. It does not load
all tasks into RAM.

To train the centralized teacher after the DEM pipeline is stable:

```bash
python tools/train_tabx_stage0.py \
  --config src/config/tabx/tabx_stage0.yaml \
  --train-cem \
  --resume /mnt/cfs/shaopt/data/MARL/maicc_results/overfit16_stage0/step_000100000 \
  --output-dir /mnt/cfs/shaopt/data/MARL/maicc_results/overfit16_stage0_cem
```

`tabx_transformer` replaces the original dense centralized attention with a
spatial-then-temporal CEM. DEM, DT and the MAICC losses keep their original
interfaces.

## 7. Build retrieval indexes

Update `retrieval.stage0_checkpoint` in
`src/config/tabx/tabx_stage1.yaml`, then run:

```bash
python tools/build_tabx_index.py \
  --config src/config/tabx/tabx_stage1.yaml
```

The builder:

- encodes every trajectory in batches;
- stores anchors at the configured stride;
- writes one persistent FAISS index per time anchor;
- stores only task/row/agent metadata beside the index;
- leaves observations in HDF5.

`max_index_rows_per_task` bounds exact-index memory. The checked-in 16-task
configuration selects at most 1024 converted trajectories per task, spread
evenly across that task.

## 8. Stage-1 training

```bash
python tools/train_tabx_stage1.py \
  --config src/config/tabx/tabx_stage1.yaml
```

The retrieval store is built once and reused. Context trajectories are loaded
from HDF5 after search, and the current query trajectory is excluded.

## Operational notes

- Start with sequence length 64 and AMP disabled.
- Use `nvidia-smi` during the first maximum-agent task.
- Do not enable the original `transformer` CEM for 20-agent TABX training.
- Resume with exactly the same model dimensions and normalization file.
- The current entrypoints are single-process. Use one selected GPU via
  `CUDA_VISIBLE_DEVICES`; multi-process distributed training is a separate
  optimization and is not required to keep the dataset out of RAM.
- Online evaluation requires the TABX simulator package/API. It is deliberately
  not guessed in this repository; offline stage-0 and stage-1 do not depend on
  that simulator.
