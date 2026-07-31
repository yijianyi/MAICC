# Original MAICC baseline on TABX

This path runs the original MAICC model, loss, Sacred entry point, sequential
training loop, and retrieval formula on TABX data. It is separate from the
factorized, streaming TABX adaptation documented in `TABX_TRAINING.md`.

## Fidelity boundary

The launcher verifies these files against upstream commit `e1640a9` before
training:

- `src/main.py`
- `src/run.py`
- `src/config/algs/MAICC.yaml`
- `src/controllers/ic_controller.py`
- `src/modules/agents/transformer_agent.py`
- `src/components/episode_buffer.py`

The working-tree `src/learners/ic_learner.py` contains TABX adaptations, so the
baseline uses the byte-identical upstream learner frozen at:

```text
baseline/original_maicc_tabx/vendor/ic_learner.py
```

No upstream file is edited or replaced on disk. The launcher registers a
schema-only TABX environment, a lazy HDF5 storage adapter, and the frozen
learner in memory, then executes the untouched `src/main.py` through `runpy`.

There are two explicit data contracts:

- `strict`: only full `L`-transition rows; Stage-0 and Stage-1 can execute the
  byte-identical upstream learner.
- `terminal_zero_fill`: also retains naturally terminated rows shorter than
  `L`. Stage-0 delegates to the upstream learner. Stage-1 uses a separate
  padding-compatible subclass that limits the shared random prefix to real
  transitions. This mode is reported as `padding_compatible_original`, not
  byte-exact.

The enforced model contract is:

```text
agent=transformer
CEM=CustomGPT2Model
n_embd=64
n_layer=8
n_head=8
hidden_dim=64
train_CEM=True
use_rtg=False
k=2
l=50
max_similarity=0.98
```

The baseline does not use:

- `FactorizedCEM`
- `TabxEpisodeSource`
- `TabxRetrievalStore`
- persistent per-anchor FAISS indices
- RMS observation normalization
- dynamic agent/time batch sizing
- retrieval self-exclusion

## 1. Prepare converted TABX data

First run the general lossless TABX converter:

```bash
python tools/convert_tabx_to_maicc.py /path/to/raw_tabx \
  --output-dir /path/to/converted \
  --sequence-length 64 \
  --stride 64 \
  --partial-policy drop \
  --max-agents 20 \
  --expected-action-dim 8
```

Then build the strict baseline dataset:

```bash
python tools/build_original_maicc_tabx_dataset.py \
  /path/to/converted/manifest.json \
  --output-dir /path/to/original_maicc_baseline \
  --trajs-per-task 0 \
  --selection-seed 7 \
  --train-fraction 0.8 \
  --validation-fraction 0.1 \
  --test-fraction 0.1
```

`--trajs-per-task 0` selects the largest common multiple of 20 that every
training task can supply. It must be at least 32. The builder:

1. retains only rows with exactly `L` transitions and `filled=1` in all `L+1`
   slots;
2. removes terminal-aligned or other overlapping windows;
3. assigns complete source episodes to train/validation/test before selecting
   rows;
4. uses the same trajectory count for every training task;
5. copies raw observations without normalization.

The convenience wrapper performs both steps:

```bash
./original_maicc_tabx_prepare.sh \
  /path/to/raw_tabx \
  /path/to/converted \
  /path/to/original_maicc_baseline
```

### Include naturally terminated short games

The general converter already represents a short natural terminal by writing
its real transitions first, setting `terminated=1` on the final real
transition, zero-filling the remaining slots, and setting `filled=0` after the
terminal next-observation slot. Retain those rows with:

```bash
python tools/build_original_maicc_tabx_dataset.py \
  /path/to/converted/manifest.json \
  --output-dir /path/to/original_maicc_padded \
  --trajs-per-task 1000 \
  --selection-seed 7 \
  --train-fraction 0.8 \
  --validation-fraction 0.1 \
  --test-fraction 0.1 \
  --include-short-terminal
```

Use `--trajs-per-task 0` if the data cannot provide 1000 eligible rows for
every task. The builder still removes overlapping windows and assigns complete
source episodes to a split before selecting rows.

For the convenience wrapper:

```bash
INCLUDE_SHORT_TERMINAL=1 ./original_maicc_tabx_prepare.sh \
  /path/to/raw_tabx \
  /path/to/converted \
  /path/to/original_maicc_padded \
  1000
```

Padding-compatible training preserves the original dense model, Stage-0 loss,
Stage-1 DT loss magnitude, Adam update, retrieval score, random candidate
permutation, and lack of self-exclusion. Its two required adaptations are:

1. Stage-1 samples `rand_t` no later than the shortest real trajectory in the
   sampled batch.
2. Retrieval at timestep `t` indexes only trajectories for which `t` is a real
   transition.

The launcher reads this mode from the manifest. Stage commands are otherwise
the same and print:

```text
padding_mode=terminal_zero_fill
baseline_execution="padding_compatible_original"
```

Before training, scan every row and estimate storage:

```bash
python tools/validate_original_maicc_tabx_dataset.py \
  /path/to/original_maicc_baseline/train_manifest.json
```

## 2. Stage-0

Byte-for-byte upstream learner, if a full batch fits:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_original_maicc_tabx.py \
  /path/to/original_maicc_baseline/train_manifest.json \
  --stage 0 \
  --seed 7 \
  --batch-size 32 \
  --device cuda
```

For 20 agents and length 64, dense CEM normally requires microbatch execution:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_original_maicc_tabx.py \
  /path/to/original_maicc_baseline/train_manifest.json \
  --stage 0 \
  --seed 7 \
  --batch-size 32 \
  --micro-batch-size 1 \
  --gradient-checkpointing \
  --device cuda
```

The effective batch remains 32. The adapter samples one original MAICC batch,
weights each microbatch by `micro_size / 32`, and performs one Adam step.

## 3. Stage-1

Pass the directory containing the numeric Stage-0 checkpoint directories:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_original_maicc_tabx.py \
  /path/to/original_maicc_baseline/train_manifest.json \
  --stage 1 \
  --checkpoint-path \
    /path/to/results/models/IC-MARL_tabx_mode0 \
  --seed 7 \
  --batch-size 32 \
  --micro-batch-size 1 \
  --gradient-checkpointing \
  --embedding-cache /path/to/embedding_cache \
  --device cuda
```

Stage-1 uses the original `InContextReplayBuffer.mips` scoring, exact
`IndexFlatIP`, `get_dem_traj_embedding`, random candidate permutation, and DT
loss. Embeddings may be stored as memory-mapped `.npy` files, but their values,
ordering, and query formula are unchanged.

### Multi-GPU data parallelism

Stage-0 and Stage-1 can each split their global batch over all visible GPUs.
The two stages remain sequential because Stage-1 consumes the final Stage-0
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_original_maicc_tabx.py \
  /path/to/original_maicc_baseline/train_manifest.json \
  --stage 0 \
  --seed 7 \
  --batch-size 32 \
  --data-parallel \
  --device cuda
```

Then run Stage-1 on the same visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_original_maicc_tabx.py \
  /path/to/original_maicc_baseline/train_manifest.json \
  --stage 1 \
  --checkpoint-path \
    /path/to/results/models/IC-MARL_tabx_mode0 \
  --embedding-cache /path/to/embedding_cache \
  --seed 7 \
  --batch-size 32 \
  --data-parallel \
  --device cuda
```

The global batch remains 32. Replica losses are weighted by the number of
valid transitions before one AMP/Adam update, including terminal-padded rows.
Stage-1 embedding construction also uses the visible GPUs. Do not combine
`--data-parallel` with `--micro-batch-size`.

This mode preserves the original model, sampling contract, loss terms,
retrieval formula, optimiser, update count, and checkpoint key format.
Multi-GPU reduction order and per-device dropout random streams are not
byte-identical to a one-GPU run, so the launcher records
`baseline_execution="data_parallel_equivalent"` (or
`"padding_compatible_data_parallel"` for terminal padding).

The wrapper script is:

```bash
./original_maicc_tabx_train.sh TRAIN_MANIFEST 0
./original_maicc_tabx_train.sh \
  TRAIN_MANIFEST 1 STAGE0_CHECKPOINT_ROOT
```

## 4. Reproducibility

Run at least three seeds:

```bash
for seed in 7 17 27; do
  SEED="${seed}" ./original_maicc_tabx_train.sh TRAIN_MANIFEST 0
done
```

Record separately whether `--micro-batch-size` and
`--gradient-checkpointing` were used. They are execution optimizations, not
model changes. The wrapper puts each seed under a separate
`results/original_maicc_tabx/seed_<seed>` directory so original MAICC's
checkpoint naming cannot overwrite another seed. The launcher records the
manifest hash, upstream commit, execution mode, and prints the Torch,
Transformers, and FAISS versions; retain those values with the experiment.

## 5. Validation

Pure fidelity and mapping tests:

```bash
python -m unittest discover \
  -s tests/original_maicc_tabx \
  -p "test_*.py" \
  -v
```

The launcher also performs the upstream hash check on every run. Use
`--dry-run` to inspect the enforced Sacred command without starting training:

```bash
python tools/run_original_maicc_tabx.py \
  /path/to/original_maicc_baseline/train_manifest.json \
  --stage 0 \
  --dry-run
```

TABX has no simulator in this repository. Offline training therefore cannot be
reported as the original paper's online return or win rate. Use a common
held-out offline evaluator for all methods, or add the TABX simulator later for
online comparison.
