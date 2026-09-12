# TABX → MAICC server training

This implementation uses the 7-task raw data at:

```text
/mnt/cfs/shaopt/data/MARL/0803_overfit_7task_train_7000record_balance_winrate
```

Current validation is 10 ally slots × 10 enemy slots (`obs_dim=342`). The
schema, masks and models derive dimensions from the manifest, so formal 20v20
data can use `max_agents=20`, `max_enemies=20`, `obs_dim=662` and must be
converted, normalized and trained again.

## Data contract

- Only `obs_flat` and `actions_behavior` supervise observations/actions.
- Public reward is `reward_team[:, 0]`; duplicated agent columns are checked,
  never summed.
- `actions_reference`, `actions_reference_distribution` and
  `reward_individual` are ignored.
- `terminated=done`, including a timeout loss. `truncation` is diagnostic.
- `completion=is_win`.
- Raw RTG is computed to the end of the complete source episode before 100-step
  windows are cut. The loader divides it by each task's P95 absolute episode
  return; online code must load the same fixed scale.
- Real dead agents remain present and keep their MAICC action targets.
  Never-present padding slots are excluded by `present` from action/observation
  loss and CEM attention.
- Observation loss retains invisible zero blocks but gives visible other-unit
  blocks weight 4, preventing the many zeros from dominating supervision.
- Incomplete record-tail episodes are dropped. Records are never concatenated.

## One-command pipeline

Install the repository requirements, then run from the repository root:

```bash
conda activate MAICC
pip install -r requirements.txt
bash tools/run_tabx_server_pipeline.sh prepare
bash tools/run_tabx_server_pipeline.sh stage0
bash tools/run_tabx_server_pipeline.sh index
bash tools/run_tabx_server_pipeline.sh stage1
```

`all` runs the four commands sequentially. For a small validation first:

```bash
python tools/train_tabx_stage0.py \
  --config src/config/tabx/tabx_stage0.yaml \
  --steps 100 --max-batch-size 2 \
  --output-dir /mnt/cfs/shaopt/data/MARL/maicc_results/0803_smoke
```

The `prepare` stage writes task-sharded HDF5 instead of loading the whole raw
dataset into memory. The normalizer is scale-only, zero-preserving, clips only
at ±10 for numerical protection, keeps ratio/binary/categorical dimensions at
scale 1, and estimates P99 absolute scale for remaining continuous features.

## Model and retrieval behavior

- `encoder_use_rtg=false`: CEM and DEM tokens are obs/action/post.
- `policy_use_rtg=true`: DT retains the original MAICC
  RTG/obs/action/post token order.
- Context length is 100; converted `EpisodeBatch` length is 101; TABX online
  episode limit remains 512.
- Stage 0 trains CEM/DEM. Stage 1 freezes every encoder parameter and creates a
  DT-only optimizer, keeping the prebuilt FAISS space stationary.
- Retrieval keys are per `(window, agent)`, as in MAICC. Padding agents do not
  enter the index.
- The full-history bucket uses every stride-50 100-step window. Early buckets
  1/4/8/16/32/64 use only true episode-start windows to limit index growth.
- Training queries exclude every candidate from the same source `record`, then
  search adaptively from 128 up to 4096 candidates. Different source records
  are preferred for `k=2`; diversity is relaxed only if necessary.
- The default FAISS factory is `HNSW32,Flat`. Change `index_factory` only after
  measuring recall, host RAM and latency on the server.

## Online environment adapter

`src/tabx/online_wrapper.py` accepts a constructed TABX environment (normally
wrapped by `TABXEnemyHeuristicWrapper`) and its `env_params`. It exposes
`reset`, `step`, `get_obs`, concat `get_state`, `get_avail_actions` and
`get_env_info`. Availability is always read from the environment. Padding
slots expose only IDLE (action 7), and concat state is computed on demand rather
than stored in a large replay buffer.

The included evaluator constructs a TABX scenario, loads the fixed
normalizer/index/checkpoint, and enables MAICC-style online memory:

```bash
python tools/eval_tabx_online.py \
  --config src/config/tabx/tabx_stage1.yaml \
  --checkpoint /mnt/cfs/shaopt/data/MARL/maicc_results/0803_7task_l100_stage1/step_000100000 \
  --task-id TASK_ID_FROM_MANIFEST \
  --episodes 100 \
  --seed 0 \
  --seed-stride 1 \
  --output-dir /mnt/cfs/shaopt/data/MARL/maicc_results/online_seed_0
```

Online memory starts empty, adds an episode only after terminal, never updates
model weights, and uses no individual reward or state. The probability of
choosing offline memory decays exponentially from 1 toward 0.01; pass
`--disable-online-memory` for the reproducible offline-only ablation. Every
independent evaluator process starts with an empty online memory.

`--seed` is the first episode seed for every task/scenario and episode `i` uses
`seed + (i - 1) * seed_stride`. Thus the defaults make every scenario use the
same sequence `0, 1, 2, ...`. Each completed episode is printed as an explicit
`WIN` or `LOSS` and appended to `episodes.csv`; `summary.csv` and `summary.json`
contain wins, losses, win rate, and the `W`/`L` result sequence.
When using `tools/eval_tabx_all_tasks_parallel.sh`, set `BASE_SEED` and
`SEED_STRIDE` in the environment. The launcher additionally writes a merged
`episodes.csv` and `episode_summary.csv` under `OUTPUT_ROOT`.

For 20v20 formal training, use a new output directory and configs with
`model.max_agents: 20`; set `MAX_ALLIES=20 MAX_ENEMIES=20 OBS_DIM=662` when
running `prepare`. A 10v10 checkpoint cannot be used as a complete 20v20
checkpoint because the observation projection changes.
