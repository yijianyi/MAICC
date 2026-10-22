#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# User configuration
#
# Every value can also be overridden without editing this file, for example:
#   GPU_ID=3 EPISODES=20 OUTPUT_ROOT=/mnt/cfs/.../test \
#     bash tools/eval_tabx_all_tasks_parallel.sh
# ---------------------------------------------------------------------------
GPU_ID="${GPU_ID:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
EPISODES="${EPISODES:-100}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
EPISODE_LIMIT="${EPISODE_LIMIT:-512}"
BASE_SEED="${BASE_SEED:-20000}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/cfs/shaopt/conda_envs/MAICC_TABX/bin/python}"
export MAICC_DATA_DIR="${MAICC_DATA_DIR:-/mnt/cfs/shaopt/data/MARL/yxq/maicc_tabx/0811_16task_l100}"
export MAICC_SEED="${MAICC_SEED:-7}"
export MAICC_RUN_DIR="${MAICC_RUN_DIR:-/mnt/cfs/shaopt/data/MARL/yxq/maicc_results/0811_16task_l100_baseline_v2_seed${MAICC_SEED}}"
CONFIG="${CONFIG:-src/config/tabx/baseline_stage1.yaml}"
CHECKPOINT="${CHECKPOINT:-$MAICC_RUN_DIR/stage1/step_000500000}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/mnt/cfs/shaopt/data/MARL/0803_overfit_7task_train_7000record_balance_winrate}"
MANIFEST="${MANIFEST:-$MAICC_DATA_DIR/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAICC_RUN_DIR/eval_parallel}"

# CPU threads used by each evaluator process. FAISS has its own
# `faiss_threads` setting under `retrieval` in CONFIG.
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer "$PARALLEL_JOBS" || die "PARALLEL_JOBS must be positive"
is_positive_integer "$EPISODES" || die "EPISODES must be positive"
is_positive_integer "$EPISODE_LIMIT" || die "EPISODE_LIMIT must be positive"
is_nonnegative_integer "$BASE_SEED" || die "BASE_SEED must be non-negative"
is_positive_integer "$CPU_THREADS_PER_JOB" || \
  die "CPU_THREADS_PER_JOB must be positive"

[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -f "$CONFIG" ]] || die "Config does not exist: $CONFIG"
[[ -d "$CHECKPOINT" ]] || die "Checkpoint does not exist: $CHECKPOINT"
[[ -d "$RAW_DATA_ROOT" ]] || die "Raw data root does not exist: $RAW_DATA_ROOT"
[[ -f "$MANIFEST" ]] || die "Manifest does not exist: $MANIFEST"

mkdir -p "$OUTPUT_ROOT"

mapfile -t TASK_IDS < <(
  "$PYTHON_BIN" - "$MANIFEST" "$EVAL_SPLIT" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

tasks = manifest.get("tasks")
if not isinstance(tasks, list) or not tasks:
    raise SystemExit(f"Manifest contains no tasks: {manifest_path}")

for task in tasks:
    if sys.argv[2] == "all" or task.get("split", "train") == sys.argv[2]:
        print(task["task_id"])
PY
)

TASK_COUNT="${#TASK_IDS[@]}"
(( TASK_COUNT > 0 )) || die "No task IDs were read from $MANIFEST"
if (( PARALLEL_JOBS > TASK_COUNT )); then
  PARALLEL_JOBS="$TASK_COUNT"
fi

cat <<EOF
[parallel-eval] gpu=$GPU_ID
[parallel-eval] split=$EVAL_SPLIT tasks=$TASK_COUNT parallel_jobs=$PARALLEL_JOBS
[parallel-eval] episodes_per_task=$EPISODES episode_limit=$EPISODE_LIMIT
[parallel-eval] checkpoint=$CHECKPOINT
[parallel-eval] output_root=$OUTPUT_ROOT
EOF

ALL_PIDS=()
INTERRUPTED=0

stop_children() {
  local pid
  INTERRUPTED=1
  trap - INT TERM
  echo "Stopping parallel evaluation processes..." >&2
  for pid in "${ALL_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${ALL_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit 130
}

trap stop_children INT TERM

FAILED=0

# Run in bounded batches so PARALLEL_JOBS can safely be changed to a smaller
# value on machines with less GPU memory or fewer CPU cores.
for ((batch_start = 0; batch_start < TASK_COUNT; batch_start += PARALLEL_JOBS)); do
  BATCH_PIDS=()
  BATCH_TASKS=()

  for ((offset = 0; offset < PARALLEL_JOBS; offset++)); do
    task_index=$((batch_start + offset))
    if (( task_index >= TASK_COUNT )); then
      break
    fi

    task_id="${TASK_IDS[$task_index]}"
    task_seed=$((BASE_SEED + task_index * EPISODES))
    task_output="$OUTPUT_ROOT/$task_id"
    task_log="$task_output/eval.log"
    mkdir -p "$task_output"

    env \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      JAX_PLATFORMS=cpu \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
      OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
      MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
      OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
      NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
      PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" -u tools/eval_tabx_online.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --task-id "$task_id" \
        --episodes "$EPISODES" \
        --episode-limit "$EPISODE_LIMIT" \
        --seed "$task_seed" \
        --raw-data-root "$RAW_DATA_ROOT" \
        --output-dir "$task_output" \
        --resume \
        >"$task_log" 2>&1 &

    pid=$!
    ALL_PIDS+=("$pid")
    BATCH_PIDS+=("$pid")
    BATCH_TASKS+=("$task_id")
    echo "[started] task=$task_id seed=$task_seed pid=$pid log=$task_log"
  done

  for i in "${!BATCH_PIDS[@]}"; do
    pid="${BATCH_PIDS[$i]}"
    task_id="${BATCH_TASKS[$i]}"
    if wait "$pid"; then
      echo "[completed] task=$task_id"
    else
      status=$?
      echo "[failed] task=$task_id pid=$pid exit_code=$status" >&2
      FAILED=1
    fi
  done
done

trap - INT TERM

# Aggregate the independently written task summaries into one table. Missing
# summaries are listed explicitly, and cause a non-zero launcher exit code.
AGGREGATE_STATUS=0
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$MANIFEST" "$EPISODES" <<'PY' || \
  AGGREGATE_STATUS=$?
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
expected_episodes = int(sys.argv[3])

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
task_ids = [str(task["task_id"]) for task in manifest["tasks"]]
rows = []
missing = []

for task_id in task_ids:
    summary_path = output_root / task_id / "summary.json"
    if not summary_path.is_file():
        missing.append(task_id)
        continue
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = [
        item for item in payload.get("tasks", [])
        if str(item.get("task_id")) == task_id
    ]
    if len(matches) != 1 or int(matches[0].get("episodes", -1)) != expected_episodes:
        missing.append(task_id)
        continue
    rows.append(matches[0])

fieldnames = [
    "task_id",
    "physics",
    "enemy_heuristic",
    "episodes",
    "seed_start",
    "seed_end",
    "wins",
    "win_rate",
    "mean_return",
    "mean_length",
    "task_package",
]
csv_path = output_root / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

total_episodes = sum(int(row["episodes"]) for row in rows)
total_wins = sum(int(row["wins"]) for row in rows)
aggregate = {
    "status": "complete" if not missing else "incomplete",
    "requested_tasks": len(task_ids),
    "completed_tasks": len(rows),
    "episodes_per_task": expected_episodes,
    "total_episodes": total_episodes,
    "total_wins": total_wins,
    "overall_win_rate": (
        total_wins / total_episodes if total_episodes else 0.0
    ),
    "macro_win_rate": (
        sum(float(row["win_rate"]) for row in rows) / len(rows)
        if rows else 0.0
    ),
    "missing_tasks": missing,
    "tasks": rows,
}
json_path = output_root / "summary.json"
json_path.write_text(
    json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(json.dumps(aggregate, indent=2, ensure_ascii=False))
print(f"Wrote combined CSV: {csv_path}")
print(f"Wrote combined JSON: {json_path}")

if missing:
    raise SystemExit(1)
PY

if (( INTERRUPTED != 0 || FAILED != 0 || AGGREGATE_STATUS != 0 )); then
  exit 1
fi

echo "All TABX task evaluations completed successfully."
