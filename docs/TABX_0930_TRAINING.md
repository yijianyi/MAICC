# 0930 数据：train_0 从原始记录到 online 评估

本流程对应用户确认的原实验预算：每阶段 **500000 次优化更新、有效 batch=32**。
数据为 `/mnt/cfs/shaopt/data/MARL/0930_4800task_train_48000record_comm_agent`，
采集容量为 10 个友方槽位、10 个敌方槽位、10 个区域槽位，观测维度 378，动作数 8。
容量指填充后的槽位数，不能改成任务中实际存活的单位数。

顺序：定位联合记录 → 全量检查 → 转 HDF5 → 训练集归一化 → 固定配置 →
Stage 0 → v4 索引 → Stage 1 → 选定任务 online。
本地仅核对代码和命令语法；以下命令需在服务器执行，每一步成功后再执行下一步。

## 已有产物时的快捷路径

`TASK_PACKAGE_ROOT` 是评估时读取 task.json 的输入目录。可以继续使用已有任务包，
该目录中的旧评估结果不会被这里的训练命令覆盖；新结果写入 MAICC_RUN_DIR。
目录中已经有结果，并不等于新版模型已经完成训练。按实际已有的产物跳过对应步骤：

| 已有产物 | 可跳过的工作 |
| --- | --- |
| 同一批任务的 task.json | 重新生成或重新采样任务包；保持 TASK_PACKAGE_ROOT 即可 |
| 完整联合 record，且已经确认来源和 RAW_DATA 路径 | 第 2 步的来源查找；不需要从 agent 数据重组 |
| 同一批 0930 训练数据的 manifest、全部 HDF5/JSONL，转换参数为 100/100 | 第 3 步重新扫描和转换；复用 MAICC_DATA_DIR，保留已有扫描报告 |
| 与该 manifest、训练划分和 10/10/10 容量匹配的归一化文件 | 第 4 步重新拟合；仍运行配置检查验证匹配关系 |
| 旧分支模型、v3 索引、online 结果 | 保留作为旧版本对照；当前稠密 CEM 仍需重新 Stage 0 → v4 索引 → Stage 1 → online |

若旧流程已经完整处理过同一批数据，设置 MAICC_DATA_DIR 为已有转换目录
（旧流程示例是 `/mnt/cfs/shaopt/data/MARL/yxq/maicc_tabx/0930_4800task_l100`），
MAICC_RUN_DIR 使用新的实验目录，直接从第 5 步生成配置、检查和训练。
保留已知的 RAW_DATA，online 仍会读取其中的原始 record 元信息。
服务器产物尚未在本机核验，不能仅凭目录名称判定其满足复用条件。

## 与旧流程的差异

| 项目 | train_0 的处理 |
| --- | --- |
| 数据入口 | 保留从 agent 的 meta.json 查找 source_record；不能把 comm_agent 当联合记录根目录 |
| 训练预算 | 每阶段 500000 次更新，最终目录 step_000500000；旧文本中的 100000 已过时 |
| batch | 有效 batch=32；默认微批次 1、累积 32 次。旧分支最新预算 32000 下实际也是 batch=32，旧文本的 batch=4 不适用于它 |
| CEM / 观测损失 | dense / obs_visible_weight=1，沿用当前模板 |
| 检索配置 | 删除稀疏 history_buckets、utility_alpha、online_search_k；使用 v4、全部精确前缀、Flat |
| 离线效用 | 索引构建只统计完整团队回报；在线入库才计算 DEM 预测个体回报 |
| 产物 | 新实验目录；旧 checkpoint 和 v3 索引不能用于本次训练 |

## 1. 环境和目录

先同步修改后的 `train_0` 完整工作区。未提交的本地修改不会随服务器切换分支自动出现。
继续使用原来能运行 MAICC/TABX 的环境。修改下面的服务器项目路径；任务包路径沿用旧流程，
需要在服务器确认仍然存在。准备步骤建议在持久终端会话中执行。

```bash
conda activate MAICC_TABX
export MAICC_REPO=/你的服务器项目目录/MAICC
cd "$MAICC_REPO"

export AGENT_DATA=/mnt/cfs/shaopt/data/MARL/0930_4800task_train_48000record_comm_agent
export MAICC_DATA_DIR=/mnt/cfs/shaopt/data/MARL/yxq/maicc_tabx/0930_4800task_l100_train0_v2
export MAICC_RUN_DIR=/mnt/cfs/shaopt/data/MARL/yxq/maicc_results/0930_4800task_l100_train0_v2_seed7_s500k
export MAICC_SEED=7
export MAICC_TRAIN_STEPS=500000
export TASK_PACKAGE_ROOT=/mnt/cfs/shaopt/data/MARL/yxq/TABX/coach_runs/seen_tasks_sample_from_4800tasks
export TABX_ROOT="$MAICC_REPO/TABX"

export CUDA_VISIBLE_DEVICES=0
export BASELINE_PYTHON=python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
mkdir -p "$MAICC_DATA_DIR" "$MAICC_RUN_DIR/configs" "$MAICC_RUN_DIR/logs"
set -o pipefail
```

`TABX_ROOT` 应包含 `src/tabx/__init__.py`，并与采集环境兼容。
可先检查服务器依赖；此命令不会训练模型：

```bash
python - <<'PY'
import sys
import numpy, h5py, yaml, torch, transformers, faiss, jax
print("Python:", sys.version)
print("Torch:", torch.__version__, "Transformers:", transformers.__version__)
print("JAX:", jax.__version__, "CUDA:", torch.cuda.is_available())
assert sys.version_info >= (3, 10)
assert torch.cuda.is_available(), "正式配置要求 CUDA"
PY
```

## 2. 定位原始联合记录

只读查看一个 agent 的来源；此操作不会重组、修改或复制数据。

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["AGENT_DATA"])
sample = next(root.rglob("obs_static.npy"), None)
if sample is None:
    raise SystemExit("没有找到 obs_static.npy，请检查 agent 数据目录")
meta = json.loads((sample.parent / "meta.json").read_text())
print("agent 目录:", sample.parent)
for key in ("task_id", "agent_index", "source_record", "n_units", "max_n_zone"):
    print(key, "=", meta.get(key))
value = meta.get("source_record")
if not value:
    raise SystemExit("meta.json 没有 source_record，需要确认原始联合记录的位置")
source = Path(value)
if not source.is_absolute():
    raise SystemExit("source_record 是相对路径，需要先确定其基准目录")
required = ("meta.json", "obs_flat.npy", "actions_behavior.npy", "reward_team.npy",
            "done.npy", "reset.npy", "truncation.npy", "episode_id.npy", "is_win.npy")
missing = [name for name in required if not (source / name).is_file()]
if missing:
    raise SystemExit(f"原始记录不存在或不完整: {source}; 缺失: {missing}")
for name in ("obs_flat.npy", "actions_behavior.npy", "reward_team.npy"):
    values = np.load(source / name, mmap_mode="r", allow_pickle=False)
    print(name, values.shape, values.dtype)
print("确认后的联合 record:", source)
PY
```

根据输出设置本次数据集的联合记录根目录，支持 `task_*/record-*` 和
`job_*/records/task_*/record-*` 布局。根目录必须对应本次计划使用的记录集合；
不要选包含其他训练集或测试集的共同上级目录。单条来源检查不能证明全量记录一致，
下一步还需核对任务数与 record 数。

```bash
export RAW_DATA=/实际的原始联合记录根目录
```

若 source_record 指向旧挂载位置，确认实际迁移位置后设置 RAW_DATA。
若原始联合记录确实已丢失，暂停数据转换。当前仓库没有 comm_agent 反向重组工具，
必须先核对所有 agent 的字段、槽位、时间轴和 episode 边界，不能直接改文件名代替重组。

## 3. 全量检查和转换

```bash
python tools/scan_tabx_dataset.py "$RAW_DATA" \
  --max-agents 10 --max-enemies 10 --max-zones 10 \
  --expected-action-dim 8 --expected-obs-dim 378 \
  --output "$MAICC_DATA_DIR/preflight.json" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/01_scan.log"
```

确认成功退出，报告 `ok: true`，实际记录集合与预期一致，再执行：

```bash
python tools/convert_tabx_to_maicc.py "$RAW_DATA" \
  --output-dir "$MAICC_DATA_DIR" \
  --sequence-length 100 --stride 100 \
  --partial-policy drop --compression lzf --invalid-record-policy error \
  --max-agents 10 --expected-obs-dim 378 --expected-action-dim 8 \
  2>&1 | tee "$MAICC_RUN_DIR/logs/02_convert.log"
```

转换前先按完整 episode 计算 RTG，再切窗口；完整短 episode 保留并填充，
长 episode 补终点对齐窗口，未完成片段丢弃。训练使用实际行为动作与团队奖励。
不同于原通信模型的序列长度配置，这里明确采用 100 步有限历史。

转换器没有自动划分验证集；本流程将目录中的任务全部作为 train。
若需留出任务，必须在归一化前确定 manifest 的 split；本次后续示例为 seen-task 评估。
若此前已用相同转换器、相同联合记录及 100/100 窗口参数生成 HDF5/JSONL，可核验后复用，
不用因为模型结构修正而重新转换。新处理默认使用以上独立目录，不覆盖已有产物。

## 4. 归一化

根据训练数据上界设置采样概率，保持旧流程约 5 万样本的目标。
填充会使实际有效样本数少于此目标；不是精确抽取 5 万条。

```bash
NORM_SAMPLE_P=$(python - <<'PY'
import json
import os
from pathlib import Path
m = json.loads((Path(os.environ["MAICC_DATA_DIR"]) / "manifest.json").read_text())
upper = sum(t["n_trajectories"] * m["sequence_length"] * t["n_agents"]
            for t in m["tasks"] if t.get("split", "train") == "train")
print(min(1.0, 50000 / max(1, upper)))
PY
)
python tools/compute_tabx_normalization.py \
  "$MAICC_DATA_DIR/manifest.json" \
  --output "$MAICC_DATA_DIR/obs_normalization.npz" \
  --split train --n-zones 10 --max-enemies 10 \
  --sample-probability "$NORM_SAMPLE_P" --max-samples 150000 \
  --seed "$MAICC_SEED" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/03_normalization.log"
```

训练、索引、评估共用这份文件；不能复制旧 16-task 归一化文件。
若复用此前完全相同训练数据的有效归一化文件，可跳过重新拟合；留出任务的旧文件
若缺少 fit_split，当前加载器会要求按 train 重新拟合。

## 5. 固定本次实验配置

从当前 train_0 模板生成独立配置，明确写入 50 万次更新及对应 checkpoint 路径。
不要继续运行旧文本中设置 7 个 history_buckets 的配置生成代码。

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml

repo = Path(os.environ["MAICC_REPO"])
data = Path(os.environ["MAICC_DATA_DIR"])
run = Path(os.environ["MAICC_RUN_DIR"])
steps = int(os.environ["MAICC_TRAIN_STEPS"])
for stage in (0, 1):
    template = repo / f"src/config/tabx/baseline_stage{stage}.yaml"
    cfg = yaml.safe_load(template.read_text())
    cfg["data"].update(manifest=str(data / "manifest.json"),
                       normalization=str(data / "obs_normalization.npz"))
    cfg["model"].update(max_agents=10, max_enemies=10, n_zones=10,
                        episode_limit=512, cem_architecture="dense",
                        obs_visible_weight=1.0, gradient_checkpointing=True)
    cfg["training"].update(device="cuda", seed=int(os.environ["MAICC_SEED"]),
                           total_steps=steps, max_batch_size=32,
                           agent_time_budget=1024, dt_loss_reduction="sum",
                           grad_norm_clip=0.0, output_dir=str(run / f"stage{stage}"))
    retrieval = cfg["retrieval"]
    for key in ("history_buckets", "utility_alpha", "online_search_k"):
        retrieval.pop(key, None)
    retrieval.update(k=2, search_k=50, max_similarity=0.98)
    if stage == 1:
        retrieval.update(stage0_checkpoint=str(run / "stage0" / f"step_{steps:09d}"),
                         index_dir=str(run / "retrieval"),
                         index_manifest=str(run / "retrieval/retrieval_manifest.json"),
                         prefix_policy="exact_all_windows", index_factory="Flat",
                         allow_approximate_index=False, max_index_rows_per_task=0,
                         index_cache_size=2, weight=0.8, ind_weight=0.2, use_utility=True)
        cfg["evaluation"] = dict(task_split="train", target_return_mode="train_only")
    output = run / f"configs/stage{stage}.yaml"
    with output.open("x") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    print(output)
PY
```

配置已有时会拒绝覆盖，应检查已有配置或为新实验换目录。
下面直接调用 Python 入口，保证使用这两份配置快照。
`run_tabx_baseline_retrain.sh all` 使用仓库模板，不会自动读取这里的 configs 目录。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/tabx -q
python tools/check_tabx_baseline.py --config "$MAICC_RUN_DIR/configs/stage0.yaml"
python tools/train_tabx_stage0.py \
  --config "$MAICC_RUN_DIR/configs/stage0.yaml" \
  --steps 20 --max-batch-size 1 --output-dir "$MAICC_RUN_DIR/smoke_stage0" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/04_smoke_stage0.log"
```

确认 loss 有限、模型可保存。check 应显示有效 batch=32、微批次 1、累积 32 次，
以及每阶段约 1600 万采样窗口。任务均匀随机采样，因此 50 万次更新不是 5 个 epoch。
查看 exact_index_bytes_upper_bound 并预留构建临时空间；新索引有 100 个前缀，
4800 任务的空间和读盘成本需按实际窗口数评估。默认 LRU 只限制常驻索引数量，
不会降低索引总存储量。

## 6. Stage 0、索引、Stage 1

```bash
python tools/train_tabx_stage0.py \
  --config "$MAICC_RUN_DIR/configs/stage0.yaml" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/05_stage0.log"

python tools/build_tabx_index.py \
  --config "$MAICC_RUN_DIR/configs/stage1.yaml" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/06_build_index.log"

python tools/train_tabx_stage1.py \
  --config "$MAICC_RUN_DIR/configs/stage1.yaml" \
  --steps 20 --output-dir "$MAICC_RUN_DIR/smoke_stage1" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/07_smoke_stage1.log"

python tools/train_tabx_stage1.py \
  --config "$MAICC_RUN_DIR/configs/stage1.yaml" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/08_stage1.log"
```

这些命令需要逐条成功执行。最终模型分别在 stage0/step_000500000 和
stage1/step_000500000。索引绑定 stage0 编码器、manifest 和归一化哈希，三者有变需重建。
构建逐桶生成 FAISS 文件，不再同时保留全部索引；离线个体效用不参与当前评分。

中断后使用对应入口及相同配置，加 `--resume /实际存在的本版本checkpoint目录`。
两阶段分别恢复，不能用旧 factorized checkpoint 恢复稠密 CEM 训练。

## 7. 选择 online 任务

沿用原流程，只评估任务包中与训练任务匹配的任务。任务包目录需包含对应 task.json。

```bash
python - <<'PY'
import json
import os
from pathlib import Path
data = Path(os.environ["MAICC_DATA_DIR"])
run = Path(os.environ["MAICC_RUN_DIR"])
root = Path(os.environ["TASK_PACKAGE_ROOT"])
m = json.loads((data / "manifest.json").read_text())
train_ids = {str(t["task_id"]) for t in m["tasks"] if t.get("split", "train") == "train"}
package_ids = set()
for path in root.rglob("task.json"):
    package_ids.update(str(t["task_id"]) for t in json.loads(path.read_text()).get("tasks", []))
selected = sorted(train_ids & package_ids)
missing = sorted(package_ids - train_ids)
print("任务包任务数:", len(package_ids), "匹配训练任务数:", len(selected))
print("未匹配任务:", missing[:10], "总数:", len(missing))
if not selected:
    raise SystemExit("没有匹配任务，请检查任务包路径和 task_id")
(run / "online_task_ids.txt").write_text("\n".join(selected) + "\n")
PY
```

本例为 seen-task online。未见任务需要预先放在同一个带 split 的 manifest 中，
归一化与索引只使用 train，随后选择 test；不能临时替换索引绑定的 manifest。

## 8. 单任务检查和正式 online

```bash
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ONLINE_CHECKPOINT="$MAICC_RUN_DIR/stage1/step_000500000"
read -r FIRST_TASK < "$MAICC_RUN_DIR/online_task_ids.txt"

python tools/eval_tabx_online.py \
  --config "$MAICC_RUN_DIR/configs/stage1.yaml" --checkpoint "$ONLINE_CHECKPOINT" \
  --raw-data-root "$RAW_DATA" --task-package-root "$TASK_PACKAGE_ROOT" \
  --tabx-root "$TABX_ROOT" --task-id "$FIRST_TASK" \
  --episodes 3 --episode-limit 512 --seed 10000 \
  --output-dir "$MAICC_RUN_DIR/online_smoke" \
  2>&1 | tee "$MAICC_RUN_DIR/logs/09_online_smoke.log"
```

确认 reset/step 和完整 3 局正常后，再按与其他方法一致的预算正式评估：

```bash
TASK_ARGS=()
while IFS= read -r task_id; do
  if [[ -n "$task_id" ]]; then
    TASK_ARGS+=(--task-id "$task_id")
  fi
done < "$MAICC_RUN_DIR/online_task_ids.txt"
if (( ${#TASK_ARGS[@]} == 0 )); then
  echo "任务列表为空，不启动评估" >&2
else
  python tools/eval_tabx_online.py \
    --config "$MAICC_RUN_DIR/configs/stage1.yaml" --checkpoint "$ONLINE_CHECKPOINT" \
    --raw-data-root "$RAW_DATA" --task-package-root "$TASK_PACKAGE_ROOT" \
    --tabx-root "$TABX_ROOT" "${TASK_ARGS[@]}" \
    --episodes 100 --episode-limit 512 --seed 10000 \
    --output-dir "$MAICC_RUN_DIR/online_seen_100ep" \
    2>&1 | tee "$MAICC_RUN_DIR/logs/10_online_seen.log"
fi
```

不要意外省略任务列表：4800 个任务各 100 局为 48 万局。默认所有友方由模型控制，
在线记忆在任务/队友条件开始时清空，任务内跨局积累，模型权重不更新。
3 局检查只用于运行验证；衰减依赖总局数，不能把它的表现直接等同于 100 局评估前 3 局。

结果保存在 episodes.csv、summary.csv、summary.json 和 run_spec.json。
中断时使用完全相同的正式命令加 `--resume`；已完成条件跳过，未完成条件整组重跑。
评估期间保持配置、代码、checkpoint、索引和任务定义一致；改变条件时使用新的输出目录。

队友实验可增加 `--teammate-config "$MAICC_REPO/src/config/tabx/teammate_modes.yaml"`，
并换输出目录。必须满足配置要求的实际友方槽位；all_model 与队友替换条件分别报告。
关闭记忆的消融使用 `--disable-online-memory` 和独立输出目录。
