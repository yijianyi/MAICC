# TABX 队友替换评估

直接加载已有 MAICC checkpoint，不需要重新训练。每个条件为「任务 × 队友模式」，
模式由被替换的零起始 ally ID 及其 heuristic policy 决定。

## 运行

在原来可运行 TABX 在线评估的 Python >= 3.10 环境中执行；继续使用已有的
PyTorch、JAX、FAISS、数据集、normalization 和离线检索索引。
`--checkpoint` 传包含 `agent.th` 的目录，`--config` 传与该模型及数据匹配的训练配置。
默认配置指向 7-task 的服务器路径，不会自动定位你提到的 16 个数据集。

先做一个任务的短跑，将下面三个变量换成实际值：

```bash
cd /home/xqyang/AIRS/MAICC
EVAL_CONFIG=/absolute/path/to/matching_training_config.yaml
EVAL_CHECKPOINT=/absolute/path/to/checkpoint_directory
EVAL_TASK_ID=actual_manifest_task_id
python tools/eval_tabx_online.py \
  --config "$EVAL_CONFIG" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --teammate-config src/config/tabx/teammate_modes.yaml \
  --task-id "$EVAL_TASK_ID" \
  --episodes 2 --seed 10000 \
  --output-dir results/teammate_smoke
```

正式运行所有 manifest 任务，每个任务的每种模式运行 100 个 episode：

```bash
python tools/eval_tabx_online.py \
  --config "$EVAL_CONFIG" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --teammate-config src/config/tabx/teammate_modes.yaml \
  --episodes 100 --seed 10000 \
  --output-dir results/teammate_eval
```

如果任务包无法从数据自动推断，沿用原评估的 `--raw-data-root`、
`--task-package-root` 参数。TABX 源码不在仓库默认位置时指定 `--tabx-root`。
不要传 `--disable-online-memory`，否则会关闭原有在线 memory 积累。

中断后在完全相同的正式运行命令末尾加 `--resume`。
已完成的任务/模式跳过；中断条件的 CSV 行丢弃，清空在线 memory 后整组重跑。
改变 checkpoint、模式、种子、episodes 或配置时使用新输出目录。
旧版评估结果没有 `run_spec.json`，不能直接在旧目录续跑。

## 默认模式与自定义

`src/config/tabx/teammate_modes.yaml` 提供七组：all_model，以及 novice/medium/expert
分别替换 ally 0、替换 ally 0 和 1。每个任务必须有至少三个有效 ally，且这些位置不能
是 padding 或 disabled 单位；否则评估报错，不会偷偷换位置或跳过条件。
若只想测单队友，删除三组 `_0_1` 配置。

可以为不同队友分别指定不同 policy：

```yaml
teammate_modes:
  - name: all_model
    policies: {}
  - name: mixed_0_1
    policies: {0: novice, 1: expert}
```

支持 random、novice、medium、advanced、expert。名称表示 TABX 自带配置，
不保证在所有任务上的实际表现严格按名称排序。不传 `--teammate-config` 时只有 all_model。
至少保留一个模型控制的有效 ally。

## Memory 与动作

每个条件都从相同 seed 序列开始，并清空在线索引、在线上下文和累计 episode 数，
恢复检索混合进度；离线索引保留。模式内跨 episode 积累在线 memory。
每回合清空模型回合内历史和每个 heuristic 的 LastVisibleTarget。

模型仍生成全部 ally 的动作，指定位置随后由 heuristic 替换；环境、模型上下文、
在线 memory 使用相同的实际执行动作。队友 heuristic 使用原始观测和独立随机数流，
不改变原环境/敌方随机流的分裂顺序。相同 seed 并不意味着不同动作后环境轨迹相同。

## 输出

- `episodes.csv`：任务、模式、agent ID、policy、seed、胜负、回报和长度。
- `summary.csv` / `summary.json`：每个条件的胜率、平均回报、平均长度及
  `delta_win_rate`、`delta_mean_return`（该条件减同任务 all_model）。胜率差单位为比例，
  -0.1 代表下降 10 个百分点；没有 all_model 时差值为空。
- `run_spec.json`：运行配置、checkpoint 路径/大小/修改时间、任务定义和条件列表，
  用于拒绝不匹配的断点续跑。

胜负继续沿用当前 TABX 环境的 `is_win[0]`，没有修改终局规则。
固定替换位置的结果会受到角色影响；第一轮用于摸底，不单凭胜率下降就断言合作坍塌。

## 验证

```bash
python -m unittest discover -s tests/tabx -p test_teammate_eval.py -v
```

测试覆盖模式校验、动作进入真实 EpisodeBatch/上下文/在线 store、环境随机流隔离、
回合内 heuristic 状态重置、跨模式 memory 清空、checkpoint 单次加载和断点续跑。
这些测试使用可控的环境及 heuristic 替身；不代替真实 TABX/checkpoint 的短跑验证。
