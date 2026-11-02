# 按标签来源分组训练 MAICC

此实验按 task 是否包含 `actions_reference_distribution.npy` 分组：

| 组名 | 分组条件 |
| --- | --- |
| `rl_label` | task 的每条原始 record 都有 distribution 文件 |
| `advance_label` | task 的每条原始 record 都没有 distribution 文件 |

组名表示按文件可用性得到的数据来源分类，不会自动验证标签的生成策略。
如果数据约定相反，可以显式指定 `--distribution-group advance_label`。
同一 task 若同时出现有文件和无文件的记录，工具会报错，不把一个 task 分到两组。
找不到原始 task 或原始目录时也会报错，不会将无法检查的数据当作 advance label。

本实验保留原 MAICC 的 behavior 动作监督；reference 硬标签和概率分布不进入损失。
不修改 CEM/DEM 特征对齐和 DT 的训练目标。

## 复用已经转换的联合轨迹

现有 MAICC 数据按 task 保存 HDF5，manifest 包含 `task_id`。分组工具创建两个
子 manifest，使用绝对路径引用原 HDF5/JSONL，不复制、重写或改成 agent-centric 数据。
原有 train/test/validation 划分和全局观测维度均保留，分组后的采样器只会看到对应组的任务。

在原有 MAICC/TABX Python 环境执行，将下面路径替换为服务器上的实际位置：

```bash
python tools/split_tabx_label_groups.py \
  --raw-root /path/to/raw_env_centric_records \
  --manifest /path/to/converted/manifest.json \
  --output-dir /path/to/converted/label_groups \
  --stage0-config src/config/tabx/baseline_stage0.yaml \
  --stage1-config src/config/tabx/baseline_stage1.yaml \
  --run-dir /path/to/results/label_groups_seed7 \
  --seed 7
```

若原 manifest 的 `source_root` 仍有效，可以省略 `--raw-root`。
可用 `--steps 100000` 给两组的两个阶段设置相同更新预算；默认保留各阶段模板预算。
其余模型、优化器和 batch 设置来自提供的配置模板，在两组之间保持一致。
生成的 `.yaml` 配置使用 JSON 语法（YAML 的子集），现有配置加载器可直接读取。
输出已存在时默认拒绝覆盖；`--overwrite` 仅更新分组清单、报告及生成配置，不删除训练产物。

输出结构：

```text
label_groups/
  label_groups.json
  rl_label/
    task_ids.txt
    manifest.json
    stage0.yaml
    stage1.yaml
  advance_label/
    task_ids.txt
    manifest.json
    stage0.yaml
    stage1.yaml
```

报告包括每组的 task ID、record 数量、distribution 检查结果和示例原始路径。
空组只有空的 task 清单，不生成无效 manifest 或训练配置；选择 `both` 时明确跳过空组，
单独选择空组则报错。只剩一组时不能完成两组对比。

## 独立训练与比较

两组的归一化文件、checkpoint、训练 CSV 和检索索引使用不同目录。
必须按各自训练子集重新拟合归一化，不能复用原完整 manifest 的归一化文件，
因为现有加载器会核对 manifest 哈希。检索索引也必须用对应组的编码器重新构建。

先查看两组完整运行命令，不启动训练：

```bash
python tools/run_tabx_label_groups.py \
  --groups-dir /path/to/converted/label_groups --stage all --dry-run
```

逐步运行，两组默认串行执行：

```bash
python tools/run_tabx_label_groups.py --groups-dir /path/to/converted/label_groups --stage normalize
python tools/run_tabx_label_groups.py --groups-dir /path/to/converted/label_groups --stage smoke
python tools/run_tabx_label_groups.py --groups-dir /path/to/converted/label_groups --stage stage0
python tools/run_tabx_label_groups.py --groups-dir /path/to/converted/label_groups --stage index
python tools/run_tabx_label_groups.py --groups-dir /path/to/converted/label_groups --stage stage1
```

`--stage all` 串联 normalize → check → stage0 → index → stage1。
`--group rl_label` 或 `--group advance_label` 可只运行一组。
`smoke` 为每组执行 2 步、batch=1 的 stage0，输出到独立的 `smoke_stage0`，不冒充正式 checkpoint。
需要自定义恢复训练时，直接向现有训练脚本传入分组后的配置和 `--resume`。

生成阶段损失对照：

```bash
python tools/run_tabx_label_groups.py \
  --groups-dir /path/to/converted/label_groups \
  --stage compare --compare-stage stage0 --tail-window 1000
```

结果为 `comparison_stage0.csv`，包含每组最后更新步数、日志行数、末段样本数及各项 loss
的均值和样本标准差。窗口单位为日志行，非必然连续训练步；stage1 可用 `--compare-stage stage1`。
比较缺少日志、空日志、重复步数或非有限 loss 时会报错。
两组任务不同，loss/胜率差异包含任务难度和数据规模的影响。正式评估应固定评估任务、
更新预算、随机种子和交互协议，并使用各组的 checkpoint、归一化和检索索引。

## 尚未转换的数据

只检查原始数据并输出 task 清单时，可以不提供 manifest 和训练模板：

```bash
python tools/split_tabx_label_groups.py \
  --raw-root /path/to/raw_env_centric_records --output-dir /path/to/raw_label_groups
```

这一步不转换数据。现有 MAICC CEM 使用联合轨迹，因此该训练流程无需 agent-centric 转换。
如果其他训练器需要 agent-centric 格式，使用提供的服务器脚本
`/mnt/cfs/shaopt/code/AnyTABX/generate/agent_centric_comm.py`，按该脚本自身参数执行；
不要将 agent-centric 数据直接输入当前联合轨迹转换器。
