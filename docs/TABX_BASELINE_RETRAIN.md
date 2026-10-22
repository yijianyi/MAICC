# TABX MAICC baseline 修复与重训

本分支保留 TABX 环境、自采行为数据和原有队友替换评估。模型仍是 TABX 适配的
MAICC，并非逐层完全相同的原论文网络。原始分支、旧 checkpoint、旧索引均保留。

## 修复范围

- 378 维观测按 10 ally 槽位、10 enemy 槽位、10 zone 槽位解释：
  own[14] + other[19×16] + zone[10×6]。归一化和 loss 共用严格布局校验；
  拒绝对不匹配维度做整除取整。可继续使用已验证的原归一化文件。
- 保留空间/时间分解的 CEM，增加上一时刻 post 汇总到各 token 的投影通路。
  post 汇总包含该步所有有效观测、动作、奖励/终止信息，先右移一步再注入时间网络。
  当前动作预测能利用过去动作和奖励，不读取当前动作、当前 post 或未来 token。
  这仍是为长 TABX 窗口采用的结构近似，不等价于论文的稠密 GPT-2 注意力。
- 第二阶段冻结的 CEM/DEM 和编码器投影保持 eval 模式；DT 保持 train 模式。
- 测试时在相似度候选集中加上
  `alpha * norm(team_episode_return) + (1-alpha) * norm(predicted_individual_episode_return)`，
  默认 alpha=0.8。训练仍只做相似度检索，并排除同源 record。
  两类回报都按完整源 episode 累加，重叠窗口每个时间步只计一次。
  个体奖励仅由冻结 DEM 的 action embedding + reward head 预测，未读取真实个体奖励。
  DEM 按覆盖窗口使用有限历史（100 步），这是长 episode 的工程近似。
  归一化区间为离线索引候选回报范围，并随已经完成的在线 episode 扩展；常数范围贡献为零。
  保留按指数概率选择离线/在线记忆的工程实现，在所选来源内重排相似度 shortlist。
  这不等价于在全部混合记忆上精确穷举最高分；论文实验应披露候选数与 HNSW 设置。
- 新索引 schema 为 v3，绑定编码器、manifest、归一化内容哈希；旧 v2 索引不能混用。
  checkpoint 新增 provenance.json，保存 Git commit、源文件哈希及数据/编码器身份。

## 不需要重新转换已有数据

使用原 `0811_16task_l100` 目录内的 HDF5、JSONL provenance、manifest.json 和
obs_normalization.npz。HDF5 本身没有因上述修复而改变。
必须从头训练 stage0、重建索引、再训练 stage1。旧 checkpoint 缺少新增的 CEM
history_projection 参数，不要用 strict=False 绕过，也不要从旧模型 resume。
新输出路径与旧训练目录不同，原归一化文件不覆盖。

服务器先检出本修复分支到独立目录，激活原来能够运行 TABX 的 Python >=3.10 环境。
不要为了重训重新安装根 requirements.txt 中的历史依赖。

```bash
export MAICC_DATA_DIR=/mnt/cfs/shaopt/data/MARL/yxq/maicc_tabx/0811_16task_l100
export MAICC_SEED=7
export MAICC_RUN_DIR=/mnt/cfs/shaopt/data/MARL/yxq/maicc_results/0811_16task_l100_baseline_v1_d256_seed7

bash tools/run_tabx_baseline_retrain.sh check
bash tools/run_tabx_baseline_retrain.sh smoke
bash tools/run_tabx_baseline_retrain.sh all
```

`smoke` 在单独目录执行两个真实 stage0 优化步骤，不替代完整训练。
`all` 依次执行检查、stage0 500000 步、索引构建、stage1 500000 步。
索引构建额外计算完整 episode 的预测个体回报，会比旧版更慢；保留磁盘空间存放新索引。
不自动覆盖已有 checkpoint；重复运行时应使用新目录或明确 resume。

也可以分开运行，检查每个阶段日志后继续：

```bash
bash tools/run_tabx_baseline_retrain.sh stage0
bash tools/run_tabx_baseline_retrain.sh index
bash tools/run_tabx_baseline_retrain.sh stage1
```

如在本修复版本的训练中断，可恢复同一阶段的新 checkpoint：

```bash
bash tools/run_tabx_baseline_retrain.sh stage0 --resume "$MAICC_RUN_DIR/stage0/step_000050000"
bash tools/run_tabx_baseline_retrain.sh stage1 --resume "$MAICC_RUN_DIR/stage1/step_000050000"
```

当前扩容配置为 n_embd=256、hidden_dim=512；层数=8、heads=8、dropout=0.1、
lr=5e-4、最大 batch=32、agent_time_budget=32000，保持修改前的训练设置。
100 步×10 槽位下有效 batch 为 32，实际显存需求须在服务器上通过 smoke 验证。
stage0 和 stage1 的模型配置同步修改；当前实现共用架构参数，因此 CEM、DEM 和 DT
都会使用 256 维表示，CEM 仍为 4 层空间注意力加 4 层时间注意力。
原 HDF5、manifest 和归一化文件可直接复用，但不能从 64 维 checkpoint resume，
也不能复用旧检索索引；须从头训练 stage0、重建索引、再训练 stage1。
脚本默认使用带 d256 标记的新输出目录；如果已经设置 MAICC_RUN_DIR，须改为新的目录。
多训练种子使用不同 MAICC_SEED 和 MAICC_RUN_DIR；每个种子重建自己的索引。

## 评估和论文记录

```bash
bash tools/run_tabx_baseline_retrain.sh eval
bash tools/run_tabx_baseline_retrain.sh eval \
  --teammate-config src/config/tabx/teammate_modes.yaml \
  --output-dir "$MAICC_RUN_DIR/eval_teammates"
```

容量默认从训练配置读取（包括 10 个 zone 槽位）。如果自采任务包无法自动定位，
传 --raw-data-root、--task-package-root、--tabx-root 的真实服务器路径。
每个任务/队友条件独立清空在线 memory，条件内跨 episode 积累；不更新权重。
现有 manifest 的 16 个任务默认全为 train。上述默认评估是这些任务上的在线/队友适应，
不能当作未见任务泛化结果；如论文要求后者，须另行确定任务划分后重训。
所有方法统一数据、任务、队友、随机种子及在线交互预算，报告适应曲线和多种子结果。
检索消融可在复制的配置中设 retrieval.use_utility=false；使用独立评估目录。

## 回归测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/tabx -q
```

测试包含 CEM 历史敏感性/当前标签与未来隔离、padding 隔离、布局拒错、窗口去重、
评分排序与来源排除，以及真实 PyTorch 优化器、合成 HDF5 和 FAISS 的两阶段闭环。
这不代替服务器真实 TABX 环境与全量数据的 smoke/在线评估。
