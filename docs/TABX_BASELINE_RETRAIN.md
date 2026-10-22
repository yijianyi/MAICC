# train_0：MAICC 的 TABX 最小适配基线

本版本恢复原 MAICC 的稠密 CEM、训练检索规则、测试评分和按智能体隔离的在线记忆。
默认仍使用 manifest 指定的有限历史（现有数据为 100 步），不是 512 步完整历史复现。
TABX 观测、有效单位屏蔽、动作编码、团队奖励和完成标志属于必要环境适配。

0930 的 `*_comm_agent` 数据从联合轨迹定位开始，完整命令见
[TABX_0930_TRAINING.md](TABX_0930_TRAINING.md)。该目录不能直接交给联合轨迹转换器。
正式 baseline 默认每阶段 500000 次更新，与用户确认的旧实验预算一致。

## 修改内容

- `model.cem_architecture: dense` 使用原 `CustomGPT2Model` 的层、位置编码和注意力边界，
  补充不存在单位的屏蔽。`factorized` 仅作为显式消融；不会静默回退。
- 保留因果屏蔽、观测布局校验、第二阶段冻结编码器的 eval 模式和产物哈希检查。
  `obs_visible_weight: 1.0` 恢复未额外加权的观测损失。
- 训练检索评分为 `(cos+1)/2 + 1[cos < max_similarity 且候选完整回报 <= 查询完整回报]`。
  完整回报按源 episode 合并窗口且去重计算；训练查询不再使用窗口回报。
  排除查询自身 episode 的全部重叠窗口；选择 k 个候选后随机打乱上下文顺序。
  不额外要求离线示范来自不同 episode；原版离线检索可以选择不同智能体的示范。
- 测试离线评分为 `(cos+1)/2 + weight*norm(team_return)`；在线评分再加
  `ind_weight*norm(predicted_individual_return*n_present)`。默认权重为 0.8、0.2。
  两项均使用训练数据确定的固定团队回报上下界，不动态扩展，不裁剪。
  `n_present` 为实际单位数量，包含已死亡的真实单位，不含填充槽位。
- 在线记忆按智能体 ID 隔离；至少有 k 个有效独立 episode 后才能返回 k 条在线示范。
  多窗口不能冒充多局。每个智能体独立选择离线/在线来源。
  保留全部已完成局，包括失败局；保留原记忆概率与动作采样的指数衰减。
- v4 索引覆盖每个历史长度，并包含各阶段窗口。查询和候选的前缀长度精确对应。
  默认 `Flat` 精确检索，候选数 `search_k: 50`。HNSW 需显式设置
  `allow_approximate_index: true`，属于需另行验证的近似变体。
- `max_batch_size: 32` 现在表示每次优化更新使用的实际窗口数。
  `agent_time_budget: 1024` 只限制微批次，余下窗口通过梯度累积完成。
  100 步 × 10 槽位下为微批次 1、累积 32 次；500000 次更新采样约 1600 万窗口/阶段。
  DT 恢复有效智能体损失求和；默认不额外裁剪梯度（`grad_norm_clip: 0`），仍检查非有限梯度。
  CSV 记录有效 batch、微批次与累积次数。激活检查点减少显存但增加计算时间。

## 数据和任务边界

现有完整 episode 转换得到的 HDF5/JSONL 可以复用，不必仅因这次修改重新转换数据。
若 provenance 无法覆盖完整 episode，索引构建会报错，须先修复数据覆盖。
同一 episode 的重叠窗口不能跨训练、验证、测试集合；任务划分必须在拟合统计前确定。

默认 `evaluation.task_split: train` 是已见任务在线适应，不可报告成未见任务泛化。
评估未见任务时使用带显式 split 的同一 manifest，并指定 `--split test`。
训练、索引、归一化只使用 train；归一化文件新增 `fit_split`。
全部任务均为 train 的旧归一化文件可继续使用；有留出任务时旧文件缺少 fit_split 会被拒绝，
需重新运行归一化工具（保留正确的 `--n-zones` 和 `--max-enemies`）。

`evaluation.target_return_mode: train_only` 下，已见任务使用其训练统计；未见任务仅使用
训练任务的归一化目标中位数和回报尺度中位数。不会读取未见任务的胜局分位数。
如协议预先规定目标，可以设 `target_return_mode: fixed`，同时指定 `target_return` 和
`return_scale`。实际来源记录在评估 `run_spec.json` 中。

所有方法必须统一离线数据权限、任务、对手、队友条件、在线交互预算及结果汇总口径。
`all_model` 与替换队友的条件分别报告。10 局或 100 局均可，但不能只为本 baseline 增加预算。
动作采样和记忆衰减都依赖总局数，比较前期/后期不能单独解释成记忆效果。

## 服务器运行

本次只做本地静态检查，没有执行单元测试、训练或 TABX 环境评估。
以下命令需在服务器原有可运行 TABX 的环境执行，不要重装根目录历史 requirements.txt。
建议先跑包含稠密 CEM、记忆隔离、评分、梯度累积和两阶段合成数据闭环的验证用例。

```bash
export MAICC_DATA_DIR=/mnt/cfs/shaopt/data/MARL/yxq/maicc_tabx/0811_16task_l100
export MAICC_SEED=7
export MAICC_RUN_DIR=/mnt/cfs/shaopt/data/MARL/yxq/maicc_results/0811_16task_l100_baseline_v2_seed7

bash tools/run_tabx_baseline_retrain.sh verify
bash tools/run_tabx_baseline_retrain.sh check
bash tools/run_tabx_baseline_retrain.sh smoke
```

`check` 输出有效 batch、微批次、采样预算和索引存储上界；`smoke` 在独立目录训练两个
stage0 步骤、batch=1。确认服务器兼容性和资源后，依次执行：

```bash
bash tools/run_tabx_baseline_retrain.sh stage0
bash tools/run_tabx_baseline_retrain.sh index
bash tools/run_tabx_baseline_retrain.sh stage1
bash tools/run_tabx_baseline_retrain.sh eval --episodes 100
```

也可用 `all` 连续执行 check、stage0、index、stage1。
`baseline_stage0.yaml` 和 `baseline_stage1.yaml` 是当前训练配置；两阶段模型设置必须一致。
每阶段默认 500000 次更新，最终 checkpoint 为 `step_000500000`。
`tabx_stage*.yaml` 保留另一份历史数据路径示例，使用前检查其观测槽位容量。

旧 factorized checkpoint、旧 v3 索引、旧优化器与本版不兼容。
必须从头 stage0 → 重建 v4 索引 → stage1；不要用 strict=False 绕过加载检查。
默认输出改为 baseline_v2，保留旧结果。只在同版本产物上使用 `--resume`。

## 存储与运行开销

精确前缀索引比原 7 个桶大得多。L 个前缀、每个前缀 V 条有效单位窗口时，
向量约占 `4*L*V*n_embd` 字节，另外每条约 30 字节元数据和文件开销。
构建时先顺序写临时向量，再逐桶生成 FAISS 文件；每次只保留一个完整索引。
临时文件在对应桶完成后删除，中断时可能保留在独立输出目录中。
不完整构建不会发布可用 manifest；重新构建需要新目录或显式 `--overwrite`。

训练和评估默认只缓存两个前缀索引；这限制索引的常驻内存，但随机前缀可能频繁读盘。
建议将索引放在服务器本地 SSD，按实际 RAM 增大 `index_cache_size`；该参数不会改变候选集。
Flat 文件的 mmap 标志不保证 FAISS 版本真正进行零拷贝映射，默认使用 ram 加有限缓存。
完整历史索引、真实 batch 32 和稠密 CEM 都会增加运行成本；尚无服务器实测耗时或显存数据。

并行评估脚本同步使用 baseline_v2 路径、`EVAL_SPLIT=train`，默认单进程，避免多个进程
同时复制大索引。可按服务器资源提高 `PARALLEL_JOBS`。并行脚本沿用每任务不同种子的规则，
单进程入口沿用每任务相同起始种子的规则；跨方法比较时必须使用同一套种子列表。

## 验收记录

服务器上保留 verify 输出、两阶段日志、训练 CSV、checkpoint provenance、索引 manifest
和每次评估的 run_spec。验证通过后再比较多个独立训练种子的胜率和适应曲线。
记忆关闭、效用关闭、factorized CEM、额外观测加权均作为独立消融，不混入参考版名称。
