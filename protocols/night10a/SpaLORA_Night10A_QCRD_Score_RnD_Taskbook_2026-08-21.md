# SpaLORA Night-10A：QCRD 跨平台提分研发任务书

日期：2026-08-21  
任务性质：模型研发 + 既有结果指标补算  
运行位置：AutoDL 持久盘；必须使用有卡模式启动  
目标：优先提高 A1、D1 和 P22，同时保护 tonsil 与空间结构；不寻找新数据集、不跑新的第三方 benchmark。

## 0. 给执行 Worker 的通俗目标

本轮不是写论文、做投稿图或继续搜数据。本轮做两件事：

1. 不重训旧模型，直接从持久盘已有结果中补齐常用评价指标，弄清当前方法到底强在哪里、弱在哪里；
2. 在冻结的现有表示上研发一个小型新模块 QCRD，让质量较好的模态帮助质量较差的模态，但不按 A1/D1/P22 名称写死权重。

当前最重要的矛盾是：已有候选经常在 P22 明显提高，却让人类淋巴下降。QCRD 要学习“这个 spot 的哪种模态更可信”，而不是预先规定某个平台永远用哪套人工权重。

## 1. 权威输入与开始口令

开始任何远端工作前，读取并逐字遵守：

- `night10a_qcrd_candidate_registry.json`  
  SHA-256：`b310e232282840772b69ad9d1c1d1de6e04c92e8b7b013a7e5247c628dfd959d`
- `metric_expansion_reference.py`  
  SHA-256：`33397cada3701108fbfb15d58fafd6f1f9e20417ac7bc14cce1e8aee993cd159`
- `authoritative_scoreboard_20260821.csv`  
  SHA-256：`951a9e0449e6e083834cd15268bb4c316b011726c80bd122dd9a7516626a0d68`
- `Night10A_Local_Evidence_and_Innovation_Audit_2026-08-21.md`  
  SHA-256：`ea920aeee3593a7903b6189119dc60b7327cee48f9d8b00e0c4283420049b34c`

这些文件位于共享目录：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_local_planning_20260821`

先回复用户一段简短中文，必须包含：

> 已读取 Night-10A 权威任务书和注册表；本轮使用有卡模式，先做只读指标补算与 QCRD 语义检查，再开始冻结表示上的候选研发。

如果任一 SHA 不一致、关键远端 raw root 不存在，停止并准确报告缺失项；不要猜路径，不要重训来代替缺失证据。

## 2. Git 与工作区

1. 以 Night-9B 科学提交 `e9bd62e2bb07c58f58c219956b8aaf090527c471` 为科学父节点；同时只读保留 Night-9C 的数据 provenance 元数据提交，不使用 E18.5 数据。
2. 新建普通分支：`revision/q2-night10a-qcrd-score-rnd-20260821`。
3. 建立保护标签，指向开始前父提交；final tag 只在全部最终交付文件进入同一个 final commit 后创建一次。
4. 所有 push 必须普通 push；禁止 force、force-with-lease、移动标签或覆盖旧 tag。
5. 历史 raw、checkpoint、embedding、partition 全部只读。开始前和结束后对实际使用文件复算 SHA，并记录 before/after。

## 3. 这一次明确不做什么

- 不继续寻找或修复 E18.5；不访问 MISAR Y；不运行 MISAR、GSE198353、GSE205055。
- 不跑新的 COSMOS、SMART、PRESENT、ARISE 等完整 benchmark。
- 不做投稿图和论文文字包装。
- 不为补 AMI/FMI 等指标重新训练旧模型。
- 不使用标签训练、选 epoch、选 seed、选 K、选 cluster endpoint 或计算 loss。
- 不以 dataset name 控制 trainable weights。
- 不构造 dense `N×N` 距离、邻接、spatial kernel 或负样本矩阵。
- 不把轻微负结果当作实现失败；不要因为某一数据集小幅下降就清空所有候选。

## 4. P0：权威性、资源与语义检查

### 4.1 文件和环境

1. 复核四个本地权威输入的 SHA。
2. 定位以下既有远端 roots，不递归复制到 Windows：
   - `/root/autodl-fs/night6c_raw_runs_20260817`
   - `/root/autodl-fs/night6d_raw_runs_20260817`
   - `/root/autodl-fs/night7b_score_rnd_20260818`
   - `/root/autodl-fs/night8b_raw_runs_20260820`
   - `/root/autodl-fs/night9b_racf_20260820`
3. 确认 CUDA、GPU、PyTorch、scikit-learn、numpy、scipy 可用。
4. 确认至少 80 GB 持久盘空间；若不足，先只读统计旧 raw 体积并报告，未经授权不得删除历史结果。
5. 记录 GPU 型号、CPU 核数、内存、库版本和 git 状态。

### 4.2 指标实现测试

将 `metric_expansion_reference.py` 和 `test_metric_expansion_reference.py` 放入工作树的 Night-10A 测试目录。必须：

- 在服务器完整运行四项 reference tests；本地 Windows 缺 sklearn 的两项 skipped 在服务器不得继续 skipped；
- 用一个历史小样本与项目已有 ARI/NMI 计算器做 parity；误差上限 `1e-12`；
- 检查跨模态配对 spot 顺序完全一致；不允许自动 inner join 后静默丢 spot；
- 对 silhouette/DB/CH 使用固定标准化和固定距离语义，并记录协议；
- FOSCTTM 和 retrieval 必须块状计算，禁止形成完整 `N×N` 矩阵。

### 4.3 QCRD 真实语义探针

在任何正式候选训练前，用合成数据和一小段真实无标签 frozen views 验证：

1. quality features 在 optimizer 建立前计算并冻结；它们没有梯度；
2. teacher 分支 `stop_gradient`；
3. 交换两模态质量后，soft teacher 方向随证据交换，而不是随 dataset name 不变；
4. 两模态同质量时 gate 接近中性，correction 不得无故放大；
5. boundary-risk 高的 spot 在 Q05 中 correction 小于内部 spot；
6. Q07 只使用 sparse reciprocal neighbors；不得调用 dense pairwise distance；
7. RNA+protein 与 RNA+ATAC 的 trainable module class、宽度和损失定义一致；允许输入 preprocessing 不同；
8. checkpoint round-trip 后 output exact 或在登记容差内一致。

P0 的硬停止仅限：权威文件不一致、原始 evidence 缺失、标签泄漏、candidate semantics 错误、spot 顺序错或基础数值测试失败。普通候选效果差不属于 P0 失败。

## 5. Stage M：不训练的指标补算

### 5.1 要补算的结果

从既有 frozen artifacts 尽可能补算：

- C00：A1、tonsil、D1、P22；
- C01：D1、P22；
- Night-7B R02 与 R08：四个核心切片；
- Night-9B F00/reference 与 N02：A1、P22；
- MISAR U00/F00 只使用既有已锁 partitions、embeddings 和已经交付的 labels/metric artifacts；**严禁第三次读取原始 MISAR Y**。

若某个指标所需 artifact 真实不存在，写 `MISSING_SOURCE_ARTIFACT`，不重训，不用相邻候选冒充。

### 5.2 指标面板

每个可用 candidate × dataset × seed 输出：

- 标签一致性：ARI、NMI、MI、AMI、FMI、homogeneity、completeness、V-measure；
- embedding 几何：silhouette、Davies-Bouldin、Calinski-Harabasz；
- 空间结构：neighbor agreement、Moran's I、Geary's C、boundary disagreement；
- 若存在成对 private views：symmetric FOSCTTM、Recall@1/5/10、paired median rank；
- runtime、peak GPU；若历史没有可比口径则标记缺失，不伪造。

规则：

- ARI/NMI/Q 仍是主指标；扩展标签指标是佐证，不以堆指标替代真实改进。
- silhouette/DB/CH 只作辅助诊断，不能单独决定候选。
- 每个数字必须带 candidate、dataset、seed、artifact SHA 和 metric protocol version。
- 输出 `metric_backfill_long.csv` 和 `metric_backfill_coverage.json`。

## 6. QCRD 实现

### 6.1 核心机制

实现注册表中的统一结构：

`z_low_corrected = normalize(z_low + gate * delta(z_low, stopgrad(z_high), spatial_context))`

其中：

- `z_low` 与 `z_high` 不是按数据集名指定，而是由训练开始前冻结的质量证据形成 soft teacher/student 权重；
- global quality 包含 frozen partition 上的无标签 silhouette/DB/CH、空间局部一致性和 graph local residual；
- per-spot quality 包含 neighborhood residual/entropy、cross-modal cosine disagreement 和 sparse MNN support；
- adapter 必须小型、低秩、显存可控；
- correction magnitude、原始表示 anchor、local boundary preservation 三项约束必须存在；
- trainable module 对两个 modality family 完全同构。

### 6.2 候选

严格实现注册表 Q00-Q07。不要追加第九个候选，不要在看到标签后改变超参数。

特别说明：

- Q00 是精确 reference alias，不训练；用于证明 evaluator 和 transform 没有漂移。
- Q01/Q02 区分 global 与 per-spot gating。
- Q03/Q04 检验 masked denoising 与 spot gating 的组合。
- Q05 重点保护组织边界。
- Q06 只加入低维 Fourier coordinate features，禁止 dense spatial kernel。
- Q07 只加入置信度加权 sparse reciprocal-MNN；不得照搬 SMART 的 dense pairwise 实现。

### 6.3 训练语义

- 使用注册表固定超参数；fixed epochs=120，无 early stopping、best epoch、seed search。
- 每个正式 unit 一个 config SHA；candidate、dataset、seed、输入 SHA、代码 commit 全部写入 manifest。
- 标签值在训练和 transform 总锁前不可进入任何进程。
- no-op 必须真实 byte/parity 可证；不能把失败 unit 改名为 no-op。
- scientific retry=0，fallback=0。基础设施问题只能在正式锁前修复并完整记录。

## 7. R1：三 seed 机制筛选

运行 Q00-Q07 × A1/tonsil/D1/P22 × seeds 0/1/2。

- Q00 为 alias；最多 84 个真实训练 unit。
- 所有训练完成后锁定 checkpoint、views、affinity、partition 和 hashes，再开启评价窗口。
- 同 seed 与准确 reference 配对；不得将 Night-7B 十 seed均值和 Night-9B 五 seed均值混作 paired reference。

### 7.1 不再使用一刀切 shortlist

从有效候选中分别保留：

1. accuracy frontier：mean paired Q 最优，ARI/NMI 分列；
2. balanced frontier：worst-dataset/worst-family paired Q 最优；
3. spatial frontier：在四项空间指标没有实质性崩塌时，paired Q 最优。

三个 frontiers 去重后最多三个候选进入 R2。某一数据集轻微负值是排名惩罚，不是语义作废。只有标签泄漏、错误输入、非有限输出、artifact 损坏或实现不符才作废整个 unit。

如果所有候选都没有任何 family mean Q 正信号且均出现明显空间崩塌，R1 后停止；保留负结果，不运行 R2。

## 8. R2：完整 seed 确认

仅对 R1 冻结的最多三个候选运行：

- A1、tonsil：seeds 0-4；
- D1、P22：seeds 0-9。

最多 90 个真实训练 unit。仍然先总锁 outputs，再评价。

输出：

- per-seed 原始指标和 paired deltas；
- mean、median、SD、worst seed；
- bootstrap 95% CI；
- 样本量允许时 exact paired sign-flip p；
- modality-family mean、worst-family、macro mean；
- runtime、peak GPU 相对 reference；
- accuracy/balanced/spatial 三条最终 frontier。

这是一轮研发确认，不宣称 pristine holdout 或 SOTA。即使出现强阳性，也只锁定为 Night-10B 端到端整合候选。

## 9. 性能、并行和截止时间

本轮必须在 AutoDL 有卡模式运行。

- GPU 训练必须实际使用 CUDA；记录每类 unit 的 GPU 利用率摘要。
- CPU transform 使用最多 4 个 worker；先做一个真实峰值内存探针，再决定是否降为 2。
- 严禁让单个 transform 单核无期限运行。单 transform 30 分钟未完成即记 `TIMEOUT`，杀死该子进程并继续彼此独立的注册单元；不能重跑同一 unit。
- 整轮 wall-clock 上限 12 小时。到上限后停止派发新 unit，等待正在运行的单元在 10 分钟内结束，否则干净终止并交付 `WALLCLOCK_BUDGET_REACHED`。
- 每完成一个 stage 才做一次简短状态更新，不高频轮询或重复输出大日志。

## 10. 测试与独立复算

最终必须包含：

- metric reference tests；
- label firewall tests；
- QCRD semantics tests；
- sparse/no-dense regression test；
- dataset-name routing negative test；
- checkpoint round-trip；
- independent recomputation of every reported aggregate；
- Q00 reference exact parity；
- artifacts before/after hash audit。

任何汇总表都必须能从 per-seed CSV 独立复算；最大误差门限 `1e-12`。

## 11. 合法终态

只能选择一个最准确的终态，并通俗解释：

- `NIGHT10A_QCRD_BALANCED_CANDIDATE_LOCKED`
- `NIGHT10A_QCRD_ACCURACY_FRONTIER_LOCKED`
- `NIGHT10A_QCRD_MULTIPLE_FRONTIERS_LOCKED`
- `NIGHT10A_QCRD_NO_POSITIVE_SIGNAL`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BLOCKED_MISSING_FROZEN_EVIDENCE`
- `WALLCLOCK_BUDGET_REACHED`

普通负结果不能写成 implementation invalid。若部分候选失败而其他候选有效，应继续独立有效单元并如实保留失败行。

## 12. 交付、D 盘与关机

D 盘只交付 compact，不下载 raw runs、checkpoints、full embeddings、affinity 或大型全历史 bundle。

目标目录：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_handoff_20260821/official_compact`

必须包含：

- `plain_language_summary.md`：先用非专业语言说明这一步做了什么、分数是否提高、代价是什么；
- `night10a_report.md`；
- `night10a_decision.json`；
- `metric_backfill_long.csv` 与 coverage；
- `per_seed_metrics_long.csv`；
- `frontier_registry.json`；
- QCRD 代码、注册表、测试和语义审计；
- resource audit、label read audit、independent recomputation；
- root-relative `delivery_index.json`，每项 size + SHA-256；
- 从 Night-9B/Night-9C lineage 到 Night-10A 的最小增量 Git bundle。

Windows 侧独立复算 compact 内全部 SHA。final commit 后只创建一次 final tag，普通 push 并核验 peel。

最后保留同一 SSH 会话，把 `/usr/bin/shutdown` 作为最后一条远端命令。命令派发后不得重连。只报告“关机命令已派发”，不要虚报控制台状态。

## 13. 最终回复格式

最终回复先写通俗结论，再写技术结果。至少回答：

1. 当前模型补齐了哪些指标，哪些仍缺证据；
2. QCRD 是否在 A1、D1、P22 上实现协同改善，tonsil 是否被保护；
3. 哪些 accuracy/balanced/spatial frontier 被锁定；
4. 运行时间和显存代价；
5. 是否存在标签泄漏、seed 选择、重试、fallback；
6. Git commit/branch/tag；
7. D 盘 compact 路径、关键 SHA、校验计数；
8. 最后一条远端关机命令的派发状态。
