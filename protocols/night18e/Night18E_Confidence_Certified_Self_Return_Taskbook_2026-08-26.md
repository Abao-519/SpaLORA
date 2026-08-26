# Night-18E：跨模态置信认证自返还（CCSR）与可迁移结构修复

日期：2026-08-26  
负责人：Worker 2；执行：Codex 2  
任务性质：针对 Night-18D 暴露的灾难性结构改坏，设计并证伪一个有数学边界的新方法模块；不是继续修 selector，也不是为了深度学习标签而堆网络。

## 1. 三个必须先说清楚的问题

1. **问题**：Night-15F 的动态 unary（每个点属于各簇的分子代价）与 Potts pairwise（空间邻点不同簇的代价）在不同数据上的相对量纲不稳定。Night-18D 在人胎盘上把好起点改坏或选择到弱起点，13/13 起点没有双指标改善。
2. **本轮新增对象**：`CCSR` 是“confidence-certified self-return”的缩写，中文为“跨模态置信认证自返还”。造这个词是为了区别 Night-15F 仅按 rejected graph mass 增加 stay cost 的旧 self-return。CCSR 对每个节点使用三视图 prototype 一致性与 unary margin，计算在当前 Potts 图能量下保证其不离开当前标签所需的最小惩罚；可信节点获得可验证的保持证书，不可信节点仍可被结构能量修正。
3. **论文意义**：经典 MRF/Potts 的 persistency/partial optimality 条件是先例，不能说成新定理。可检验的新贡献只可能是：把跨模态证据转成节点级可信集合，再用能量量纲内的最小认证自返还连接多尺度空间图和 alpha-expansion，从而同时避免连续组织的过平滑与离散细胞类型的灾难性破坏。

## 2. 权威父级与必须完整吸收的结果

先独立复核以下 compact/report/source，不依赖对话摘要：

- Night-15F：完整连续多尺度能量、原始 self-return、alpha-expansion、逐 lane frozen configs 与 matched ablation。
- Night-16H：结构可行域、四条 RNA+ATAC 强输入 authority 和 89-candidate bank；只作为输入/对照，不继承 selector 贡献叙事。
- Night-18D：human placenta authority、label-free carrier、13 starts、全部 130 partitions、`SCIENTIFIC_NEGATIVE` 与 28/28 Windows compact。
- Night-16A：已有“统计量到参数 + Ridge selector”的科学负结果。CCSR 不能只是换名字重做 Night-16A。

冻结事实：Night-18D placenta family-default full 为 `0.079067/0.132742`，no-op medoid 为 `0.351753/0.533451`，L2 为 `0.370597/0.526850`，paired dual gain 为 `0/13`。任何新结果必须用相同起点、相同表示、相同图和相同求解预算做配对归因。

## 3. 数学合约

对当前起点标签 `p_i`，先按 Night-15F 的同一多视图 prototype 语义得到 unary `U_i(k)`，按同一多尺度冲突感知图得到非负 Potts 边容量 `w_ij`。所有量必须包含最终实际进入求解器的 beta/scale，不可拿未缩放中间量做证书。

定义：

- `m_i = min_{k != p_i} [U_i(k) - U_i(p_i)]`：当前标签相对最有竞争力替代标签的 unary margin；它可以为负。
- `d_i = sum_j w_ij`：节点的最终 incident Potts capacity。
- 三视图支持：retained、RNA/view1、second/view2 各自在使用当前 partition prototypes 时是否把 `p_i` 作为最低 unary 标签。不得用真实 annotation。
- 可信集合 `C(q,s)`：至少 `s` 个视图支持当前标签，并且某个预先登记、无标签的 robust margin/rank 条件达到 `q`。`q`、`s` 是方法超参数，但语义对全部 lane 相同。

对 `i in C`，增加离开当前标签的最小认证惩罚：

`g_i = max(0, d_i - m_i + epsilon_i)`，

即给 `k != p_i` 的 unary 加 `g_i`，给 `p_i` 不加。`epsilon_i` 必须是同量纲、预先登记的严格正数（例如由有限精度和数据内 robust unary scale 机械得到），不能读取标签。

对 `i not in C`，可以保留 Night-15F rejected-mass self-return，也可以设为零；两者必须作为 matched arms 比较。不得为了结果擅自把“不认证”节点也称为认证。

### 3.1 必须验证的保证

在非负 Potts pairwise 下，如果上述不等式使用最终 `d_i` 且为严格正余量，那么把任何候选解中一个认证节点恢复为 `p_i` 时，最坏 pairwise 增量不超过 `d_i`，而 unary 至少减少 `d_i + epsilon_i`。因此任何能量最小解都不应改变认证节点（考虑整数容量取整误差后仍须成立）。

不要把这条充分条件说成原创的普适 persistency 理论。必须：

- 给出清晰推导与已知 persistency/partial-optimality 先例；
- 用 N<=7、K<=3 的随机 tiny Potts 图穷尽全部标签组合验证；
- 测试负 margin、孤立点、零边、容量取整、多个认证节点同时存在；
- 在真实 lane 输出 `certified_changed_count=0`，否则 fail closed 为实现/数值语义错误。

## 4. 实验分层

### Stage P0：真实端到端语义

至少在 placenta、P22、MISAR、人海马四条真实 RNA+ATAC lane 上完成：

`carrier/representation -> three sparse graphs -> start -> unary/pairwise -> CCSR -> alpha-expansion -> artifact reload -> independent evaluator`

逐 lane 列出真实 tensor shape、K、图 nnz、起点 SHA、最终容量单位、`m_i/d_i/g_i` 分布、认证比例、认证节点改动数、wall/RAM/GPU。禁止 dense N×N。P0 先用每 lane 一个起点；4/4 通过后再扩大。

### Stage A：公开 benchmark development（允许透明 label-assisted HPO）

Discovery lanes：P22 K9、MISAR K7、human placenta K10。可以使用公开标签在 partitions 全部先生成、锁 SHA 后，由独立 evaluator 做跨运行 benchmark HPO。候选生成器、配置 bank 与标签 evaluator 必须物理/代码分离；不得把标签放进 unary、pairwise、confidence、solver 或 within-run checkpoint。

搜索既要给分数空间，也必须配平预算：

- 同一组 Night-15F/base configs、相同 starts、相同图和 cycles 同时跑旧 self-return 与 CCSR；
- CCSR 超参数的语义只包含 view-support 数、margin rank/quantile、未认证节点是否保留旧 self-return，以及必要的数值 epsilon 规则；
- 可以做足够的 Sobol/确定性网格搜索，但先写全 config bank 与 hash，失败行不删除；
- 每条 lane 可给出 transparent per-dataset development frontier，同时必须生成一个由三条 discovery study-balanced 选出的 `RNA_ATAC_SHARED_CCSR` 配置。

主 matched arms 至少包括：

1. `NO_OP_STRONG_START`；
2. `L2_LOWPASS_MATCHED`；
3. `NIGHT15F_ORIGINAL_SELF_RETURN`；
4. `CCSR_FULL`；
5. `CCSR_CERTIFICATE_DISABLED`（相同 confidence 计算但 `g_i=0`）；
6. `CCSR_RANDOM_MASK_MASS_MATCHED`（相同认证节点数/规模，节点位置确定性置换）；
7. `CCSR_UNARY_MARGIN_ONLY`（不使用跨模态支持）；
8. `CCSR_VIEW_SUPPORT_ONLY`（不使用 margin rank）；
9. `CCSR_PROTECT_ALL`，作为保守 no-op/safety sensitivity；
10. 若成本允许，单点更新与 alpha-expansion 的同能量对照。

### Stage B：冻结机制确认

只有 Stage A 至少 2/3 discovery lanes 的 CCSR 在相同强起点和相同预算下实现 ARI/NMI 双升，并且没有被旧 self-return、certificate-disabled 或 random-mask 控制复现，才进入 Stage B。

Stage B 在人海马 K7 上只使用 `RNA_ATAC_SHARED_CCSR`，先锁 partition/config/hash，后开标签；不能按人海马结果调整公式或参数。人海马虽然历史上用于其他候选开发，仍可作为“该新机制的冻结确认”，但不得称 pristine blind external test。

如果 Stage A 不通过，直接封口；不得靠增加神经网络、改 endpoint 或扩大到 protein 来救本轮。若 Stage A+B 形成可信方法信号，才可追加 RNA+protein 的少量真实 P0/兼容性测试，不能把不同模型分流包装成统一模型。

## 5. 分数与归因判定

所有主表必须给绝对 ARI/NMI/AMI/FMI、Moran/Geary、cluster sizes、min cluster、changed observations、认证比例、paired starts 胜数、wall/RAM/GPU。

分类：

- `CERTIFIED_METHOD_SIGNAL`：至少 2/3 discovery 双升，冻结 human confirmation 双升；CCSR 胜过同预算旧 self-return 与关键 matched controls，且认证节点 0 改动、非认证节点确有方法特异变化。至少一条 lane 某指标提升 >=0.01，不能只靠数值噪声。
- `CERTIFIED_LOCAL_SIGNAL`：证书正确且一部分 lane 有不可由控制解释的双升，但跨 study 或效应门不足。
- `SAFETY_ONLY_SIGNAL`：主要作用是避免灾难性下降或收窄 worst-case，分数没有独立提升。
- `HEAD_OR_START_ONLY`：提升来自强起点、endpoint 或 label-assisted oracle，CCSR 本身被控制复现。
- `SCIENTIFIC_NEGATIVE`：实现正确，但 CCSR 没有独立收益。
- `IMPLEMENTATION_FAILURE`：证书、容量量纲、精确重放或真实路径语义不成立。

公开 benchmark 的 per-dataset HPO 可用于开发上限，但必须和 `RNA_ATAC_SHARED_CCSR`、冻结 confirmation 分开写。不能用“证明了迁移”描述逐数据集调参结果。

## 6. 新颖性审计

源码与论文至少对照：经典 Potts/alpha-expansion、MRF persistency/partial optimality、contrast-sensitive CRF、Night-15F self-return，以及 SpatialGlue、spaMGCN、BANKSY、PRAGA 等空间多组学/空间图方法。

禁止主张：Potts、prototype、alpha-expansion、unary dominance 或 persistency 本身原创。只有当结果支持时，保留以下窄表述：

“一种将跨模态 prototype 一致性定义的可信集合，与 Potts 能量量纲内最小离开惩罚连接起来的认证自返还机制，使高置信分子结构具有可验证的保持性，同时把空间修正限制在跨模态不确定节点。”

若 `UNARY_MARGIN_ONLY` 与 full 等价，跨模态部分不成立；若 `PROTECT_ALL`/no-op 同样好，方法只有 safety/no-op；若 random mask 复现，证据位置无贡献。必须如实降级。

## 7. 数据扩展的并行只读审计

在不影响主实验的前提下，更新高质量数据表，重点查官方/原论文已经给出 manual annotation 的 MISAR E13.5 与 E18.5：

- 只审计权威 accession、processed file 名称、大小、下载方式、标签来源和能否构成真实 RNA+ATAC；
- 当前根盘约 2.2 GiB 可用，未经大小确认不得下载；不下载 10.7 GB SMART_data.zip 或 9.38 GB raw FASTQ；
- `/autodl-fs/data` 只剩约 64 inode，严格禁止写入；
- 若存在 <=500 MB 的单文件 processed subset 且来源/哈希可闭合，可下载到根盘受控目录；否则只交 capability/space request，不因数据下载阻塞本轮。

## 8. 资源、Git、交付

- 根盘持续保持 >=1.5 GiB；working+delivery 尽量 <250 MB；禁止向 `/autodl-fs/data` 新增文件。
- 优先 CPU；若没有可训练网络，不要为了“有 GPU”而制造训练。GPU 仅在真实需要时使用并报告。
- 旧 raw、branch、tag、delivery 不修改；新 branch/tag 普通提交，不 force push。GitHub SSH 失败一次后保留 bundle，不循环重试。
- 保存失败与工程修正 ledger；正式前数学/维度错误可以修并完整重跑，正式后不能按分数改公式。
- 交付 compact：源码、公式/证明、tiny exhaustive tests、配置 bank/hash、全部指标/controls、label-flow、资源/磁盘、新颖性、数据扩展审计、fresh-process replay、报告与决策 JSON。Windows 独立复算 size/SHA/extras。
- **不要关机**。本轮完成后 AutoDL 保持开机，由 Worker 2 判断是否达到项目里程碑以及是否继续下一任务。

## 9. 执行自由度

科学对象、标签边界和归因对照冻结；具体代码结构、缓存、并行、数值稳定实现由 Codex 2 自主决定。若发现上述充分条件在当前 alpha-expansion/整数容量实现下需要更严格的误差项，应先给出推导和 tiny exhaustive 证据，再做保守修正；不要为了逐字照搬任务书而保留错误公式。若已有正式文献表明同一跨模态认证构造已被完整提出，应立即降级新颖性并报告，不要换名字规避碰撞。
