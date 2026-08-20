# SpaLORA Night‑9A 高效拓扑迁移关系融合研发任务书

版本：2026‑08‑20 REV1  
任务性质：RNA+ATAC family-specific score-preserving efficiency R&D  
工作名：`ETRF = Efficient Topology-Transferred Relational Fusion`

## 0. 第一条回复、算力与总目标

第一条回复必须原样为：

> 已读取 Night‑9A 高效拓扑迁移关系融合任务书，开始 P0‑AUTHORITY‑SEMANTICS‑PROFILE。本轮只研究 RNA+ATAC：保留 R02 的准确性优势，同时把第二套完整 G00 训练替换为低成本拓扑迁移专家；MISAR Y 永久封存，严禁第三次读取。

请由用户在 AutoDL **有卡模式**开机后执行。不得调用 AutoDL API。

总 wall time 上限 6 小时；单个正式 chain 上限 20 分钟。不得无限等待。GPU 利用率可能因 1,949 spots 的小型 full-batch 图训练而间歇波动，但任何单元若长期 CPU-only 或 CUDA 未实际参与训练，必须停止并审计，不能静默继续。

本轮目标按优先级排列：

1. 保住已经确认的 R02 分数优势；
2. 将端到端 runtime 从 `2.261×` 压到 `<=1.50×`；
3. 把“双完整骨干”改造成可泛化、可写成方法创新的“单主骨干 + 拓扑迁移专家”；
4. 不为了提速牺牲空间保护，不利用 MISAR 标签重新选模型。

## 1. 当前里程碑与成本根因

Night‑8B 已确认，在固定统一 head 下 F00/R02 相对 U00：

- mean ΔARI `+0.013742`；
- mean ΔNMI `+0.020982`；
- mean ΔQ `+0.017362`；
- ARI/NMI/Q 都是 `10/10` wins；
- 空间保护全部通过；
- 但 runtime ratio=`2.261×`，资源门失败。

独立拆账：

| 项目 | mean seconds |
|---|---:|
| U00 G04 backbone | 27.6034 |
| U00 head | 0.6433 |
| U00 end-to-end | 28.2467 |
| F00 G00+G04 backbones | 56.3029 |
| F00 R02 adapter | 6.9800 |
| F00 head | 0.5836 |
| F00 end-to-end | 63.8665 |

`1.50×` 对应 `42.3701 s`，至少需节省 `21.4964 s`。仅优化约7秒的 adapter 不够；必须避免第二套约28.7秒的完整 G00 从头训练。

## 2. 方法思想与创新边界

F00 的科学价值来自：G00、G04 两种拓扑视角 + `RECON+MNN` 关系 adapter。Night‑9A 不删除关系 adapter，而是以 G04 为唯一完整主骨干，廉价构造第二个 G00 拓扑专家：

1. **Weight transfer 路线**：同 seed 的 G04 checkpoint 严格加载到形状相同的 G00 模型，在 G00 图上零样本 forward 或固定短程微调。
2. **Precomputed topology projection 路线**：借鉴 SGC/SIGN，用 G00/G04 的固定稀疏传播算子把 G04 latent views 映射成 virtual G00 views，无第二骨干训练。
3. 所得三份 virtual-G00 views 与原三份 G04 views继续进入完全不变的 R02 `RECON+MNN/equal/160 epochs` adapter。

参考源码/论文：

- SGC：`https://proceedings.mlr.press/v97/wu19e.html`
- SIGN：`https://arxiv.org/abs/2004.11198`
- GNN checkpoint transfer/fine-tuning：`https://github.com/snap-stanford/pretrain-gnns`

必须实际阅读相应论文和源码关键实现，保存 `source_code_inspiration_audit.md`，说明哪些思想被采用、哪些没有照搬、如何适配 SpaLORA。不得只复制论文摘要。

## 3. 权威文件与 Git

机器可读注册表：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night9a_efficient_topology_transfer_planning_20260820/SpaLORA_Night9A_Efficient_Topology_Transfer_Registry_2026-08-20.json`

任务书与注册表冲突立即停止。

Git：

- base commit：`991ba9dcbd7108c2b3b4c9b4b8e233c5f394da5a`
- base tag：`night8b-cardinality-safe-eval-final-20260820`
- branch：`revision/q2-night9a-efficient-topology-transfer-rnd-20260820`
- protection tag：`baseline/pre-night9a-efficient-topology-transfer-rnd-20260820`
- final tag：`night9a-final-20260820`
- worktree：`/root/autodl-fs/SpaLORA-night9a`
- raw root：`/root/autodl-fs/night9a_efficient_topology_transfer_20260820`

普通 push，禁止 force/force-with-lease/移动 tag。

必须核验注册表列出的全部 Windows SHA，包括 Night‑7B R02 candidate lock 与 Night‑8B report、decision、resource、statistics、deltas、delivery index。

## 4. P0‑AUTHORITY‑SEMANTICS‑PROFILE

### 4.1 历史证据只读

定位并只读复核：

- P22 Night‑7B 的 10-seed G00/G04 views、R02 embeddings/checkpoints、C06/endpoint 与历史 metrics；
- MISAR Night‑8B 的 10-seed G00/G04 checkpoints/views、R02 adapter、F00/U00 affinities与20个统一-head partitions；
- 相关 manifests、config SHA、observation SHA；
- Night‑8B 原 raw 297/297 与后续 head-recovery/eval roots 不变。

历史 teacher 不得重训、覆盖、touch 或移动。

### 4.2 状态兼容性

在不读取任何标签的条件下证明：

1. G04 与 G00 模型类、parameter names、shape、dtype完全相同；
2. 同 seed G04 `state_dict` 可 `strict=True` 加载到 fresh G00 model；
3. zero-shot G00 forward 产生完整三视图且有限；
4. 0/40/80/160/320 epochs 路线固定 final epoch，无 best epoch/early stop；
5. optimizer/scheduler/seed/RNG 语义与注册表一致；
6. projection operator 对所有数据集和 seed 使用同一函数，不能接收 dataset name、labels、ARI/NMI/Q 或 teacher score。

任一不符停止 `IMPLEMENTATION_SEMANTICS_INVALID`。

### 4.3 真实性能剖析

用不进入科学结果的独立 smoke seed，分别记录：

- G04 full backbone；
- historical G00 full backbone；
- strict load + zero-shot G00 forward；
- 40/80/160/320 G00 warm fine-tune；
- 四种 precomputed projection；
- R02 adapter；
- affinity + head。

必须记录 wall clock、CUDA event、peak GPU、peak RSS、CPU/GPU utilization timeline。若注册表的预算预测明显错误，可停止报告，但不得在看标签后改候选。

P0 smoke 产物必须标为 invalid for science，不计正式 units。

## 5. 固定候选

候选共9个，详见 JSON：

- `E00_ZERO_SHOT_TOPOLOGY_SWAP`
- `E01_WARM_FULL_E40`
- `E02_WARM_FULL_E80`
- `E03_WARM_FULL_E160`
- `E04_WARM_FULL_E320`
- `E05_SGC1_RESIDUAL50`
- `E06_SGC2_MEAN`
- `E07_DELTA_TOPOLOGY_RESIDUAL50`
- `E08_DELTA_TOPOLOGY_RESIDUAL100`

所有候选最终都必须接完全相同的 R02 adapter：`RECON+MNN`、equal fusion、160 fixed epochs、C06 mean endpoint。不得把 adapter 删掉后冒充 R02 压缩。

### 5.1 Weight transfer

对同 seed：

1. 只训练/复用一次完整 G04；
2. strict load G04 final state 到 G00；
3. E00 直接 forward；E01–E04 分别训练固定 40/80/160/320 epochs；
4. optimizer 为 Adam，lr=`1e-4`，weight decay=0，无 scheduler；
5. 保存 final checkpoint、views、initial/final state SHA、loss curve、runtime与fresh-process reload audit。

### 5.2 Precomputed projection

canonical operator：float64 CSR、非负、sorted indices、zero diagonal、row normalize。对 omics1/omics2 分别用 G00/G04 的 spatial与feature normalized adjacency均值；fused view用两模态 operator均值。

严格使用注册表中的四条公式。输出转回 row-L2 float32。两次独立调用必须 array SHA 完全一致。

## 6. Head 与 teacher 口径

为避免再次被不稳定 spectral endpoint主导，本轮所有 teacher/candidate/comparator都使用同一 `EIGEN_KMEANS100` 实现：

- K 使用各数据集历史锁定值；
- 同一 dataset 下 U00、full F00、E00–E08 head/config完全相同；
- teacher full F00 优先复用已有 affinity；若 P22 只有 embedding+C06，可按历史锁定 endpoint **一次性只读重建 affinity**，并验证其与历史 endpoint语义；
- head 不允许接收 method、labels或metrics；
- 任何 candidate partition failure原样保留，不重试、不 fallback。

## 7. R1：宽筛

固定运行：9 candidates × P22/MISAR × seeds 0,1,2 = `54` chains。

每个 chain 包含：virtual/short-G00 expert、R02 adapter、endpoint、uniform head、checkpoint/reload/partition audits。科学 retry=0。

端到端资源核算不得因复用历史 G04 checkpoint 而把主骨干成本记成0。每个 candidate 的标准成本固定为：同 dataset/seed 的一次完整 G04 主骨干历史实测时间 + topology transfer/projection实测时间 + R02 adapter实测时间 + endpoint/head实测时间。U00仍为一次完整G04+head。另表报告本次实际增量wall time，但不得用增量时间替代方法级端到端时间。

### 7.1 MISAR

MISAR `Y` 已达到 lineage 2次读取上限，永久禁止第三次读取。本轮只计算不依赖标签的 teacher fidelity：

- candidate vs full F00 canonical partition ARI/NMI；
- affinity top-k neighbor Jaccard（k固定10、20）；
- normalized Frobenius/cosine distance；
- cluster-size distribution distance；
- runtime、peak GPU。

不得从旧 metrics 反推逐 spot标签。

### 7.2 P22 DEV-1

只有54个 R1 outputs、manifests、SHAs、runtime全部锁定并普通 push 后，才可打开一次 P22 labels 评价 seeds 0–2。之后不得修改 E00–E08 代码、公式或超参数。

计算 candidate 对 full F00 与 U00 的 ARI/NMI/Q、空间指标、paired deltas。

### 7.3 R1 shortlist

严格使用 JSON 门：runtime、GPU、P22 score preservation/gain/spatial、MISAR teacher fidelity全部通过后，按预注册排序取最多3个。若0个通过，立即终止 `NIGHT9A_NO_EFFICIENT_R02_REPLACEMENT_KEEP_FULL_F00`，不得补候选。

## 8. R2：DEV-2 与最终单候选锁

仅 shortlist candidates，固定 P22/MISAR seeds 3、4，最多12 chains。所有输出锁定/push后才能打开 P22 seeds3–4 labels。

把 seeds0–4 合并，重新应用相同硬门和排序，只锁定1个最终候选。不得因为某个候选“接近门槛”而放宽阈值。

输出 `night9a_final_candidate_lock.json`，锁定 config SHA、代码 SHA、候选ID、两数据集 seeds0–4证据和未来 seeds5–9执行表。

## 9. R3：锁定确认

只运行最终候选，P22/MISAR seeds5–9，共最多10 chains。全部输出锁定/push后一次性打开 P22 seeds5–9 labels；之后不得返回训练或候选。

### 9.1 P22确认

必须同时报告：

- final efficient candidate vs U00；
- final efficient candidate vs full F00；
- seeds5–9 与全10 seeds；
- ARI/NMI/Q wins、paired bootstrap、空间保护；
- runtime与GPU gate。

### 9.2 MISAR无标签确认

继续只比较 candidate 与 full F00 teacher partitions/affinities，绝不读取 Y。

若10/10 canonical partitions 与 full F00 在标签置换意义下完全相同，可数学继承 Night‑8B 已存的 aggregate/per-seed metrics，并必须写成“由 partition identity 推得，未重新读取 Y”。若不是10/10 exact，禁止继承或估计 MISAR分数，只报告 high-fidelity程度。

## 10. 终态

- runtime/GPU/P22全部确认，且MISAR 10/10 partition exact：`NIGHT9A_BALANCED_EFFICIENT_R02_CONFIRMED_WITH_MISAR_EXACT_INHERITANCE`；
- runtime/GPU/P22确认，MISAR达到 high-fidelity但非exact：`NIGHT9A_EFFICIENT_R02_P22_CONFIRMED_MISAR_HIGH_FIDELITY`；
- DEV门通过并锁定候选，但R3证据不足：`NIGHT9A_EFFICIENT_R02_LOCKED_FOR_NEW_EXTERNAL_VALIDATION`；
- 无候选通过：`NIGHT9A_NO_EFFICIENT_R02_REPLACEMENT_KEEP_FULL_F00`；
- 正负混合：`NIGHT9A_PARTIAL_OR_MIXED_EVIDENCE`。

无论结果如何，Night‑8B full F00 的准确性结论保持不变；本轮只能决定是否有更高效的替代实现。

## 11. 预算与禁止事项

- 正式 chain max `76`；总尝试 max `80`；global infrastructure corrections max4；scientific retry=0；fallback=0。
- 禁止 MISAR Y 第三次读取；
- 禁止从旧分数反推标签；
- 禁止改动 teacher artifacts；
- 禁止重训 U00/full F00来美化计时；
- 禁止改候选 epoch、公式、adapter、endpoint或head；
- 禁止 best epoch、seed search、补跑失败、删除负结果；
- 禁止本轮运行 A1/D1/tonsil 科学实验；只允许代码/hash回归证明 RNA+protein universal path未改变；
- 禁止 third-party benchmark或SOTA声明；
- 禁止 raw/checkpoints/views/embeddings/affinities/partitions下载到Windows；
- 禁止 force push、移动tag或提前建final tag；
- 禁止 AutoDL API。

## 12. 测试、交付与关机

至少测试：

- authority SHA/Git peel；
- G04→G00 strict state compatibility；
- candidates 9/9真实构造；
- formula/config SHA唯一且锁定；
- projection determinism；
- warm-start initial state exact等于同seed G04 final state；
- fixed epochs/no early stopping；
- unchanged R02 adapter semantics；
- same head across comparator/teacher/candidates；
- R1/R2/R3 key coverage与预算；
- P22 stage label windows；
- MISAR Y access count=0、lineage remains2；
- teacher roots before/after invariant；
- independent metric/resource/fidelity recompute `<=1e-12`。

最终 compact：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night9a_handoff_20260820/official_compact`

目标 `<10 MB`。raw与checkpoints只保留远端持久盘。compact包含任务书、注册表、源码审计、profile、candidate registry、逐阶段manifests/metrics/fidelity/runtime、失败记录、final lock/decision/report、测试、Git、delivery index、planner tar和增量bundle。

final tag只能在报告、测试、delivery index全部进入final commit后创建一次并普通push。

完成所有远端工作、Git push、D盘复制与Windows校验后，在同一保留SSH会话把 `/usr/bin/shutdown` 作为最后一条远端命令；之后不重连。

最终先用通俗中文回答：

1. 为什么旧F00慢；
2. 新候选少训练了什么；
3. P22分数保住/提高了多少；
4. MISAR是exact继承、high-fidelity还是无法继承；
5. runtime从2.261×降到多少；
6. 这能否成为论文中的“高效拓扑迁移关系融合”创新点。
