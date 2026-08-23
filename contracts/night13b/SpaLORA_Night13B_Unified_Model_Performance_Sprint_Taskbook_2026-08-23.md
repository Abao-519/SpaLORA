# SpaLORA Night-13B：统一模型性能冲刺

## 1. 本轮要解决什么

Night-13A 已证明数据和两种真实模态路径可以运行，但只得到很弱的 simple concatenation 工程锚点，且外部强基线因为规划边界过严而根本没有真正进入 runner。本轮不再把主要算力用于复现外部方法，而是优先回答：

> 能否在 RNA+protein 与 RNA+ATAC 上使用同一套模型主体，通过可解释的融合改进，稳定超过各自最强的项目内部参考？

这是一次公开标签可参与开发选择的性能冲刺。标签可以用于 ARI/NMI 评价、候选淘汰和超参数选择；为保持论文仍是无监督空间聚类方法，标签不进入训练 loss、梯度或单次运行的 checkpoint 选择。

## 2. 对 Night-13A 的正式判定

Night-13A 保持 PARTIAL_ENGINEERING_AND_DEVELOPMENT_BENCHMARK：两个真实 P0 和六行 simple seed-0 数值是有效工程起点，但不能当作竞争性方法证据。

Night-13B 必须先建立 errata ledger，保留原文件不改，并更正：

1. SpatialGlue commit 被写成 41 位；正确 40 位是 7c976d811d27ace51ce47ae0ad94a068a7d222fa。
2. P5S1/S2/S3 的 RNA/ATAC accession 应为 GSM9247581/8997、GSM9247582/8998、GSM9247583/8999。
3. MISAR E15.5 S1 已有官方 Figshare Y：1949/1949 对齐、K=7、carrier SHA-256 为 2f5862cff045b6a296f3cbc0a978d8576bd36f2d4c9419b75c6760c9c7e9d2e3；它是历史已用的公开开发标签，不是无标签数据。
4. registry 漏掉 Night-12A 已闭合的 P10S1/S2/S3 RNA+protein 重复；补为 UNLABELED_ONLY。
5. donor 列目前是样本/切片代号而非闭合的真实 donor ID；证据不足时写 UNKNOWN，不得推导 donor 独立性。
6. tonsil s1/s2/s3 作为同一 study block；s1/s2 是相邻连续切片，三片不能在统计中算三张独立选票。
7. Night-13A runner 对 tonsil s2/s3 先按 final_annot 非空过滤，再建立表示和聚类，实际只用了 4518/4460 点。Night-13B 要先用全部 4519/4521 个配对点建立 embedding/partition，最后只在有标签的 4518/4460 点计算指标，并重算相关锚点。

## 3. 战略调整

本轮外部方法只做两件轻量工作：

- 从官方论文和 supplement 登记 A1、D1、tonsil、P22、MISAR 可核对的 reported best 数字和协议，形成 SOTA context board；
- 必要时阅读正式源码寻找可迁移的机制，但不做 SpatialGlue、SMART、ARISE 的大规模复现。

reported SOTA 只作为冲刺目标，不能与本项目 common endpoint 数值直接混称公平胜负。正式外部复现与公平 comparison 留到自己的候选达到稳定水准之后。

本轮真正的即时门槛不是 Night-13A simple anchor，而是同协议下能闭合的最强内部参考：RNA+protein 的 C00/G04+H05，RNA+ATAC 的 F00/R02，以及 P22 上的 N02 历史开发上界。G04 是构图方式，决定空间点怎样连边；H05 是统一聚类 head，决定怎样把表示划成空间区域；N02 是曾在 P22 有明显增益、但在 A1 退化的层级融合配置。

## 4. 统一模型原则

“统一”不等于所有输入矩阵维度相同。允许 RNA、protein、ATAC 使用各自标准预处理和输入 projection；模型可以根据显式 modality schema 选择输入 adapter。进入公共 latent 后，下列内容必须是一套：

- encoder/fusion 模型类和前向语义；
- 训练 loss 的定义；
- 稀疏图与融合公式；
- clustering endpoint；
- 全局超参数选择规则。

禁止读取 A1、P22、tonsil 等 dataset 名称后切换另一套 backbone、loss 或后处理。输入维度、模态类型、样本数量和坐标尺度可触发机械的 shape/resource rule。

Codex 2 有权在 discovery 阶段自由组合和淘汰机制。优先考察但不限于：

- 把 N02 层级表示作为对通用表示的受控 residual，而不是直接替换通用表示；
- 全局或 spot-level 的连续融合权重；
- sparse spatial/feature graph residual；
- cross-reconstruction；
- loss/gradient balancing；
- shared/private regularization。

这些是搜索原语，不预先声称为论文创新。每个正式候选必须有独立 ID、公式摘要、来源/clean-room 说明、resolved config 和参数量。若 discovery 中修改公式，就创建新候选 ID；不把修改覆盖在旧结果上。

## 5. 数据角色

### 有标签 discovery

- A1 lymph node，K=10；
- canonical tonsil slice 1，K=4；
- P22 mouse brain，K=9。

A1 与 tonsil slice 1 代表两个 RNA+protein study block，等权计分；P22 是 RNA+ATAC discovery unit。

### 候选冻结后的内部 confirmation

- D1 lymph node，K=10；
- tonsil slices 2/3，K=4，先逐切片报告，再合成一个 tonsil study-level effect；
- MISAR E15.5 S1，K=7。

这些数据历史上已经使用过，报告称 internal confirmation / public benchmark confirmation，不称 pristine blind test。

### 无标签真实压力测试

- SPOTS spleen replicates 1/2；
- P10S1/S2/S3；
- P5S1/S2/S3。

它们不参与有标签候选排名。只有候选表现达到继续门时，才按资源优先做真实 forward、checkpoint reload、稀疏 endpoint、repeat stability、邻域保持、Moran/Geary、模态一致性和资源审计。

GSE tonsil A1/D1 永久 audit-only；legacy placenta 因第二模态 provenance 未闭合而排除。

## 6. 执行节奏

### B0：修正基础和竞争参考

1. 建 errata ledger 与完整 registry v2。
2. 修 runner：全体配对 observation 先建表示/partition，标签 mask 只进入 evaluator。
3. 加最小回归测试，并重算 simple anchors；主表增加 total observations、evaluated observations、ordered-ID hash、embedding/partition SHA。
4. 定位并验证 C00、F00、N02 的 checkpoint、embedding、partition、manifest 和 SHA。能精确复用则写 LOCKED_REUSE；缺工件时按冻结配置完整 replay，并写 EXACT_REPLAY，不能冒充 reuse。
5. 在相同 observations、相同 known-K common endpoint 下形成 strong_internal_reference_board。native endpoint 只作补充。

### B1：真实双家族 P0

候选统一模型至少在 A1 和 P22 上各走一次：

load → preprocessing → forward/loss → backward finite-gradient smoke → checkpoint strict reload → fresh-process numerical round-trip → fusion → common clustering endpoint。

同时列出真实 tensor shapes。P0 只用于排除代码/语义错误，不要求此时提分。

### B2：自适应 discovery 搜索

建议从 8–20 个有机制差异的候选开始，先用 seed 0 在 A1、tonsil s1、P22 粗筛。Codex 2 可以根据早期结果：

- 立即淘汰被内部参考全面支配的候选；
- 把剩余预算用于最有希望机制的邻域；
- 增减候选数量；
- 调整正常工程 batch、学习率、epoch 和 loss weight 搜索范围；
- 在理由和结果都写入 search ledger 的前提下提前停止明显无效路线。

不设置“全局只能修一次”的 correction cap。纯工程错误正常修复并重跑受影响单元。探索期允许基于公开标签做 HPO；每次科学配置变化都有新 ID。

### B3：多 seed 收敛与冻结

优先候选进入至少 seeds 0/1/2；最终 1–2 个候选进入 seeds 0–4。排名主量为：

- 每个数据集的 absolute ARI、absolute NMI；
- 相对最强可比内部 reference 的 ΔARI、ΔNMI；
- RNA+protein 以 A1 与 tonsil study 等权；
- 两个模态家族分别汇总，再看较弱家族和 macro；
- 运行时间、显存、RAM 作 Pareto tie-break。

Moran’s I、Geary’s C、AMI、FMI、homogeneity 和 V-measure必须报告，但不能用高空间平滑掩盖低 ARI/NMI。

冻结唯一候选、唯一全局配置和 seeds 后，才开启 D1、tonsil s2/s3、MISAR confirmation。confirmation 后不得返回修改候选再重算同一终态；若要继续，另开下一轮 discovery。

### B4：可选无标签扩展和 SOTA context

候选在两家族都有正向迹象时，再运行无标签真实重复。若没有正向迹象，优先封口，不为凑工作量消耗 P5/P10 大规模资源。

从官方论文/supplement 建 SOTA context board，记录 method、dataset、absolute metric、seed/endpoint/K、数据版本和直接可比性。这里只追踪目标，不复现整套外部方法。

## 7. Codex 2 的自主权与底线

Codex 2 对候选组合、搜索顺序、计算预算分配、正常工程修复和提前淘汰拥有自主权，无需逐项等待用户或 Worker 批准。应以“尽快找到跨家族可用的统一候选”为目标，而不是机械跑满表格。

仅保留以下不可协商底线：

1. 不伪造、不删除失败、不只报最好 seed；
2. 不按 dataset 名称切换整套模型；
3. 同一比较使用同 observations、known K 和 common evaluator；
4. 不把标签放进无监督候选的 loss/梯度/checkpoint 选择；
5. 不修改历史 raw、旧工件或旧 tag；普通 push，不 force；
6. 不把公开开发/内部 confirmation 夸大成盲测或 SOTA。

## 8. 终态分类

- IMPLEMENTATION FAILURE：真实路径未按设计执行，不能评价方法。
- INFRASTRUCTURE FAILURE：资源/环境阻塞，不能评价方法。
- SCIENTIFIC NEGATIVE：实现正确，但搜索后没有候选稳定超过强内部参考。
- LOCAL SIGNAL：只在一个家族或部分 study 有提升。
- CONFIRMED MILESTONE：同一模型主体在两个家族的 study-balanced ΔARI/ΔNMI 均为正，多 seed 稳定，冻结后的 internal confirmation 没有整体翻转。
- PAPER-READY EVIDENCE：Night-13B 不直接授予；还需要外部强基线、公平完整指标、消融、生物解释和至少一个真正冻结后的外部确认来源。

即使候选接近或超过论文 reported number，也只能先写“reported-SOTA context reached/approached”，不能写正式 SOTA。

## 9. 精简交付

必须交付：

- errata ledger、registry v2；
- corrected anchor board 和 strong internal reference board；
- candidate registry、search ledger、全部 run/failure manifest；
- absolute ARI/NMI 主表、study/family 汇总和资源表；
- SOTA context board；
- real-path P0 与 tests；
- final report、plain summary、decision JSON；
- 增量 bundle、root-relative compact index 和 Windows 独立复算。

大型 checkpoint/embedding 留在远端，只在 compact 交付必要的 config、manifest、hash、表和小型 partition/prediction evidence。最终回复先给三件事、结果分类和绝对指标，再给导师汇报版，Git/hash 放技术附录。

