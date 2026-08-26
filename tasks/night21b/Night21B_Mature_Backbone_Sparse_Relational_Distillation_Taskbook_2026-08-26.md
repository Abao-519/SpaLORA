# Night-21B：成熟骨干上的稀疏关系蒸馏与强表示保护

日期：2026-08-26

## 0. 本轮要解决的真实问题

Night-21A 已证明：AMCF 的工程实现成立，但当前“三尺度纹理 + 分层注意力 + 有界残差”组合没有形成科学协同。这个结论不能外推为“组合创新全部无效”，因为 Night-21A 的 `CARRIER_ONLY` 是 retained embedding 经重新标准化、归一化和固定 KMeans 后的 common-head 起点，并不是各 lane 历史最高分对应的最强可复用数值表示；A1、tonsil s1、P22、placenta 的起点均明显低于项目既有分数前沿。

Night-21B 不再扩展 AMCF。它改用一个有正式论文、官方源码和允许修改许可证的成熟空间多组学图学习骨干，并只新增一个针对性模块：在骨干学习新表示时，保护强 carrier 中已经存在的局部邻域和边界关系，避免图传播把原本可分的结构抹平。

临时工作名为 `MSRD`（Mature-backbone Sparse Relational Distillation，成熟骨干稀疏关系蒸馏）。这个名字只是工程代号，不是最终论文命名。

本轮不是复现大赛，也不是把外部骨干冒充自研。必须把三类结果分开：

1. `BACKBONE`：成熟公开骨干本身的成绩；
2. `OUR_MODULE`：在同一骨干、同一训练预算和同一 endpoint 下，稀疏关系蒸馏相对骨干与强 carrier 的净贡献；
3. `SCORE_FRONTIER`：允许透明 benchmark HPO 后的绝对最高分，但不能反向充当方法贡献证据。

## 1. 必须先读的 authority

完整读取并复核，不要只看摘要：

- `D:/文档/ChatGPT/博士第一篇科研论文项目/night21a_delivery_20260826/official_compact/outputs/night21a_handoff/night21a_report.md`
- `D:/文档/ChatGPT/博士第一篇科研论文项目/night21a_delivery_20260826/official_compact/outputs/night21a_handoff/absolute_metrics_and_controls.csv`
- `D:/文档/ChatGPT/博士第一篇科研论文项目/night21a_delivery_20260826/official_compact/SpaLORA/night21a_amcf.py`
- Night-19C、Night-16B、Night-16C、Night-16H 和 Night-14B/15B 的正式报告、分数表、表示/分区工件及 manifest；自行按文件名定位最新 verified compact，不凭记忆抄数字。
- 项目治理文件：
  - `D:/文档/ChatGPT/博士第一篇科研论文项目/research_governance_20260821/PROJECT_MEMORY_PLUGIN_RESEARCH_WORKFLOW_2026-08-21.md`
  - `D:/文档/ChatGPT/博士第一篇科研论文项目/research_governance_20260821/Spatial_MultiOmics_Reading_List_2025_2026.md`

官方主候选骨干：

- spaMGCN 论文及官方仓库：`https://github.com/hongfeiZhang-source/spaMGCN`
- 仓库当前公开为 MIT License。必须固定实际读取的 commit，阅读 `model/`、`train/`、`utils/`、config 和相关 notebook，不要只读 README。

相关先例仅用于碰撞审计和设计边界：SpatialGlue、BANKSY、关系知识蒸馏、图对比学习、局部度量学习。优先原论文和官方源码。

## 2. Stage 0：磁盘、来源和“强 carrier”闭合

### 2.1 AutoDL 磁盘门

连接后第一件事运行并保存：`df -h`、项目目录和缓存目录占用、最大 30 个文件/目录。

- 不下载新数据。
- 不复制历史 raw。
- 官方源码只允许浅克隆或下载固定 commit 的最小代码快照；下载前先估计大小，建议上限 200 MiB。
- 优先复用现有 Python/PyTorch 环境，不新建大型 conda 环境，不升级 CUDA/PyTorch。
- 若系统盘使用率达到或超过 93%，或可用空间低于 8 GiB，停止新增安装/下载，完成本地只读审计并报告是否需要扩容。不要通过删除历史证据来腾空间。
- 本轮所有新工件总量目标低于 1 GiB；checkpoint 只保留必要的 best/final 和 replay 证据。

### 2.2 强 carrier 不是“最高分 partition”的同义词

为 A1、D1、tonsil s1/s2/s3、P22、MISAR、placenta 建立 `frontier_carrier_bridge.csv`，每条 lane 至少记录：

- 历史最高可信 ARI/NMI partition 的来源、SHA、head、K、是否 label-assisted HPO；
- 对应的 pre-clustering 数值表示是否仍然存在且可重放；
- 若存在：shape、dtype、spot ID hash、表示来源、common endpoint 复算 ARI/NMI；
- 若不存在：明确标记 `PARTITION_ONLY_NOT_A_REUSABLE_CARRIER`，不得用标签或最高分 partition 反推表示；
- 本轮可用的最强数值 carrier 及其 common-head 成绩；
- Night-21A carrier 与该数值 carrier 的差异原因。

任何 representation 必须先按 ordered spot ID 对齐，并做 fresh-process hash/replay。不能因为历史 partition 分数高就宣称它对应的 embedding 也强。

### 2.3 骨干来源审计

固定 spaMGCN 官方 commit、许可证和文件清单。检查：

- 两模态 encoder、图构建、多尺度卷积、融合、训练 loss、初始化和 clustering endpoint 的真实代码路径；
- 官方 notebook 是否含 dataset-specific 超参数、标签驱动 checkpoint/epoch/cluster selection、dense `N×N` 或隐式全对全距离；
- A1、P22、placenta、tonsil 的输入能否映射到官方语义；
- 不能原样运行时，区分 `OFFICIAL_REPLAY` 与 `SOURCE_FAITHFUL_PORT`。若做稀疏端口，保留官方源码快照，并在小输入上验证端口与原算子的数值等价或清楚说明不可等价处。

如果 spaMGCN 缺少不可替代语义或无法在现有资源内完成真实 A1/P22 P0，不要中途换另一个骨干并混在同一 revision；以 `BACKBONE_P0_BLOCKED` 封口并给出下一骨干建议。

## 3. Stage P0：真实端到端训练语义

先在 A1 与 P22 各做一个真实 seed：

`raw feature-level modalities -> registered preprocessing -> sparse graphs -> backbone forward/loss -> backward/optimizer update -> checkpoint strict reload -> fused embedding -> common endpoint`

必须报告真实 shape、参数量、optimizer steps、参数变化、finite loss/gradient、checkpoint fresh-process 数值重放和 exact K。不能用合成 adapter 代替真实混合路径。

P0 通过后再对 tonsil s1 与 placenta 各做一个 seed。工程修复可以在 formal freeze 前进行，记录原因；不要因为一次正常训练波动就当实现失败。

## 4. 自研模块：稀疏有符号关系蒸馏

### 4.1 要解决的缺陷

成熟图骨干可以学习跨模态与空间结构，但也可能破坏强 carrier 已有的局部可分性。点对点 L2 anchor 过于僵硬，并且不同 latent 坐标系之间未必可直接比较。本轮不要求学生逐点复制教师，而是保护“哪些近邻应保持接近、哪些空间邻边位于分子边界两侧”。

### 4.2 冻结公式前的源码推导

在首次读取评价标签之前，根据官方骨干输出、强 carrier 和稀疏图冻结具体公式。建议对象如下，Codex 可在保持问题定义不变的情况下做更优但可解释的实现：

- `E+`：强 carrier 的 mutual-kNN 关系中，同时得到空间近邻或跨模态一致性支持的稀疏正边；
- `E-`：真实空间邻边中，在强 carrier/两模态局部关系上显示边界分离的稀疏负边；
- `L_pos`：学生表示保持 `E+` 局部邻接/排序；
- `L_boundary`：学生表示不跨越 `E-` 的冻结 margin；
- `L_MSRD = lambda_pos * L_pos + lambda_boundary * L_boundary`；
- 总损失为官方骨干无监督损失加 `L_MSRD`。所有关系都在稀疏边或抽样负边上计算，不构造 dense `N×N`。

关系阈值、margin 和权重必须来源于 carrier/图的可观测尺度或显式 family-level search grid。不要用 dataset 名称写分支。RNA+protein 和 RNA+chromatin 可以各有一套冻结数值参数，但公式、代码路径和参数语义必须一致。

训练 checkpoint 选择只能使用无标签训练 loss、有限性和表示/partition 稳定性；评价标签不得进入 loss、gradient 或 within-run checkpoint selection。

### 4.3 匹配对照

必须至少包含：

- `T0_STRONG_CARRIER_COMMON_HEAD`
- `B0_BACKBONE_ONLY`
- `B1_BACKBONE_POINTWISE_ANCHOR`（简单 L2/余弦 anchor）
- `B2_POSITIVE_RELATION_ONLY`
- `B3_BOUNDARY_RELATION_ONLY`
- `FULL_SIGNED_RELATIONAL_DISTILLATION`

如果使用 post-hoc convex blend 或不同聚类 head，必须作为单独 score arm，不能混入上述最小贡献对照。

## 5. Discovery、HPO 与冻结迁移

### 5.1 Discovery

- RNA+protein：A1、tonsil s1。
- RNA+chromatin：P22、placenta。

允许透明的公开 benchmark HPO。候选配置先生成、编号、哈希和运行，再由独立 evaluator 读取公开标签；保留完整 candidate board 和失败运行。可以根据结果进行最多两次有明确机制理由的设计修订，但每次修订必须形成新 cycle，旧 cycle 整体保留，不能只删负结果。

训练步数不得再任意固定为很短的 60 steps。优先沿用官方合理预算，或使用无标签 loss/表示稳定性停止；先做学习曲线，证明训练已进入稳定区。

Discovery 首先用 seed 0 探索；每个家族 top-3 配置随后运行 training seeds 0–2，matched endpoint seeds 至少 10 个。BEST、median、mean、min 和胜出 seed 数全部报告。

### 5.2 Family freeze transfer

每个家族只冻结一套公式和超参数：

- protein freeze 后运行 D1、tonsil s2、tonsil s3；
- chromatin freeze 后运行 MISAR；若某 lane 缺精确 carrier/标签协议则报告缺口，不制造替代。

这些数据历史上已被项目使用，因此只能称 `frozen transfer`，不能称 pristine blind confirmation。

## 6. 评价与终态分类

每条 lane 主表至少包括 total/eval、K、ARI、NMI、AMI、FMI、Moran、Geary、最小簇、BEST/median/mean/min、双胜率、训练墙钟、GPU 时间、峰值 GPU/RSS。

建立两张互不混淆的表：

1. `method_contribution_board.csv`：共同 endpoint、匹配训练预算，对比 T0/B0/B1/B2/B3/FULL；
2. `score_frontier_board.csv`：允许透明 head/HPO 的绝对分数，与历史可信 frontier 和公开 context 比较。

终态按证据选择，不强迫正结论：

- `BACKBONE_P0_BLOCKED`：成熟骨干真实路径不闭合；
- `BACKBONE_ONLY_SIGNAL`：公开骨干提高，但 FULL 不超过骨干与 carrier；
- `NO_RELATIONAL_METHOD_SIGNAL`：FULL 无稳定独立贡献；
- `RELATIONAL_MODULE_LOCAL_SIGNAL`：FULL 在至少两个 lane 超过 carrier、backbone 和全部原子臂；
- `FAMILY_FROZEN_RELATIONAL_SIGNAL`：至少一个家族冻结后在多个 transfer studies 维持双指标正效应；
- `CROSS_FAMILY_METHOD_MILESTONE`：两个家族 family-balanced ARI/NMI 都为正，FULL 在不少于四个 primary lanes 超过 carrier、backbone 和全部原子臂，且多 seed 不依赖单一幸运 seed。

任何绝对分数刷新另记 `SCORE_FRONTIER_ADVANCE`，不能代替自研模块贡献。

## 7. 数据扩展短名单（只审计，不下载）

从 spaMGCN 与 SEPAR 原论文、官方仓库/notebook 和 GEO/Zenodo authority 中建立 `dataset_expansion_shortlist.csv`。优先：

- 与当前任务相同的 RNA+protein 或 RNA+chromatin；
- 有可追溯专家标注或可审计 domain assignment；
- 有 feature-level 双模态、坐标、spot ID 和明确 K；
- 不是当前物理样本的重复包装；
- 数据规模和许可适合后续 AutoDL。

每项记录 accession、物种、组织、平台、两模态、N、标签来源、K、下载大小估计、是否与现有资产重复、推荐角色。Human melanoma 只有 K=2 时可以保留为压力测试，但不能单独承担主要泛化证据。

## 8. 资源、复现与交付

- 记录全部 seed、配置、失败、设计 cycle、标签 load event、训练/评价隔离、GPU/RAM/磁盘。
- 不修改历史 raw 和历史 final。
- 不 force push。GitHub key 缺失时只尝试一次普通 push，随后用可验证 bundle 交付。
- Compact 只保留源码、配置、表格、必要 checkpoint/replay、小型日志和报告；不得复制 raw、大缓存、完整第三方数据。
- Windows 独立复算 compact root-relative size/SHA index，报告 missing/size/SHA/extras。

最终报告正文先用通俗中文给出：问题、实际做了什么、论文意义、明确终态；再给绝对指标、贡献对照、数据扩展短名单、导师汇报版和技术审计。

AutoDL 本轮结束后，把 `/usr/bin/shutdown` 作为最后一条远端命令派发；随后不得重连。只表述“命令已派发”，不要声称控制面板已关机。

## 9. Codex 自主权

这不是逐行实现处方。Codex 应完整阅读官方源码和现有项目代码后，自主选择最稳妥的兼容实现、训练预算和稀疏关系损失细节；只要不改变“成熟骨干 + 单一关系保护模块 + 匹配贡献对照 + family freeze”的研究问题即可。发现公式、维度、许可证、数据 authority 或磁盘证据不确定时，先从源码/权威文件闭合；确实无法闭合再 fail-closed，不要猜。
