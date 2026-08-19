# Night-8A 文献、源码与外部数据侦察

日期：2026-08-20  
用途：为下一轮“以公平、可泛化的数值提升为核心”的模型研发提供依据。本文不是论文结论，也不把任何第三方结果冒充为本项目结果。

## 1. 当前证据起点

1. 当前最可靠的统一主候选仍是 `C00_G04_H05_CONFIRMED`。
2. Night-6D 的同 seed 配对结果：
   - D1：mean ΔARI `+0.031700`，mean ΔNMI `+0.028965`，mean ΔQ `+0.030332`，空间门通过。
   - P22：mean ΔARI `+0.016361`，mean ΔNMI `+0.046475`，mean ΔQ `+0.031418`，空间门通过；但 ARI 稳定性弱于 NMI/Q。
3. Night-7A 显示 C00 相对 G00/H00：A1 ΔQ `+0.008270`、tonsil `+0.083227`、D1 `+0.030332`、P22 `+0.031418`。
4. Night-7B 的 `R02` 是明确的 P22 前沿候选：相对 C00，P22 ΔQ `+0.07171`，但 A1、D1、tonsil 均有轻微下降，所以不能冒充统一赢家。
5. Night-7C recovery 已证明简单 router 与 weighted-MNN 没有解决冲突；继续沿这条窄路搜索的预期收益低。
6. 因 A1 与 D1 属于同一套人类淋巴来源，后续排名不能把它们当作两个完全独立生物数据集重复加权。

## 2. 关键结论：同一代码框架不等于所有模态使用完全相同配方

当前较强方法普遍承认不同测序技术与模态有不同噪声、稀疏性和图结构。源码与教程中常见的是：同一方法框架保持统一，但根据 `RNA+protein`、`RNA+ATAC/epigenome` 等模态家族选择预先声明的图参数、损失权重或观测模型。

这意味着：

- 可以采用“模态家族条件化”的方法；
- 不应使用数据集名称、标签、ARI/NMI 或人工观察结果来决定配方；
- 同一模态家族的规则必须在独立数据上复现，否则只能算开发集适配，不能算泛化创新。

因此，Night-8A 把当前数据分为：

- `RNA_PROTEIN`：A1、D1、tonsil；
- `RNA_EPIGENOME`：P22；
- 将来新增数据只能通过输入 assay metadata 归类，不允许写 `if dataset == "P22"` 之类的分支。

## 3. 源码级发现

### 3.1 SMART（Nature Communications, 2026）

官方论文：https://www.nature.com/articles/s41467-026-70821-5  
官方仓库：https://github.com/Xubin-s-Lab/SMART-main  
本次读取的上游快照提交：`75676546...`；仓库 GPL-3.0。

源码而非摘要所显示的核心：

- 每种模态各自使用 GraphSAGE 编码器/解码器，再拼接并通过全连接层形成共同表征；本地快照 `smart__model.py:27-33`。
- 损失为重构与按模态 triplet loss；本地快照 `smart__train.py:128-161`。
- 正样本来自模态内 MNN，负样本来自较远 spot；教程针对不同平台使用不同空间邻居数、最远负样本比例、学习率、权重衰减和训练轮数。
- 优点是轻量、可扩展；论文还使用 AMI、FMI、homogeneity、V-measure、Moran’s I，以及多切片场景下的 iLISI/kBET 等指标。

可迁移启发：模态家族条件化、轻量 GraphSAGE、固定的远距离度量学习。  
限制：Night-7B 已显示简单 MNN family 只带来很小的描述性增益，所以 triplet 仅作为对照模块，不作为唯一主线；不复制 GPL 源码，采用独立实现。

### 3.2 SpaMosaic（Nature Genetics, 2026）

论文：https://www.nature.com/articles/s41588-026-02573-3  
官方仓库：https://github.com/JinmiaoChenLab/SpaMosaic  
文档：https://spamosaic.readthedocs.io/en/latest/  
本次读取的上游快照提交：`33319c...`；MIT。

关键源码/文档思想：

- 对比学习 GNN 同时做特征与图重构；
- 支持 vertical、horizontal 与 mosaic integration；
- 对 protein 与 epigenome 推荐不同 graph-reconstruction 权重；
- dev 分支强调 sparse message passing 与 decoupled training，说明大图必须避免密集 `N×N` 路径。

可迁移启发：跨模态一致性损失、稀疏图、模态家族条件化图重构。Night-8A 优先实现不需要负样本的 VICReg-style shared alignment，以减少同域 spot 被当成假负样本的风险。

### 3.3 MultiGATE（Nature Communications, 2025）

论文：https://www.nature.com/articles/s41467-025-63418-x  
官方仓库：https://github.com/cuhklinlab/MultiGATE  
本次读取的上游快照提交：`7e846af...`。

核心思想：模态内空间图自编码器、跨模态特征关系和 same-spot CLIP 对齐共同优化。其成人海马数据报告了较大的 ARI 提升，也把调控关系作为聚类之外的生物学任务。

可迁移启发：same-spot cross-modal alignment 是高优先级；但不沿用“attention 权重天然可解释”的叙事，因为 Night-3B 已证明 attention 稳定不等于 faithful 或有益。

### 3.4 COSMOS（Nature Communications, 2025）

论文：https://www.nature.com/articles/s41467-024-55204-y  
本次读取的上游快照提交：`56ea355...`。

核心思想：双通道 DGI 捕获全局互信息，再用 WNN 与空间正则融合。

源码审计警告：本地快照 `COSMOS__cosmos.py:235-239` 的 accelerated spatial regularization 分支中，`c1` 与 `c2` 都索引 `cell_random_subset_1`，可能让抽样坐标距离退化为零。不能把论文描述或官方代码当作无误真理；DGI 只作为独立重写的对照模块。

### 3.5 ARISE（Bioinformatics, 2026）

论文：https://academic.oup.com/bioinformatics/article/42/7/btag465/8721301  
本次读取的上游快照提交：`fefdd84...`；未发现明确许可证。

可用思想：以 RNA 特征相似性与空间邻接交集构建共享边，再对辅助模态做 RNA-anchored message passing，适合稀疏/noisy epigenome。

不可照搬之处：本地快照 `ARISE__train.py:76-125` 把真实标签传入每个 epoch，并按 best ARI 保存 embedding。这个训练终点对公平无标签评价不可接受。Night-8A 只能独立实现 RNA-anchor 拓扑，必须固定 epoch/无标签停止，绝不能按 ARI/NMI 选 checkpoint。

### 3.6 SpaMCA、SpaMode 与 PRAGA

- SpaMCA：masked multi-view graph AE、instance/cluster contrastive、cluster projector；Night-7B 的简单 MASK family 已表现较差，因此不把 masking 当主线，但保留 prototype alignment 思路。
- SpaMode：shared/private variational representation、PoE/MoE 与模态判别；其仓库未发现明确许可证，且 Night-7B 的简单 MoE 并未等价测试完整 shared/private 机制。
- PRAGA：可学习邻接与 split/merge prototype；密集 `N×N` 邻接对 P22/未来大数据风险过高，不能进入主实现，除非提供等价稀疏算法。

这些仓库只作概念审计；未确认许可证的源码不得复制。

## 4. 从其他深度学习方向迁移的主设计

Night-8A 工作名为 `MF-SPC`：Modality-Family-conditioned Shared-Private Prototype Consensus。

它不是把多个流行名词简单堆在一起，而是围绕一个具体冲突组织：RNA+protein 与 RNA+epigenome 的共同生物信号、模态私有信号和空间图可靠性不同。

候选机制：

1. `SP`：每模态编码为 shared 与 private 子空间；shared 用于跨模态融合，private 保留重构所需的模态特异信息，并用 covariance/orthogonality penalty 限制泄漏。
2. `RR`：对 same-spot shared embeddings 使用 VICReg-style invariance/variance/covariance 约束；避免 InfoNCE 把同一空间域内不同 spot 当作负样本。
3. `PROTO`：模态 shared 与 fused embedding 的软原型分配保持一致；使用弱平衡而非把所有真实簇强迫等大。
4. `RNA_ANCHOR`：仅在 assay metadata 判定为 `RNA_EPIGENOME` 时启用 RNA-anchored shared-edge topology；protein 家族保持 C00 图策略。
5. `DGI` 与 `SMART_TRIPLET`：分别作为全局互信息、远距离 metric-learning 对照，而非默认必加组件。

工作假设：C00 是稳健基础；R02 暗示 epigenome 需要不同的结构/终点。新的方法应至少做到：保住 human lymph、保住 tonsil 的大幅收益，并将 P22 的 specialist 收益变成可在独立 RNA+ATAC 数据上复现的模态家族规则。

## 5. 评价体系升级

### 5.1 选择指标

- Primary：ARI、NMI、`Q=(ARI+NMI)/2`。
- 重要性加权：先合并 A1 与 D1 为 `Q_HLN`，再计算 `0.45*Q_HLN + 0.45*Q_P22 + 0.10*Q_tonsil`。
- 空间保护：neighbor agreement、Moran’s I、Geary’s C、boundary disagreement，沿用已有锁定方向和容差。

### 5.2 次级、描述性指标

- 带标签：AMI、FMI、homogeneity、completeness、V-measure、per-class/region recall（若标签语义允许）。
- 无标签：silhouette、Davies-Bouldin、Calinski-Harabasz、图连通性、modularity、空间自相关与运行时/显存。
- 多切片/多批次未来指标：iLISI、kBET、batch ASW、graph connectivity；生物保真与批次混合必须分别报告。

scMultiBench 的 2025 Nature Methods 注册报告在 64 个真实、22 个模拟数据上比较 40 种方法，明确指出不同任务与指标会导致不同排名；因此不能把单个 ARI 当成全部，但本阶段仍把 ARI/NMI/Q 作为清晰的首要提分目标。来源：https://www.nature.com/articles/s41592-025-02856-3

## 6. 外部数据候选与优先级

### 第一优先：独立 RNA+ATAC / RNA+epigenome 验证

1. `MISAR-seq mouse brain`（OEP003285；SMART 处理版见 Zenodo 17093158）
   - 多个发育时间点/section，约千到两千级 spots；计算量适中。
   - 与 P22 不同技术/研究，是检验 `RNA_EPIGENOME` 家族规则最关键的候选。
   - SMART 文中部分所谓 ground truth 来自原研究的无监督/人工整理注释，必须先审计 provenance，不能默认是专家金标准。
2. `GSE205055` 中 mouse embryo E13 等 section
   - 官方 GEO：https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055
   - 与 P22 同一 SuperSeries，适合作为同研究 replication，不算真正 study-independent holdout。

### 第二优先：RNA+protein 复现

3. Tonsil section 2/3（Zenodo 12654113）
   - 能检验同组织不同 section 和蛋白 panel 变化；但与现 tonsil 同来源，属于内部 replication。
4. SPOTS spleen replicate（GSE198353）
   - 两个 replicate 的模态与坐标可用；缺少官方人工域标签，进入 label-free replication。

### 第三优先：规模与新模态压力测试

5. STARmap+RIBOmap（Zenodo 8041114）
   - 约 5.8 万对齐 spots，可检验 RNA+translatome 与扩展性。
   - 数据规模大、对齐过程复杂，不在 Night-8A 直接下载或正式运行。

数据策略：Night-8A 只做 provenance、文件列表、大小、许可证、spot 对齐、模态与坐标完整性预检；最多在服务器持久盘选择性下载一个中等规模 MISAR section。不得把 1–2 GB 以上的原始包下载到 Windows D 盘。

## 7. 明确不做的事

- 不按数据集名字硬编码参数。
- 不用真实标签、ARI/NMI、H&E 人工观察、seed 搜索或 best epoch 参与训练与 checkpoint 选择。
- 不删除失败 run，不回填负结果，不只汇报“最好 seed”。
- 不宣称当前已经 SOTA；正式 SOTA 结论必须在相同预处理、相同 spots、相同 K、相同聚类终点和多个 seeds 下与官方方法公平比较。
- 不复制无许可证代码；即使有许可证，也优先独立实现并记录来源。
- 不再运行 Night-7C router/weighted-MNN 变体，除非出现新的机制证据。

## 8. 阶段性科学判断

项目已跨过“旧模型是否完全失败”的阶段：C00 在 D1/P22 已有可重复的正增益，R02 还显示 P22 存在更高的可达前沿。当前缺口不是没有信号，而是尚未把不同模态家族的最优规律统一成一个有清晰机制、能在独立数据泛化的方法。

Night-8A 的成功标准不是堆更多图，而是产出以下之一：

1. 一个同时保住 human lymph 与 P22 的 `MF-SPC` 候选；或
2. 一个明确的 modality-family policy，并在独立 MISAR RNA+ATAC 上冻结验证；或
3. 即便未涨分，也用严格消融排除 shared/private、RR、prototype、RNA-anchor、DGI、triplet 中的无效模块，收缩下一轮搜索空间。
