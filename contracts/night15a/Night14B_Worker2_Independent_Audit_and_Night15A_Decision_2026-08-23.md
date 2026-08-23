# Night-14B 独立复核与 Night-15A 决策

日期：2026-08-23  
负责人：Worker 2

## 我现在需要知道的三件事

1. Night-14B 的最高分是真实、可重放的开发突破，但当前不能确定其中有多少来自 RNA+ATAC、多少来自坐标与聚类 head。
2. 独立复核确认 TSPR（可训练的拓扑状态传播模块）没有独立增益；真正有效的是空间滤波、坐标增强、PCA 和 KMeans/GMM 组合。最关键的缺失对照是 coordinate-only，即只给空间坐标、不提供任何分子信息。
3. 下一步不退回旧分流器，也不立即包装论文；保留有效链，先闭合多模态贡献，再做降方差与 raw-feature 新核心。若分子信息有独立贡献，再把它发展成原创方法。

## 终态复核

- Night-14B 官方终态 `BACKBONE_OR_HEAD_SIGNAL` 成立。
- 它属于开发阶段的 `LOCAL SIGNAL`，不是 SOTA、confirmed milestone 或 paper-ready evidence。
- 45/45 compact 文件及 SHA-256 独立复算闭合；final commit/tag peel 与报告一致。
- 36/36 formal partition fresh-process byte-exact，8/8 训练 checkpoint/reload 通过。

## 复核发现的关键事实

| 数据与链路 | 无坐标最佳 ARI/NMI | 加坐标、refinement 前最佳 | 最终 BEST_RUN | 解释 |
|---|---:|---:|---:|---|
| P22 K=9 | 0.4805/0.6429 | 0.5608/0.6750 | 0.5683/0.6845 | 约 0.08 ARI 来自坐标增强；refinement 只再加约 0.0075 |
| MISAR K=7 | 0.3792/0.5669 | 0.5041/0.6248 | 0.5099/0.6290 | 约 0.125 ARI 来自坐标增强；refinement 只再加约 0.0058 |
| MISAR K=12 | 0.3117/0.5434 | 0.3795/0.5276 | 0.4143/0.5329 | 坐标提高 ARI，但 NMI 不同步，仍远低于公开背景 0.644 ARI |

Night-14B 没有 coordinate-only lane，所以不能把上述增益全部归因于多组学融合。这个缺口不抹去分数，但决定了论文故事究竟是“空间几何分割”还是“空间多组学整合”。

## 种子与搜索压力

- P22 与 MISAR 的 BEST_RUN 多为 `backbone seed 0 + endpoint seed 0`，也是开发搜索使用的单元。
- P22 K=9 和 MISAR K=7 各经历约千级候选评价；最高格属于强标签驱动开发结果。
- MISAR K=7 的均值与中位数仍明显超过旧 support-only 参考，因此不是纯偶然峰值。
- P22 的均值 ARI 接近 N02 历史高位，但均值/中位数 NMI 尚未超过；P22 更接近 best-run-only signal。

## P22 的 K 协议

1. `P22_COSMOS_K9`：继续使用项目现有 `MouseBrain_groundtruth.csv` 与 K=9；这是当前直接可复算的主开发协议。
2. `P22_PAPER_K18`：SpatialGlue/3d-OT 等公开流程存在 K=18 协议，但必须取得其精确 h5ad、annotation、mask 与 observation ID 后独立运行。
3. K=9 与 K=18 是两个问题，不能互换标签或横向比较绝对 ARI。
4. 已知专家类别数用于固定 K 是允许的，但报告中必须写成“known-K unsupervised clustering”；标签可用于公开 benchmark HPO 与评价，不进入模型输入和无监督 loss。

## SEPAR 数据集审查

SEPAR 论文覆盖 DLPFC 12 切片、Stereo-seq 小鼠嗅球、MERFISH、osmFISH、MISAR-seq、Spatial-CITE-seq、DLPFC 多切片和结直肠癌 VisiumHD/Visium/Xenium。这里面只有 MISAR-seq 与 Spatial-CITE-seq 是与当前双模态核心直接同类的空间多组学；其余数据可用于验证聚类 head 的通用性，但不能冒充多模态主证据。

优先级：

1. 立即主线：P22 K=9、MISAR E15.5 K=7/K=12。
2. 立即补协议：P22 K=18 官方 artifact；SEPAR 使用的 MISAR exact processed artifact/label。
3. 已有多组学扩展：A1、D1、tonsil s1/s2/s3；作为统一模型的旁路验证，不要求本轮全部提升。
4. 新增可行性核对：SEPAR 的 Spatial-CITE-seq GSE213264，先与现有 tonsil 切片做 observation/file hash 去重。
5. 辅助而非主线：DLPFC、osmFISH、MERFISH、Stereo-seq、CRC；仅当统一 head 可直接复用且成本低时加入。

## 方法方向决策

### 停止作为论文核心

- TSPR：保留为负消融，不再投入主算力。
- 单纯 bilateral + PCA + GMM：作为有效工程骨架保留，但不单独声称原创模型。
- 单纯多 seed consensus/ensemble：可用来稳分，但已有大量先例，只是工程设施。

### 新工作核心：MCDF（暂定工作名）

MCDF 是 `Modality-Contribution-constrained Diffusion Fusion`，中文为“受模态贡献约束的多尺度扩散融合”。造这个词是为了指代一个尚待实验验证的真实模块：它同时学习几何、RNA、ATAC 和联合图的多尺度传播权重，并显式阻止纯坐标通道吞没分子证据。它与 Night-14B 的区别是：Night-14B 固定组合滤波与 head；MCDF 要让分子贡献成为可训练、可消融且必须独立成立的部分。

候选结构不是死命令，Codex 2 可在源码审查和真实 smoke 后提出更好结构：

- 四类稀疏专家：identity、geometry-only、RNA-guided、ATAC-guided/joint bilateral；
- 每类包含少量多尺度邻域/扩散深度；
- 内容门给出凸组合权重，同一代码结构服务 RNA+ATAC 数据；
- raw RNA 采用 HVG/PCA 或轻量 masked encoder，raw ATAC 采用 TF-IDF/LSI、gene activity 或可审计的稀疏编码；
- 模态 dropout consistency、masked reconstruction、anti-collapse 与 coordinate-dominance penalty 可作为候选训练目标；
- KMeans、GMM、Leiden 或 label-free medoid/consensus head 均可比较，head 稳定化不作为论文唯一创新点。

## Night-15A 的成功分级

不要求每个数据集、每个种子都提升；BEST_RUN、均值、中位数和完整失败分开报告。

1. `GEOMETRY_ONLY_SIGNAL`：coordinate-only 已解释主要高分，多模态贡献未成立。
2. `STABLE_HEAD_SIGNAL`：有效流水线在新种子上更稳定，但新模型无独立增益。
3. `MULTIMODAL_CONTRIBUTION_SIGNAL`：fused 明确优于 coordinate-only、RNA-only 和 ATAC-only，且至少两个未参与 HPO 的种子复现方向。
4. `RAW_FEATURE_BACKBONE_SIGNAL`：raw-feature backbone 对 MISAR K=12 或 P22 形成显著增益。
5. `METHOD_SIGNAL`：MCDF 在最强 preprocessing/head 对照之上有独立增益，并通过真实 checkpoint/reload。
6. `NO_ADDED_METHOD_SIGNAL`：工程正确但新模块没有增益。
7. 实现或设施失败按既有协议如实分类。

## 分数目标

- P22 K=9：保留 BEST_RUN ARI ≥0.55，并把 median/mean 推过 0.5063/0.6562。
- MISAR K=7：保留 BEST_RUN ≥0.50，优先把 median ARI 推到 ≥0.45。
- MISAR K=12：先过 0.50，再冲公开背景 0.644；不把不同 artifact/protocol 的数字写成公平胜负。
- P22 K=18：在 exact annotation 下建立第一版绝对 ARI/NMI，再追同协议高位。

## 最终决策

授权启动 Night-15A。资源分配建议：约 70% 用于 raw-feature backbone 与 MCDF，20% 用于稳定 head/seed 方差，10% 用于 exact artifact 与新增数据闭合。先做便宜的贡献对照，再决定大规模训练；不再把主要算力用于复现外部整表。
