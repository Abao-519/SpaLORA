# SpaLORA 指标驱动研发依据（2026-08-13）

## 结论先行

Night-3B 与 Night-4A 之后，项目应暂停投稿图、论文包装和正式大规模 benchmark，进入新的 metric-driven R&D cycle。目标不是让某一个已看过的数据集出现漂亮单点，而是在统一、标签隔离、多 seed 协议下找到可泛化候选。

当前最有信息量的研发方向依次为：

1. 以等权为先验的受限融合，以及基于局部可预测性的无监督可靠性权重；
2. RNA 特征图与空间图交集形成的高置信共识边，并以残差方式共享给两模态；
3. 在训练前冻结的 MNN 正样本/远距离负样本上施加 fused-embedding triplet loss；
4. label-free neighbor-aware contrastive；
5. 两层残差非线性稀疏图编码器或轻量 DGI，作为竞争机制而非预先叠加。

## 现有证据为什么要求更换研发母体

Night-3AF 的每数据集最优机制不同：

| 数据集 | 最优既有变体 | ARI | NMI |
|---|---|---:|---:|
| A1 | IGE | 0.2469 | 0.3725 |
| Placenta | C1 | 0.6825 | 0.7337 |
| P22 | ILN | 0.4635 | 0.5864 |

Night-3B 又显示：

- Corr1 是唯一跨数据集明确支持的原组件；
- 删除 Corr2 后，A1、Placenta、P22 的 ARI 均高于相应 FULL_IGE；
- Placenta 的 `UNIFORM_WITHIN` 为 ARI 0.7506 / NMI 0.7662；
- P22 的 `UNIFORM_CROSS` 为 ARI 0.4565 / NMI 0.5886，五个 seed 全胜 FULL_IGE；
- P22 learned cross-attention 的 spot-level 极端权重比例约 94.9%，但极端门控没有转化为更高 ARI/NMI；
- IGE 只在 step 0 进行一次梯度均衡，训练后半程的实际梯度份额会重新漂移。

因此，`FULL_IGE` 继续作为历史参考，但不再被假定为研发母体的正确答案。

## 当前代码瓶颈

### 1. 表示能力不足

`night4a_local_repo/SpaLORA/model_corrected.py` 中的 `GraphLinear` 只有 `A @ X @ W`。没有激活、归一化、bias 或残差，模型的大部分非线性只来自注意力打分中的 `tanh`。

### 2. 注意力缺少可靠性约束

当前融合只对两个 score 做 softmax，没有温度、均匀先验、熵约束或模态质量信息。这与 Night-3B 的 attention 近饱和且功能失配结果一致。

### 3. Corr2 结构缺少实证支持

Corr1/Corr2 实现为跨 decoder/encoder 的循环 latent MSE，而不是同一 spot 两模态的直接 shared-latent 对齐。Corr2 删除在三个数据集的 ARI 上都没有造成损失，应先移除或由直接 label-free alignment 替代。

### 4. 图构建制度差异过大

当前二值 KNN 的空间邻居数为 A1=18、Placenta=1、P22=16。同一注意力模块实际接收的是连通性完全不同的图。下一轮优先测试“不删除原空间图、只对 RNA-feature 与 spatial 的交集边加权”的保守共识图，避免图交集导致断连。

### 5. evaluator 是次级变量

固定 `L2 normalize -> PCA20 -> mclust EEE` 可能影响绝对数值，但不能把调聚类器当作方法创新。研发阶段主表继续用锁定 evaluator；可以对历史 embedding 做 label-free 的 evaluator 敏感性诊断，但不得只报告更有利的聚类器。

## 一手论文与官方源码审查

### SMART

- 论文：Nature Communications 2026，https://www.nature.com/articles/s41467-026-70821-5
- 官方源码：https://github.com/Xubin-s-Lab/SMART-main
- 核心：模态独立 GraphSAGE encoder/decoder；PCA 空间 MNN 正样本与远距离负样本；triplet margin loss。
- 本地源码快照：`method_source_review_20260813/SMART_src`
- 许可证：GPL-3.0。不得把源码直接复制进不明确兼容的工程；本项目只依据论文独立实现算法思想并引用。

### ARISE

- 论文：Bioinformatics 2026，https://academic.oup.com/bioinformatics/article/42/7/btag465/8721301
- 官方源码：https://github.com/XiangxiangWang-code/ARISE
- 核心：RNA feature graph 与 spatial graph 的交集作为高置信共享拓扑，特别针对稀疏辅助模态。
- 本地源码快照：`method_source_review_20260813/ARISE_src`
- 仓库未发现明确 LICENSE。只允许 clean-room 实现论文思想，不复制源码。

### COSMOS

- 论文：Nature Communications 2025，https://www.nature.com/articles/s41467-024-55204-y
- 官方源码：https://github.com/Lin-Xu-lab/COSMOS
- 核心：WNN/局部可预测性模态权重、DGI、空间正则；论文明确显示空间正则过强会损害 ARI。
- 本地源码快照：`method_source_review_20260813/COSMOS_src`
- 许可证：MIT，但仓库包含对 pyWNN/PyG 等工作的适配声明。若复制实现必须保留许可证与声明；本轮优先独立实现公式并引用。

### SpaMFG

- 论文：Bioinformatics 2026，https://academic.oup.com/bioinformatics/article/42/7/btag457/8722297
- 官方源码：https://github.com/LiangYu-Xidian/SpaMFG
- 核心：空间表达模式驱动的特征分组、跨模态组匹配、MOFA 融合。
- 本地源码快照：`method_source_review_20260813/SpaMFG_src`
- 仓库未发现明确 LICENSE；模块复杂、特征平方级操作较多、组件消融不足，不进入第一轮。

### SpatialCOC

- 论文：Nature Communications 2026，https://www.nature.com/articles/s41467-026-71882-2
- 官方源码：https://github.com/xjtu-omics/SpatialCOC
- 核心：连续坐标 INR 与 CCA-style cross-omics correction。
- 本地源码快照：`method_source_review_20260813/SpatialCOC_src`
- 许可证：GPL-3.0。真实 A1 上的模块级直接证据不如前三项，不进入第一轮核心候选。

### SMODEL、soFusion、PRESENT、CoMo

- SMODEL 更接近多基础聚类的后处理集成，不优先用于改进表示。
- soFusion/PRESENT 的概率解码器要求重新回到原始 counts 与分布假设，第一轮改动过大。
- CoMo 与 SpaLORA 骨架相近；只借鉴 label-free neighbor-aware contrastive，不采用依赖伪聚类类别数的 cluster-aware 选择。

## 公开论文数值只能作为方向标尺

| 数据集 | 方法 | ARI | NMI | 限定 |
|---|---|---:|---:|---|
| A1 | FULL_IGE | 0.2469 | 0.3725 | 当前锁定五 seed |
| A1 | SpatialGlue（SpaMFG 表） | 0.297 | 0.393 | 单论文协议 |
| A1 | SpaMFG | 0.316 | 0.420 | 单论文协议 |
| A1 | ARISE | 0.3427 | 0.4182 | 单论文协议 |
| P22 | FULL_IGE | 0.3952 | 0.5531 | 当前锁定五 seed |
| P22 | COSMOS | 0.63 | 未给精确值 | 固定 seed/论文协议 |
| P22 | SpatialGlue（COSMOS 表） | 0.43 | 未给精确值 | 单论文协议 |

预处理、K、聚类器、seed 与标签合并差异使这些数值不能构成公平排名。新候选必须先在本项目锁定协议内证明稳定增益，之后才值得投入现代 baseline 复现。

## 数据分层

- A1、Placenta：开发集，允许进行候选筛选；今后不再承担独立泛化证明。
- P22：锁定内部鲁棒性门；Night-5A 不运行新候选 P22，只选出最多两个候选后停下交接。
- D1 lymph node：最终 within-study accuracy confirmation；在候选和现代 baseline 完全冻结前不得打开结果。
- GSE198353 rep1/rep2：独立 label-free 层；当前条码、矩阵与坐标尚未建立无歧义 model-ready cache。
- A1/D1 tonsil：无可审计语义标签，仅 label-free。
- GSE213264：缺可复现 paired spatial coordinates，当前排除。

## 学术与软件边界

允许：阅读论文和官方代码、独立实现思想、如实引用、在开发集进行预注册网格搜索、组合多个有理论联系的模块。

禁止：复制无许可证源码、隐藏失败、筛 seed、用 P22/D1 标签反复调参、把公开论文单点当作自己的公平胜利、把开发集 bootstrap 当确认性统计。

这种边界不是保守包装，而是让数值提升在投稿时真正经得起复现和反驳。
