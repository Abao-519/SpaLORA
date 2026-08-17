# SpaLORA：Night-6A 后研发决策

日期：2026-08-17  
决策状态：`PROCEED_TO_NIGHT6B_GRAPH_AND_CLUSTERING_RESCUE`

## 1. 当前项目状态

- Night-3B 方法级结论仍为 `MIXED_EVIDENCE`。
- Night-5 的 C04/B01 在 A1 与 placenta 上有可复现的开发提升，但 P22 仅为 `P22_PARTIAL_OR_MIXED_EVIDENCE`，没有确认性显著优势。
- Night-6A 因预锁定 `obs` 反序列化而终止为 `IMPLEMENTATION_SEMANTICS_INVALID`；其训练不得作为有效科研证据。
- 当前最需要提升的是 A1 与未来 P22/D1 上的 ARI/NMI；placenta 只保留为补充证据，不再主导研发方向。
- 现阶段不做正式 benchmark、不做投稿图、不包装论文结论。

## 2. 为什么下一轮不再优先改 loss

Night-5/6 已覆盖：attention shrink、RNA anchor、MNN triplet、DGI、diffusion、Laplacian、PCGrad、MinNorm、Barlow、neighbor InfoNCE 和多种组合。它们证明部分路线有局部收益，但没有形成跨重要数据集的稳定确认性优势。

Night-6A 的两步语义探针还显示，C04/B01 当前训练标量主要由 RNA reconstruction 贡献。继续围绕 loss 权重做小变体，既容易重复已有搜索，也未必触及最终聚类性能瓶颈。

下一轮把 encoder 固定在当前最可靠的 C04/B01 语义上，把研发重心移到：

- 空间图是否过宽；
- feature graph 是否过宽或度量不合适；
- 空间边与 feature 边是否应取交集；
- fused embedding 是否丢失模态互补信息；
- 固定 EEE mclust 是否限制了最终分区；
- 是否能用完全 label-free 的 sparse affinity / view consensus 改善分区。

## 3. 源码层面的主要启发

### SpaBalance

论文与源码：

- https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202512973
- https://github.com/nudt-bioinfo/SpaBalance

源码事实：

- 10x 数据的 `construct_neighbor_graph` 默认 `n_neighbors=3`；
- feature graph 默认 k=20、correlation；
- 使用 shared/private 表示与梯度协调；
- 官方 tutorial 随机 seed 和部分复现实务并不够严格，不能把其公开数值直接当成我们的公平基准。

可借鉴：更局部的空间图、保存 private/shared 表示。  
不直接照搬：其随机 seed、dense adjacency 和未经统一的 benchmark 设定。

### SpaMosaic

官方 tonsil 教程：

- https://spamosaic.readthedocs.io/en/latest/tutorials/integration/vertical/Tonsil_vertical.html
- https://github.com/JinmiaoChenLab/SpaMosaic

源码/教程事实：

- tonsil section 1 使用 10 个空间邻居；
- vertical integration 保存 modality-specific embeddings；
- 最终表示可由模态表示平均或拼接；
- 教程以 mclust 做空间域识别，并对同名数据使用 `n_cluster=6`。

可借鉴：多 view 输出、平均/拼接对照、k=10 图尺度。  
待审计：为什么其 K=6 与其他文献的 4/7 口径不同。

### SpaMICS

论文与源码：

- https://www.sciencedirect.com/science/article/pii/S1566253525005019
- https://github.com/SZU-CGC/SpaMICS

源码事实：

- feature k=10、spatial k=10；
- 用 feature graph 与 spatial graph 的支持交集细化空间图；
- 学习 shared low-rank 与 modality-private affinity，最后做 spectral clustering；
- 代码读取真实标签以决定 K，并在训练循环中定期评价；还包含 N×N 可学习矩阵。

可借鉴：feature-spatial intersection、shared/private sparse affinity、spectral cluster head。  
不直接照搬：标签参与、训练期评价、N×N dense 参数和论文特定超参数。

### SMODEL

论文与源码：

- https://www.nature.com/articles/s42003-025-08372-6
- https://github.com/liying-1028/SMODEL

源码事实：

- 输入多个 base partitions，构造 co-association/ensemble 信息；
- 联合分子与空间图；
- A1 脚本直接使用真实 `Y` 决定 K，且有数据特定常数和 MATLAB 依赖。

可借鉴：冻结多个 view/seed 后做 label-free co-association。  
不直接照搬：任何 ground-truth 依赖或数据集特定调参。

## 4. Night-6B 的核心假设

### H1：当前 k=18 对 10x A1/tonsil 过宽

若 H1 成立，k=3/6/10 中至少一个全局规则应在 A1+tonsil 上提升 Q，并减少过度空间平滑。

### H2：feature-spatial intersection 比粗暴空间剪枝更有针对性

Night-6A 是按坐标/权重直接剪空间边；SpaMICS 的源码做法是只保留同时得到分子邻域支持的空间边。二者不是同一机制，不能用 Night-6A 的负诊断否定 H2。

### H3：最终 clustering head 是被忽略的性能瓶颈

当前只使用 fused embedding + PCA20 + EEE mclust。若 H3 成立，concat、稀疏 affinity、view consensus 或很轻的空间正则应在不重训 encoder 的情况下提升 Q。

### H4：正确的 tonsil ontology/K 会显著改变可解释性和基线难度

本假设必须先通过 data-steward contract 验证；在真实唯一值审计前，不预言 K 或指标方向。

## 5. 决策边界

Night-6B 允许大胆搜索，但必须遵守：

- 候选是预先注册的全局规则，不为 A1/tonsil 分别定制；
- A1+tonsil 是开发集；D1/P22 是完全封存的未来确认集；
- known-K 是公开 benchmark protocol 输入，必须由独立 ontology contract 固化并与训练进程隔离；
- 保存 balanced frontier 与 accuracy frontier，不能因空间指标不完美而抹去高准确率候选；
- 同时也不能只看 ARI/NMI 后悄悄隐藏严重空间破坏；
- 任何实现错误影响的结果一律 invalid，不得当作候选失败；
- 不以单 seed、最佳 seed 或中间 epoch 选候选；
- 不复用 Night-6A invalid checkpoint 作为正式证据。

## 6. 预期成功标准

Night-6B 不是保证成功，而是一次高信息密度的研发轮。

最低有用结果：

- 彻底解决 tonsil ontology/K 与标签隔离；
- 明确 k=18 是否是瓶颈；
- 明确多 view/affinity head 是否优于固定 fused mclust；
- 得到完整负结果也能关闭一大类路线。

候选冻结的目标：

- balanced candidate：A1+tonsil 五 seed macro ΔQ 至少 +0.020、两个数据集均不退化、空间门通过；
- accuracy-frontier candidate：macro ΔQ 至少 +0.030、最差数据集不超过轻微退化；即使空间门失败也保留，但必须单独标记。

只有冻结后的候选，下一轮才允许进入 D1/P22。

