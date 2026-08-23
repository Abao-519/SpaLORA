# Night-13B Worker 2 独立证据重分级

日期：2026-08-23  
审查对象：`night13b_delivery_20260823/official_compact`

## 一、终结论

Night-13B 的 `LOCAL SIGNAL` 分类可以保留，但准确表述应收窄为：

`DETERMINISTIC_COMMON_ENDPOINT_REPRESENTATION_LOCAL_SIGNAL`

它不是实现失败，也不是已确认的统一模型里程碑。A1 与 P22 的 common-KMeans 表示层提分是真实数值信号；但“5/5 跨 seed 稳定”“15 种机制搜索”“已经训练好的统一模型”均不成立。

## 二、交付复核

- compact：35 个 indexed payload，目录实际 36 个文件（含 index）；missing、size、SHA、extras、forbidden 全为 0。
- compact index SHA-256：`8c93bef89ae5c03073b53c7ba095a4e32880d7adc6ac409ef6fb0e5b41abe45e`。
- bundle SHA-256：`630a593e4e2cdf8cb7628540aff5ff07c82f362ba843ae3aca4886dc704fef9d`。
- 35 行 formal 数值均 finite，mask、ordered observation、逐行算术和原报告的分阶段聚合均闭合。
- tonsil 表示是在完整配对 observations 上建立，只有 evaluator 使用标签 mask；此处没有再现 Night-13A 的 mask 错误。

本地 compact 是增量 bundle，缺少完整 parent 链，因此本机只独立验证了 bundle hash 与唯一 annotated-tag head；远端 peel 结论来自已交付审计，不冒充本地完整 peel。

## 三、B10 实际是什么

真实数据流为：

`两个模态分别预处理 -> 标准化拼接 -> PCA -> B10 多尺度空间图 residual -> 固定 seed-0 KMeans`

B10 只接收一个已经融合好的 embedding 与两个稀疏空间图。它没有直接接收 RNA 与第二模态张量，也没有 optimizer 或参数更新；唯一 backward 只是 P0 梯度探针。正式运行中 `threshold_offset=0`，所以有效学习参数为 0。

公式为：

`g = sigmoid(s * (mean[1-cos(z,P4z)] - t))`

`beta = m * g`

`z_out = (1-beta)z + beta[(1-g)P4z + gP18z]`

其中 gate 是每个数据集一个全局标量。它没有按数据集名称换模型，但会根据整套数据内容连续决定空间平滑强度。因此，它是统一的软路由后处理层，不是已经训练的端到端统一融合模型。

## 四、五个 seed 的问题

`run_model(..., seed)` 接收 seed 但没有使用；KMeans 始终为 `random_state=0`。每个数据集五行的 embedding SHA、partition SHA、ARI 和 NMI 完全相同。

因此：

- 35/35 作为工程执行行数成立；
- 科学上只有 7 个确定性结果，而不是 35 个独立结果；
- SD=0、5/5 双胜和跨 seed 稳定不能作为证据；
- 下一版确定性表示只登记一次，聚类不确定性应由多个 endpoint seeds、consensus、空间块 bootstrap 或图扰动评价。

## 五、真实数值边界

### Common endpoint 表示层

| 数据集 | B10 ARI/NMI | 同口径参考 | 结论 |
|---|---:|---:|---|
| A1 | .2693/.3817 | C00 common 均值 .2305/.3760 | 双指标局部提升 |
| D1 | .2508/.3643 | simple .2428/.3630 | 小幅提升 |
| P22 | .4619/.6042 | N02 common 均值 .4203/.5771 | 双指标局部提升 |
| tonsil | 基本等于 simple，s2 相对 RNA-only 混合 | 数据集最强内部参考 | 未形成稳定提升 |
| MISAR | .1592/.2966 | simple .2021/.3562 | 明显下降 |

### Native 完整管线高水位

| 数据集 | B10 common | 历史 native 强线 | 结论 |
|---|---:|---:|---|
| A1 | .2693/.3817 | C00/H05 .2692/.4087 | ARI近似持平，NMI较低 |
| D1 | .2508/.3643 | C00/H05 .2412/.3777 | 混合 |
| P22 | .4619/.6042 | F00 .4677/.6334；N02 .5063/.6562 | 未超过完整管线高水位 |

所以，B10 证明的是固定 common endpoint 下的表示层信号，不是完整方法已经超过历史强线。

按真实研究单位重聚合后：protein 的两个 study block 约为 `ΔARI +.0120 / ΔNMI +.0030`；ATAC 的 P22 与 MISAR 等权约为 `ΔARI -.0007 / ΔNMI -.0163`。统一 ATAC 家族尚未改善。

## 六、为什么 MISAR 不能只靠继续调 threshold 修复

MISAR 的 beta 只有约 .0369，主要是弱 P4 传播；但 ARI、NMI、Moran 同时下降，Geary 同时变差。A1 与 MISAR 的全局 discrepancy 又很接近，效用方向却相反，说明单一全局均值不是可靠的传播风险量。

搜索表还显示很小的 beta 变化就能让 tonsil 与 A1 的 KMeans partition 大幅跳变。因此目前无法区分“表示真实变好”和“微扰碰巧把固定 KMeans 推入另一个局部盆地”。先做 endpoint 稳健性审计比继续调全局 gate 更重要。

## 七、原创性判断

B10 的精确公式未发现逐字相同方法，但概念空间已经拥挤：近期方法已经覆盖双层图注意力、多尺度/高阶图、多级融合、图对比、共享/私有分解和自适应聚合。例子包括 [MultiGATE](https://www.nature.com/articles/s41467-025-63418-x)、[SpaFusion](https://www.sciencedirect.com/science/article/abs/pii/S1566253525004452)、[SpaMV](https://www.nature.com/articles/s41467-026-74718-1)、[SMART](https://www.nature.com/articles/s41467-026-70821-5) 和 [CoMo](https://pubmed.ncbi.nlm.nih.gov/42001472/)。

因此，“根据内容自适应选择平滑强度”不足以单独支撑二区论文。B10 最多保留为 graph-residual head、消融或诊断工具。

仍值得验证的窄问题是：能否用两个模态对每条空间边的共同支持度，连续抑制高风险传播，同时保留 identity residual。它与全局 gate 不同，也比按 RNA 单边锚定更对称。但该对象必须先作为候选接受碰撞审查和真实数据检验，不能预先宣布为原创点。

## 八、决策

授权 Night-13C，采用两段式、可提前停止的路线：

1. 用缓存 embedding 低成本复核 B10 的 endpoint、图与 beta 稳健性，并把同一 head 施加到 C00/F00/N02 强表示；
2. 若信号稳健，再把 B10 降格为辅助层，集中实现真正接收两个模态、optimizer steps 大于 0、具有真实训练 seed 的统一 core；若信号不稳健，立即结束 B10 主线，不再围绕 threshold 做网格搜索。

Night-13C 不做完整外部方法复现，不新增数据下载。当前首先解决自己的方法语义与性能。
