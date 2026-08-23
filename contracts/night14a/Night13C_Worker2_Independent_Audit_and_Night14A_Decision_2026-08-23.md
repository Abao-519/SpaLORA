# Night-13C 独立复核与 Night-14A 决策

日期：2026-08-23  
角色：Worker 2（规划与科学决策）

## 我现在需要知道的三件事

1. Night-13C 真正检验的是一组被简单拼接 PCA 强力锚定的浅层原型，不是成熟统一模型的上限。
2. 59/59 compact、报告、决策、核心源码和训练 runner 已独立复核；本轮科学负结论对 E10/X10/S10/E11/E12 这些具体实现成立。
3. 下一步停止调 B10 和浅层 simple-anchor 路线，转为“成熟统一 backbone + 跨模态拓扑冲突滤波”的性能开发冲刺；不再要求所有数据集同时上涨。

## 1. 权威输入与复核结果

- Night-13C compact：`night13c_delivery_20260823/official_compact`
- compact index SHA-256：`bfdd7996fe742f19e1f9cd9782e85e5fb082d40409507c449cac47587cae189a`
- bundle SHA-256：`ac7b69ccef576b56650ce4185b947365fb4fff551ed93ff2ae3bb3c8526fa46d`
- Windows 独立复算：59 个 indexed payload 全部 size+SHA 通过；0 missing、0 mismatch、0 extras。

关键文件独立 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `night13c_report.md` | `dc8fb13023d29bf53be484d5d3282ede426f71542fee6e0f108562cd26b9ff8e` |
| `night13c_decision.json` | `140fbf4d887aee6a3a76584444359c658eba1add5755f7d6f2cf8b51a76a9159` |
| `night13c_candidate_summary.csv` | `6076c4fd11bb05d8b8323d7d56fba9b2ca16704ed06636bf8f9e0bf6b01646dc` |
| `night13c_core.py` | `721df9f3218ef74d076d2d2ddbe9b8b50ed57ac1098cee0426e72a990b6fe0b` |
| `night13c_stage_b.py` | `58de699fb59b0cc24c89dd43a49b9da323112f624f105f50f076e6b5380b8b01` |

## 2. Night-13C 负结果应怎样解释

终态 `NIGHT13C_B10_ENDPOINT_FRAGILE` 和分类 `SCIENTIFIC_NEGATIVE` 可以保留，但适用范围必须收窄为：

> 在 Night-13C 的弱 simple-anchor、浅层网络、短训练和强回拉协议下，五个具体可训练原型没有超过强参考。

它不能外推为“统一可训练空间多组学方法不可行”。理由如下：

1. 原始 RNA/ADT 或 RNA/ATAC 已先被压为 30+30 或 30+50 维。
2. backbone 只有双线性 adapter、一个 tied Linear–GELU–LayerNorm core 和线性 decoder，总参数约 1.2 万到 2.8 万。
3. 训练只有 80 步；E11/E12 也只有 160 步，且 cross-reconstruction 多数仍接近零预测水平。
4. 训练 loss 直接要求 learned embedding 贴近 simple-concat PCA 坐标。
5. 推理时 learned embedding 再对齐回 simple anchor，最终输出中 70%–95% 仍是弱 anchor。
6. 这种候选却直接与 C00/N02 等历史强表示比较，尤其在 P22 上起点不对等。
7. E10 的 edge target 来自同一弱表示的平均 cosine，相当于自我回归，无法识别“传播这条边是否真正有益”；其 edge reliability 几乎退化为常数。
8. E11/E12 不是新机制，只是从分别重训的 checkpoint 使用不同 residual，且混入 GPU 非确定性，不能作为干净消融。

## 3. 一个被主报告低估的重要信号

P18 是“较宽空间邻域的稀疏传播算子”。P18-only 在相同 endpoint 分布下显示：

| 数据集 | Identity ARI/NMI | P18-only ARI/NMI | 解释 |
|---|---:|---:|---|
| P22 | 0.2798 / 0.4266 | 0.4786 / 0.6356 | 强正向 |
| MISAR E15.5 | 0.1582 / 0.3096 | 0.3420 / 0.5411 | 强正向 |
| A1 | 0.2276 / 0.3517 | 0.1962 / 0.3373 | 负向 |
| D1 | 0.2398 / 0.3558 | 0.1787 / 0.3289 | 负向 |
| tonsil s1 | 0.1564 / 0.2749 | 0.0570 / 0.1514 | 强负向 |

这说明“空间传播有没有用”不是全局常数：RNA+ATAC 当前表示明显受益于宽尺度低通，而 RNA+protein 的组织边界被同一操作破坏。B10 只用一个切片级标量，无法区分切片内部的安全边、冲突边和边界边。

这不是重新回到两个模型的分流器。下一步要做的是一套相同的图频率处理结构：每个位置都拥有 identity（不传播）、low-pass（向可靠邻居聚合）和必要时的 high-pass（保留/强调局部差异）通道；通道权重由真实模态内容、空间关系和局部冲突产生，而不是由 `dataset == P22` 之类的名称判断产生。

## 4. 文献碰撞与可迁移经验

- SMART 已经证明 GraphSAGE、重构和 MNN metric learning 能作为较成熟且可扩展的统一骨架，并明确允许逐数据集调整邻居数和 triplet-loss 权重。
- SpaBalance 处理的是训练阶段的跨组学梯度冲突，并采用 shared/private 与注意力；它没有直接解决“某条空间边是否适合传播”的拓扑冲突。
- ACM-GNN 在一般图学习中按节点混合 aggregation、diversification 和 identity 通道，说明低通/高通/保真联合建模是成熟可迁移思想。
- Deep Safe Multi-View Clustering 说明增加视图可能降低聚类表现，并提出相对单视图的安全学习问题。
- SpatialSyn 已覆盖 static/dynamic graph、边界过平滑和 context-aware reweighting，所以“自适应图”本身不够新。我们的区别必须具体落在：跨模态对称、边级连续冲突证据、identity safety residual、稀疏图频率通道，以及对负迁移的显式实证。

因此，工作代号定为 **TCF（Topology-Conflict Filter，拓扑冲突滤波器）**。造这个词只是为了给真实代码对象一个稳定代号，不是最终论文命名。它对应的是“由两个模态对每条空间边的支持与冲突，控制 identity/低通/高通响应的稀疏模块”；它不同于 B10 的切片级单标量，也不同于按数据集名称选择模型。

## 5. Night-14A 科学决策

1. 停止：B10 threshold/slope 微调、S10 shared/private、E11/E12 residual 邻域、强 simple-anchor MSE、cosine-average 自回归 edge target。
2. 保留：真实双家族 runner、稀疏 edge-list、checkpoint/fresh-process、C00/F00/N02 强工件、X10 modality-dropout 作为辅助训练、paired endpoint 评价设施。
3. 起点：先从 SMART 与 SpaBalance 中选一个适合当前资产、许可证可交付且真实 P0 性能较强的成熟统一 backbone；也允许基于官方源码语义做可追溯 clean-room 实现。
4. 新增：TCF 作为唯一主要新模块；exact formula、normalization、loss 和小型参数搜索由执行 Codex 根据源码与真实表现决定，不预先把它写死。
5. 评价：公开标签可用于开发评价和透明的数值超参数选择，但若论文声称无监督，标签不得进入模型 loss、gradient 或单次训练 checkpoint selection。
6. 统一性的含义：结构、信息流和目标统一；RNA/ADT 与 RNA/ATAC 的输入 adapter/preprocessing 可以不同，且各数据集可有透明的数值超参数。禁止的是 dataset-name-driven backbone switching，不是正常调参。
7. 晋级不再要求七个数据集全部双升。按研究单位平衡的总体收益、强参考差值、端点稳健性和最坏回撤共同判断；若主要证据只在 RNA+ATAC 成立，论文范围可以诚实收窄，而不是判整个项目死亡。

## 6. 对论文的当前判断

当前仍不是论文级证据，但已经从连续负结果中得到一个可检验、范围适中的原创问题：

> 空间多组学模型不应默认每条空间边都能安全传播；跨模态局部一致与冲突应共同决定图频率响应，从而在利用空间连续性的同时保护组织边界并降低负迁移。

这比“根据模态家族选两套模型”更像方法论文：它是一套可复用模块、一个明确失效机制和一个可做消融的因果链。Q2 发表的可行性取决于 Night-14A 能否在强 backbone 上获得稳健数值、跨端点复现和至少一项真实机制证据，当前不预先承诺发表等级。

