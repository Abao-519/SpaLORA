# Night-17C P0 报告

## 我现在需要知道的三件事

1. **问题**：候选关系蒸馏能否不再从随机残差破坏强表示，而是从确定性 relation-smooth 表示精确零起步，只让可信节点学习小残差？
2. **实际动作/层级**：本轮修改的是表示层，不是标签评价器或简单聚类 head。统一编码器在 P22、MISAR、人海马上真实训练 40 steps，公开标签只在 partition 锁定后用于 family-level benchmark HPO。
3. **论文意义**：种子 0 的严格门为 3/3，多种子在 P22/MISAR 稳定；同预算结构 head 也在 3/3 超过 matched zero/permuted carrier。但没有在至少两个数据集刷新 Night-16H 的绝对双指标前沿，人海马多 seed 仍不稳，melanoma residual 反而略坏，所以分类只能是 **LOCAL SIGNAL**。

## 绝对指标（Z01，同一 endpoint）

| 数据 | 最强 matched control ARI/NMI | seed0 full | ΔARI/ΔNMI | seeds 0-2 mean | seeds 0-2 min | dual-win seeds |
|---|---:|---:|---:|---:|---:|---:|
| P22 K9 | 0.393901 / 0.584713 | 0.481650 / 0.610499 | +0.087749 / +0.025787 | 0.480888 / 0.609897 | 0.480492 / 0.609503 | 3/3 |
| MISAR K7 | 0.361113 / 0.538358 | 0.363714 / 0.540974 | +0.002601 / +0.002616 | 0.363704 / 0.541180 | 0.363653 / 0.540974 | 3/3 |
| Human hippocampus K7 | 0.195737 / 0.278234 | 0.209619 / 0.279205 | +0.013883 / +0.000971 | 0.199789 / 0.270691 | 0.193964 / 0.260301 | 1/3 |

MISAR 的关系候选高度一致，IQR 退化后 gate 使用原始可信度；因此它没有 zero-gate 节点（zero fraction 0），但仍通过 smooth anchor 限制残差。这个事实不隐藏，也不把它包装成普遍的节点拒绝证据。

## Night-16H 同预算结构 head 归因

| 数据 | learned head | zero-residual head | permuted head | learned 相对最强 matched Δ | Night-16H 主结果 |
|---|---:|---:|---:|---:|---:|
| P22 K9 | 0.425892 / 0.625361 | 0.425646 / 0.625077 | 0.424941 / 0.623601 | +0.000246 / +0.000285 | 0.587533 / 0.708974 |
| MISAR K7 | 0.366453 / 0.554718 | 0.364301 / 0.551403 | 0.365054 / 0.552748 | +0.001399 / +0.001970 | 0.534624 / 0.656772 |
| Human hippocampus K7 | 0.633605 / 0.578094 | 0.547099 / 0.555968 | 0.556763 / 0.559197 | +0.076842 / +0.018897 | 0.596178 / 0.585490 |

这里的 learned/zero/permuted 三条都使用 89 个候选、同一稀疏结构可行域和同一固定 selector。人海马 ARI 刷新，但 NMI 回撤；P22/MISAR 没追上 Night-16H 的强历史起点，因此不能称 score-frontier milestone。

## 安全性、限制与失败

- Melanoma K2：smooth 0.980593/0.953369，Z01 full 0.975775/0.944451，permuted 0.980585/0.952323；训练 residual 没有独立收益。
- 人海马的多 seed 仅 1/3 双胜，说明节点 gate/小残差还没有稳定解决该数据的优化方差。
- `ONE_RAW_VIEW*_ADAPTER` 只关闭一个 raw adapter；retained carrier 仍可能含双模态信息，不能称真正单模态。
- full-bank 只作敏感性；headline teacher 是剔除历史 authority 的 `UNBIASED_BANK`。
- 本轮是公开 benchmark development，不是盲测、SOTA、confirmed milestone 或 paper-ready evidence。

## 导师汇报版

1. 我们把失败的随机残差改成了从强 relation-smooth 表示精确零起步的统一模型。
2. 模型只在候选关系可信的节点上学习，低可信节点保持原表示。
3. seed0 在 P22、MISAR、人海马 3/3 都超过 frozen/smooth/zero-residual 强参考，并且没有被置换关系双指标压过。
4. P22、MISAR 三个训练 seed 都稳定；人海马只有一个 seed 双胜，稳定性仍不足。
5. 同样 89-candidate 结构 head 下，learned carrier 在三条数据都超过 zero/permuted matched carrier，说明不是纯 head 假象。
6. 但 P22/MISAR 没追上 Night-16H 的历史强前沿，人海马也只是 ARI 上升、NMI 有取舍。
7. melanoma 上 full 还略差于 smooth/permuted，因此当前只能定为局部方法信号。
8. 下一步若继续，应针对人海马优化方差和跨起点绝对前沿，而不是扩大同一超参网格。

## 技术说明

完整表、公式冻结、重放、资源、label-flow 和修正 ledger 与本报告同目录。服务器按连续自主任务要求保持开机，`shutdown_dispatched=false`。
