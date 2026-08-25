# Night-17C：零起点可信关系精炼（工作名）

## 1. 这一步解决什么问题

Night-17B 证明候选关系在 P22 上能稳定改善表示，但随机起始的残差网络会破坏 human hippocampus、MISAR 和 melanoma 中已经较强的非训练关系平滑结果。Night-17C 不再扩大原 SFRD 网格，而是检验一个新的、可证伪的对象：**模型是否能从强关系平滑表示精确起步，只在候选关系足够可信的节点上学习有限残差，从而保留已有结构并在至少两个主要 RNA+ATAC 单元上产生独立表示增益。**

工作名不是论文最终名称。

## 2. 方法对象

核心仍为一套 RNA+ATAC 公式，不按数据集名称切换模型：

1. 从 Night-16H 结构可行候选得到稀疏 pair posterior；主教师使用 `UNBIASED_BANK`，完整候选库只作 benchmark-development sensitivity。
2. 先计算 Night-17B 已登记的 deterministic feasible-relation smooth 表示 `z_smooth`。
3. 编码器读取真实 `view1`、`view2`、原 retained carrier 与 `z_smooth`，输出受限 residual。
4. 最后一层零初始化，使 optimizer step 0 的输出与 `z_smooth` 数值一致；低可信节点的 residual gate 必须趋于 0，形成可审计 self-return。
5. pair loss 使用候选共聚类概率的软目标，并由候选分歧产生的 uncertainty 降权；保留 identity/smooth anchor、masked raw-view consistency 和最小 anti-collapse 约束。
6. 所有 pair 仅来自注册稀疏空间边和有上限的 feature-neighbour 边；不构造 dense N×N。

Codex 可根据真实梯度、尺度和数值稳定性决定精确实现，但不得用评价标签进入 input、loss、gradient 或 within-run checkpoint selection；不得用 dataset-name 分支替代统一公式。

## 3. 真实 P0 与对照

主要单元：P22 K=9、MISAR K=7、human hippocampus K=7。Melanoma K=2 只作低复杂度安全单元。

每个主要单元至少保留：

- frozen retained + 相同 endpoint；
- deterministic feasible-relation smooth + 相同 endpoint（最关键强参考）；
- zero-start full model；
- zero-residual control；
- permuted relation；
- unweighted relation；
- full-bank sensitivity；
- unbiased-bank primary；
- 两个 raw-view-adapter ablation，准确命名，不冒充真正单模态。

必须证明 step 0 与 `z_smooth` 一致、正式训练有非零梯度和参数变化、checkpoint strict load、fresh-process representation 与 partition replay。

## 4. 科学门

只比较相同 endpoint、相同候选预算。配置可以在公开 benchmark 上透明做 family-level HPO，但候选先锁定再评价。

主门：同一配置下，unbiased-bank full model 在至少 2/3 主要单元上同时严格超过 coordinate-wise strongest matched control（frozen retained、deterministic smooth、zero-residual），且 learned full 不得被 permuted relation 同时超过。未满足则停止，不补 seed、不在原结果上改门。

若主门通过：

1. 再运行训练 seeds 0–2；
2. 把新表示接入与 Night-16H 相同预算的结构可行 energy/head，比较是否刷新至少两个单元的绝对分数前沿；
3. melanoma 仅检查安全性，不计入 2/3 主门。

若主门失败：终态为 `SCIENTIFIC_NEGATIVE`，保留最强局部信号，下一轮离开 residual-distillation 家族。

## 5. 资源和交付

- 复用现有环境、carrier、candidate bank；不下载数据、不复制 raw、不新建环境。
- 系统盘接近 91%，只保留代码、表格、checkpoint 和必要小型表示；先估算再写。
- 先完成单 seed P0，严格门通过才多 seed。
- 服务器保持开机，当前不是最终项目里程碑时不要派发 shutdown。
- 报告正文先写问题、做了什么、论文意义和明确结果分类，再写哈希与 Git。
