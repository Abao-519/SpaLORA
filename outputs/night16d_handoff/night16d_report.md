# SpaLORA Night-16D report

## 我现在需要知道的三件事

1. 本轮把 Night-16C 的三态边从“后处理权重”改成了三种真正不同的可训练表示运算：域内支持边做低通，共同边界做高通/分离，模态冲突边保留各自私有信息。
2. 工程链路成立：RNA+protein 与 RNA+chromatin 使用同一模型、同一损失、同一 endpoint；A1 与 P22 两条真实 GPU 路径均完成真实参数更新、checkpoint 严格加载和 fresh-process 回放。原始强起点、same-head teacher 和 generic residual 已明确分开。
3. 科学结果是 **SCIENTIFIC_NEGATIVE**。合成 teacher、激进搜索和 exact retained-embedding 插件筛选都没有在两个真实 study 上给出 ARI/NMI 双升；因此按任务书提前停止，没有冻结家族配置，也没有把起点或 head 的分数包装成新表示方法成功。

## 绝对指标与独立贡献

| 数据集 | 输入强起点 ARI/NMI | same-head teacher | generic residual | 最佳 full（平衡规则） | 结论 |
|---|---:|---:|---:|---:|---|
| A1 | 0.276004/0.421690 | 0.267086/0.409567 | 0.264431/0.408595 | 0.276119/0.421587 | 无双指标独立增益 |
| tonsil_s1 | 0.236683/0.317365 | 0.217995/0.277958 | 0.220969/0.281350 | 0.236683/0.317365 | 无双指标独立增益 |
| P22 | 0.595958/0.718006 | 0.502921/0.643029 | 0.505297/0.646072 | 0.590751/0.713603 | 无双指标独立增益 |
| MISAR_E15_5_S1 | 0.541637/0.666949 | 0.400219/0.568530 | 0.400203/0.569990 | 0.522092/0.649507 | 无双指标独立增益 |

A1 的 retained full 出现极小 ARI-only 增量（约 +0.000115）但 NMI 下降；tonsil s1 的 max-ARI full 到 0.242427，但 NMI 从 0.317365 降到 0.310661。P22 与 MISAR 的所有 retained full 都低于输入强起点。这些 trade-off 保留在逐 lane frontier，不构成表示方法信号。

## 真实 P0

| 家族真路径 | view1 | view2 | N | sparse nnz | optimizer steps | fresh-process | peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | [3484, 30] | [3484, 30] | 3484 | 24178 | 40 | PASS | 75.3 |
| P22 | [9196, 30] | [9196, 50] | 9196 | 64466 | 40 | PASS | 178.1 |

两次独立 GPU 训练在 A1、P22 上分别锁出相同 partition 与相同指标；embedding 的浮点字节跨训练不完全相同。每个 checkpoint 在自己的 fresh process 中 embedding 与 partition 均精确回放（4/4）。

## 为什么提前停止

任务书允许“快速筛选完全无表示增益”时停止扩网格。最终语义下共保留 424 行候选，其中 68 行旧 endpoint 语义标为 superseded；其余候选没有形成跨 study 双指标独立增益。继续堆同义超参数无法回答新机制问题，只会放大标签辅助搜索。

## 结论与论文意义

终态：`NIGHT16D_CMBF_RL_SCIENTIFIC_NEGATIVE`；分类：`SCIENTIFIC_NEGATIVE`。这否定的是当前“强起点插件式三态残差表示”实现，不是否定三态边诊断本身。论文现在仍应把 Night-15/16 的高分归于强起点与结构化 head；Night-16D 不能作为已证实的统一 trainable representation contribution。

## 导师汇报版

1. 我们把 support、boundary、conflict 三类边做成了三种独立的可训练表示运算，而不再只是同一个平滑权重。
2. 两种模态家族共用同一代码、损失和 endpoint，真实 GPU 参数确实更新，checkpoint 回放也闭合。
3. 原始强起点现在是 byte-exact no-op control，same-head teacher 与 generic residual 不再冒充它。
4. 在 A1、tonsil s1、P22、MISAR 的有界筛选中，没有一个统一配置形成跨 study 的 ARI/NMI 双升。
5. retained embedding 只带来 A1 或 tonsil 的单指标取舍，P22/MISAR 反而下降。
6. 因此本轮是实现正确的科学负结果，未进入 family freeze，也没有多 seed 放大。
7. 下一步若继续，应改变表示目标或起点生成证据，而不是继续微调同一残差网格。

## 技术审计摘要

- 标签进入 producer/loss/gradient：0；评价由独立脚本在 partition 锁定后读取公开 annotation。
- dense N×N：0；全部图运算沿注册 CSR 边。
- Moran/Geary 以 cluster one-vs-rest indicator 做 macro 汇总，且已通过类别标签置换不变性测试；full 与 evaluation-mask 簇大小分别报告。
- targeted tests：13/13；candidate failures：0。
- peak GPU：318.2 MiB；peak RSS：1076.4 MiB。
- Git branch：`revision/q2-night16d-cmbf-rl-representation-20260824`；final tag：`night16d-final-20260824`；final commit 由提交后的 delivery verification 独立登记。
- shutdown：未派发；按 Night-16D 夜间联动要求保持 AutoDL 在线。
