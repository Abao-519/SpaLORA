# Night-19A 稀疏证据梯度仲裁 P0 报告

## 我现在需要知道的三件事

1. **问题**：Night-17C 的关系保持目标与强起点保护是否真的持续冲突？答案是“是”。修正 zero-start ramp 的合约解释后，P22、MISAR、人海马 3/3 主 lane 以及 placenta 都在两个 seed 的 step 5/20/40 复现 `relation–anchor` 负方向冲突。
2. **实际动作所在层**：本轮没有改 selector 或图割 head，而是在同一个 32 维 zero-start 残差网络的梯度层，把注册稀疏空间/feature-neighbour 边按证据分层；低支持冲突分量投影更强，高支持更弱，并对合成 relation 梯度做 L2 量级匹配。这里的起点是**登记的 relation-smoothed carrier**，不是 raw retained 的 byte-exact 拷贝。
3. **论文含义**：冲突诊断是阳性，但方法结果是阴性。`FULL` 在三条主 lane 都没有同时超过坐标级最强 matched control，因此没有独立分数贡献，不能写成新方法成功。主分类为 **SCIENTIFIC_NEGATIVE**，多 seed 与后续 stage 均未授权。

## 绝对结果

| lane | FULL ARI/NMI | 最强 matched control ARI/NMI | FULL 差值 | 结论 |
|---|---:|---:|---:|---|
| P22 K9 | 0.481496 / 0.610371 | 0.482339 / 0.610676 (`TOPOLOGY_DISABLED`) | -0.000843 / -0.000304 | 未通过 |
| MISAR K7 | 0.360299 / 0.538696 | 0.363714 / 0.540974 (`STANDARD_WEIGHTED_SUM`) | -0.003415 / -0.002279 | 未通过 |
| Human hippocampus K7 | 0.194597 / 0.274609 | 0.209619 / 0.279205 (`STANDARD_WEIGHTED_SUM`) | -0.015022 / -0.004596 | 未通过 |
| Placenta K10（压力测试） | 0.373976 / 0.527426 | 0.381948 / 0.532974 (`STANDARD_WEIGHTED_SUM`) | -0.007971 / -0.005548 | 安全阈值内，但没有增益 |

这些数值使用相同 KMeans endpoint。`ARBITRATION_DISABLED_SAME_LOSSES` 与 `STANDARD_WEIGHTED_SUM` 在四条 lane 的分区和指标完全一致，说明分层重写本身没有偷换损失。P22 中 FULL 与 mass-matched permuted 的指标完全相同；MISAR 与 human 的 permuted 还优于 FULL，因此内容位置证据没有独立贡献。

## 与既有强结果的边界

Night-16H 的固定 selector 为 P22 `0.587533/0.708974`、MISAR `0.534624/0.656772`、人海马 `0.596178/0.585490`。本轮 Stage-A 使用的是 relation-smoothed zero-start representation 加共同 KMeans endpoint，不是 Night-16H 的 89-candidate 结构选择器；因此绝对落差同时包含 consumer/endpoint 差异，不能全部归因于训练表示。但 FULL 相对本轮 exact matched controls 仍为 0/3，足以否定本次仲裁的独立贡献。

## D0：冲突可识别，但不是任意负余弦都算通过

REV1 保留余弦阈值 `-0.05` 和梯度 norm ratio 阈值 `1e-3`，把 step 0/1 解释为 zero-start ramp，仅在 step 5/20/40 要求每 seed 至少两个可操作冲突点。结果 3/3 主 lane 与 placenta 均通过，复现 pair 都是 `relation__anchor`。step0 的零范数余弦保存为 NA；从未伪填 0。

原 REV0 结果完整保留，但它把 step1 约 `2e-4–7e-4` 的早期比例纳入所有负冲突点的 minimum，从而结构性产生 false negative。该周期标为 `SUPERSEDED_IMPLEMENTATION_SEMANTICS_INVALID`，没有参与 Stage-A 决策。

## 机制归因与先例

PCGrad、CAGrad、GradNorm、OGM-GE、MMPareto、SpaBalance，以及计算图分块、层级/模块解耦、只在严重冲突时投影和不确定性聚合均已有明确先例。本轮唯一可能区分的对象只是“注册稀疏 edge evidence 对 relation-gradient strata 与强起点保护方向之间的连续投影，并以 mass-matched permuted evidence 隔离内容位置”。正式结果没有支持该对象，故不产生新颖性主张。

## 工程与标签流

- D0 REV1：4 lane × 2 seeds = 8/8 producer；8/8 checkpoint/artifact 独立进程重放 exact。
- Stage A：4 lane × 8 arms；4/4 artifact/checkpoint 独立进程重放全部表示与分区 exact。
- 生产端标签读取为 0。四条 Stage-A artifact 全部锁定后，四个独立 evaluator 才读取公开 annotation；没有参数网格、label-HPO 或 within-run checkpoint 选择。
- 最终测试覆盖固定 named-parameter 坐标、unused gradient 补零、零范数 NA、step1 ramp 排除、inactive min-norm 任务排除、置乱质量匹配以及 standard/disabled 等价。
- `/autodl-fs/data` 没有新写入；未下载数据、未新建环境；AutoDL 保持开机，`shutdown_dispatched=false`。

## 失败与限制

- positive D0 只证明梯度方向冲突存在并可操作，不证明冲突必须被消除；Recon 等工作也提醒小冲突可能有益。
- FULL 对 human 的损失最大，提示按 evidence strata 对 relation 梯度投影仍可能破坏对聚类有用的更新。
- placenta 仅说明未超过预注册 0.01 额外回撤阈值，不能称 safety gain。
- 单 seed Stage-A 是按预注册门停止后的完整证伪，不支持稳定性或里程碑结论。

## 导师汇报版

这轮先验证了过去可训练核心为什么经常拉坏强起点：关系损失与强起点保护梯度在三条 RNA+ATAC 主数据以及 placenta 上都持续反向，而且两个 seed 可复现。我们随后实现了一个很窄的稀疏证据仲裁，只削弱低支持 edge strata 中与保护方向冲突的分量，同时保持 relation 梯度总量级。工程上，固定参数坐标、零梯度任务处理、mass-matched 置乱、checkpoint 与新进程重放都闭合。科学上 full 在 P22、MISAR、人海马均没有赢过同预算强对照，permuted 在部分数据还更好，因此 edge evidence 的具体位置没有独立贡献。结论是“冲突诊断阳性、仲裁方法阴性”，分类 `SCIENTIFIC_NEGATIVE`。这也说明下一步不应继续微调 gradient surgery，而应回到能真正改变强表示或候选生成的对象。

## 技术附录

- Taskbook SHA-256: `19107b229b1fbd06a6c6dafee797b469057ec8d75eab7dcec6e6ae22b8ad2e62`
- D0 REV1 contract SHA-256: `8d15260779f20ef322b70eacc313ad02d2e90241c5f180f3f8881f366955b1e1`
- Stage-A REV1 contract SHA-256: `d1a5275f8b76865de555d37a939b9e77dcc080590a8fc4bedcb1fd047c8a01a5`
- Parent Night-18E commit: `8d9b28aff48b338d55f1a91a1129d96f5ded0856`
