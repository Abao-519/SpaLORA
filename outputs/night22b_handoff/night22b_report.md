# Night-22B frozen RNA+chromatin junction transfer report

## 我现在需要知道的三件事

1. **问题**：本轮没有再调参数，而是把 Night-22A 的 `GEOM_LEIDEN_FEATURE + J01_GRAPH_LEAN` 原样冻结，问它能否在 MISAR 与人海马两个 held-out 研究上独立超过起点、纯 head 和每一个原子臂。
2. **实际结果**：父级 P22/placenta 15/15 候选重放完全一致；但 MISAR 的冻结 32 个 Leiden resolution 中没有 exact K=7 起点，不能补网格。人海马 FULL 为 **0.068806/0.148261**，低于 start **0.069277/0.146839** 的 ARI，也明显低于 `SHARED_GRAPH_ONLY` **0.082287/0.170063**。
3. **论文结论**：分类是 **NO_FROZEN_TRANSFER_SIGNAL**。Night-22A 的 chromatin 局部 junction 信号没有跨研究成立；该 junction 主线按约定终止，不追加 HPO，也不触发 GSE205055 下载。Melanoma 的高分只证明简单图原子臂很强，不能挽救 primary 判定。

## 主迁移判决

| lane | frozen primary start | FULL ARI/NMI | Δ vs coordinate-wise strongest control | independent pass |
|---|---|---:|---:|---|
| MISAR K7 | unavailable under frozen exact-K Leiden grid | NA | NA | false |
| human hippocampus K7 | GEOM_LEIDEN_FEATURE | 0.068806/0.148261 | -0.013482/-0.021801 | false |

MISAR 不是程序崩溃：冻结 generator 在标签关闭阶段按原 32 个 resolution 运行完毕，但没有任何 exact-K=7 解。扩网格会改变已冻结方法，因此按规则 fail-closed。人海马 FULL 实际改变 16 个 observation、保持 exact K、无 singleton、每簇有最小尺度内部边，但没有 matched score contribution。

## 起点敏感性与 secondary robustness

- MISAR + Ncut-K24：start 0.427961/0.556508，FULL 0.427803/0.556238；双降。
- MISAR + Night-16H：start 0.535306/0.658265，FULL 0.535348/0.658406，但与 `CLUSTER_GRAPH_ONLY` byte-exact，相同增益不能归因于 FULL。
- Human + Ncut-K24：FULL 0.156061/0.217678，低于 shared-only 0.165778/0.232719。
- Human + Night-16H：FULL 0.595824/0.588737，与 cluster-only byte-exact；ARI 略低于 start。
- Melanoma primary Leiden：FULL 0.928204/0.859914，高于 start，但低于 shared-only 0.951795/0.898555 和 additive 0.961342/0.915599；它是 secondary K2，不计主门。

这些敏感性说明失败并非单纯“起点太弱”：把 Night-16H 强分区接入后，FULL 仍等价于更简单的 cluster-only 原子臂。

## 数学与工程证据

- 使用 Night-22A 原始五图 bank、J01 数值参数、Adam/240 steps、hard exact-K 与六个 matched arms；没有根据 transfer 分数修改任何公式。
- 8 个可用 start bank 均完成真实 loss/backward、参数改变、strict checkpoint reload；两轮独立 Python fresh-process replay 共 16/16 bank PASS，每个 bank 5 个 trainable candidate 精确一致。
- Producer 只读取显式 numeric carrier/representation/sparse graph 与预锁 start；标签读取次数为 0。独立 evaluator 在 bank/checkpoint SHA 验证后才打开 reference。
- 15 项 targeted tests 全通过。父级 P22/placenta 各 15 个 candidate exact replay PASS。

## GSE205055 与停止线

主迁移未出现 signal，因此没有下载或运行 GSE205055 ME13_50um。这个决定避免在负迁移后把外部数据变成新的调参集。Junction 论文主线在本轮关闭；下一步若继续研究，应回到更强的上游 representation 或新的、预注册的直接分区对象，而不是补 Leiden resolution 或继续修 J01。

## 导师汇报版

Night-22A 在 P22 和 placenta 上曾出现很小的 chromatin junction 正信号，所以本轮把公式、五张图、训练预算和起点规则全部冻结，去做 MISAR 与人海马迁移。父级重放完全一致，说明实现接续可靠。MISAR 暴露了第一个严格问题：冻结 Leiden resolution 网格无法产生 exact K=7，我们没有为它扩网格。人海马能完整训练并产生新分区，但 FULL 只有 0.0688/0.1483，低于更简单的 shared-only 原子臂。换成 Ncut 或 Night-16H 强起点后，FULL 仍然不独立，往往被原子臂支配或与 cluster-only 完全相同。Melanoma 的高分同样主要来自 shared/additive 图臂，而非 FULL。结论是这条 junction 没有跨研究冻结迁移证据，应当停止，不再追加 HPO；这比把 sensitivity 的较高数字替换成 headline 更可信。

## 技术状态

- Parent: `1309979e43abc37f4b42b61af48489e611753935` / `night22a-final-20260827`。
- Frozen contract SHA-256: `2dcbeaf42c4ada5351e0e1f2a30e79d1e1715e6709bf74dca77c47c7d263d957`。
- 根盘余量约 22.5 GiB，超过 15 GiB 安全线；新增外部下载 0。
- 普通 push、final tag、incremental bundle、compact 与 Windows 独立复算在最终封口步骤登记。
