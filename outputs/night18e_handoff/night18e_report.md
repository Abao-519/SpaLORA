# SpaLORA Night-18E report

## 我现在需要知道的三件事

1. **数学证书闭合了，但方法分数没有成立。** CCSR（可信度认证自返还）在实际整数容量 alpha-expansion 子问题上通过穷举等价测试，4 条真实 P0 与 discovery 的认证节点均 0 改动；然而严格独立门是 0/3，终态为 `SCIENTIFIC_NEGATIVE`。
2. **实际改的是最终结构能量的离开代价。** 三个分子视图先按 prototype 支持与 unary margin 的**秩**确定可信节点，再按该节点可能节省的全部 incident Potts capacity 添加最小离开惩罚。它不是幅度校准，标签不进入 producer、confidence、unary、pairwise 或 solver。
3. **固定局部点不等于保护整张分区。** Placenta 的 start2 从 no-op `0.417146/0.527825` 降到 full `0.229201/0.307250`；P22 authority 也从 `0.595958/0.718006` 降到 `0.592875/0.713130`。因此只能说“受保护点不移动”，不能说“避免总体灾难”。

## 绝对指标主表

| lane / role | N/eval/K | ARI / NMI | Δ vs same-start no-op | AMI / FMI | Moran / Geary | changed / trusted / certified changed | min cluster |
|---|---:|---:|---:|---:|---:|---:|---:|
| P22 K9 / shared start0 | 9196/9196/9 | 0.592875 / 0.713130 | -0.003083 / -0.004875 | 0.712635 / 0.653157 | 0.930232 / 0.072381 | 85 / 0.494 / 0 | 168 |
| MISAR K7 / shared start0 | 1949/1949/7 | 0.541798 / 0.665842 | +0.000162 / -0.001107 | 0.664081 / 0.633339 | 0.922652 / 0.079224 | 4 / 0.482 / 0 | 125 |
| Placenta K10 / shared start0 | 1662/1662/10 | 0.163449 / 0.256887 | -0.188303 / -0.276565 | 0.247414 / 0.289777 | 0.520751 / 0.492437 | 643 / 0.496 / 0 | 53 |
| Human hippocampus K7 / P0 diagnostic | 2500/2500/7 | 0.344464 / 0.428937 | +0.178730 / +0.165748 | 0.426528 / 0.474497 | 0.716027 / 0.272818 | 875 / 0.327 / 0 | 51 |

Human 的 P0 full 虽高于其 no-op 与随机掩码控制，但 Stage A 已先失败，故没有资格成为冻结 Stage-B confirmation，也不改变终态。

## Stage-A shared profile 与配对稳定性

机械规则选出 `Q50_S2_KEEP1_E1E6`：先最大化通过 study 数，再看最差 study median ΔARI、平均 ΔARI/NMI、较低 protected fraction。其 3-start 汇总为：

| lane | ARI best/median/mean/min | NMI best/median/mean/min | no-op 双升 starts | 严格独立 starts |
|---|---|---|---:|---:|
| P22 | 0.592875 / 0.512060 / 0.531267 / 0.488865 | 0.713130 / 0.641517 / 0.660141 / 0.625774 | 2/3 | 0/3 |
| MISAR | 0.541798 / 0.359122 / 0.419969 / 0.358988 | 0.665842 / 0.547066 / 0.586178 / 0.545625 | 2/3 | 0/3 |
| Placenta | 0.229201 / 0.182601 / 0.191750 / 0.163449 | 0.307250 / 0.301087 / 0.288408 / 0.256887 | 0/3 | 0/3 |

所谓“严格独立”要求 full 对 no-op 双升，并同时严格双胜旧 self-return、certificate-disabled 和 count-matched random-mask；没有任何 paired start 满足。P22/MISAR 低起点上的表面双升均被关键控制解释。

## 证书与实现边界

- 最终核心 SHA：`be91e224f0dea152fbc39cab935b4ecdcacd76c447d1ce24b512f48de051fd19`；execution contract SHA：`5a2a0847e5e85ae0a04932fcf753e78d66beb804a46999dbb7808b92ab60f6b4`。
- 正式 cut 与 tiny test 共用同一 int64 量化函数；8/8 targeted tests PASS，正式 cut 的整数能量对每个枚举 alpha 子空间等于穷举最优。
- 7/7 fresh-process replay 的 ordered IDs、candidate IDs 与 partitions 精确一致。
- 711 个 formal 候选全部 PASS；producer 标签读取 0，独立 evaluator 在分区锁定后读公开标签。
- margin confidence 是 raw margin 的 midrank percentile；robust scale 不改变排名，只作诊断。
- 原始 rejected-mass stay 仅留给未认证节点，认证节点换为 certificate gap；现有对照并非 total-stay-mass matched，不能单独归因 stay 质量。

## 数据扩展审计

E13.5/E18.5 没有进入实验。已下载的 268.2 MB `spatial_ATAC-RNA-seq_MB.zip` 经 accession 解析属于 GSE205055，不是 MISAR stage，且没有 reference annotation。SIVA 的 1.65 GB processed archive元数据包含 MISAR stages，但当前根盘需保留 1.5 GiB，未下载；E13.5 K7/E18.5 K10 的逐点 label、ID、mask、hash 仍未闭合，且两者属于同一 MISAR study block。

## 导师汇报版

1. 本轮把“高置信点不应被空间后处理改坏”写成了实际整数 Potts 量纲里的可验证离开惩罚。
2. 量化 graph-cut 的 tiny 穷举、真实四 lane P0、源码 hash 和 fresh-process replay 都闭合，认证节点确实 0 改动。
3. 但该局部证书没有转化为整张分区的可靠增益：Stage-A 严格门为 0/3。
4. P22 authority 比 no-op 和旧路径更低；MISAR 只有极小 ARI 上浮同时 NMI下降；placenta 出现明显整体退化。
5. Human P0 有局部高分，但因为 discovery gate 先失败，它不是冻结确认，不能用来挽救结论。
6. 因此 Night-18E 终态是 `SCIENTIFIC_NEGATIVE`：理论/工程对象正确，科学方法贡献不成立。
7. 经典 persistency、Potts 与 alpha-expansion 全部按先例处理；本轮不提出论文新颖性声明。

## 资源与终态

全程 CPU，GPU time 0；formal working 小于 20 MiB，根盘在封口时仍高于 1.5 GiB 保留线。`shutdown_dispatched=false`，AutoDL 保持开机。
