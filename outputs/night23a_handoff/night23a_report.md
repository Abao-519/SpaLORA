# Night-23A 跨研究边界边蒸馏报告

## 我现在需要知道的三件事

1. **边关系能迁移，但不是四研究一致。** 共享 logistic 在 P22、MISAR 的 held-out teacher-edge AUROC 分别为 0.8652、0.8685，human 只有 0.4484；三主研究 pooled AUROC 为 0.8229，Stage A 按预注册门通过。这个结果说明训练研究中学到的多模态局部关系含有可迁移信息，但在人海马上的线性关系发生明显域偏移。
2. **可识别的边，不等于可用的分区边权。** 同一 exact-K 稀疏分区器下，FULL 在 P22/MISAR/human 的 ARI/NMI 分别为 0.4210/0.5979、0.1604/0.4047、0.4630/0.4613，三条都被匹配原子对照双指标支配，主门 0/3；macro ΔARI/ΔNMI 为 -0.1213/-0.0807。
3. **终态是 `RELATIONAL_TRANSFER_WITHOUT_PARTITION_GAIN`。** 本轮得到的是一个真实、可复算的跨研究 edge-identifiability 诊断，不是新的空间分区方法。按冻结合约停止四研究最终训练、placenta 和 GSE205055，不回到 Night-22 junction，也不追加 head/HPO 挽救。

## 算法实际做了什么

每个研究只从注册 spatial CSR 和 retained/RNA/ATAC 三个数值 view 机械构建稀疏 union edge。30 个特征全部在研究内转为 rank/robust relation statistics；模型看不到数据集名、组织名、观测 ID、teacher cluster 编号或真实标签。训练研究的 Night-16H partition 只转换为“同簇/跨簇”二元边关系，因而对 cluster label permutation 不变，也不依赖不同研究的 K 数值可比。

四折 LOSO 中，预测 artifact、checkpoint 和 edge hashes 先锁定并 fresh-process exact replay，held-out teacher 随后才由诊断 evaluator 打开。Stage B 则完全不读取 held-out teacher：冻结 edge probability 后，十个 matched arms 使用同一 edge union、同一 sparse normalized adjacency、同一 deterministic eigensolver 和同一 KMeans exact-K endpoint。FULL 使用 `c=2|p-0.5|; w=(1-c)+cp`：低置信边退回 raw union，高置信同域边保留，高置信边界边衰减。

## Stage A：跨研究边关系可识别性

| Held-out study | Logistic AUROC | Logistic AUPRC | MLP AUROC | Spatial AUROC | Intersection AUROC | Shuffled AUROC |
|---|---:|---:|---:|---:|---:|---:|
| P22 K9 | 0.8652 | 0.9575 | 0.8320 | 0.6569 | 0.6149 | 0.3421 |
| MISAR K7 | 0.8685 | 0.9457 | 0.8223 | 0.6239 | 0.5894 | 0.5893 |
| Human hippocampus K7 | 0.4484 | 0.8124 | 0.7071 | 0.7160 | 0.5032 | 0.5937 |
| Melanoma K2（secondary） | 0.6213 | 0.9744 | 0.7688 | 0.8350 | 0.5007 | 0.3173 |
| 三主研究 pooled | **0.8229** | **0.9327** | — | 0.6586 | 0.5990 | 0.4219 |

Stage A 的正面证据仅能表述为“跨研究 teacher-edge 统计可部分迁移”。Human 的 logistic 失败、melanoma 的空间距离更强，禁止把 pooled 数字写成普适边界识别器。

## Stage B：绝对分区指标与匹配贡献

| Study | FULL ARI/NMI | strongest matched control ARI/NMI | ΔARI/ΔNMI | FULL 最小簇 | 独立通过 |
|---|---:|---:|---:|---:|---|
| P22 K9 | 0.4210 / 0.5979 | 0.4568 / 0.6213（MLP direct） | -0.0358 / -0.0233 | 242 | 否 |
| MISAR K7 | 0.1604 / 0.4047 | 0.4346 / 0.5588（retained-only） | -0.2742 / -0.1541 | 75 | 否 |
| Human hippocampus K7 | 0.4630 / 0.4613 | 0.5169 / 0.5261（MLP direct） | -0.0540 / -0.0648 | 170 | 否 |
| Melanoma K2（secondary） | 0.9001 / 0.8163 | 0.9001 / 0.8163（多臂等价） | 0 / 0 | 271 | 不计票 |

FULL 不是 no-op：相对 raw union 的最优标签对齐后 changed spots 为 P22 1867、MISAR 479、human 178；melanoma 为 0。它确实改变了分区，却没有带来独立质量增益，因此失败不能归因于“模块没接进去”。所有 exact-K 与 fresh replay 均通过，没有空簇；主失败是 learned reliability 到 partition capacity 的语义不匹配/域偏移，而非工程崩坏。

与 Night-16H 正式结果相比，本轮任何臂都没有刷新 frontier：P22 0.5875/0.7090、MISAR 0.5346/0.6568、human 0.5962/0.5855、melanoma 0.9758/0.9428 仍是对应权威高位。Night-23A 中较好的 MLP-direct/retained-only 数字只是 matched head/context，不是 XBED 贡献。

## 贡献边界与新颖性

ARISE 已覆盖 RNA-feature/空间图硬交集与层级融合；MMSpa 已覆盖研究内 noisy-edge removal；PRAGA 已覆盖动态模态图和 prototype aggregation；stGuide/stMixer 已覆盖 reference/query 或跨切片迁移。teacher supervision、边分类和 pseudo-label graph self-training 也都是成熟对象。因此能够保留的窄对象只能是“cluster-label/K 不变 edge target + study-balanced LOSO + held-out no-teacher inference + sparse exact-K consumer”的联合协议，不能把任何组件单独称原创。

本轮联合协议只通过了 edge diagnostic，没有通过 partition contribution gate，所以暂不值得冻结论文方法名，也不进入 placenta 外部确认。最直接的科学结论是：**用分区 teacher 学到的边概率可以预测另一个研究的 teacher edges，但不能直接视为最优 Potts/spectral capacity。**

## 失败、限制与下一步边界

- Human 的 logistic 域偏移显著，MLP 虽改善 edge AUROC，却仍无法使 FULL 分区优于对照；这提示 calibration 与 partition loss 并不一致。
- teacher 来自 Night-16H selector，属于项目历史 benchmark-development 资产；即使标签没有进入本轮 producer，也不能把整个教师链称为原始数据端到端无标签。
- Stage B 采用 fixed spectral exact-K consumer；结果否定的是当前 reliability-to-capacity 公式与该 consumer 的联合对象，不等价于否定所有跨研究边预测任务。
- 预注册门禁止 placenta/GSE205055，因此不存在外部确认结果，也不得把“没跑”写成负数据集结果。
- 这条 XBED 分区线到此停止。若未来复用，只宜把 edge model 作为独立 boundary diagnostic，不能在没有新数学对象和外部证据时继续调 capacity/head。

## 5–8 句导师汇报版

我们这轮不再改 Night-22 的 junction，而是把 Night-16H 的高分分区压成与类别编号无关的同域/跨域边关系，训练一个跨研究共享边模型。模型在 P22 和 MISAR 上的 held-out AUROC 都约 0.87，三主研究 pooled AUROC 约 0.82，证明局部边关系里确实有可迁移统计信号。可惜在人海马上线性模型 AUROC 只有 0.45，说明跨组织校准不稳定。更关键的是，把这些概率真正接进统一 sparse exact-K 分区器后，FULL 在三个主要研究上全部被原子对照双指标支配，方法门是 0/3。FULL 确实改变了数百到上千个 spot，并且所有重放和 exact-K 检查通过，所以这是科学负结果，不是代码没生效。最终分类是 `RELATIONAL_TRANSFER_WITHOUT_PARTITION_GAIN`：边诊断有信号，但不能写成分区方法贡献。我们按预注册规则没有再跑 placenta、GSE205055 或追加 HPO，也没有复活旧 junction。论文上最多保留为“跨研究边可识别但直接容量映射失败”的负证据和算法诊断资产。

## 技术审计摘要

- Parent: `1d0a7116b4404d66600e412bc2cfa8eddabf1c1c` / `night22b-final-20260827`
- Taskbook SHA-256: `2f052d68013c9a0843f25f67b6a557794d515b17000a8e079dab5b55f239f039`
- Stage-A freeze: `8ce934643c5c1e425811842a8a5c98ad0395db71`
- Stage-B freeze: `c30798835351e4b149df5cd02d1e4ac26dda3230`
- Stage-A checkpoint/prediction replay: 4/4 PASS；Stage-B partition replay: 4 lanes × 10 arms PASS。
- Targeted tests at Stage-B freeze: 10/10 PASS。
- Root free space at handoff build: 22.52 GiB；working artifacts: 6.94 MiB。
- Shutdown was not dispatched when scientific handoff was built; final dispatch is an external last-command step after compact verification.
