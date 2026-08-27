# Night-23B 结果报告

## 我现在需要知道的三件事

1. **问题**：Night-23A 已能跨研究预测部分边关系，但这些关系没有变成好分区。本轮先问一个更基础的问题：即使把近乎完美的 teacher 同域/边界关系直接交给固定消费者，它能否恢复 teacher 分区。
2. **实际动作所在层**：实现的是保留 retained carrier 节点几何的稀疏 signed consumer：carrier emission 与正负关系谱坐标共同进入同一个 exact-K endpoint。它不是候选 selector，也没有训练新的 predictor。
3. **论文意义与分类**：严格 oracle gate 仅 1/3 primary 通过，所以问题首先卡在固定消费者可识别性，而非仅是 calibration。终态为 **SCIENTIFIC_NEGATIVE / RELATION_CONSUMER_NOT_IDENTIFIABLE**；Stage B 与 placenta 均未授权，Night-23 edge-distillation 主线到此关闭。

## Oracle 消费者上限（诊断，不是方法成绩）

下表 ARI/NMI 是输出分区相对 teacher partition 的 recovery，不是相对生物学 benchmark annotation。整个 Night-23B 没有打开 benchmark labels。

| lane | N | K | carrier recovery ARI/NMI | signed oracle recovery ARI/NMI | strict dual gain | min cluster | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| P22_K9 | 9196 | 9 | 0.588518/0.672850 | 0.991029/0.984552 | +0.311702 | 220 | PASS |
| MISAR_K7 | 1949 | 7 | 0.554080/0.680890 | 0.754779/0.833851 | +0.152961 | 137 | FAIL |
| HUMAN_HIPPOCAMPUS_K7 | 2500 | 7 | 0.046353/0.113564 | 0.911433/0.884055 | +0.770492 | 57 | FAIL |
| MELANOMA_TUMOR_K2 | 833 | 2 | 0.130328/0.084594 | 1.000000/1.000000 | +0.869672 | 276 | PASS |

统一选择的原子消费者为 `ORACLE_SIGNED_RELATION`, relation scale=1。P22 几乎精确恢复；MISAR 的恢复仍只有 0.755/0.834 且 dual gain 0.153；human 的 ARI 高但 NMI 0.884 未达冻结门。primary 严格通过数为 1/3。melanoma 是 secondary，虽达到 1.0/1.0，不参与主门。

## 根因归因

- **coverage**：Night-23A union edge 上同时存在 teacher 同域和边界边，P22/MISAR/melanoma 的正边连通分量恰等于 K；human 有 12 个正连通分量、3 个孤立点。
- **consumer**：signed relation 显著优于 carrier-only，但统一固定 consumer 仍不能在至少 2/3 primary 同时满足高恢复与足够增益；因此消费者上限本身不稳健。
- **predictor/calibration**：本轮没有进入 Stage B，不能新增归因；Night-23A 的 predictor 排序信号不能补偿这个 oracle consumer 缺口。
- **三态机制**：正边吸引、负边排斥、缺边擦除的数值语义已由性质测试闭合，但没有成为方法证据，因为 oracle gate 先失败。

## 贡献边界与新颖性

Signed Laplacian、must/cannot-link、概率校准和 boundary affinity 都是成熟原子先例。本来只有“跨研究置换不变 relation + nested source calibration + carrier-preserving tri-state sparse consumer”的联合对象可能检验；由于 Stage B 被禁止，这个联合对象没有获得正面证据，也不冻结论文名称。

## 失败与限制

- 当前 fixed consumer 只探索了预冻结的小型 sparse signed-embedding family；科学结论是该消费者族不可识别，不等同于数学上所有 possible signed partition objective 均不可能。
- Stage A 使用 teacher 作为 oracle 诊断，因此任何 recovery 高分都不能写入方法主表。
- 没有 benchmark ARI/NMI、AMI/FMI、Moran/Geary；这是 label firewall 的主动结果，不是漏报。edge disagreement 是对 teacher relation 的稀疏结构指标。
- 未运行 GPU、MLP、LOSO Stage B 或 placenta。

## 导师汇报版

Night-23A 的边分类器在部分数据上 AUROC 不低，但分区失败的根因一直不清楚。Night-23B 先把预测误差拿掉，直接用 teacher 同域/边界边测试一个保留强分子表示的 signed 稀疏消费者。P22 和 melanoma 能近乎完整恢复 teacher，说明实现和关系对象并非完全无效；然而 MISAR 只有 0.755/0.834，human 的 NMI 也只有 0.884，严格门只有 1/3。按照预注册规则，我们没有继续训练校准桥，也没有读取 benchmark labels 或跑 placenta。这说明当前瓶颈不只是边预测精度，而是“局部边关系如何唯一确定全局 K 分区”的消费者可识别性。Signed clustering、约束聚类和概率校准都有充分先例，因此本轮不做新颖性包装。项目层面应关闭 Night-23 edge-distillation 主线，而不是继续调整阈值。

## 技术审计

- Stage-A freeze commit: `005e3427ea28e0102668ef4feaeac6d98b271d59`
- 四条 formal bank 均 fresh-process exact replay。
- Targeted tests 由最终封口时的真实 pytest 输出登记。
- GPU: 未使用；formal producer peak RSS 最大约 446.9 MiB。
- `shutdown_dispatched` 在 sealed science handoff 中记录封口时事实；最终远端命令另行本地登记。
