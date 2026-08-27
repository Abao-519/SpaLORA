# Night-23C 结果报告

## 我现在需要知道的三件事

1. **问题**：Night-23B 错误地用 worst-first 规则阻断了 scale=2 的 2/3 oracle 可识别证据。本轮先勘误，再真实检验 source-only learned tri-state bridge。
2. **实际层级**：共享 MLP 预测 held-out union edges；inner study-wise cross-fit 只用 source teacher 做 midrank calibration 和阈值；WITHIN 吸引、BOUNDARY 排斥、UNKNOWN 擦除，随后进入固定 scale=2 carrier-preserving signed consumer。
3. **论文意义与分类**：oracle consumer 确实可识别 2/3，但 learned bridge 为 0/3，macro 相对 carrier 为 ΔARI -0.239115、ΔNMI -0.292939。终态 **SCIENTIFIC_NEGATIVE / NO_CALIBRATED_RELATION_BRIDGE_SIGNAL**；Night-23 主线永久关闭。

## Night-23B 只读勘误

原候选、指标和 replay 保持有效；其 narrow decision 改判为 `IMPLEMENTATION_FAILURE / DECISION_SEMANTICS_INVALID`。scale=2 的 P22/human 通过冻结 oracle gate，证据为 `ORACLE_CONSUMER_IDENTIFIABLE_2_OF_3`。这只授权本轮 Stage B，不是方法成绩。

## 绝对 benchmark 指标与匹配贡献

| lane | carrier ARI/NMI | FULL ARI/NMI | coordinate-wise strongest control | FULL-carrier Δ | min cluster | independent gate |
|---|---:|---:|---:|---:|---:|---|
| P22_K9 | 0.471486/0.600288 | 0.038072/0.077535 | 0.524833/0.637898 | -0.433414/-0.522752 | 637 | FAIL |
| MISAR_K7 | 0.366279/0.542777 | 0.045425/0.104498 | 0.543587/0.581646 | -0.320854/-0.438279 | 133 | FAIL |
| HUMAN_HIPPOCAMPUS_K7 | 0.043363/0.087154 | 0.080285/0.169369 | 0.429098/0.426792 | +0.036922/+0.082215 | 64 | FAIL |

P22/MISAR 的最强 matched control 均是 `RETAINED_ONLY_UNSIGNED`；human 最强是 `MLP_DIRECT_NONNEGATIVE`。FULL 在三条 lane 都没有同时超过全部原子臂。human 虽高于弱 carrier endpoint，但远低于 MLP direct，不能算独立贡献。

## Relation 与三态诊断

- P22/MISAR held-out MLP AUROC 为 0.844/0.849，说明排序信号仍在；human 仅 0.436。
- source-only threshold 在 P22/MISAR 将约 96%/92% 边判为 BOUNDARY，但真实 boundary purity 仅约 0.199/0.264；负边被大规模误用，signed consumer 分区崩坏。
- human 的 boundary purity 仅约 0.127；FULL 也被 MLP direct 原子臂显著支配。
- UNKNOWN 在三条 lane 约 2%，且 FULL 中确实擦除，没有恢复为 raw-union 强连接。三态实现正确，但 calibration/transfer 科学对象失败。

## 标签流与真实性

三条 checkpoint、prediction 和 9-arm banks 全部先物化/hash，并完成 3/3 fresh-process exact replay。producer 没有打开 held-out teacher 或 benchmark annotation；held-out teacher 只用于锁定后的 relation AUROC/AUPRC，benchmark labels 只由独立 evaluator 读取。MLP 参数真实更新、checkpoint strict reload 通过、所有 arms exact-K。

## 失败、限制与新颖性

Platt purity cycle 在 source-only 阶段无可行阈值；Platt utility cycle 发生 boundary collapse；两者均在任何 held-out teacher/reference 打开前被 supersede。正式 midrank 公式只跑一次完整 LOSO，未按分数修改。Signed clustering、约束聚类、概率校准和 boundary affinity 都是成熟先例；联合对象得到负证据，不冻结方法名。GPU 峰值显存没有在 producer 中登记，这是资源审计限制；CPU peak RSS 与 wall time完整保留。

## 导师汇报版

Night-23B 的数值没有错，错的是决策顺序：scale=2 实际能在 P22 和 human 两条 primary 上恢复 teacher，因此我们先发布只读勘误。Night-23C 随后真正执行了原本被阻断的 learned Stage B，每折只用另外两个研究的 teacher relations 训练和校准。工程路径、三态作用、checkpoint 与 fresh replay都成立。P22/MISAR 的 MLP 边排序仍有约 0.84 AUROC，但跨研究阈值把绝大多数边错判成 boundary，导致 signed partition 大幅下降。三条 primary 的 FULL 均未双胜全部 matched controls，宏观 ARI/NMI 也显著为负。human 仅相对弱 carrier KMeans endpoint 上升，但被 MLP direct 原子臂解释。按冻结门 placenta 没有运行，GSE205055 没有下载。结论是 oracle 边关系可以被消费者利用，但当前 learned relation-to-partition bridge不能跨研究校准；Night-23 路线到此永久结束。

## 技术摘要

- Formula freeze: `a976c722669bc6e621fb436664c5a9f34eaa3565`
- Evaluator gate freeze: `7693b94beaa8304eae138db8440a18244e8226d2`
- Formal replay: 3/3 exact
- GPU: used; peak memory not instrumented
- Max RSS: 772.4 MiB; formal producer wall sum: 27.0 s
