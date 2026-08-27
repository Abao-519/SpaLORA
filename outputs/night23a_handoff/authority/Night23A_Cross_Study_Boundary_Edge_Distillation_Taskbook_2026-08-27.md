# Night-23A：跨研究边界边蒸馏（XBED）可识别性与方法 P0

日期：2026-08-27  
工作名：`XBED`（Cross-study Boundary-Edge Distillation），仅为工程代号；证据成立前不得冻结论文名称。  
性质：新方向的关系迁移可识别性测试 + 共享边模型 + held-out 分区 P0。

## 1. 为什么结束旧线并启动本轮

Night-22B 已证明 Night-22A junction 不可冻结迁移：MISAR 的主起点不可用，人海马 FULL 被更简单 shared-only 支配；换成 Night-16H 强起点后，FULL 又与 cluster-only byte-exact。因此 `J01`、椭圆发射和直接 `N×K` junction 主线永久结束，Night-23A 不得继承其公式或追加 HPO。

项目目前最强、最可迁移的资产反而是 Night-16H 的固定结构可行域和跨证据 selector partitions。它们跨研究得分较高，但只是候选选择结果，论文创新偏软。Night-23A 尝试把这些分区转换成更一般的计算对象：**不蒸馏类别编号，而只蒸馏局部边是否跨越空间域边界**。这种二元关系与 K、组织名称和 cluster label permutation 无关，可以训练一个真正跨研究共享的边模型。

## 2. 科学问题

能否仅从训练研究中冻结、运行时不读真实标签的 teacher partitions，学习一个共享的、内容驱动的稀疏边可靠性函数；并在 held-out 研究上不生成 teacher、不读取标签、不使用数据集名称，只根据多模态局部关系预测边权，随后通过固定 exact-K 图分区获得优于未学习图和简单交集规则的空间域？

## 3. 与先例的边界

必须审计论文和源码：ARISE、MMSpa、PRAGA、SMART、S3RL/SEPAR、SpatialGlue、stMixer、图结构净化/边可靠性、pseudo-label graph self-training、跨图迁移。

已知先例包括：

- ARISE：RNA feature graph 与 spatial graph 的固定交集；
- MMSpa：单数据集内为边界识别进行 edge removal；
- PRAGA：单数据集内原型感知的自适应模态图；
- SMART：MNN 正负样本和 triplet metric learning；
- 普通 pseudo-label/self-training：用聚类标签训练节点或边关系；
- stMixer/stGuide：跨切片或 reference-to-query 的标签迁移。

Night-23A 只有在以下联合对象成立时才可保留窄主张：**类别置换/K 不变的 teacher 边关系 + 跨研究共享训练 + held-out 无 teacher 推理 + 稀疏 exact-K partition**。任何单个组件不得称原创。不得复制第三方受限源码。

## 4. 数据与 teacher authority

### LOSO 开发研究

使用 Night-16H 已闭合的四个真实 RNA+chromatin 单元：

1. P22 K9，N=9196；
2. MISAR E15.5 K7，N=1949；
3. Human hippocampus K7，N=2500；
4. Slide-tags melanoma tumour K2，N=833（低复杂度，只作辅助折叠）。

每个研究的 teacher 必须是 Night-16H 固定 selector partition 或严格 LOSO authority，不能使用真实标签、Night-22A/22B 分数选择的新 partition，不能重做 label-HPO。记录 teacher partition SHA、候选来源和结构可行性。

### 冻结确认

Placenta K10，N=1662。它不参与 XBED 架构、边特征或超参数选择。若 LOSO 方法门通过，用四个 Night-16H teacher studies 训练最终共享边模型后，在 placenta 上无 teacher 推理。Placenta 曾参与历史项目开发，因此称 frozen cross-study confirmation，不称 pristine blind test。

## 5. 输入与跨研究不变量

每个研究读取已注册的 retained、view1、view2 和 spatial CSR。允许各 view 原始维数不同；边模型只能使用在研究内标准化后的关系特征，不直接拼接固定维度节点向量。

稀疏 union edge set 至少可包括：

- registered spatial edges；
- retained reciprocal kNN；
- view1 reciprocal kNN；
- view2 reciprocal kNN。

可用 edge features：各 view 的距离/相似度分位数、reciprocal-rank、mutual indicator、空间距离分位数、局部密度/degree、跨模态邻域重叠与冲突、端点局部尺度。不得包含数据集名称、组织名称、真实标签、teacher cluster ID、绝对 observation ID。

所有特征计算保持稀疏，禁止无必要 dense `N×N`。

## 6. Teacher relation 与可识别性 Stage A

训练研究中，对 union edge `(i,j)` 定义置换不变 teacher relation：同一 teacher cluster 为 1，不同为 0。可以使用候选库中标签关闭的 partition agreement 作为置信权重，但不得读取 benchmark reference。

先实现两个低成本模型：

1. 正则化 logistic/线性 edge classifier；
2. 小型共享 MLP edge classifier。

执行四折 leave-one-study-out（LOSO）：每折仅用其余三个研究的 teacher edge relations 训练；held-out teacher 只在 edge prediction 锁定后用于诊断 AUROC/AUPRC/calibration，不进入训练、模型选择或 partition producer。

Stage-A 继续门建议：三个主要研究 P22/MISAR/人海马中至少 2 个 held-out edge AUROC > 0.65，三研究 pooled AUROC > 0.70，且 MLP 或线性模型稳定优于仅按空间距离、硬 graph intersection 和随机 teacher 对照。若关系本身不可迁移，终态 `NO_CROSS_STUDY_EDGE_IDENTIFIABILITY`，停止，不运行大规模 partition HPO。

数值阈值用于止损，不是论文理论结论；Codex 可在不查看 held-out 真实标签的前提下根据训练折 calibration 修正实现细节。

## 7. Held-out partition Stage B

若 Stage A 通过：

1. 用每折训练研究拟合的冻结 edge model 预测 held-out union-edge reliability；
2. 用固定、稀疏、保证 exact K 的分区器生成硬分区。优先使用 sparse normalized-Laplacian embedding + 固定 KMeans 或经源码核验的确定性 exact-K 方法；不得依赖可能找不到 K 的有限 Leiden resolution grid；
3. partition、edge model checkpoint、ordered IDs、edge probabilities 和 hashes 全部锁定并 fresh-process replay 后，独立 evaluator 才读取真实标签。

匹配对照至少包括：

- raw/equal-weight union graph；
- ARISE-like hard intersection clean-room heuristic；
- spatial-only、retained-only、view1-only、view2-only；
- logistic edge model；
- MLP 去掉 teacher confidence；
- shuffled-teacher negative control；
- FULL XBED。

所有对照使用相同 union edges、exact-K partitioner、seed 和预算。

## 8. 方法门与分类

主要方法票只看 P22、MISAR、人海马；melanoma 不计票。

- `CROSS_STUDY_EDGE_PARTITION_SIGNAL`：FULL 在至少 2/3 主要 LOSO held-out 研究上 ARI/NMI 双指标严格超过所有匹配非 FULL 对照；三研究 macro ΔARI、ΔNMI 均为正；剩余研究任一指标回落不超过 0.02。
- `RELATIONAL_TRANSFER_WITHOUT_PARTITION_GAIN`：edge identifiability 通过，但 partition 方法门失败。
- `NO_CROSS_STUDY_EDGE_IDENTIFIABILITY`：Stage A 失败。
- `IMPLEMENTATION_FAILURE`：真实路径、teacher authority、checkpoint/replay 或标签隔离失败，不能形成科学结论。

若 LOSO 方法门通过，冻结架构、边特征、训练策略和 partitioner，用四个 teacher studies 训练最终模型并运行 placenta：

- placenta FULL 双指标胜过全部 matched controls：登记 `FROZEN_EXTERNAL_EDGE_SIGNAL`；
- 否则保留 LOSO 信号，但不得称确认里程碑。

任何 score frontier 单独登记，不能替代 matched contribution。真实标签只用于锁后评价，不参与 edge model loss、gradient 或 checkpoint selection。

## 9. 允许的开发自由与禁止事项

- 允许基于训练折 teacher-edge loss/AUROC 调整模型、采样、class weighting 和特征；不得查看 held-out 真实标签来决定这些修改。
- 不设置僵硬 correction-cycle 数量；实质公式变更必须重新跑完整 LOSO，并保留简明 ledger。
- 不要求复杂深网：若 logistic 与 MLP 相当，应优先简单模型并如实报告；不能为“看起来像深度学习”堆层。
- 不得让 held-out teacher 进入推理；不得用真实标签构造边、triplet、loss 或 early stopping；不得以数据集名称选择模型。
- 不复活 Night-22A/22B junction，不在同轮增加 selector/head 挽救失败 partition。

## 10. 条件式新数据

只有达到 `CROSS_STUDY_EDGE_PARTITION_SIGNAL` 且 placenta 不出现严重退化，才继续 GSE205055 ME13 authority closure。下载优先数据盘，单轮新增不超过 2 GiB；找不到权威 K/mask/annotation 时只保留无标签工程单元，不制造标签。

## 11. 工程、资源和交付

- 先做五研究真实 shape、union edge 数、正负 teacher edge 比例、checkpoint/partition P0；避免某个大研究主导训练，按研究平衡采样。
- 记录训练/推理 wall time、CPU/GPU time、peak GPU/RSS、edge 数和复杂度；根盘至少保留 15 GiB。
- targeted tests 至少覆盖：类别置换不变、K 不变 edge target、study-balanced sampler、held-out teacher isolation、exact-K、节点置换等变、稀疏性、checkpoint/fresh-process replay。
- 交付中文报告、绝对指标主表、edge-transfer table、matched contribution、collision matrix、teacher/label flow、源码、测试、失败 ledger、compact index、Windows 独立复算、普通 Git push 和增量 bundle。

最后将 `/usr/bin/shutdown` 作为严格最后一条远端命令；派发后不重连，也不声称控制面板状态。
