# Night-17F — Direct Relation-Distilled Cut

## 我现在需要知道的三件事

1. **问题**：我们检验了 Night-17B 的“可行候选共聚类软后验”是否在不经过节点 embedding 的情况下，直接放到稀疏图割边权上就能产生独立价值。结论是否定的：预注册门为至少 2/3 主 lane 同时双升且胜过全部匹配对照，实际是 **0/3**。
2. **实际动作所在层**：本轮只改聚类候选生成的 pairwise edge capacity。强表示、强起点、动态 prototype unary、三尺度稀疏图和 alpha-expansion 均保持不变；没有训练网络，也没有把关系再压进节点表示。
3. **论文含义**：P22 和人海马有小幅绝对分数上升，但 P22 与等权 analytic posterior 得到 byte-exact 同一分区，人海马又被 uniform/disabled 对照支配，MISAR 完全 no-op。因此这不是新的 relation-specific 方法信号；Night-17B 至 Night-17F 的 relation-distillation 链条到此停止。终态分类是 **SCIENTIFIC_NEGATIVE**。

## 绝对指标主表

以下“direct”均为 3 个冻结 global mix 中按公开 benchmark ARI、再按 NMI 选出的开发诊断峰值；它是透明 label-assisted diagnostic，不是迁移或自动模型输出。

| dataset | N | K | Night-16H strong start ARI/NMI | direct posterior ARI/NMI | ΔARI/ΔNMI | AMI | FMI | Moran / Geary | changed spots | min cluster | gate explanation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P22 | 9,196 | 9 | 0.587533 / 0.708974 | **0.588603 / 0.710379** | +0.001070 / +0.001405 | 0.709878 | 0.649634 | 0.932930 / 0.069442 | 33 | 166 | 与等权 analytic posterior 的分区 SHA 和全部指标完全相同，独立性失败 |
| MISAR E15.5 | 1,949 | 7 | 0.534624 / 0.656772 | 0.534624 / 0.656772 | 0 / 0 | 0.654963 | 0.627422 | 0.922360 / 0.079545 | 0 | 125 | no-op；后两档 mix 下降 |
| Human hippocampus | 2,500 | 7 | 0.596178 / 0.585490 | **0.599919 / 0.589516** | +0.003741 / +0.004026 | 0.587678 | 0.690629 | 0.784433 / 0.207666 | 15 | 30 | uniform 0.602963/0.593359、disabled 0.601848/0.594073 均支配 direct |

主结果没有空簇或 exact-K 失败。完整 cluster sizes 分别为：P22 `[1141,622,1520,789,1071,1522,549,1816,166]`；MISAR `[293,445,181,265,353,287,125]`；human `[30,58,295,660,159,873,425]`。

## 公式与匹配对照

对每个注册图尺度上的边 `(i,j)`，先在 Night-16H 的结构可行候选中计算

`p_ij = Σ_m q_m 1[z_mi = z_mj]`，

其中 `q_m` 只由候选锁定前的 molecular、topology 和 persistence 三个证据轴的 percentile-softmax 得到。entropy 形成 uncertainty `u_ij`，边因子为

`f_ij = (1-λ) + λ[ε + (1-ε) p_ij (1-u_ij)]`。

它只非负缩放原 Night-15F Potts capacity，因此仍可使用同一 alpha-expansion。λ 只取 0.25、0.50、0.75；ε=0.05，其余 base config 三条 lane 完全相同。persistence 是为了 faithful 重放 Night-17B posterior 的既有权重轴，不代表 Night-16H 已支持 persistence 贡献。

每个 λ 都同时生成：disabled、uniform-mass-matched、within-scale deterministic permuted、equal-weight analytic posterior 和 weighted direct posterior。匹配 arm 每尺度的 base-weighted mass 最大绝对误差为 `1.37e-12`，最大相对误差为 `4.68e-16`。

## 关键归因

- P22 说明“共聚类关系进入图割”可以改动少量 spot，但 weighted candidate evidence 没有贡献：weighted 与 equal-weight analytic byte-exact。
- Human 的增益主要可由总平滑容量/继承的 base energy 解释；关系位置并非必要。
- MISAR 的强起点是固定点，direct posterior 没有产生改动。
- 三条 producer 都读取 89 个候选；UNBIASED_BANK 实际为 P22 70、MISAR 70、human 58。Human 少于 70 是因为其 89 个候选中只有 77 个通过统一结构可行域，其中 58 个同时属于 unbiased start family；这来自锁定 authority，不是结果后删行。

## 为什么没有 Stage B

Stage B 的 edge learner 只有在 Stage A 至少 2/3 lane 双升且不被四个 matched controls 解释时才获授权。实际 lane pass 为 0/3，所以未训练 edge predictor、未追加阈值、未搜索新 selector、未跑 melanoma，也没有多 seed。这样避免用更复杂模型救一个已被对照否定的 target。

## 失败、修正和限制

- 第一次真实 producer 直接执行时缺少仓库根 `PYTHONPATH`，三条均在 import 前退出且未生成 partition；补上既有模块路径后完整重跑。
- P22/MISAR 的 unbiased 候选各为 70；human 因统一 feasibility 只有 58。预检查最初把 70 错写成全 lane 固定值，human fail-closed；修正为验证 89/89 完整对齐并记录实际 feasible-unbiased count，随后只完整重跑受影响 human lane。公式未变。
- 第一次 human evaluator 指向没有 annotation 的 numeric source H5AD，因缺 `true_label` fail-closed；改为已审计的官方 result/reference H5AD 后重新评价。没有修改分区。
- Fresh-process 三条 partition NPZ 均 byte-exact；评价表除不可复现的 `wall_seconds` 外，所有字段 exact。
- 本轮是 in-study teacher consumer diagnostic。候选 bank 中的关系来自同一 study 的多个锁定候选，即使 headline 使用 unbiased subset，也不能当外部泛化或新数据确认。

## 新颖性与数据审计

关系后验、候选集成、学习/手工 affinity、Potts/CRF 及 alpha-expansion 都有明确先例。本轮 clean-room 对象只是把 Night-16H 候选共聚类概率直接接到现有稀疏 cut；结果又未通过贡献门，因此不提出原创性主张。GSE213264 的空间 CITE-seq 路径可以闭合 RNA+protein 与生发中心生物学语义，但全域互斥 K-class 权威标注仍未闭合，不能制造 ARI/NMI ground truth；本轮没有下载数据。

## 导师汇报版

1. 本轮问的是：候选分区里隐含的“哪些空间邻边应当同域”能不能不经过神经表示，直接改善图割。
2. 我们固定了 Night-16H 的强表示、强起点和图割，只用三档统一关系强度，并配齐等总质量、随机位置、等权 posterior 和禁用关系对照。
3. P22 从 0.5875/0.7090 小升到 0.5886/0.7104，但等权 posterior 完全复现该结果，说明复杂候选权重没有贡献。
4. 人海马从 0.5962/0.5855 升到 0.5999/0.5895，但 uniform 和 disabled 都更高，说明增益主要不是关系边位置。
5. MISAR 完全不动，因此三条主 lane 的独立贡献门为 0/3。
6. 按预注册规则没有启动 edge learner，也没有靠扩大网格救结果。
7. Night-17B 到 Night-17F 的 relation-distillation 路线据此封闭，下一步不应再在同一关系 target 上继续包装。
8. 工程路径、标签后置、质量匹配和精确重放均闭合；终态为科学负结果，而不是实现失败。

