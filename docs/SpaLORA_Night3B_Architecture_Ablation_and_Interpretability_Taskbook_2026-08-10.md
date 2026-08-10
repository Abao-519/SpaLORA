# SpaLORA Night‑3B：架构/损失消融与可解释性任务书

日期：2026‑08‑10  
执行者：Codex（通过 SSH 操作 AutoDL）  
科学基线：commit `384e66587149a687b3eac4a6d1918d8d4972dc06`  
基线 tag：`night3af-final-20260810`  
建议分支：`revision/q2-night3b-ablation-interpretability-20260810`  
建议保护标签：`baseline/pre-night3b-20260810`  
最终标签：`night3b-final-20260810`  
输出根目录：`outputs/night3b_handoff`

## 0. 本轮唯一目标

Night‑3AF 已预注册通过，Night‑3B 不再搜索新创新点，也不调 IGE 参数。本轮只回答两个审稿问题：

1. 四个 loss term 和双层 attention 分别贡献了什么？
2. IGE 的数据自适应系数、weighted-gradient influence 和 spot-level attention 能否形成可信、可复现的解释证据？

本轮结束后只允许给出 `KEEP_FULL / SIMPLIFY_CANDIDATE / MIXED_EVIDENCE` 三种架构建议；不得在同一夜根据标签结果继续改模型或追加变体。

## 1. 不可违反的约束

### 1.1 禁止事项

- 不搜索或增加 seed；固定 `[0,1,2,3,4]`。
- 不读取标签决定模型、loss、attention、threshold、epoch 或变体。
- 不修改 IGE 公式、epsilon、weight sum、PCA、HVG、feature graph、cluster number 或 evaluator 既有指标。
- 不动态更新 IGE coefficient；IGE 仍是初始化计算、正式训练前冻结。
- 不恢复 ASR、low-expression rescue 或 gene-level weighting。
- 不调学习率、epoch、embedding dimension、loss factor 或 attention 温度。
- 不因某个数据集结果不好而停止、重跑、换 seed、删结果或增加补救变体。
- 不覆盖 Night‑3AF/Night‑3A‑R/Night‑3A/Night‑2C 的历史报告与结果。

### 1.2 允许事项

- 新增显式架构开关与 loss mask；
- 新增只读评价指标、attention 统计、图和报告；
- 修复导致测试或数值契约失败的实现错误，但必须记录；
- 对 registered loss-drop 允许被删除项 coefficient 精确为 0；其他项必须为有限正数。

## 2. 启动与保护

1. 验证 `night3af-final-20260810` 指向：

   `384e66587149a687b3eac4a6d1918d8d4972dc06`

2. 工作树若有未知修改，不得覆盖；先报告并停止。
3. 从该 tag 创建本轮分支与保护标签。
4. 对下列历史清单重新执行 SHA‑256：

   - Night‑3AF：既有内部 `SHA256SUMS`，期望 709/709；
   - Night‑3A‑R：211/211；
   - Night‑3A：198/198；
   - Night‑2C：913/913。

5. 将历史核验输出写入 `outputs/night3b_handoff/protection_preflight/`。
6. 继续使用 Night‑3AF 已发布的三个 immutable deterministic cache；禁止重新预处理。

如果基线 tag、cache manifest 或历史保护失败，才允许硬停止。

## 3. 固定的八个架构变体

运行顺序在标签隔离窗口开始前写入并锁定：

| 代码 | 定义 | 科学问题 |
|---|---|---|
| `FULL_IGE` | Night‑3AF 原始 corrected architecture + 四项 IGE | 同批可复现基线 |
| `DROP_RNA_RECON` | 删除 RNA reconstruction loss | RNA reconstruction 是否是主要增益来源 |
| `DROP_MOD2_RECON` | 删除 modality‑2 reconstruction loss | 第二模态重建是否必要 |
| `DROP_CORR1` | 删除 omics‑1 consistency/correspondence loss | Corr‑1 的独立贡献 |
| `DROP_CORR2` | 删除 omics‑2 consistency/correspondence loss | 后期 influence 较低的 Corr‑2 是否仍必要 |
| `UNIFORM_WITHIN` | 两个 within-modality attention 固定为 `[0.5,0.5]`；cross attention 保持学习 | spatial/feature 自适应融合是否必要 |
| `UNIFORM_CROSS` | within attention 保持学习；cross-omics attention 固定为 `[0.5,0.5]` | 模态自适应融合是否必要 |
| `UNIFORM_ALL` | 三个 attention 全部固定为 `[0.5,0.5]` | 双层 attention 整体贡献 |

总计：

`8 variants × 3 datasets × 5 seeds = 120 runs`

不得加入 spatial-only、feature-only、dynamic IGE、random weights、temperature sweep 或其他临时变体。本轮结果回来后再决定是否有必要追加。

## 4. Loss-drop 的确定公式

设 active loss 集合为 `S`，在对应架构、对应 seed 的同一初始化状态上计算 raw RMS gradient `g_i`。

对所有 `i ∈ S`：

```text
geo_S = exp(mean(log(g_i + eps), i in S))
r_i   = geo_S / (g_i + eps)
c_i   = 4 * r_i / sum(r_j, j in S)
```

对被删除的 loss：

```text
c_i = 0
```

因此：

- `FULL_IGE` 和三个 attention ablation 的四项初始 weighted-gradient share 各约 `1/4`；
- 四个 loss-drop 变体的三项 active weighted-gradient share 各约 `1/3`；
- 所有变体 active coefficient 总和固定为 4；
- loss-drop 不继承某个表现较好的 Night‑3AF seed 权重，不跨 seed 复制权重；
- 只允许使用该变体、该 seed、该初始化状态的 label-free gradient。

## 5. 架构开关实现要求

### 5.1 参数初始化保持可比

所有八个变体实例化相同参数集合、相同顺序、相同 Xavier 初始化。attention ablation 只改变 forward 中的组合规则，不删除参数对象，以保持共享参数的初始化和 RNG 流一致。

### 5.2 Uniform attention

每个 observation 的两路权重必须精确为：

```text
alpha[:, 0] = 0.5
alpha[:, 1] = 0.5
combined = 0.5 * first + 0.5 * second
```

禁止使用经过 softmax 的近似值、可训练常数或额外温度。

### 5.3 输出契约

保持原有输出 key 与 shape。uniform 分支也必须输出对应的 `alpha` 数组，以便解释与测试。

## 6. P0‑ARCH：运行前硬门

P0‑ARCH 不读取语义标签。

### 6.1 FULL_IGE 等价性

新 runner 的 `FULL_IGE` 必须与 Night‑3AF 在同一 cache、同一 seed、同一初始 state 下满足：

- CPU shared-forward：output、四项 raw loss、IGE coefficient、total loss、gradient、单步 Adam state 与更新参数 exact；
- GPU 独立 forward：落入 Night‑2C 已采用的数值包络；
- seed 0 的三数据集 coefficient 与 Night‑3AF `ige_weights.csv` 一致到既有 GPU 包络；
- 未启用任何架构开关时，模型 source path 与默认行为保持兼容。

若 FULL_IGE 等价性失败，硬停止，不运行 120 次实验。

### 6.2 变体数值探针

对 `8 variants × 3 datasets × seed 0 = 24 cells` 做探针：

- cache SHA 精确匹配；
- raw loss/gradient/coefficient/total loss 有限；
- active coefficient 有限且为正；drop coefficient 精确为 0；
- active coefficient sum 为 `4 ± 1e-6`；
- FULL/attention ablation 初始 active gradient share 各约 0.25；
- loss-drop 初始 active gradient share 各约 1/3；
- uniform alpha 必须逐元素 exact 0.5；
- probe 不改变 model、optimizer、`.grad` 或 RNG state；
- sparse adjacency 从 cache 到 forward 保持 sparse；
- 训练 payload 不含 label/ground-truth 字段。

24/24 与所有测试均通过后才授权主实验。

## 7. 标签防火墙

沿用 Night‑3AF 的 `ScientificWindow`，但修订 integrity check：

- 允许以“integrity only”读取 ground-truth 文件字节计算 SHA；
- 禁止解析 label column、obs label、类别数或任何语义值；
- 所有训练 run manifest、embedding、cluster、attention、loss trajectory、gradient trajectory 完成并 fsync；
- 先生成并锁定 `locked_120_run_manifest.json`；
- 再关闭 scientific window；
- 之后才允许 evaluator 读取标签。

标签防火墙应阻止语义访问，不得再次因单纯 SHA‑256 文件读取而硬停止。

## 8. 执行 120 次主实验

### 8.1 顺序

在运行前生成固定 ordinal，按以下层级排序：

1. dataset：`a1, placenta, p22`；
2. variant：按本任务书表格顺序；
3. seed：`0,1,2,3,4`。

禁止读取已有 ARI/NMI 后改变顺序或选择性重跑。

### 8.2 每个 run 的必需输出

- `embedding.npz`
- `attention.npz`
- `clusters.csv`
- `observation_ids.csv`
- `loss_trajectory.csv`
- `gradient_influence_trajectory.csv`
- `coefficient_probe.json`
- `checkpoint_index.csv`
- `run_manifest.json`
- `failure.json`（只在失败时）

Manifest 至少记录 dataset、variant、seed、ordinal、cache SHA、source SHA、initial/final state SHA、active loss mask、attention mode、coefficients、epochs、timing、CPU/GPU peak memory、artifact SHA。

## 9. 评价指标

在 120/120 锁定后，使用与 Night‑3AF 完全相同的：

- ARI、NMI、AMI、FMI、homogeneity、V-measure；
- Hungarian macro/weighted F1、balanced accuracy、per-domain F1；
- spatial neighbor agreement；
- mean one-vs-rest cluster Moran’s I；
- silhouette 与 Davies–Bouldin；
- runtime、peak GPU allocated/reserved、peak CPU RSS。

新增 evaluation-only 指标：

### 9.1 Mean one-vs-rest Geary’s C

对每个 predicted cluster 的 binary indicator 在同一 symmetric spatial kNN graph 上计算 Geary’s C，再对 cluster 取平均。更低表示局部空间连续性更强。

必须新增 dense-vs-sparse 单元测试，并报告 cluster 数不变、常数向量、空边图等边界情况。

### 9.2 Boundary disagreement

报告 `1 - spatial_neighbor_agreement`，仅作为同一量的可读表达，不得当作独立证据重复计数。

## 10. 配对消融表

对每个 ablation 计算：

```text
FULL_IGE - ABLATION
```

分别输出每个 dataset、每个 seed 的 ARI/NMI/空间指标差异，以及：

- 五 seed mean、SD、median、min、max；
- 5 个 seed 中 FULL 胜出的次数；
- 描述性 paired bootstrap 95% CI，固定 seed `20260810`、10000 次；
- 15 个 dataset-seed cell 的总体 win count；
- 三数据集等权 macro mean，不能按 spot 数加权。

五 seed 不是五个独立生物样本，不得把普通显著性检验 p-value 当作核心结论。

## 11. 预注册架构判读规则

这些规则只生成建议，不允许同夜继续改模型。

### 11.1 Simplification dominance

若某 ablation 同时满足：

1. 在至少 2/3 数据集上，平均 ARI 与 NMI 均不低于 FULL；
2. 三数据集等权 macro ARI 至少比 FULL 高 `0.01`；
3. 没有一个数据集同时出现 neighbor agreement 与 Moran’s I 各下降超过 `0.03`；

则标记为 `SIMPLIFY_CANDIDATE`。

### 11.2 Component support

若删除/均匀化某组件后，在至少 2/3 数据集出现以下任一项：

- FULL−ablation ARI `≥0.02`；
- FULL−ablation NMI `≥0.02`；

且 FULL 没有在对应数据集造成 neighbor 与 Moran 同时下降超过 `0.03`，则该组件标记为 `SUPPORTED`。

其他情况标记为 `MIXED`，不得写成“无贡献”。

### 11.3 方法级建议

- 没有 simplification dominance，且至少一个 attention 组件与至少两个 loss term 为 SUPPORTED：`KEEP_FULL`；
- 存在 simplification dominance：`SIMPLIFY_CANDIDATE`；
- 其余：`MIXED_EVIDENCE`。

无论分类为何，必须完成全部 120 runs、全部评价与报告。

## 12. 可解释性分析

只对 `FULL_IGE` 做主要解释；ablation 用于机制比较。

### 12.1 IGE coefficient 与 influence

输出：

- 每个 dataset/seed 的四项 coefficient；
- coefficient mean±SD、CV、min/max；
- 初始与后半程 weighted-gradient share；
- 四项 share 的 entropy 与最大/最小比；
- coefficient 与最终 ARI/NMI 的 seed-level Spearman 只作探索性描述，不据此调参。

### 12.2 Spot-level attention

对六个 attention 通道输出：

- mean、SD、median、IQR、5th/95th percentile；
- normalized two-way entropy；
- `<0.05` 或 `>0.95` 的 spot 比例；
- 同一 observation 跨 seed 的 Spearman stability；
- seed 0 与“ARI 距五 seed mean 最近、平局取最小 seed”两套预注册地图，禁止选最好 seed。

### 12.3 与 label-free QC 的关联

在不读取标签的阶段计算 attention 与以下变量的 Spearman 相关：

- RNA per-spot log library size、RNA feature L2 norm；
- modality‑2 per-spot row sum 与 L2 norm；
- spatial graph degree、feature graph degree；
- 两模态局部邻域一致性诊断。

使用 BH-FDR；同时报告相关系数，不只报告 q-value。Placenta 第二模态继续称为 `ATAC-derived / TF-associated regulatory features`，不得擅自改称原始 peak-level ATAC counts。

### 12.4 与 spatial domain 的事后关联

只能在 `locked_120_run_manifest.json` 生成后读取 ground truth：

- 每个 attention channel 按 annotated domain 画 violin/box plot；
- Kruskal–Wallis + BH-FDR；
- 报告 epsilon-squared effect size；
- 同时给出 per-domain 样本量；
- 不依据结果反向修改模型。

### 12.5 A1 空间权衡专项

因为 Night‑3AF 中 A1 IGE−C0 Moran’s I 为 `−0.04568`：

- 对 FULL 与三个 attention ablation 对比 Moran、Geary、neighbor agreement、ARI/NMI；
- 绘制固定 seed 0 与 mean-nearest seed 的边界图；
- 判断 ARI 增益是否来自更细边界、局部碎片化或 domain 合并/拆分；
- 报告 per-domain F1，不能只给总体 ARI。

### 12.6 P22 异质性专项

- 绘制五 seed 的 FULL−ablation ARI/NMI/Geary 分布；
- 报告 P22 中哪些 domain 的 F1 对 IGE 最敏感；
- 比较 UNIFORM_CROSS 与 UNIFORM_WITHIN，判断强 RNA/cross/spatial attention 是否与 ARI 下降相关；
- 禁止搜索“能让 P22 变好”的 seed 或 threshold。

## 13. 必需图表

1. `ablation_ari_nmi_forest.{png,pdf}`
2. `ablation_spatial_tradeoff.{png,pdf}`
3. `loss_term_ablation_heatmap.{png,pdf}`
4. `attention_ablation_heatmap.{png,pdf}`
5. `ige_coefficients_by_dataset.{png,pdf}`
6. `weighted_gradient_share_trajectories.{png,pdf}`
7. `attention_distribution_by_dataset.{png,pdf}`
8. `attention_entropy_and_stability.{png,pdf}`
9. `attention_qc_correlations.{png,pdf}`
10. `attention_domain_association.{png,pdf}`
11. `a1_boundary_tradeoff_maps.{png,pdf}`
12. `p22_seed_heterogeneity.{png,pdf}`

所有图必须使用全 seed；空间地图按预注册 seed 规则，不得 best-seed cherry-picking。

## 14. 必需交付物

至少生成：

- `night3b_report.md`
- `night3b_completion.json`
- `night3b_gate_status.json`
- `p0_arch.json`
- `locked_120_run_manifest.json`
- `scientific_window_label_firewall.json`
- `per_seed_metrics.csv`
- `summary.csv`
- `paired_ablation_deltas.csv`
- `component_support_matrix.csv`
- `per_domain_metrics.csv`
- `geary_metrics.csv`
- `ige_coefficients.csv`
- `gradient_influence_trajectories.csv`
- `attention_spot_summary.csv`
- `attention_seed_stability.csv`
- `attention_qc_correlations.csv`
- `attention_domain_association.csv`
- `resource_usage.csv`
- `failure_index.json`
- `protocol_deviations.json`
- `tests_final.log`
- `SHA256SUMS`

## 15. 报告必须明确回答

1. P0‑ARCH 是否通过，24/24 探针是否通过？
2. 120/120 是否完成，failure JSON 是否为 0？
3. 训练期间 semantic label access 是否为 0？
4. FULL_IGE replay 相对 Night‑3AF 是否一致？
5. 四项 loss 的删除分别造成什么变化？
6. within、cross、all-uniform attention 分别造成什么变化？
7. 方法级建议是 KEEP_FULL、SIMPLIFY_CANDIDATE 还是 MIXED_EVIDENCE？
8. A1 Moran 下降得到什么解释？
9. P22 ARI 异质性由哪些 domain/attention 分支驱动？
10. spot-level attention 是否稳定且具有 domain/QC 关联？
11. IGE 的额外运行时间与 peak memory 是多少？
12. 是否发生任何偏离、调参、seed 搜索或历史文件变化？

## 16. Git、归档与关机

1. 全部测试、实验、评价、图和报告完成后提交 Git。
2. 创建 tag `night3b-final-20260810`。
3. 检查工作树 clean、tag 指向最终 commit。
4. GitHub 只尝试 push 一次；若仍因 HTTPS 凭据失败，记录并停止重试。
5. 生成完整 Git bundle 并执行 `git bundle verify`，再临时 clone/checkout 最终 tag。
6. 生成 artifacts archive、内部 `SHA256SUMS`，解压到新临时目录逐项验证。
7. 将报告、CSV、图、bundle、archive、delivery index 下载到本地持久位置并双端校验。
8. 生成 shutdown checklist 与 confirmation。
9. `/usr/bin/shutdown` 必须是最后一条远端命令。
10. SSH 断开后不得重连验证。

## 17. 给规划者的最终摘要模板

```text
Night‑3B 已严格完成并关机。
P0‑ARCH：PASS/FAIL；探针：x/24。
主实验：x/120；failure JSON：x；测试：x passed, x failed。
训练期 semantic label access：0/存在。
FULL_IGE replay：一致/存在差异（说明）。
方法级架构建议：KEEP_FULL / SIMPLIFY_CANDIDATE / MIXED_EVIDENCE。
四项 loss support：RNA ...；Mod2 ...；Corr1 ...；Corr2 ...。
Attention support：within ...；cross ...；all ...。
A1 ARI/Moran/Geary 结论：...。
P22 seed/domain 异质性结论：...。
IGE coefficient、gradient influence 与 attention 解释结论：...。
最终 commit/tag：... / night3b-final-20260810。
GitHub push：成功/失败原因；bundle/archive 校验：...。
/usr/bin/shutdown 已作为最后一条远端命令；之后未重新连接。
```

