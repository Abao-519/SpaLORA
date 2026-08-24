# SpaLORA Night-16A 报告：高分驱动的自校准统一空间能量

## 我现在需要知道的三件事

1. **绝对分数确实继续前进了，但这是内部开发上限。** 9 条协议中 6 条至少一项刷新、6 条 ARI/NMI 双升；最重要的是 D1 从可信无微小簇的 `0.288876/0.413762` 提到 `0.338775/0.435972`，最小簇 51，ARI 距公开 ARISE context `0.3427` 约 0.0039。这个 D1 结果来自公开标签驱动的跨运行筛选，不能当成自动模型输出。
2. **真正的自校准链在工程上闭合、科学上没有追上。** 它把样本规模、有效秩、跨模态边重叠/冲突、图尺度稳定性、multi-start 稳定性和形态可靠性转成同一套能量参数，再用整组留出研究的 Ridge 校准器选最终 candidate。9/9 partition 和 9/9 校准器选择都在两个新进程中精确重放，但严格自动输出在 9/9 协议上都低于旧 best-so-far。
3. **对论文的含义是“分数前沿推进 + 自校准机制负结果”。** Night-16A 找到了更好的 D1 非病理性开发分区，也完成了统一消费者和标签后置评价的真实框架；但初始化/selector 仍无法无标签复现 oracle 高分，因此不能登记 `SELF_CALIBRATING_METHOD_SIGNAL`、`CONFIRMED MILESTONE` 或 paper-ready。

## 明确终态

- 主分类：`SCORE_FRONTIER_ADVANCE` + `LOCAL_SIGNAL`
- 严格自校准组件：`SCIENTIFIC_NEGATIVE`
- 状态：`NIGHT16A_SCORE_FRONTIER_ADVANCE_WITH_SELF_CALIBRATION_GAP`
- 证据层级：公开 benchmark 开发；不是盲测，不是 SOTA 声明。

## 本轮改了模型的哪一层

Night-16A 没有再造数据集专用出口，而是在 Night-15F/15G 的连续多尺度能量前增加两层统一消费者：第一层由可观测统计量生成 modality logits、三尺度权重、unary 温度、pairwise、self-return、size prior、trust 和 optional-view 权重；第二层在同一 KMeans/GMM start 生成器上，用跨研究留出的 Ridge checkpoint 按结构质量、prototype margin、模态共同支持和簇平衡选最终 partition。RNA+protein 与 RNA+ATAC 使用相同公式；只有冻结的 family-level 数值 checkpoint 可不同。图像缺失时 optional weight 精确归零。

## 绝对指标主表

`内部最高`使用公开标签做跨运行开发选择；`严格自动`在当前整组研究上先冻结 partition，再由独立 evaluator 打分。

| 数据集 | N/eval | K | 旧 best ARI/NMI | 新内部最高 ARI/NMI | 内部 ΔARI/ΔNMI | 严格自动 ARI/NMI | 自动 ΔARI/ΔNMI | 自动 AMI/FMI | 自动 Moran/Geary | 内部/自动最小簇 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 lymph node K10 | 3484/3484 | 10 | 0.2760/0.4217 | 0.2760/0.4217 | 0.00000/0.00000 | 0.1885/0.3279 | -0.08745/-0.09381 | 0.3236/0.3353 | 0.4824/0.5164 | 114/98 |
| D1 lymph node K10 | 3359/3359 | 10 | 0.2889/0.4138 | 0.3388/0.4360 | 0.04990/0.02221 | 0.1336/0.2570 | -0.15524/-0.15680 | 0.2522/0.2790 | 0.3521/0.6482 | 51/130 |
| tonsil s1 K4 | 4326/4326 | 4 | 0.2362/0.3169 | 0.2365/0.3171 | 0.00032/0.00023 | 0.1015/0.2303 | -0.13469/-0.08662 | 0.2296/0.4237 | 0.6979/0.3001 | 478/356 |
| tonsil s2 K4 | 4519/4518 | 4 | 0.2576/0.3119 | 0.2583/0.3143 | 0.00070/0.00245 | 0.1471/0.2277 | -0.11045/-0.08415 | 0.2271/0.4368 | 0.6527/0.3464 | 445/723 |
| tonsil s3 K4 | 4521/4460 | 4 | 0.3499/0.3093 | 0.3506/0.3098 | 0.00076/0.00050 | 0.0681/0.1787 | -0.28179/-0.13061 | 0.1778/0.4777 | 0.7138/0.2937 | 294/138 |
| P22 K9 | 9196/9196 | 9 | 0.5939/0.7142 | 0.5940/0.7145 | 0.00005/0.00027 | 0.4418/0.5756 | -0.15213/-0.13865 | 0.5749/0.5265 | 0.8565/0.1460 | 167/159 |
| P22 author assignment K18 | 9196/9196 | 18 | 0.7397/0.7542 | 0.7411/0.7543 | 0.00138/0.00012 | 0.3606/0.5451 | -0.37908/-0.20907 | 0.5422/0.4278 | 0.4406/0.5660 | 24/39 |
| MISAR E15.5 K7 | 1949/1949 | 7 | 0.5414/0.6668 | 0.5414/0.6668 | 0.00000/0.00000 | 0.3489/0.5270 | -0.19257/-0.13979 | 0.5244/0.4830 | 0.9130/0.0917 | 125/71 |
| MISAR E15.5 K12 | 1949/1949 | 12 | 0.4532/0.5986 | 0.4532/0.5986 | 0.00000/0.00000 | 0.3186/0.5451 | -0.13456/-0.05357 | 0.5411/0.4423 | 0.8509/0.1525 | 42/64 |

内部 D1 的完整簇大小为 `[1081,86,1243,116,193,292,58,188,51,51]`；旧 singleton 高分 `0.350729/0.416032` 仅保留为病理性参照。P22 K18 的内部 balanced profile 最小簇为 24（低于 N 的 1%），因此即使 ARI/NMI 较高也明确登记为 small-cluster sensitivity。所有 lane 的完整簇大小、AMI/FMI、Moran/Geary 和 partition SHA 在 `main_results_table.csv`。

## 自动模型与内部最优的差距

严格 family-level multi-start 的主要结果为：A1 `0.1885/0.3279`、D1 `0.1336/0.2570`、P22 K9 `0.4418/0.5756`、P22 K18 `0.3606/0.5451`、MISAR K7 `0.3489/0.5270`、MISAR K12 `0.3186/0.5451`、tonsil s1/s2/s3 分别 `0.1015/0.2303`、`0.1471/0.2277`、`0.0681/0.1787`。这不是小幅不稳定，而是系统性的 initialization/selection gap。统一 generic medoid/consensus 也没有恢复 Night-15F/15G 的高质量 authorities。

## 分组转移与消费者对照

- lymph-node A1+D1、tonsil s1/s2/s3、P22 K9/K18、MISAR K7/K12 都按整组留出；同一 study 的次级 K 没被当作额外独立数据。
- family constants 相对 global constants 有时改善，但仍没有一个 study 保持旧 best；因此 family calibration 也不能升格为迁移信号。
- multi-start 相对固定 `KMeans(retained, seed=0)` 有混合结果：它改善 D1、MISAR K7、tonsil s2/s3，但固定单起点在 P22 K9/K18 和 tonsil s1 更高。初始化银行本身尚未被可靠排序。
- 详细 study-balanced 均值见 `grouped_transfer_summary.csv`；逐 lane global/family/single/multi 见 `consumer_and_calibration_scope_ablation.csv`。

## 关键 matched ablation

- 去掉 self-return 在部分 lane 分数上升，但 D1 最小簇降到 3、P22 K18 降到 1且指标严重崩塌；它是重要的稳定器，却不是普适增益来源。
- 固定单尺度和固定等模态权重多数与 full 接近，说明当前严格 selector 尚未真正利用丰富的连续校准自由度。
- A1、D1、tonsil s3 的 morphology missing/permuted 对严格自动输出几乎无影响；自动链没有重现 Night-15G 的形态增益。
- 这组结果支持“统一公式已实现”，不支持“每个子机制均有独立普适贡献”。

## 参数压缩与统计量映射

Night-15G 可暴露最多 38 个嵌套数值字段；Night-16A 用 7 个全局/家族控制量生成全部下游能量参数。严格 selector 使用 11 个结构描述量和 9 个 start-class 指示量，且对当前 held-out study 不打开指标。零值比例目前只记录、未进入最终公式；这一点已在 `data_statistics_to_parameters.csv` 如实标为 inactive。

## 真实 P0 与复现

| lane | retained | RNA/view1 | second/view2 | optional | registered graph [N,N,nnz] | byte exact |
|---|---:|---:|---:|---:|---:|---:|
| A1 lymph node K10 | [3484, 64] | [3484, 30] | [3484, 30] | [3484, 48] | [3484, 3484, 24178] | true |
| D1 lymph node K10 | [3359, 60] | [3359, 30] | [3359, 30] | [3359, 48] | [3359, 3359, 23310] | true |
| MISAR E15.5 K7 | [1949, 64] | [1949, 30] | [1949, 50] | None | [1949, 1949, 13666] | true |
| MISAR E15.5 K12 | [1949, 64] | [1949, 30] | [1949, 50] | None | [1949, 1949, 13666] | true |
| P22 K9 | [9196, 64] | [9196, 30] | [9196, 50] | None | [9196, 9196, 64466] | true |
| P22 author assignment K18 | [9196, 64] | [9196, 30] | [9196, 50] | None | [9196, 9196, 64466] | true |
| tonsil s1 K4 | [4326, 16] | [4326, 30] | [4326, 30] | None | [4326, 4326, 29868] | true |
| tonsil s2 K4 | [4519, 60] | [4519, 30] | [4519, 30] | None | [4519, 4519, 31286] | true |
| tonsil s3 K4 | [4521, 60] | [4521, 30] | [4521, 30] | [4521, 48] | [4521, 4521, 28312] | true |

- 两次最终 9/9 energy/partition fresh-process replay：精确一致。
- 两次冻结 Ridge checkpoint/selection replay：9/9 candidate ID 精确一致，4 个 held-out-study checkpoint 均重新加载。
- 针对性测试：5/5；最终 replay 9 lanes 合计约 8.10 秒，实测峰值 RSS 377.79 MiB；GPU 时间和峰值显存均为 0。

## 失败、限制和论文边界

1. 自校准 selector 在所有 9 条协议上都没有达到旧 best；最主要瓶颈是从通用 starts 找不到开发高分 authority，而不是能量重放失败。
2. D1 新高依赖 Night-15G 已经由公开标签筛过的形态分区再做结构修复，所以只属于内部 score frontier。
3. internal arena 曾发现 writer 元数据 bug；v1 输出已保留为 superseded，strict/fixed 全部重跑 v2，内部 ledger 的参数列从权威 registry 恢复，partition 和指标未改。
4. optional morphology 在严格自动链中没有可见增益；不能把 Night-15G 局部形态信号推广成自动模型结论。
5. Potts/CRF、alpha-expansion、多尺度图、prototype unary、图像融合和 meta-calibration 均有明确先例；本轮自动性能为负，不做新颖性锁定。

## 导师汇报版

1. Night-16A 同时做了两件事：继续刷新公开 benchmark 开发分数，并把历史逐数据集参数整理成统一的统计量驱动能量。
2. D1 获得了本轮最有价值的进展：无极小簇 ARI/NMI 达到 `0.3388/0.4360`，ARI 已接近公开 ARISE context。
3. 但这个高分仍是公开标签辅助开发结果，不是自动校准器自己找到的。
4. 严格自动链对 lymph node、tonsil、P22 和 MISAR 都做了整组留出，当前数据集标签只在 partition 锁定后评价。
5. 工程上它已 9/9 完整运行，并完成 partition 与校准器 checkpoint 的双重 fresh-process 重放。
6. 科学上自动结果在 9/9 协议都低于旧 best，说明统一初始化和无标签 final selector 仍是核心瓶颈。
7. 因此本轮应记为“分数前沿推进 + 局部信号”，自校准方法本身是科学负结果。
8. 下一阶段若继续，应优先研究能否在不继承标签筛选 authority 的前提下生成高质量 starts，而不是继续加复杂的能量参数。

## 技术附录

- 父级：`750d3fe9df52208b7618b878eb2701cd613b23b0` / `night15g-final-20260824`
- 分支：`revision/q2-night16a-score-driven-self-calibrating-energy-20260824`
- final tag：`night16a-final-20260824`
- 最终 commit、bundle SHA、compact N/N 与 index SHA 在 commit 后生成的 `delivery_verification.json` 中记录。
- 全部工作 ledger 保留；compact 不含 raw、embedding、partition array、image 或 checkpoint binary。
