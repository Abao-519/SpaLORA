# SpaLORA Night-15F — 连续多尺度图能量与大邻域求解分数冲刺

## 我现在需要知道的三件事

1. **问题**：Night-15E 已有的多模态边界、prototype 和可靠性质量保持，换成一次可移动整批观测的“大邻域”求解后，能否带来实质分数，而不只是再磨局部参数。
2. **实际做了什么**：在同一连续能量中同时放入三个已登记稀疏图尺度、动态多模态 prototype unary、冲突感知 pairwise、显式 rejected-mass stay cost 与簇大小弱正则；再用 sparse s-t min-cut 完成 alpha-expansion。公开标签只在候选运行结束后用于 known-K、跨运行 HPO 和评价，未进入 unary、edge、energy 或 move acceptance。
3. **对论文意味着什么**：9/9 lane 的公开开发 balanced profile 均相对 Night-15E ARI/NMI 双升，P22 K=18 与 tonsil s3 是本轮最明显增益；但这是逐 lane 标签 HPO 的 development ceiling。大邻域求解只在 5/9 lane 有 matched 独立增益，尚不足以升格为盲测、SOTA、`CONFIRMED_MILESTONE` 或 paper-ready 证据。

## 明确终态

- **Classification：`LOCAL SIGNAL`**
- Status：`NIGHT15F_MULTISCALE_EXPANSION_ENERGY_DEVELOPMENT_SIGNAL`
- Evidence tier：`PUBLIC_BENCHMARK_DEVELOPMENT_SCORE_CEILING`
- Method subtype：`DIRECT_CLUSTER_ENERGY_SIGNAL_WITH_PARTIAL_LARGE_NEIGHBORHOOD_SUPPORT`

## 第一次出现的术语

- **Unary（单点代价）**：每个 spot 若属于每个 cluster，需要支付多少由 retained、RNA 和第二模态 prototype 决定的分子代价。
- **Pairwise（邻边代价）**：稀疏空间邻边两端被分到不同 cluster 时支付的代价；边权由两个模态的一致、冲突和绝对可靠性共同决定。
- **Alpha-expansion（大邻域扩张）**：固定一个 cluster `alpha`，一次最小割可让任意一批 spot 同时转入它，而不是 ICM 每次只改一个 spot。它是经典计算机视觉先例，不是本轮原创。
- **Rejected-mass stay cost（拒绝质量留驻代价）**：不可靠边被拒绝的传播质量不再被 row-normalization 强行放大，而是显式形成“离开当前状态要付费”的连续 unary；每个动态 outer cycle 随当前分区刷新。

## Balanced absolute metrics（主结果）

| Lane | K | Night-15E ARI/NMI | Night-15F ARI/NMI | ΔARI/ΔNMI | AMI | FMI | Moran | Geary | 改动 spot | 最终簇大小 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | 10 | 0.275293/0.419908 | 0.275543/0.420138 | +0.000250/+0.000230 | 0.416352 | 0.416170 | 0.549704 | 0.453505 | 4 | [275,775,575,181,311,793,159,109,121,185] |
| D1 | 10 | 0.254658/0.389044 | 0.255044/0.389356 | +0.000386/+0.000311 | 0.385363 | 0.410256 | 0.517906 | 0.487816 | 2 | [44,246,227,210,1,479,140,822,145,1045] |
| MISAR K=7 | 7 | 0.540932/0.666617 | 0.541424/0.666798 | +0.000492/+0.000181 | 0.665042 | 0.633032 | 0.922608 | 0.079241 | 2 | [295,438,178,267,355,291,125] |
| MISAR K=12 | 12 | 0.452367/0.597708 | 0.453176/0.598629 | +0.000809/+0.000921 | 0.595027 | 0.560140 | 0.912409 | 0.086473 | 27 | [281,64,478,135,42,147,143,108,63,242,68,178] |
| P22 K=9 | 9 | 0.589191/0.710042 | 0.593912/0.714243 | +0.004721/+0.004201 | 0.713749 | 0.654072 | 0.929483 | 0.073216 | 133 | [1147,620,1490,793,1033,1519,614,1814,166] |
| P22 K=18 | 18 | 0.728307/0.750114 | 0.739685/0.754218 | +0.011378/+0.004104 | 0.752612 | 0.768065 | 0.869740 | 0.132359 | 722 | [764,2276,1013,578,624,593,591,174,45,23,451,143,356,142,26,431,680,286] |
| tonsil s1 | 4 | 0.233825/0.314270 | 0.236220/0.316884 | +0.002395/+0.002614 | 0.316285 | 0.503050 | 0.726051 | 0.273928 | 18 | [478,1300,1142,1406] |
| tonsil s2 | 4 | 0.254798/0.305252 | 0.257560/0.311871 | +0.002762/+0.006619 | 0.311294 | 0.514929 | 0.765454 | 0.233830 | 50 | [1012,1395,443,1669] |
| tonsil s3 | 4 | 0.328311/0.289524 | 0.341107/0.300966 | +0.012797/+0.011442 | 0.300311 | 0.611217 | 0.693268 | 0.313890 | 91 | [2508,567,1169,277] |

RNA+protein 五条 lane 的简单均值增益为 `+0.003718 ARI / +0.004243 NMI`；RNA+ATAC 四条为 `+0.004350 / +0.002352`。这是 lane mean，不是独立样本数统计；P22 两种 K 与 MISAR 两种 K 各自共享同一数据。

### 脆弱性

D1 保留了一个 singleton：最终簇大小含 `1`。它来自 Night-15E authority 的已有小簇结构，本轮仅改 2 个 spot，没有回填或隐藏。这一 lane 的微增不能被当成稳健聚类改善。

## Balanced、max-ARI 与 max-NMI 分开

| Lane | Balanced ARI/NMI | Max-ARI profile | Max-NMI profile |
|---|---:|---:|---:|
| A1 | 0.275543/0.420138 | 0.276004/0.419256 | 0.274785/0.420355 |
| D1 | 0.255044/0.389356 | 0.267516/0.381042 | 0.254633/0.389439 |
| MISAR K=7 | 0.541424/0.666798 | 0.541424/0.666798 | 0.540385/0.667802 |
| MISAR K=12 | 0.453176/0.598629 | 0.457408/0.595854 | 0.453176/0.598629 |
| P22 K=9 | 0.593912/0.714243 | 0.593912/0.714243 | 0.593912/0.714243 |
| P22 K=18 | 0.739685/0.754218 | 0.739685/0.754218 | 0.739685/0.754218 |
| tonsil s1 | 0.236220/0.316884 | 0.236220/0.316884 | 0.236220/0.316884 |
| tonsil s2 | 0.257560/0.311871 | 0.257560/0.311871 | 0.257560/0.311871 |
| tonsil s3 | 0.341107/0.300966 | 0.342806/0.300504 | 0.341107/0.300966 |

D1 max-ARI 是明显 ARI/NMI 权衡，不能写成双指标胜；A1 与 tonsil s3 也分别保留了单指标 profile。Balanced 是唯一主 profile。

## Matched contribution audit

所有对照从相同 Night-15E initial partition、相同 frozen numeric config 与相同输入开始，只替换一个组件。

| Ablation | Full 双指标更好 | Exact tie | Ablation 双指标更好 | 结论 |
|---|---:|---:|---:|---|
| `SINGLE_SITE_SAME_ENERGY` | 5/9 | 4/9 | 0/9 | alpha-expansion 在 MISAR K12、P22 K9/K18、tonsil s1/s2 有独立增益，其他 lane 不支持额外价值。 |
| `REGISTERED_SCALE_ONLY` | 9/9 | 0/9 | 0/9 | 三尺度连续混合在冻结配置上得到一致支持。 |
| `NO_SELF_RETURN_STAY` | 9/9 | 0/9 | 0/9 | 显式 stay cost 是必要的稳定器。 |
| `NO_SIZE_PRIOR` | 3/9 | 3/9 | 1/9 | 大小项只在部分 lane 有用，非普适机制。 |
| `PAIRWISE_ZERO_KEEP_STAY` | 8/9 | 0/9 | 1/9 | pairwise 对多数 lane 有用，但 tonsil s3 不支持。 |
| `PURE_DYNAMIC_UNARY` | 9/9 | 0/9 | 0/9 | 仅靠动态 prototype unary 不足。 |

tonsil s3 的 full 是 `0.341107/0.300966`；去掉 size prior 后仍为 `0.330677/0.290924`，依然高于 Night-15E `0.328311/0.289524`，所以不是“只靠簇均衡”。但 pairwise 置零后为 `0.341188/0.301228`，略高于 frozen full，说明该 lane 的主要贡献更接近多尺度 unary + stay + size，而非显式邻边耦合。

## 求解语义与复现

- `n<=8` 的任意稀疏/有孔随机图上，对每个 alpha binary subspace 做了穷举；quantized s-t cut 与穷举最优在 `2e-7` 容差内一致。
- 每个 accepted alpha move 都以原始浮点能量复核，只有严格下降才接受；2 次 fresh-process replay 均为 9/9 partition SHA、ARI、NMI、cluster sizes exact。
- 动态 prototype unary 会在 outer cycle 之间重算。因此只声明**每个 frozen-unary cycle 内** `end <= start`；不声明跨 cycle 的单一全局能量单调，也不声明全局最优。
- 5415 个候选行全部保留，failed row 为 0；工程错误发生在产生正式候选行前或使受影响脚本完整重跑。

## 资源

- 搜索位置：Windows 65 MB local compute kit，CPU only。
- 记录的候选计算时间合计：895.12 秒；GPU time 0，peak GPU 0 MiB。
- Peak RSS 未做进程级仪器化，诚实登记为 `NOT_INSTRUMENTED`；宿主 RAM 15.19 GiB。
- dense N×N：0。
- AutoDL 未承担本轮科学搜索；按夜间联动指令保持有卡开机，`shutdown_dispatched=false`。

## 最重要限制

1. 9/9 来自逐 lane 公开标签 HPO，不是一个全局自动 policy，也不是 held-out blind confirmation。
2. A1 仅有万分位级双升，仍远低于公开 ARISE context ARI 约 0.3427；本轮没有缩小到可宣称竞争的程度。
3. Alpha-expansion 的独立贡献是 5/9，而非 9/9；普通 Potts、graph cut、prototype 与 multiscale 均有明确先例。
4. 现有 lane 全部参与开发，尚无新冻结外部有标签单元。
5. 已确认 GSE263617 A1/D1 官方页面存在小型 H&E/position 资产，但按指令留给下一轮 morphology-view 研究，本轮未下载、未进入模型。

## 导师汇报版（7句）

1. 这轮把之前有效的多模态边界与可靠性机制写进同一套连续稀疏聚类能量，并用计算机视觉的 alpha-expansion 做大邻域更新。
2. 公开 benchmark 开发口径下，9 条 lane 相对 Night-15E 的 ARI/NMI 都是双正，其中 P22 K18 与 tonsil s3 的提升最明显。
3. P22 K9 达到 `0.5939/0.7142`，P22 K18 达到 `0.7397/0.7542`，tonsil s3 达到 `0.3411/0.3010`。
4. 与同能量的单点更新相比，大邻域求解在 5 条 lane 更好、4 条相同，因此它是部分成立的组件，而不是普适答案。
5. 三尺度混合和 rejected-mass stay cost 的 matched 对照较一致；size prior 与显式 pairwise 只在部分数据成立。
6. 这些值是逐数据集公开标签 HPO 的开发 ceiling，不是盲测、SOTA 或论文完成。
7. 下一步最合理的是用已闭合的 A1/D1 H&E 资产测试一个可缺失的 morphology 第三视图，同时保留本轮能量作为统一后端。

## 技术附录摘要

- Parent：`2504391f1ea9d6ae02b7b5b958140eba396cd56d` / `night15e-final-20260824`
- Branch：`revision/q2-night15f-multiscale-expansion-energy-20260824`
- Intended final tag：`night15f-final-20260824`
- Tests：7/7；fresh-process replay：2/2 × 9/9。
- Night-15E authority：39/39，index SHA `95f57bb955d929946587c04d700a42e3e3970fe68e7a1c21ea844a9ebbabb64a`，bundle SHA `5bf4c580a53ca274e5478160d83123ddffc54f9e391830bed18af13d1c17098c`。
- Final commit、bundle 与 Windows compact hashes 在提交完成后的 delivery manifest / compact index 中登记。
- AutoDL：`KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS`，未派发 shutdown。
