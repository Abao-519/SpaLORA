# SpaLORA Night-15D：可靠性质量保持、动态原型与多尺度稀疏能量

## 我现在需要知道的三件事

1. **问题是什么。** Night-15C 已证明直接 Potts-style 聚类能量有用，但它把每个 spot 的邻边总质量都归一成 1：即使所有邻边都很弱，也会被强迫接受一整份空间平滑。本轮要判断，保留“绝对可信边质量”、把被拒绝的质量退回自身，再让 RNA 与第二模态按局部 prototype 置信度贡献，能否在 RNA+ATAC 和 RNA+protein 同时继续提高绝对 ARI/NMI。
2. **实际做了什么。** 本轮没有重训大模型。它在冻结的强表示之后、最终聚类之前加入一个共享的 `reliability-energy` 组件超集：稀疏边可以保留绝对质量，拒绝的质量可形成 self-loop；两个模态各自产生动态簇原型代价，并可按局部 margin 连续融合；另有邻域均值、残差和多步 diffusion 组成的稀疏多尺度特征。九条 lane 都调用同一个核心，但离散模块与数值参数由每条公开 benchmark 的标签 HPO 从共享超集中选择，**不是模型自动 gate**。
3. **对论文意味着什么。** Night-15D 得到一个跨两类模态、9/9 lane 相对 Night-15C 稳定线均 ARI/NMI 双升的**公开 benchmark 开发里程碑**。它比 Night-15C 更强，也首次轻微抬动 A1 和 D1；但它不是盲测、不是 SOTA、不是 `CONFIRMED_MILESTONE`、不是 paper-ready。Potts/ICM、多尺度图特征、相似性边和 prototype unary 都有明确先例，目前只可主张组合机制及其跨家族经验信号。

## 明确终态

- 终态：`NIGHT15D_CROSS_FAMILY_RELIABILITY_ENERGY_DEVELOPMENT_MILESTONE`
- 分类：`LOCAL SIGNAL`
- 证据层级：`PUBLIC_BENCHMARK_DEVELOPMENT_MILESTONE`
- 贡献范围：`UNIFIED_CLUSTER_ENERGY_COMPONENT_SIGNAL`
- 相对 Night-15C 稳定线 ARI/NMI 双升：**9/9 lanes**
- 覆盖模态家族：RNA+ATAC、RNA+protein，2/2
- 最终 fresh-process replay：完整 9/9 重放两次，两次 partition 文件、数组 SHA、指标、cluster sizes 和 config 全部 exact
- targeted tests：10/10
- GPU：0 秒 / 0 MiB；dense N×N：0；新数据下载：0

按项目六类治理口径，本轮只能归为 `LOCAL SIGNAL`。“开发里程碑”只描述公开标签参与 HPO 后的可复算 9/9 开发结果；由于逐 lane 标签 HPO 且不是盲测，不能升格为 `CONFIRMED_MILESTONE`，更不等价于论文级确认。

## 绝对指标主表

| Lane | K | Night-15C 稳定 ARI/NMI | Night-15D ARI/NMI | ΔARI/ΔNMI | AMI | FMI | Moran / Geary | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P22 | 9 | .582526/.701575 | **.587186/.708094** | +.004660/+.006519 | .707589 | .648351 | .930700/.071804 | 主公开 K=9 协议 |
| P22 author assignment | 18 | .727678/.748984 | **.727952/.749674** | +.000274/+.000690 | .748045 | .757247 | .834625/.166877 | 同一 P22 的不同 K 协议，不是独立样本 |
| MISAR E15.5 S1 | 7 | .535477/.654013 | **.540665/.665900** | +.005188/+.011887 | .664140 | .632402 | .921524/.080253 | 公开 K=7 Y |
| MISAR prediction | 12 | .439555/.574298 | **.451195/.596512** | +.011640/+.022214 | .592886 | .558367 | .909716/.090179 | 仍以公开 K=7 Y 评价 12 簇，不是 exact SEPAR K12 annotation |
| A1 | 10 | .273021/.417368 | **.274380/.418549** | +.001359/+.001181 | .414753 | .415186 | .544252/.458917 | 首次双升，但幅度很小 |
| D1 | 10 | .243757/.380534 | **.245193/.383665** | +.001436/+.003132 | .379610 | .405514 | .480907/.525051 | 首次双升，但幅度很小 |
| tonsil s1 | 4 | .220592/.304094 | **.231668/.312166** | +.011075/+.008071 | .311563 | .499496 | .713455/.286732 | tonsil study block |
| tonsil s2 | 4 | .236471/.278675 | **.251476/.301339** | +.015006/+.022663 | .300753 | .510992 | .745450/.254039 | 4519 total / 4518 evaluated |
| tonsil s3 | 4 | .311243/.278154 | **.325526/.287363** | +.014284/+.009208 | .286680 | .608817 | .677423/.330950 | 4521 total / 4460 evaluated |

MISAR K=12 需要额外解释：Night-15C 稳定线是 `.439555/.574298`；更早可审计的 NMI 高位是 `.590609`。Night-15D 的 `.451195/.596512` 同时高于 Night-15C 稳定 ARI/NMI，也高于历史 NMI 高位约 `.005903`。不过标签协议仍不是 exact SEPAR K=12，不能与 SEPAR 原文数字作严格胜负。

逐 observation 数、ordered-ID SHA、cluster sizes、最终 config、partition SHA 和完整精度见 `absolute_metrics_main_table.csv`。

## 这次提分究竟来自哪一层

Night-15C 的 row normalization 会把任意非零邻边重新缩放到总质量 1。Night-15D 增加两个可靠性算子：

- `mass`：以原始空间 degree 为分母，弱 conductance 的邻边总质量可以小于 1；
- `self`：保留接受的邻边质量，同时把被拒绝的质量退回自身，相当于在不可信边附近保护当前分子证据。

另有两个可选组件：每个模态独立形成动态 prototype unary，再由局部 margin 连续融合；以及仅在稀疏图上计算邻域均值、低/高频残差与多步 diffusion 的多尺度特征。更新仍是同步条件更新，并非全局 MAP 或 graph-cut 最优保证。

Matched component 对照显示：

- A1/D1 中，`self` 明显优于同强度的 row 或 mass 版本，支持“拒绝质量回流自身”的机制线索；D1 还需要多尺度特征与 switch trust。
- P22 K=9 的增益依赖多尺度特征和 modality-margin unary；MISAR K=7 的 margin unary 也有独立正贡献。
- MISAR K=12 去掉 multiscale 后 ARI/NMI 分别回撤约 `.010861/.010527`，解释了为什么 Worker2 的 mass/multiscale-only粗探针为负，而更正交的共享组件组合仍可抬升。
- tonsil s1 主要由 sub-stochastic mass 与多模态边支持；s2 主要由 margin unary；s3 对 margin unary、自回流和多模态边均敏感。
- P22 K=18 只改变 5 个 observations，增益极小，不能把它包装成强机制证据。

完整 90 行 matched ablation 和 full-minus-ablation 差值分别在 `matched_component_ablation.csv` 与 `matched_component_summary.csv`。

## “同一方法”与“逐数据集 HPO”的边界

九条 lane 的 core、函数签名、稀疏图操作、prototype 代数和更新规则相同，core 不读取 dataset、tissue、family 或 label 字符串；不存在按数据集名称切换 backbone/flow。

但本轮没有冻结一个所有数据共用的单配置。每个 lane 在同一个 superset 内独立选择 edge rule、normalization、feature bank、unary mode、pairwise strength、temperature、trust 与 steps。选择由公开标签 HPO 完成。因此准确表述是“共享可靠性能量组件超集 + 逐 lane 开发 HPO”，不能写成自动内容门、统一零标签选择器或外部确认。

## 时序审计与工程修正

Worker2 发现最早的 `working/replay1`、`working/replay2` 完成时间早于最终源码和测试写入。根因是 AST firewall 把内部 cluster-assignment helper 的参数名 `labels` 也视作潜在标签输入；随后仅把该 helper 参数重命名为 `partition`，未改变数组、公式、config 或数值运算。

因此两份旧 replay 已明确标记为 **superseded**，没有被用作最终证据。最终源码 SHA 为 `8b212f95830f3895b534da487bb5d8d993ddc8a4c95faed37738a0b7816d7a0e`，其写入时间早于 `final_replay1`、`final_replay2` 与最终 tests。此后重新执行了两次完整 9/9 fresh-process replay，并再次运行全部 10 个 targeted tests；结果 exact。详见 `engineering_changelog.csv` 与 `exact_replay_audit.json`。

## 失败、限制与诚实边界

- Worker2 的 A1/D1 探针只是方向证据；formal 结果由独立 clean implementation 从权威 initial partition SHA 复算。
- Worker2 的 MISAR K=12 mass/multiscale-only probe 没有超过 Night-15C，负结果保留；本轮没有在同一邻域无意义扩网格，而是从共享 superset 测试正交的 multiscale/trust 组合。
- 43,150 条搜索行全部保留；matched ablation 90 行另存。没有删除坏配置或隐藏失败。
- 9 lanes 并非九个独立科学重复：P22 K=9/K=18 是同一切片；MISAR K=7/K=12 是同一切片；tonsil s1/s2/s3 属于一个 study block。
- A1、D1 与 P22 K=18 的增益很小，容易在新的冻结协议中消失；下一轮应优先做无回看确认，而不是继续在这些标签上细调。
- 本轮证明的是 clustering-energy 组件信号，不是新 backbone、端到端表示学习或完整 SpaLORA 方法已成功。

## 资源与复现

- 完成 search ledger：43,150 rows；matched component ablation：90 rows；最终 replay：18 rows。
- 搜索 wall sum：1,829.83 s；ablation 行 wall sum：6.05 s；两次最终 replay wall sum：5.88 s。
- 冻结 replay 实测 peak RSS：218.71 MiB。
- GPU：0 s / 0 MiB；AutoDL GPU run：0；新下载：0；dense N×N：0。
- 最终 replay：9/9 × 2 exact；tests：10/10。
- AutoDL 按夜间自主联动指令保持有卡开机，**未派发 shutdown、stop、poweroff 或 halt**。

## 导师汇报版

1. Night-15C 的普通 row-normalized Potts 已不再微调，本轮针对弱边也被强制平滑的问题做了语义修正。
2. 新组件允许保留绝对可信边质量，并把被拒绝的边质量退回自身，同时可用两个模态的 prototype 置信度和稀疏多尺度特征。
3. 在公开标签参与逐数据集 HPO 的口径下，RNA+ATAC 与 RNA+protein 共 9/9 lane 相对 Night-15C 都实现 ARI/NMI 双升。
4. P22 K=9 达到 `.5872/.7081`，MISAR K=7 达到 `.5407/.6659`，MISAR 12 簇预测达到 `.4512/.5965`。
5. A1/D1 首次双升，但只有千分位量级；P22 K=18 也仅是极小增益，不能夸大。
6. 两次最终 fresh-process 的分区 SHA 和指标完全一致，旧的时序不合格 replay 已作废并保留审计记录。
7. 每条 lane 使用同一个组件超集，但具体模块与参数由公开标签 HPO 选择，不是自动 gate，也不是盲测。
8. 当前可讲的是统一组合机制的跨家族开发信号；各基础零件都有先例，方法新颖性和外部泛化仍需下一轮冻结验证。

## 技术附录

源码碰撞边界、shared-superset 语义、冻结 registry、完整 ledger、replay、标签用途、资源、失败记录、Git bundle 和 Windows root-relative compact index 由同目录对应文件记录。最终 commit/tag 与 bundle/index SHA 写入 compact 生成后的 `delivery_manifest.json`，避免在被提交文件内制造自引用哈希。
