# SpaLORA Night-15C：直接聚类能量与边界保护分数冲刺

## 我现在需要知道的三件事

1. **问题是什么。** Night-15B 已证明 SAPR 的伪分区残差路线没有新增证据。本轮改问一个更直接的问题：保留已有强分子表示不重训，只在最终聚类阶段同时考虑“每个 spot 更像哪个簇”和“空间邻边是否应支持同簇”，能否稳定提高绝对 ARI/NMI。
2. **实际做了什么。** `unary`（一元项）是每个 spot 到各分子簇中心的代价；`pairwise`（二元项）是相邻 spot 之间的支持；Potts/CRF 是把两者放进同一个离散聚类更新规则；“各向异性边界”表示不同空间边有不同平滑强度，而不是整张切片一刀切地平滑。本轮在 65 MB local compute kit 上比较了静态 Potts、双模态边界 Potts、稀疏谱、伪 Fisher、稀疏分区集成和动态 centroid-unary Potts，共保留 **68,798** 行完成记录，并对九条最终配置各做两次 fresh-process exact replay。
3. **对论文意味着什么。** 结果分类为 **`DIRECT_CLUSTER_ENERGY_SIGNAL`**：同一直接能量形式在 RNA+ATAC 和 RNA+protein 都有稳定增益，但它仍是公开标签驱动 HPO 的开发信号，不是盲测、不是 SOTA，也不是论文方法已经成立。最稳妥的下一步是把它当作可复用 clustering head，做冻结超参数、多 seed/切片确认和与现有 CRF/BANKSY/动态图方法的公平消融。

## 明确终态

- 终态：`NIGHT15C_DIRECT_CLUSTER_ENERGY_SIGNAL`
- 分类：`DIRECT_CLUSTER_ENERGY_SIGNAL`
- 稳定双指标上涨：6/9 lanes
- 稳定仅 ARI 上涨：1/9 lanes
- 公开标签 development HPO 在统一网格中选择 registered no-op：2/9 lanes；这不是模型内容自适应 gate
- 两个模态家族均有信号；RNA+protein 的正证据来自同一 tonsil study block，A1/D1 没有新增，不能把三个 tonsil slice 当三次独立外部确认。

## 绝对指标主表

稳定结果采用 full-SVD、`OMP/MKL/OPENBLAS=1`；tonsil s3 采用不做 PCA 的标准化双模态 60D。每行做了两次独立进程回放，partition SHA 与全部指标逐字节一致，所以 BEST/median/mean/min 相同。

| Lane | K | 历史高位 ARI/NMI | 稳定 BEST=median=mean=min ARI/NMI | ΔARI/ΔNMI | AMI | FMI | Moran / Geary | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P22 | 9 | .569117/.685131 | **.582526/.701575** | +.013409/+.016444 | .701059 | .644281 | .934619/.067192 | 双升 |
| P22 author assignment | 18 | .687795/.723330 | **.727678/.748984** | +.039884/+.025653 | .747350 | .757002 | .833270/.168239 | 双升；不同 K 协议 |
| MISAR E15.5 S1 | 7 | .514300/.628963 | **.535477/.654013** | +.021177/+.025049 | .652187 | .628128 | .919626/.081798 | 双升 |
| MISAR prediction K=12 | 12 | .414275/.590609 | .439555/.574298 | +.025280/-.016311 | .570467 | .548103 | .897260/.102379 | 仅 ARI；仍用公开 K7 Y 评价，不是 exact SEPAR K12 annotation |
| tonsil s1 | 4 | .207947/.296928 | **.220592/.304094** | +.012645/+.007166 | .303485 | .491964 | .698354/.301921 | 双升 |
| tonsil s2 | 4 | .213006/.267406 | **.236471/.278675** | +.023465/+.011269 | .278072 | .500157 | .710889/.288577 | 双升；4519 total / 4518 eval |
| tonsil s3 | 4 | .196933/.249779 | **.311243/.278154** | +.114309/+.028375 | .277462 | .600989 | .682160/.326215 | 双升；4521 total / 4460 eval |
| A1 | 10 | .273021/.417368 | .273021/.417368 | 0/0 | .413563 | .414133 | .531123/.472025 | development HPO 选择 `steps=0` registered no-op |
| D1 | 10 | .243757/.380534 | .243757/.380534 | 0/0 | .376458 | .405122 | .422032/.584373 | development HPO 选择 `steps=0`；`5.55e-17` 舍入残差按 `1e-10` 容差判 NONE |

完整数值、ordered-ID SHA、partition SHA、簇大小、资源和 resolved config 在 `absolute_metrics_main_table.csv`。

## 增益究竟来自哪里

本轮没有训练新 embedding，GPU 使用为 0。最终 head 使用相同代码形式：

1. 从注册的初始分区计算分子 centroid-distance unary；
2. 仅在稀疏空间图的真实边上计算两个模态的相似度；
3. 以 `either_similar=max(s1,s2)`、`both_similar=min(s1,s2)` 或 spatial-only 形成 row-normalized conductance；
4. 同步更新簇标签，并在每步重算 centroid unary；若候选更新会丢失请求的 K，该行保留更新前分区并记录 collapse guard；
5. 公开标签只在跨 run HPO、known K 和 evaluator 中出现。

在七条激活 lane 上，完整 unary+pairwise 相对同初始化的 dynamic-unary-only 均为 ARI/NMI 双正增益；A1/D1 则是公开标签 development HPO 在所有 lane 共享的数值网格中选择 `steps=0` registered no-op，并不是模型从内容中自动推断出来的拒绝门。这个对照说明信号不是纯多数邻居投票，也不是只靠重新算质心。详细见 `minimal_energy_contribution_table.csv`。不过，该同步更新是 Potts-style 条件更新，不应夸大为严格全局 MAP 求解。

## 数值环境敏感性

Worker2 的 tonsil s3 峰值 `.317410/.275343` 被独立复现，但只在 randomized PCA 且 `OMP=MKL=8` 时得到；同公式在线程 1、full-SVD 或不同 BLAS 环境下会产生不同 partition SHA。十步离散更新放大了 PCA 的微小浮点差异。因此：

- `.317410/.275343` 只登记为环境锁定的开发峰值；
- stable 方法证据使用 no-PCA concat，`.311243/.278154`，两次 fresh process byte-exact；
- P22、MISAR K=7、tonsil s1 的 randomized PCA 也在线程 1/8 之间改变 partition SHA，虽然分数差很小；
- 正式稳定配置统一改用 full-SVD thread 1，避免把线程数伪装成科学 seed。

全部对照在 `numerical_environment_sensitivity.csv`。

## 算法家族淘汰

| 家族 | 代表 lane 结果 | 决定 |
|---|---|---|
| 静态 Potts | 仅局部改善 | 停止扩展 |
| 双模态边界静态 Potts | P22/MISAR 局部双升，但弱于动态版本 | 作为消融保留 |
| 普通/加权稀疏分区集成 | 无代表 lane 双升 | 淘汰 |
| 稀疏谱 | 显著低于强表示 head | 淘汰 |
| 伪 Fisher / prototype metric | 无代表 lane 双升 | 淘汰，避免重复 SAPR 自蒸馏 |
| 动态 unary + anisotropic sparse Potts | 6 个稳定双升 lane，跨两个家族 | finalist |

没有为了“用 GPU”而启动 edge learner：确定性核心已经在 matched head 上产生信号，且没有独立证据说明可学习边权会再增益。

## 失败、限制与诚实边界

- `smoke_dynamic_p22` 的缺失 import 是工程错误，已加最小回归测试并完整重跑受影响 lane。
- 初版 tonsil s3 audit 没覆盖 selected medoid，保留为不完整审计；随后按 partition SHA 做了 exact replay。
- 全量 no-PCA 九 lane 网格是在未写出首 lane 时由执行者主动中止，以把 CPU 让给 full-SVD；它是 `ABORTED_RESOURCE_REALLOCATION`，不是崩溃。要求的 no-PCA 单点和 focused s3 网格均已完成。
- 汇总初版把 D1 的 `5.55e-17` 浮点残差误判为 DUAL；在任何 commit/tag 前加 `1e-10` 容差和回归测试，正确计数为 6 DUAL、1 ARI_ONLY、2 NONE。分区与原始分数未改。
- P22 K=9/K=18 是同一数据的两套协议；MISAR K=12 不是 exact K12 annotation；tonsil s1/s2/s3 属于同一 study block。不能把九行当九个独立科学重复。
- 所有标签用途都是公开 benchmark development。标签未进入 unary、pairwise、edge、模型输入或训练 target，但标签用于 HPO 与 BEST 选择。

## 资源与复现

- 完成 run ledger：68,798 rows；失败/中止另列 engineering changelog。
- 最终 fresh-process replay：9/9 pairs、18/18 rows，partition 与指标 exact。
- 已计完成 arena wall sum：1,745.77 s；peak RSS 204.13 MiB。
- GPU：0 s / 0 MiB；新数据下载：0；dense N×N：0。
- targeted tests：10 passed。
- AutoDL：按夜间自主联动指令保持开机，**未派发 shutdown**。

## 导师汇报版

1. Night-15B 的 SAPR 已停止，本轮没有再包装伪标签残差。
2. 我们保留强 embedding，直接把分子簇代价和双模态边界支持放进稀疏 Potts-style 聚类更新。
3. 稳定配置在 P22 K=9/K=18、MISAR K=7 和 tonsil 三切片上均实现 ARI/NMI 双升；A1/D1 是公开标签 development HPO 在统一网格中选择 registered no-op，不是模型自动 gate。
4. P22 K=9 为 `.5825/.7016`，P22 K=18 为 `.7277/.7490`，MISAR K=7 为 `.5355/.6540`。
5. tonsil s3 的线程敏感峰值被识别并降级为开发峰值；稳定 no-PCA 版本仍达到 `.3112/.2782`。
6. 所有九条最终配置都通过两次 fresh-process byte-exact 回放，GPU 未使用。
7. 这说明直接聚类能量值得作为统一辅助 head 继续，但不是 SOTA 或独立论文贡献；Potts/CRF、BANKSY 邻域特征和 PRAGA 动态图均是明确先例。
8. 下一轮应冻结开发超参数，增加真正外部/未调参确认和 matched literature ablation，而不是再扩展伪分区自蒸馏。

## 技术附录

Night-15B authority、source commit/license、代码、配置、测试、全量 ledger、fresh-process audit、标签防火墙、资源、Git bundle、root-relative compact index 和 Windows 独立复算分别由同目录对应文件记录。最终 Git commit/tag 记录在 compact 生成后的 `delivery_manifest.json`，避免在被提交文件中制造自引用哈希。
