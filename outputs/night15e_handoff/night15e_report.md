# SpaLORA Night-15E：连续可靠性能量、开发分数上限与新增真实单元

## 我现在需要知道的三件事

1. **问题是什么。** Night-15D 虽然在九条公开 benchmark lane 上都提分，但依靠逐 lane 选择离散模块和数值参数，仍像一个共享组件菜单。本轮把边模式、质量归一/self-return、特征尺度、模态 unary 和 trust 尽量写进一条始终存在的连续公式，并分别回答两件事：逐 lane 数值 HPO 能把分数推到哪里；不看 held-out study 标签的增量参数策略能否转移。
2. **实际做了什么。** 在冻结的 Night-15D authority partition 之后加入同一个稀疏连续可靠性能量层：绝对 conductance 质量与被拒绝质量 self-return、双模态边一致/冲突、动态 prototype margin、保留表示与两模态 unary、多尺度低/高频特征和 trust-aware move 都由连续权重混合。formal、adaptive、multi-initial 共保留 8,087 行 score-ceiling 运行；另做 891 行联合策略、54 行 matched ablation、两次完整 9/9 新进程重放。还新增了 GSE213264 Human tonsil 的真实 RNA+protein 工程路径。
3. **对论文意味着什么。** 平衡 profile 相对 Night-15D 达到 **9/9 ARI/NMI 双正**，因此仍可归为跨两模态家族的 `LOCAL SIGNAL`；但这是公开标签参与逐 lane 跨运行 HPO 后的开发分数上限，不是盲测、自动配置选择或完整零标签部署。全局/家族/leave-one-study-out 增量策略几乎退化为 no-op，并在 held-out lymph-node/tonsil 出现下降，所以方法收敛问题尚未解决。

## 明确终态

- 终态：`NIGHT15E_CONTINUOUS_RELIABILITY_PUBLIC_BENCHMARK_SCORE_CEILING`
- 六类治理分类：`LOCAL SIGNAL`
- 证据层级：`PUBLIC_BENCHMARK_DEVELOPMENT_SCORE_CEILING`
- score ceiling：相对 Night-15D **9/9 balanced lanes ARI/NMI 双正**
- incremental policy transfer：`NO_MEANINGFUL_INCREMENTAL_POLICY_TRANSFER`
- 新单元：`GSE213264_UNLABELED_REAL_PATH_READY`
- 两次最终 fresh-process replay：9/9 × 2 exact
- targeted tests：7/7
- GPU：0 秒 / 0 MiB；dense N×N：0

这不是 `CONFIRMED_MILESTONE`。Night-15D authority partition 本身已经过逐 lane 公开标签 HPO；Night-15E 的 global/family/LOO 审计只能说明“增量层参数选择没有读取 held-out study 标签”，不能把整条流程重新解释成零标签可部署。

## 一、SCORE CEILING：绝对指标主表

冻结选择规则为：先要求 ΔARI、ΔNMI 均严格大于 0，再最大化 `min(ΔARI, ΔNMI)`，最后比较 `ARI + 0.35×NMI`。max-ARI 和 max-NMI profile 另列在 CSV 中，没有拼成一个虚构的“最好结果”。

| Lane | K | Night-15D ARI/NMI | Night-15E balanced ARI/NMI | ΔARI/ΔNMI | AMI | FMI | Moran / Geary | config |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P22 | 9 | .587186/.708094 | **.589191/.710042** | +.002005/+.001948 | .709540 | .650091 | .931697/.070751 | `P22__CEM_I05_011` |
| P22 author assignment | 18 | .727952/.749674 | **.728307/.750114** | +.000355/+.000440 | .748488 | .757563 | .835200/.166092 | `P22_3DOT_K18__CEM_I00_020` |
| MISAR E15.5 S1 | 7 | .540665/.665900 | **.540932/.666617** | +.000267/+.000717 | .664860 | .632625 | .922742/.079097 | `...G049_af33b6b74c20` |
| MISAR 12-cluster prediction | 12 | .451195/.596512 | **.452367/.597708** | +.001172/+.001196 | .594094 | .559422 | .910540/.089258 | `...CEM_I01_038` |
| A1 | 10 | .274380/.418549 | **.275293/.419908** | +.000913/+.001359 | .416120 | .415992 | .548701/.454521 | `A1__CEM_I05_011` |
| D1 | 10 | .245193/.383665 | **.254658/.389044** | +.009465/+.005379 | .385050 | .409914 | .517291/.488440 | `D1__CEM_I02_000` |
| tonsil s1 | 4 | .231668/.312166 | **.233825/.314270** | +.002157/+.002105 | .313669 | .501303 | .719589/.280459 | `tonsil_s1__CEM_I02_040` |
| tonsil s2 | 4 | .251476/.301339 | **.254798/.305252** | +.003322/+.003914 | .304669 | .513522 | .750158/.249606 | `tonsil_s2__CEM_I02_008` |
| tonsil s3 | 4 | .325526/.287363 | **.328311/.289524** | +.002784/+.002162 | .288844 | .610429 | .683139/.324994 | `tonsil_s3__CEM_I05_007` |

MISAR K=12 仍使用项目登记的 12-cluster prediction 对公开 K=7 Y 评价，不是 exact SEPAR K=12 annotation。其 Night-15E NMI `.597708` 也高于更早登记的历史 NMI `.590609`，但不能与不同 annotation/protocol 的 SEPAR 原文数字作严格胜负。

### 必须保留的脆弱性

D1 的 balanced profile 产生一个 singleton cluster：cluster sizes 为 `[44,246,227,210,1,480,140,822,145,1044]`，`min_cluster_size=1`。该结果未回填、未删簇、未换配置；它说明公开标签 HPO 可以把 endpoint 推到脆弱解。D1 的绝对提升是真实可复算的开发峰值，但不应作为稳定泛化证据。

## 二、INCREMENTAL POLICY TRANSFER：没有形成可部署策略

联合 global config 主要在 D1、P22 K=18、tonsil s1/s2 产生千分位以内增益，其余多为 no-op，tonsil s3 NMI 还轻微下降。family-numeric policy 同样接近 no-op。leave-one-study-out 的关键结果是：

- held-out lymph-node：A1 为 `-.000952/-.000553`，D1 为 `-.001416/-.002347`；
- held-out tonsil：s1/s2 小幅正，s3 为 `-.000372/-.000562`；
- held-out MISAR：两个 K lane 均 no-op；
- held-out P22：K=9 no-op，K=18 仅 `+.000041/+.000137`。

因此冻结结论是 `NO_MEANINGFUL_INCREMENTAL_POLICY_TRANSFER`。准确表述只能是“在固定 Night-15D authorities 上审计 Night-15E 增量参数的迁移”；不能写成 full deployable policy，更不能把 9/9 ceiling 掩盖迁移失败。

## 三、NEW UNLABELED ENGINEERING UNIT：GSE213264 Human tonsil

从 NCBI GEO 官方 processed tar 下载并只提取 Human tonsil RNA/Protein：

- RNA：`2492 × 28417`，protein：`2492 × 283`；
- 两模态 spot-ID 集合 2492/2492 byte-exact 相同，但原始行顺序不同；正式路径按字符串 spot ID 显式 join，绝未按行号拼接；
- ID 均解析为 `xCoordxYCoord`，坐标范围 `1..50 × 1..50`，2492 个坐标唯一；
- 稀疏图 `2492 × 2492`、17,696 nnz，dense N×N=0；
- preprocessing → continuous core → endpoint → serialize → 两次 fresh-process reload 均通过。

K=8 仅是**公开作者报告的 RNA cluster count 用作工程 known K；没有读取逐 spot reference labels**。K=7 是作者报告的 protein cluster count 敏感性。原作者 Seurat 等脚本产生的是 author-derived unsupervised clusters，不是人工/专家 spatial-domain ground truth；当前没有可逐 spot复算的权威 reference partition，因此本轮不报 ARI/NMI。

## 连续公式的贡献边界

Night-15E 的 core 没有 dataset/family 分支，九条 lane 使用同一函数和同一公式。与 Night-15D 不同，edge union/intersection、row/mass/self 行为、模态 unary、多尺度特征和 trust 都以连续数值存在；但逐 lane 数值仍由公开标签 HPO 选择，因此只是 score ceiling。

Matched ablation 不支持“所有组件普适有效”的说法：

- `TRUST_TERM_NEAR_ZERO` 在 A1、D1、MISAR K=7/K=12、P22、tonsil s2 等多条 lane 与 full 完全或近乎相同，trust 尚无普适证据；
- P22 K=9 的 `NO_REJECTED_MASS_SELF_RETURN` 反而略优 full；
- P22 K=18 的 `RETAINED_UNARY_DOMINANT` 略优 full；
- rejected-mass self-return 在 protein lanes 和部分 MISAR 有明显 matched 贡献，但不是普适定律。

所以最多可说：连续组合总体有跨家族开发信号，其中“保留绝对 conductance 质量并将拒绝质量回流自身”在部分数据有实质证据。Potts/ICM、prototype unary、相似性边、多尺度图和动态路由均有先例；方法新颖性仍待进一步收敛。

## 回放、资源与工程审计

- formal + adaptive + multi-initial score-ceiling ledger：8,087 行，全部保留；joint policy：891 行；matched ablation：54 行。
- 两次最终 replay：9/9 × 2，partition array、formal SHA、ARI、NMI、config、cluster sizes 全部 exact。
- 最终 core、replay runner 与 frozen registry 的 mtime 均早于两次 replay；之后未修改 core 或 replay 语义。
- targeted tests：7/7；标签进入模型/energy/loss=0；core dataset-name reads=0；dense N×N=0。
- observed search wall lower bound：806.24 s；资源 replay：4.04 s，peak RSS 326.36 MiB；GSE213264 P0：18.70 s。
- GPU：0 s / 0 MiB。本轮是 Windows local-first CPU 执行。
- 新下载：1 个 NCBI GEO 官方 processed tar，31,877,120 bytes，SHA-256 `468b47cfdb271483177b58670d846840be6198e97f69deb1d2b68a73a35c49a6`。
- 历史 raw 修改：0；失败/混合 ablation 删除：0。
- 夜间联动要求下 AutoDL 保持有卡开机；`shutdown_dispatched=false`，未派发 shutdown/poweroff/halt/stop。

## 主要失败与限制

- 9 lanes 不是 9 个独立科学数据：P22 K=9/K=18 共用切片，MISAR K=7/K=12 共用切片，tonsil 三片属于一个 study block。
- 9/9 ceiling 依赖逐 lane 公开标签数值 HPO；没有一个统一冻结策略复现这些收益。
- D1 singleton 暴露 endpoint 脆弱性。
- trust、self-return、unary 和 multiscale 的作用随 lane 改变；目前无法把每个组件都写成普适贡献。
- GSE213264 仅闭合无标签工程路径，没有人工参考，因此不提供科学外部确认。
- 结果仍是聚类后端/增量能量层信号，不是端到端新表示模型已成立。

## 导师汇报版

1. 我们把 Night-15D 的离散组件菜单改写成同一条连续稀疏可靠性能量公式，两个模态家族共用核心。
2. 逐数据集公开标签 HPO 的分数上限上，九条 lane 相对 Night-15D 都实现 ARI/NMI 双升。
3. P22 K=9 达到 `.5892/.7100`，MISAR K=7 达到 `.5409/.6666`，D1 增幅最大但出现一个 singleton cluster。
4. 这不是自动策略：全局、家族和 leave-one-study-out 增量参数基本 no-op，held-out lymph-node 还下降。
5. 消融也显示 trust 在多条 lane 不起作用，P22 某些简化版还略优 full，不能声称每个组件都获得普适支持。
6. 新接入 GSE213264 Human tonsil，RNA/蛋白按 spot ID 显式对齐并完成两次新进程回放，但没有人工 reference，所以不报 ARI/NMI。
7. 当前分类仍是 `LOCAL SIGNAL`，准确证据层级是公开 benchmark 开发分数上限，不是 blind、SOTA 或论文确认。
8. 下一步若继续，重点应是无逐 lane 标签回看的内容标定/冻结策略和真正外部参考，而不是继续堆 ceiling 参数。

## 技术附录

完整精度指标、max-ARI/max-NMI profiles、8,087 行压缩 ledger、54 行 ablation、policy audit、GSE provenance、两次 replay、源码碰撞审计、Git bundle、root-relative compact index 和 Windows 独立复算均由同目录文件记录。final commit/tag 与 bundle/index SHA 写入交付 manifest，避免 commit 自引用。
