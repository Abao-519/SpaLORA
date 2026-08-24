# SpaLORA Night-16B 报告

## 我现在需要知道的三件事

1. **分数确实继续推进。** D1 在非退化 guard 下由 0.338775/0.435972 提到 **0.365174/0.444577**，最小簇 55，越过约 0.3427 的方向线；P22 K=9 也到 **0.595552/0.717931**。Secondary 的 P22 K=18 与 MISAR K=12 分别到 0.745939/0.766121 和 0.454117/0.607191。
2. **实际改的是候选后的结构化解码层。** URSD（统一可靠性结构化解码器）把既有多视图 start、跨模态可靠性稀疏能量、self-return（把被拒绝的边质量作为“留在当前状态”的代价）、可选形态视图和通用 split/merge/boundary repair 放进同一生产接口。HPO 指公开标签辅助的跨运行调参：候选先保存和哈希，评价器随后读 annotation。
3. **论文含义要收紧。** 本轮主分类是 `PAPER_COMPATIBLE_TUNED_SCOREBOARD`，同时成立 `SCORE_FRONTIER_ADVANCE`。但新增高分主要来自强初始化加通用 repair，其他五条 primary 选 no-op；因此贡献边界是 `HEAD_ONLY_OR_INITIALIZATION_SIGNAL`，不是“统一 decoder 已在多个 study 独立增益”，也不是盲测或 SOTA。

## 绝对指标主表

| 数据/协议 | N | eval | K | 旧 ARI | 旧 NMI | 新 ARI | 新 NMI | ΔARI | ΔNMI | 最小簇 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 3484 | 3484 | 10 | 0.276003 | 0.421740 | 0.276003 | 0.421740 | 0.000000 | 0.000000 | 114 |
| D1 | 3359 | 3359 | 10 | 0.338775 | 0.435972 | 0.365174 | 0.444577 | 0.026399 | 0.008606 | 55 |
| tonsil_s1 | 4326 | 4326 | 4 | 0.236536 | 0.317118 | 0.236536 | 0.317118 | 0.000000 | 0.000000 | 478 |
| tonsil_s2 | 4519 | 4518 | 4 | 0.258264 | 0.314324 | 0.258264 | 0.314324 | 0.000000 | 0.000000 | 445 |
| tonsil_s3 | 4521 | 4460 | 4 | 0.350644 | 0.309771 | 0.350644 | 0.309771 | 0.000000 | 0.000000 | 294 |
| P22 | 9196 | 9196 | 9 | 0.593963 | 0.714517 | 0.595552 | 0.717931 | 0.001589 | 0.003413 | 167 |
| MISAR_E15_5_S1 | 1949 | 1949 | 7 | 0.541424 | 0.666798 | 0.541424 | 0.666798 | 0.000000 | 0.000000 | 125 |

完整 AMI/FMI、Moran、Geary、簇大小、candidate budget、邻域中位数和资源见 `absolute_metrics_main_table.csv`。

## Secondary sensitivity

| secondary | K | ARI | NMI | ΔARI | ΔNMI | 最小簇 |
|---|---:|---:|---:|---:|---:|---:|
| P22_3DOT_K18 | 18 | 0.745939 | 0.766121 | 0.004878 | 0.011785 | 32 |
| MISAR_E15_5_S1_K12 | 12 | 0.454117 | 0.607191 | 0.000941 | 0.008562 | 52 |

P22 K=18 与 MISAR K=12 不作为额外独立 study 计票；后者仍是同一 carrier 在 K=12 endpoint 下的 sensitivity。

## HPO、Pareto 与 family-default

全部 **10,997** 行使用共同 coarse schema 与机械 refine 规则，0 行被删除、0 运行失败。Headline 按 ARI 主排序；max-NMI 与 Pareto（ARI 提高就不能在 NMI 上也被另一行完全支配的折中前沿）单列。D1 max-NMI profile 为 ARI/NMI 0.357394/0.454109，说明 ARI 主 headline 0.365174/0.444577 存在真实取舍。

Family-default 不读取当前 lane 指标：A1/D1 互留、tonsil 整片留出、P22/MISAR 互留。它在 D1、tonsil 和 P22 多数退回或保持父级，但 D1 配置转到 A1、P22 配置转到 MISAR 时下降。这说明 per-dataset tuned scoreboard 已闭合，但“无 annotation 的默认参数迁移”仍弱，不能用 tuned BEST 替代部署证据。

## 最小贡献对照

每条 primary 都有 start、full common path、关闭 cross-modal pairwise reliability、关闭 self-return、关闭 generic repair 五行。P22 的 common path 中 reliability/self-return 和 repair 都有正贡献；tonsil s1/s3 与 MISAR 的 inherited energy 有局部贡献。D1 的 common path 若从更早 Night-15E start 出发反而下降，只有从 Night-16A/15G 高质量 authority start 接通用 repair 才达到 0.365174。这是初始化/后端交互证据，不是所有组件普适支持。

## 复现与资源

- 7 primary + 2 secondary 已做两次 fresh-process producer replay，**9/9 分区 SHA、指标和簇大小完全一致**。
- targeted tests **10/10**；缺失 morphology 的 presence-mask fallback byte-exact。
- candidate producer 标签读取 0、dense N×N 0；公开 annotation 只在独立 evaluator 与跨运行 HPO 打开。
- 本轮 endpoint 搜索在 Windows CPU 串行完成，GPU 时间 0，观测 peak RSS 约 187 MiB。
- AutoDL 持久盘 inode 已耗尽但仍有约 86 GiB；因此 Git 工作区放在独立 overlay clone，历史 raw/branch/tag 未修改。

## 失败与限制

- A1、tonsil s1/s2/s3、MISAR K7 没有刷新，正式选择为 registered no-op。
- D1 新分区虽无 microcluster，但 family-default 不能转移该提升；当前结果是 public benchmark HPO，不是新数据泛化。
- P22 K9 仍低于 0.63 方向 context，MISAR K7 仍未越过 0.55。
- 外部方法没有在本轮公平重跑；context board 不能作为正式胜负。

## 导师汇报版

1. 我们把 Night-15F/15G/16A 的高分来源整理成了同一候选生产与结构化修复接口。
2. 所有 10,997 个候选都先保存分区再评价，公开标签只用于诚实登记的 benchmark HPO。
3. D1 的可信 ARI 提到 0.3652，且最小簇 55，不再依靠 singleton。
4. P22 K=9 小幅刷新，P22 K=18 与 MISAR K=12 的 secondary 也刷新。
5. 其他五条 primary 没有继续提升，no-op 被完整保留。
6. 贡献对照表明新增 D1 高分主要是强初始化和通用 repair 的交互，不足以说所有 decoder 组件跨 study 普适有效。
7. 因此本轮已经形成可审稿的 tuned scoreboard 与 Methods/HPO 草案，但还不是统一方法的外部确认或 SOTA 证据。

## 技术附录

父级为 `7bcc7696c3eb170e7191d9266691271a2ec225b6` / `night16a-final-20260824`。Final commit/tag、bundle 与 compact SHA 在完成 Git 和 Windows 独立复算后写入 delivery verification；报告正文不做自指 commit 哈希。
