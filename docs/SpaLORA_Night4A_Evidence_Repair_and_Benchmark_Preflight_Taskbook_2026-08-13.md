# SpaLORA Night‑4A：证据修复与公平 Benchmark 入场预检任务书

日期：2026‑08‑13  
执行者：负责 AutoDL/代码执行的 Codex  
规划与验收者：论文规划 Worker  
阶段性质：**证据修复 + 数据/方法端到端可复现性预检；不是正式 benchmark，不得产出胜负结论**

## 0. 任务唯一目标

把 Night‑3B 的 `MIXED_EVIDENCE` 转化为一个可审计、无事后选择的 Night‑4B 入场条件：

1. 不改动 Night‑3B 原始结果，修正 lower-is-better win-count 语义并重绘出版候选图；
2. 在标签防火墙下验证 held-out 与独立空间多组学数据是否真的可用于公平比较；
3. 在目标 AutoDL 环境中完整跑通核心 baseline 的官方教程，锁定官方 commit、环境、输入输出与失败；
4. 只输出 `READY_FOR_NIGHT4B / READY_WITH_REDUCED_MATRIX / BLOCKED_PREFLIGHT`，不得运行正式方法比较，不得根据任何 ARI/NMI 选择架构、数据或 baseline。

## 1. 当前权威基线与输入

### 1.1 科学基线

- Night‑3B final commit：`16b04499d41a1e45727eea46a4b6e138e35f599a`
- Night‑3B final tag：`night3b-final-20260810`
- Night‑3AF parent：`384e66587149a687b3eac4a6d1918d8d4972dc06`
- Night‑3B 方法级结论：`MIXED_EVIDENCE`
- 允许带入 Night‑4B 规划的 SpaLORA 候选只有：
  - `FULL_IGE`
  - `UNIFORM_CROSS`
  - `UNIFORM_WITHIN`

本轮不得新增第四个模型变体，不得删除 Corr2，不得根据现有三数据集重新设计 attention/loss。

### 1.2 本地权威文件

- 独立审计：`D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night3B_Independent_Evidence_Audit_and_Decision_2026-08-13.md`
- 最小证据表：`D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night3B_Minimal_Evidence_Table_2026-08-13.csv`
- Night‑3B 交付根：`C:\Users\李昌赫\Documents\Codex\2026-08-07\https-github-com-abao-519-spalora\work\night3b_handoff_20260810`
- Git bundle：`C:\Users\李昌赫\Documents\Codex\2026-08-07\https-github-com-abao-519-spalora\work\night3b_handoff_20260810\SpaLORA_night3b_20260810.bundle`
- 完整归档：`C:\Users\李昌赫\Documents\Codex\2026-08-07\https-github-com-abao-519-spalora\work\night3b_handoff_20260810\night3b_artifacts_20260810.tar.gz`
- 精简包：`C:\Users\李昌赫\Documents\Codex\2026-08-07\https-github-com-abao-519-spalora\work\night3b_handoff_20260810\night3b_planner_handoff_20260810.tar.gz`

### 1.3 启动前必须独立核对的 SHA‑256

- bundle：`776ac67c79788fd2993b402b84efb974ae5fca64a213662b3911cfe321cb005b`
- full archive：`f5030232b19ee92888c4c4b710dca1282c757d373c4a080b06fe1f6beacac649`
- planner handoff：`ed77df03cc48b4b153b82d9275c95f8a3eacf428a6410478ba5a38ec1e2c4017`

任一 SHA 不一致、final tag 不能 peel 到指定 commit、baseline 不是 final 祖先：**立即停止并报告，不得尝试“修好后继续”。**

## 2. 不可违反的科研与操作边界

### 2.1 禁止事项

1. 禁止修改或覆盖 Night‑3B handoff、两个 tar.gz、bundle、原 CSV/JSON/NPZ/图。
2. 禁止在 `bundle_verify_repo` 中继续工作；该目录已被 2026‑08‑13 审计取出工作树，只能作为过程偏差记录。
3. 禁止重跑 Night‑3B 的 120 个训练，禁止重跑 Night‑3AF。
4. 禁止读取 baseline 或 SpaLORA 的 ARI/NMI 后决定是否保留某方法、某数据集或某变体。
5. 禁止 seed 搜索、超参数搜索、按标签调 resolution/K/邻居数、删除失败运行或只展示有利 seed。
6. 禁止拿论文中已公布的表格数字和本项目结果直接做胜负结论。
7. 禁止把 GSE263617 的 D1/tonsil 称为“独立研究验证”；它们是同一研究/平台的 held-out sections。
8. 禁止把无可靠人工/独立标签的数据集强行用于 supervised clustering accuracy 表。
9. 禁止为某个方法单独提供更大的 tuning budget；本轮本来就不允许正式 tuning。
10. 禁止静默修改第三方方法。所有兼容性 patch 必须先保留原始失败，再记录 diff、理由和是否改变科学计算。

### 2.2 允许事项

- 在**新隔离 worktree/branch** 中添加 Night‑4A 脚本、测试、报告和重绘图。
- 从锁定 Night‑3B 表/NPZ 和固定 ground truth 生成更正表与新图；必须保留原文件不变。
- 下载官方数据、官方代码、官方教程；保存下载 URL、时间、大小、SHA‑256、license/terms。
- 为完成官方教程做一次有界的环境兼容性修复；不得改模型公式、默认科学参数或评价定义。
- 运行 baseline 官方教程自身所需的小规模训练；这些输出仅用于“能否端到端执行”，不得进入论文 benchmark。

## 3. Git 与隔离环境

### 3.1 建议命名

- branch：`revision/q2-night4a-preflight-20260813`
- baseline tag：`baseline/pre-night4a-20260813`
- final tag：`night4a-preflight-final-20260813`

### 3.2 必须满足

1. 从经验证的 bundle 恢复完整历史；不得把 handoff 内验证目录当工作仓库。
2. 从 `night3b-final-20260810` 新建隔离 worktree。
3. 写入前记录 clean status、HEAD、tag peel、ancestor check、`git fsck`。
4. 对 Night‑3B 原始交付根建立只读保护清单；结束时复核关键 SHA。
5. 第三方方法各自使用独立 clone、独立环境、独立 cache；不得让依赖互相污染。

## 4. 总体运行顺序

必须按以下顺序，不能因为某个结果“看起来好”而跳步：

1. `P0-PROTECT`：输入、Git、历史结果保护；
2. `P1-EVIDENCE-REPAIR`：从锁定结果修正表头语义并重绘图；
3. `P2-DATA-PREFLIGHT-LABEL-FREE`：下载/定位、哈希、spot 对齐、坐标与模态兼容性；
4. 锁定数据资格规则、预处理规则与候选矩阵；
5. `P2-DATA-POSTLOCK-LABEL-AUDIT`：只在锁后检查标签来源、K、域规模与可评价性；
6. `P3-BASELINE-TUTORIALS`：在目标环境完整执行官方教程；
7. `P4-MATRIX-LOCK-PROPOSAL`：生成 Night‑4B 提案，不运行正式 benchmark；
8. 打包、下载本地、验证、Git 归档；
9. 远端最后一条命令严格为 `/usr/bin/shutdown`，之后不得重连。

## 5. P0‑PROTECT：入场门

### 5.1 必做检查

- 三个大文件 SHA 与第 1.3 节完全一致；
- bundle verify、final/baseline tag peel、ancestor、fsck 全通过；
- Night‑3B `delivery_index.json` 的 55/55 条重新核对；
- 原 `SHA256SUMS` 1186/1186 重新核对；
- 新 worktree 以外的历史工作树不写入；
- 记录当前 CPU/GPU/RAM/磁盘、CUDA、驱动、Python、R、网络与代理状态；
- 记录可用磁盘预算，数据与各方法环境分别估算空间。

### 5.2 失败策略

任何保护门失败：输出 `BLOCKED_PREFLIGHT`，保留日志，停止。不得在损坏或来源不明的基线上继续。

## 6. P1‑EVIDENCE‑REPAIR：只修证据，不改结果

### 6.1 lower-is-better 语义修复

从原始 `paired_ablation_deltas.csv` 生成新文件，例如：

- `outputs/night4a_evidence_repair/night3b_corrected_win_counts.csv`
- `outputs/night4a_evidence_repair/metric_direction_registry.json`

规则：

- ARI/NMI/neighbor/Moran：`Δ = FULL − ABLATION > 0` 才是 FULL win；
- Geary C/boundary disagreement：`Δ < 0` 才是 FULL win；
- 同时保留 `positive_delta_count`，避免再次混淆。

不得覆盖原 `paired_ablation_deltas.csv`。必须加单元测试，覆盖 higher-is-better、lower-is-better、zero/tie、缺失值和 5/15 seed 汇总。

### 6.2 数值不可变性

独立重算并验证：

- 120-row per-seed 主键覆盖；
- summary mean/SD/median/min/max；
- paired seed deltas、dataset summaries、macro summaries；
- 原预注册 component/method decision；
- 方法级仍为 `MIXED_EVIDENCE`。

除更正的 win-count/字段名外，所有数值输出必须与锁定原表一致；误差要求是由相同浮点输入重算可达到的 exact/机器精度，不得新设事后容差。

### 6.3 图表重绘

仅从锁定 CSV/NPZ 和固定 ground truth 重绘 PNG/PDF；不得重新训练。至少修复：

1. forest plot 的全部 dataset/ablation y 标签可见；
2. P22 图显示七变体完整名称或明确 legend；
3. attention 图显示六通道完整名称或明确 legend；
4. domain 图显示真实域名，并说明五 seed 是优化重复；
5. gradient trajectory 显示 seed-level ribbon/points；
6. heatmap 同时显示 seed-level uncertainty 或 paired seed points；
7. A1 maps 加 ground-truth panel；预测 cluster 先对 GT 做后处理 Hungarian color alignment，并明确这是仅用于可视化；
8. spatial trade-off 同时呈现 ARI、neighbor、Moran、Geary，避免文字重叠；
9. 图内/图注明确 `Δ = FULL − ABLATION` 及指标方向；
10. P22 cross-attention 负面边界必须可见，不得只挑有利面板。

每张图均需：

- PNG + vector PDF；
- 记录源表 SHA、脚本 SHA、生成命令、尺寸、字体；
- 实际渲染 PDF 后视觉检查；
- 检查标签缺失、重叠、裁切、legend 映射、颜色语义；
- 生成 `figure_qa_manifest.json`，逐图 PASS/FAIL 和问题说明。

若任一图未通过，不得标为 publication-ready；允许交付为 diagnostic，但报告必须写明。

## 7. P2‑DATA‑PREFLIGHT：先数据，后标签

### 7.1 数据分层

#### A. 已开发/已观察证据集

- Human lymph node A1
- Human placenta
- P22 mouse brain

这些数据不得再用于同类架构搜索。它们可进入未来完整 benchmark 的“legacy/development evidence”层，但不能承担独立泛化证明。

#### B. Held-out within-study confirmation 候选

官方来源：[NCBI GEO GSE263617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617)

- Human lymph node D1
- Human tonsil A1
- Human tonsil D1

三者与已用 lymph node A1 同属 SpatialGlue 研究/平台，只能称 `held-out sections` 或 `within-study section generalization`。

#### C. Truly independent 候选

优先检查：

1. [NCBI GEO GSE198353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353)：SPOTS mouse spleen replicate 1/2，RNA+protein，独立研究、多 section；
2. [NCBI GEO GSE213264](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213264)：spatial‑CITE‑seq human tonsil RNA+protein，独立研究；只有在坐标、paired spot IDs 与独立标签满足资格门时才可进入 accuracy benchmark。

若上述任一不合格，可提出一个替代候选，但必须来自论文官方数据入口、GEO/SRA/Zenodo/figshare 等权威来源，并先写明替代原因；不得在看到 SpaLORA 表现后换数据。

### 7.2 Label-free 入场检查

在读取 semantic label values 前完成并锁定：

- 原始/处理文件 URL、accession、下载时间、大小、SHA‑256；
- license/terms、引用与数据来源；
- RNA 与第二模态是否同 spot/cell；
- observation IDs 集合与顺序；
- 坐标、图像、feature names、稀疏/稠密、raw/count-like/normalized 状态；
- spot 数、features 数、缺失率、重复 IDs、非有限值；
- 第二模态的真实语义，不得把派生 regulatory features 写成 raw peaks；
- 能否在不看标签的情况下建立 immutable cache；
- 固定预处理规则是否可由平台/模态类型决定，而非由 ARI/NMI 决定。

输出：

- `data_source_registry.csv/json`
- `raw_file_manifest.json`
- `observation_alignment_report.json`
- `label_free_cache_manifest.json`
- `data_preprocessing_lock.json`
- `semantic_label_access_log.jsonl`

### 7.3 数据资格门必须预先写入锁

一个数据集进入未来 accuracy benchmark 必须同时满足：

1. 两模态配对且 observation IDs 可无歧义对齐；
2. 有可复现空间坐标；
3. 输入来源、预处理与 feature 语义可审计；
4. semantic domain label 来自实验/人工/独立注释，不是任何被比较方法的预测；
5. label 与 observation IDs 可无歧义连接；
6. 至少两个非空域；报告每域 n，极小域不得被静默合并；
7. 文件可公开获取或有明确可再分发边界；
8. 没有因预览 SpaLORA/基线表现而被纳入或排除。

缺少标签但其他条件合格的数据可进入：

- label-free representation/continuity/资源/生物学定性层；
- 不得进入 ARI/NMI/AMI 排名表。

### 7.4 锁后标签审计

只有 `data_preprocessing_lock.json` 和候选矩阵 SHA 固定后，才允许读取 semantic label values，并记录：

- label 文件/列/来源；
- K、各域 n、missing/duplicate；
- 是否存在同一标签的事后合并版本；
- 哪个版本是原论文/人工权威版本；
- 未来所有方法是否使用同一版本。

不得用标签调整邻接图、HVG、PC 数、latent dimension、learning rate、epoch、resolution 或数据纳入。

## 8. P3‑BASELINE‑TUTORIALS：目标环境端到端可复现性

### 8.1 核心五方法

必须优先使用论文/作者给出的官方代码入口：

1. [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue)
2. [Seurat / WNN](https://github.com/satijalab/seurat)
3. [COSMOS](https://github.com/Lin-Xu-lab/COSMOS)
4. [ARISE](https://github.com/XiangxiangWang-code/ARISE)
5. [SMART](https://github.com/Xubin-s-Lab/SMART-main)

候选扩展只做可复现性筛选，不看性能：

- [PRESENT](https://github.com/lizhen18THU/PRESENT)
- [SpaMFG](https://github.com/LiangYu-Xidian/SpaMFG)

### 8.2 “跑通”的定义

README 可读、能 import、能加载 notebook 均不算跑通。每个方法必须在目标 AutoDL 机器上从干净独立环境完成一条官方教程的完整流程，至少到：

1. 官方示例数据完整读取；
2. 官方预处理完成；
3. 模型/集成算法完整运行；
4. 生成 integrated embedding 或官方定义的联合表示；
5. 生成官方聚类/下游输出；
6. 输出可重新读取且哈希固定；
7. 记录 wall time、peak CPU RAM、peak GPU memory（若用 GPU）；
8. 从零复现命令和环境 lock 可执行。

Seurat WNN 必须实际生成 modality weights、WNN/SNN graph 与 cluster，不得只运行单模态预处理。COSMOS/SpatialGlue/SMART/ARISE 同理，必须完成其官方多组学教程的终点。

### 8.3 方法锁与失败保留

每个方法记录：

- 官方 URL、clone 时间、commit SHA、tag/release、license；
- 原始 requirements/environment、实际 lock、Python/R/CUDA；
- 官方教程文件 SHA、数据 SHA、完整命令；
- 原始 attempt 1 完整日志；
- 若失败，root cause；
- 最多一次 compatibility-only retry，必须先保存 diff；
- 是否改了科学公式/参数。只要改了，状态不得超过 `UNREPRODUCIBLE_AS_OFFICIAL`；
- 最终状态：`COMPLETE_OFFICIAL / COMPLETE_COMPAT_ONLY / UNREPRODUCIBLE / TASK_INCOMPATIBLE`。

不得删除失败方法后假装它从未预注册。方法是否进入 Night‑4B 只能按以下预先标准决定：任务匹配、官方端到端可复现、输入公平、许可证/资源可承受；不能按表现决定。

### 8.4 本轮禁止输出的内容

- 不得把教程示例的 ARI/NMI 与 SpaLORA 比较；
- 不得在项目数据上跑五方法正式结果；
- 不得计算方法排名；
- 不得为某方法调参以提高结果；
- 不得在看到某方法失败后用一个更弱但有利于 SpaLORA 的方法静默替代。

## 9. P4‑MATRIX‑LOCK‑PROPOSAL：只提案，不执行

输出 `night4b_benchmark_matrix_proposal.csv/json`，每一格写明：

- dataset×method 是否任务匹配；
- 输入模态与 spot 对齐是否支持；
- 官方教程状态；
- 固定 commit/env；
- method-native preprocessing 与 common fairness layer；
- 是否需要 K；K 如何在不调参的前提下固定；
- seeds 固定为 `0,1,2,3,4` 是否可控制；
- 预计 runtime/memory/disk；
- 预先排除原因（若有）；
- 未来输出 embedding/cluster/resource 的统一接口。

同时生成 `night4b_protocol_draft.json`，但不得运行，其中至少包括：

1. 三层数据角色：development / held-out architecture confirmation / untouched independent benchmark；
2. 统一 seeds 0–4；
3. 固定 run order；
4. 标签防火墙；
5. K/cluster protocol；
6. ARI、NMI、AMI、Homogeneity、neighbor agreement、Moran、Geary、per-domain、runtime、CPU/GPU peak；
7. 失败运行保留规则；
8. 不给 SpaLORA 单独 tuning budget；
9. baseline 与 SpaLORA 使用同一 observation set、标签版本和评价器；
10. 未来 architecture confirmation 仅比较 `FULL_IGE / UNIFORM_CROSS / UNIFORM_WITHIN`，不得追加变体；
11. architecture confirmation 数据不得同时作为“完全独立最终验证”来夸大；
12. independent benchmark 在架构规则锁定前不得读取结果。

Night‑4B 的最终阈值与正式任务书仍由规划 Worker 在审计 Night‑4A 后签发；执行 Codex 不得自行启动。

## 10. Night‑4A 决策规则

### 10.1 `READY_FOR_NIGHT4B`

必须全部满足：

1. P0 保护全部通过；
2. Night‑3B 原始结果哈希未变；
3. lower-is-better 表修复通过测试，原判定仍 `MIXED_EVIDENCE`；
4. 全部拟用于论文的 Night‑3B 新图通过实际 PNG/PDF 视觉 QA；
5. GSE263617 的 3 个 held-out sections 通过 label-free/cache/label audit；
6. 至少一个 truly independent、至少含两个 section/replicate 的 paired spatial multi-omics family 通过资格门；
7. 核心五方法官方教程全部达到 `COMPLETE_OFFICIAL` 或 `COMPLETE_COMPAT_ONLY`；
8. 未来矩阵与协议在没有正式结果的情况下锁定。

### 10.2 `READY_WITH_REDUCED_MATRIX`

允许的最低条件：

- 数据门同上；
- SpatialGlue 与 Seurat WNN 必须完成；
- ARISE/SMART/COSMOS 至少 2/3 完成；
- 每个被排除方法都有任务匹配或官方可复现性原因，且排除发生在任何项目 benchmark 结果产生前；
- 方法数和覆盖仍足以回应审稿人的 SpatialGlue、COSMOS、Seurat v5 与现代 baseline 质疑。

### 10.3 `BLOCKED_PREFLIGHT`

任一情况成立：

- 输入/Git/哈希保护失败；
- 没有合格的 truly independent 数据；
- 核心方法完成度低于 reduced matrix 门；
- 需要标签指导改数据/超参数才能继续；
- 证据修复改变了原方法级结论；
- 无法保证未来 benchmark 的共同 observation/label/evaluation protocol。

即使 blocked，也必须交付所有失败日志和已完成清单，不得“再试几个 seed/参数”绕过。

## 11. 必需测试

至少新增并通过以下类别：

1. metric direction registry；
2. corrected win counts；
3. Night‑3B table recomputation；
4. original-artifact immutability；
5. figure labels/legend/domain mapping manifest；
6. data observation ID exact set/order；
7. label-free cache schema 与禁用 semantic columns；
8. label access event logging；
9. baseline tutorial completion contract；
10. benchmark matrix 不含结果列、无 outcome-driven exclusion；
11. run-order/seeds/K/evaluator protocol lock；
12. Git/tag/bundle/protection checks。

测试报告必须区分 Night‑4A 新专项测试与全仓测试，不得再泛称“所有测试通过”。

## 12. 交付物

远端建议根：`/root/autodl-fs/SpaLORA-night4a/outputs/night4a_handoff/`

必须包括：

- `night4a_report.md`
- `night4a_completion.json`
- `night4a_gate_status.json`
- `protocol_deviations.json`
- `metric_direction_registry.json`
- `night3b_corrected_win_counts.csv`
- 独立重算表与 exact-diff 报告
- 新图 PNG/PDF、caption draft、`figure_qa_manifest.json`
- `data_source_registry.csv/json`
- `raw_file_manifest.json`
- `observation_alignment_report.json`
- `data_preprocessing_lock.json`
- `label_free_cache_manifest.json`
- `semantic_label_access_log.jsonl`
- `baseline_reproducibility_matrix.csv/json`
- 每方法 source commit/env/license/tutorial/input/output SHA 与 attempts 日志
- `night4b_benchmark_matrix_proposal.csv/json`
- `night4b_protocol_draft.json`
- 资源汇总
- 全部新测试日志与失败日志
- `SHA256SUMS`
- Git bundle
- 精简 planner handoff
- 完整 artifacts archive
- `delivery_index.json`
- `shutdown_confirmation.json`

报告首页必须明确：

- Night‑3B 原始结果是否改变：必须为 `NO`；
- Night‑3B 决策是否改变：预期为 `NO / MIXED_EVIDENCE`，若不是则立即升级；
- 本轮是否运行正式 benchmark：必须为 `NO`；
- semantic label access 是否发生在数据/协议锁后；
- 核心五方法各自状态；
- held-out 与 independent 数据各自资格；
- 最终门：三种之一；
- 所有偏差、compatibility patch 和失败。

## 13. 本地交付与 Worker 接棒

完成后把所有交付复制到一个新的本地目录，例如：

`C:\Users\李昌赫\Documents\Codex\2026-08-13\SpaLORA-night4a\work\night4a_handoff_20260813`

然后只需在 Codex 对话中告诉规划 Worker：

1. 本地目录绝对路径；
2. report、planner handoff、full archive、bundle、delivery index、shutdown confirmation 的绝对路径；
3. 三个大文件 SHA‑256；
4. final commit/tag；
5. Night‑4A gate；
6. 远端最后一条命令是否为 `/usr/bin/shutdown`，之后是否未重连。

无需用户手工上传；两个对话通过共享本地路径交接。规划 Worker 将独立读取后决定是否签发 Night‑4B，执行 Codex不得自行继续。

## 14. GitHub 与关机

- push 前先用无副作用方式检查凭据是否存在；没有凭据时不要盲目重试。
- 若 push 失败，只允许一次已记录尝试，随后依靠 Git bundle 完整交付。
- 下载并本地校验所有交付后，远端最后一条命令必须严格为：

```sh
/usr/bin/shutdown
```

- 命令之后不得重新连接服务器。

## 15. 允许的最终科学表述模板

Night‑4A 结束时只能写类似：

> Night‑3B evidence repair preserved all locked numerical results and the preregistered `MIXED_EVIDENCE` decision. Candidate external datasets and baseline implementations were evaluated for provenance, task compatibility, and end-to-end reproducibility before any formal benchmark outcome was generated. The project is [READY_FOR_NIGHT4B / READY_WITH_REDUCED_MATRIX / BLOCKED_PREFLIGHT] under the preregistered entry gates.

禁止写“SpaLORA 优于现代方法”“最终架构已被外部验证”或任何未在本轮执行的 benchmark 结论。

