# SpaLORA Night-7A：CPU 跨图共识与外部 benchmark 预检任务书

日期：2026-08-18  
任务性质：无卡 CPU；零科学训练；固定候选开发选择；外部方法与新数据预检  
权威父提交：`e8a49fb874209b2bd4474691ee5d03ee7639c0c7`  
权威父标签：`night6d-final-20260817`  
计划分支：`revision/q2-night7a-cpu-consensus-preflight-20260818`  
保护标签：`baseline/pre-night7a-cpu-consensus-preflight-20260818`  
最终标签：`night7a-final-20260818`

## 0. 任务定位

Night-6D 已确认：预锁定的 `G04_SP10_F10_EUC_UNION/H05_EQUAL3_AFFINITY_SPECTRAL` 相对同轮 fresh `G00/H00`，在 D1 和 P22 的 mean ΔQ 分别为 `+0.030332`、`+0.031418`，Holm p 分别为 `0.001953`、`0.008789`，空间门均通过。该结论是当前权威证据，Night-7A 无权倒写。

但 2×2 机制分解显示，H05-only 在 D1/P22 的 ΔQ 为 `+0.04157/+0.05631`，而 G04-only 为 `+0.01244/-0.01480`，交互项也均为负。Night-6C 的 A1/tonsil 又表明 G04 在部分开发情境有补益。因此，本轮只回答：能否在不训练、不看标签的前提下，把已保存的 G00/G04 六个表示 affinity 做固定共识，得到比单一 G04/H05 更均衡且足以补偿双图成本的最终结构。

本轮同时完成两个工程预检：

1. 逐仓库检查近期方法的真实代码、标签使用、终点、环境和许可证，形成公平 benchmark 可运行清单；
2. 只读取公开 metadata，锁定一个 fresh 人类 RNA+protein 数据候选和一个 fresh 小鼠 RNA+ATAC 数据候选，不下载大型原始数据、不读取 per-spot labels。

本轮不训练 encoder，不跑 diffusion，不用 GPU，不跑正式外部 benchmark，不制投稿图，不声称 SOTA。

## 1. 权威文件与优先级

启动前必须逐字节核验：

1. `SpaLORA_Night6D_Independent_Planner_Audit_and_Night7A_Decision_2026-08-18.md`
2. `SpaLORA_Night7A_Consensus_Registry_2026-08-18.json`
3. 本任务书
4. Night-6C compact：`D:\文档\ChatGPT\博士第一篇科研论文项目\night6c_handoff_20260817\official_compact`
5. Night-6D compact：`D:\文档\ChatGPT\博士第一篇科研论文项目\night6d_handoff_20260817\official_compact`

冲突优先级：本任务书 > Night-7A registry > Night-6D/Night-6C 交付证据。候选公式、候选顺序、数据集、seed、K、门槛或排序不得根据结果修改。

Night-6C internal `77/77`、external `4/4`、post-dispatch `3/3`；Night-6D internal `68/68`、external `5/5`、post-dispatch `3/3`。必须独立按各自索引根规则复核，不得仅相信报告中的数字。

## 2. 允许终态

最终科学/工程终态只能是：

- `KEEP_CONFIRMED_G04_H05_FOR_EXTERNAL_VALIDATION`
- `LOCK_NEW_CONSENSUS_FOR_FRESH_EXTERNAL_VALIDATION`
- `NO_FINAL_STRUCTURE_READY`
- `BLOCKED_INPUT_INTEGRITY`
- `BLOCKED_SOURCE_ARTIFACTS`
- `LABEL_FIREWALL_BREACH`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BUDGET_EXHAUSTED`
- `INFRASTRUCTURE_BLOCKED`

任何 Night-7A 失败都不得使已成立的 Night-6D confirmation 失效；它只表示本轮没有产生可采纳的新结构或预检不完整。

## 3. P0-AUTHORITY：实例、Git、目录与预算

### 3.1 实例边界

1. 用户必须已在 AutoDL 控制台用“无卡模式”开机；执行者不得调用 AutoDL API。
2. 登录后记录 CPU、内存、磁盘和环境；设置 `CUDA_VISIBLE_DEVICES=""`。
3. 本轮 `torch.cuda.is_available()` 必须为 false，所有结果中 GPU allocation 必须为零。若实例意外带 GPU，也禁止使用；无法强制隔离则 `INFRASTRUCTURE_BLOCKED`。
4. 科学训练、checkpoint forward、diffusion 均严格为零。

### 3.2 Git 与新目录

1. 从 `e8a49f...` 建新 worktree `/root/autodl-fs/SpaLORA-night7a`，不得直接修改 Night-6D worktree。
2. 创建计划分支和保护标签，普通 push。同名 ref 若存在且不一致，立即停；禁止删除、移动或 force。
3. 三份规划 authority 原样放入 `protocols/night7a/`，记录 SHA。
4. 新目录固定为：
   - consensus raw：`/root/autodl-fs/night7a_consensus_20260818`
   - external sources：`/root/autodl-fs/night7a_external_sources_20260818`
   - lightweight metadata：`/root/autodl-fs/night7a_dataset_metadata_20260818`
   - repo output：`outputs/night7a_handoff`
5. Night-6C/Night-6D raw、cache、compact 和 Git refs 全部只读；不得覆盖。

### 3.3 硬预算

- scientific training：`0/0`
- checkpoint forward：`0/0`
- diffusion：`0/0`
- 正式 consensus transforms：`360/360` 上限
- implementation/infrastructure transform corrections：最多 12
- 总 transform attempts：最多 372
- 正式 benchmark runs：`0/0`
- fresh label reads：`0/0`
- Windows D 盘新增 compact 目标 `<20 MB`；绝不下载 views、affinity、checkpoint、cache 或外部大数据
- 外部源码/metadata 只保留在 `/root/autodl-fs`；不得下载 Zenodo `data_integration.zip` 等大型 archive

通过后写入 `p0_authority_and_budget.json`，明确：`P0-AUTHORITY PASS; BEGIN P0-SOURCE`。

## 4. P0-SOURCE：历史证据与 60 个 views 文件

### 4.1 只读输入矩阵

固定数据集与 seed：

| dataset | role | K | seeds | G00/G04 来源 |
|---|---|---:|---|---|
| A1 | development | 10 | 0–4 | Night-6C；seed 0–1 R1，2–4 R2 |
| tonsil | development | 4 | 0–4 | Night-6C；seed 0–1 R1，2–4 R2 |
| D1 | development after confirmation | 10 | 0–9 | Night-6D |
| P22 | development after confirmation | 9 | 0–9 | Night-6D |

共 30 个 dataset-seed 单元、每个单元 G00/G04 各一个 `views.npz`，必须定位 60/60。只允许从权威 `raw_artifact_manifest.csv` 的 absolute path 解析；不得以目录 mtime、文件名猜测或“最新 attempt”替代。

每个文件必须核验：

- byte size 与 SHA-256；
- dataset、graph、seed、stage 和唯一 attempt；
- observation IDs、顺序和数量；
- 六个所需表示中每个数组的 key、shape、dtype、finite；
- 同 dataset/seed 的 G00 与 G04 observation IDs 完全相同；
- label-free coordinates 的 observation order 完全一致；
- 不读取 checkpoint、不进行 forward；本轮只消费已锁定 views。

若一个权威 views 文件缺失、SHA 不符或顺序不能证明一致，终止为 `BLOCKED_SOURCE_ARTIFACTS`，不得重训补齐。

### 4.2 历史预测复用

从锁定 transform manifest 精确定位同 dataset/seed 的：

- G00/H00：所有 candidate 的 paired reference；
- G00/H05：C01 parity；
- G04/H05：C00 parity 和 Night-6D confirmed comparator；
- G04/H00：机制/完整性诊断，非候选 reference。

这些是模型预测，不是 ground truth，可以在开标签前读取。必须核验 cluster file SHA、spot ID 和顺序。比较 partition 时用 deterministic first-occurrence canonical relabeling；不得仅比较数值 label 名称。

### 4.3 历史 metric 文件

开标签前允许计算 byte SHA，但禁止为 candidate 实现或早停解析 Night-6C `per_seed_metrics.csv`、Night-6D `d1_p22_per_seed_metrics.csv` 中的逐 seed 科学值。规划 authority 已披露的 aggregate 结论不构成新的访问；候选空间在本轮启动前已固定。

生成 `source_reuse_manifest.json`、`source_views_index.csv`、`source_prediction_index.csv`。60/60 与全部 prediction SHA 通过后进入 P0-SEMANTIC。

## 5. P0-SEMANTIC：实现硬门

### 5.1 复用 H05 的唯一合法方式

从 Night-6D 父提交中定位并调用已经过测试的 H05 affinity 构造函数。禁止凭论文描述重新写一个“近似 self-tuning kernel”。若必须为 sparse 组合抽取公共函数，只允许语义保持重构，并先完成以下手工/合成测试：

1. row-L2、Euclidean k10、排除 self、observation-ID lexical tie-break；
2. local sigma、kernel 数值、maximum symmetrization 与 zero diagonal；
3. `S_g=(A_g1+A_g2+A_g3)/3`；
4. spectral precomputed、`discretize`、`n_init=20`、`random_state=2020`、known K；
5. sparse 运算不产生常驻 dense `N×N`；
6. NaN/Inf、zero degree、connected components、symmetry error、diagonal error均有显式审计；
7. 12 个候选公式逐项合成测试，并为 C07、C08、C09、spatial blend 写有手算期望值的小矩阵测试。

### 5.2 真实 parity 门

在不打开任何 ground truth 的情况下，对 30 个 dataset-seed 单元全部执行：

- 从 G00 三 views 重建 `S_G00`，spectral partition 必须与权威 G00/H05 partition 完全等价；
- 从 G04 三 views 重建 `S_G04`，spectral partition 必须与权威 G04/H05 partition 完全等价；
- C02 必须与六个单 view affinity 的算术均值在 sparse canonical representation 上误差 `<=1e-12`；
- 重复运行同一 transform，consensus SHA 与 canonical partition SHA 必须完全相同。

若真实 H05 parity 不是 30/30 × 2：不得放宽容差、换 spectral solver、重装成“更合适”版本或以 ARI 接近 1 代替 exact partition；终止 `IMPLEMENTATION_SEMANTICS_INVALID`。

通过后把源代码 SHA、环境、60 个 parity 结果锁入 `p0_semantic_contract.json`，提交一次 pre-science implementation commit。正式 360 cells 开始后，candidate registry 不得改。

## 6. C1-LOCK：360 个无标签 consensus transforms

### 6.1 固定顺序与全量执行

顺序固定：dataset `[a1, tonsil, d1, p22]` → seed 升序 → registry candidate order。12 个候选必须对全部 30 个 dataset-seed 单元尝试，共 360 个正式 cells。

不得因为某个候选早期表现、运行时间、连通性或人工观感而跳过剩余 cells。正式阶段看不到 ground truth，因此任何“表现不好”都不能成为停止理由。

### 6.2 每 cell 必须保存

远端 raw 中每 cell 至少保存：

- `clusters.csv`：spot ID + cluster；
- `consensus_affinity.npz`：canonical CSR；
- `candidate_audit.json`；
- input views SHA、six affinity canonical SHA、S_G00/S_G04 SHA；
- candidate-config canonical SHA、代码 commit、环境；
- output affinity SHA、cluster SHA、canonical partition SHA；
- shape、nnz、symmetry max error、diagonal max abs、finite、min/max；
- zero-degree count、connected-component count/size、spectral warning；
- runtime、process peak RSS、GPU allocation=0；
- status、attempt、correction ID、fallback=false、label access=false。

不得把这些大文件复制到 D 盘；D 盘只保存 manifests、指标、代码和测试。

### 6.3 失败与纠正

- 固定 candidate 在合法输入上发生 eigensolver/数值失败：原样保留，记 `scientific_numerical_failure`，不换 solver、不加 epsilon、不重跑该 cell；继续其他 cells。
- 在开标签前发现全局实现或基础设施 bug：允许纠正，但必须保留旧 attempt、明确 invalid 原因，对所有受影响 cells 统一重跑并计入最多 12 次 correction attempts；不得只补对结果有利的 cells。
- 一旦标签窗口打开，任何实现纠正、补跑、重新聚类或 candidate 变化都禁止。若此后发现语义错误，终止 `IMPLEMENTATION_SEMANTICS_INVALID`。
- 到 372 总 attempts 仍不完整，终止 `BUDGET_EXHAUSTED`。

### 6.4 总锁

生成不可变 `locked_consensus_transform_manifest.json`。只有以下全部成立才能开标签：

1. 360 cells 均有 success 或原样保留的 numerical-failure record；
2. success 的 clusters/affinity/audit 均有 byte SHA；
3. 30×2 H05 parity 仍通过；
4. candidate registry SHA、代码 commit、transform order、预算已锁；
5. evaluator 进程尚未启动；
6. per-spot ground truth 读取/反序列化计数均为零。

## 7. B1-SOURCE-PREFLIGHT：现代方法真实代码审计

本阶段不运行正式数据，只审计源码与做 toy/input-adapter smoke test。按 registry 顺序至少检查 SpatialGlue、Seurat/WNN、COSMOS、SMART、PRESENT、MultiGATE、SpatialCOC、SpaMode、SpaMCA、ARISE、GROVER、MultiSP、SpatialEx。

### 7.1 每个仓库必须记录

- canonical repository URL、resolved commit、default branch、submodule；
- LICENSE 文件与实际许可；README 自述但无 LICENSE 时不能写成已获许可；
- Python/R/CUDA/系统依赖和可复现环境风险；
- 真实入口、输入格式、是否要求 histology/private weights；
- 联合 representation 输出、最终 cluster endpoint、known K 的处理；
- 源码中 labels/ARI/NMI 的读取位置、best epoch/best seed/best cluster number 的选择；
- dataset-specific epoch、K、mask、loss、resolution 等硬编码；
- 官方教程是否真正到达共同 evaluator 所需终点；
- 在统一协议下需要的 fixed-final、label-free adapter；
- 预计 GPU、运行时间、内存、磁盘；
- 失败命令、唯一兼容性纠正与最终 readiness status。

### 7.2 预检状态

每个方法只允许以下之一：

- `READY_COMMON_PROTOCOL`
- `READY_WITH_FIXED_ENDPOINT_ADAPTER`
- `SOURCE_ONLY_LICENSE_BLOCKED`
- `BLOCKED_PRIVATE_ASSET_OR_THIRD_MODALITY`
- `BLOCKED_NO_JOINT_CLUSTER_ENDPOINT`
- `BLOCKED_ENVIRONMENT`

official tutorial 中若读取标签并保留 best ARI，必须照实记录；正式 benchmark 不得复现这种选优。任何 repo 无兼容许可证时只学习思想，不复制源码。

### 7.3 Toy smoke test

只对在现有 CPU 环境中无需大规模安装即可运行的高优先级方法/adapter 做最小 synthetic smoke test。不得为了“凑齐成功”反复改上游代码，不得用 A1/D1/P22/tonsil 正式输入，不得计作 benchmark 成功。

输出 `external_method_source_audit.csv`、`external_method_readiness.json`、`label_selection_code_audit.md`、`toy_smoke_test_manifest.json`。

## 8. B2-DATA-PREFLIGHT：fresh 数据 metadata 审计

只访问公开 landing page、API metadata、README、archive listing 或 HTTP headers。禁止下载大型 archive，禁止读取任何 per-spot annotation vector。

### 8.1 人类 RNA+protein 优先候选

优先审计 SpaMode 公开的三个人类 tonsil sections（Zenodo record 12654113）：

- section 是否与当前 tonsil/A1/D1 独立；
- RNA/protein 是否同 spot 配对，barcode 与坐标来源；
- 是否有官方人工/组织学 domain annotation；谁制作、如何制作、是否从同一 omics 特征聚类派生；
- section 数、spot 数、平台、文件清单、archive size/checksum、许可；
- 能否在不读 per-spot labels 的情况下锁 known K；
- 是否适合 primary fresh external confirmation，还是只能 development/replication。

### 8.2 小鼠 RNA+ATAC 优先候选

审计 GSE205055 / OEP003285 的 MISAR mouse embryo/brain 数据：

- 找出真正同一 section 的 paired RNA+ATAC 和空间坐标；
- 明确 tissue/development stage；
- 查明 annotation 是否独立、是否有组织学依据；
- 若只有作者聚类或 annotation 来源不清，降级为 exploratory，不得包装成 ground truth。

### 8.3 label-free 与 simulation

- GSE198353 两个 spleen replicate 维持 label-free 角色，除非找到可审计的官方人工域标签；
- simulation 可用于边界、噪声、模态缺失和机制测试，但不能替代真实外部验证。

每个候选生成一份 data contract draft，包含 URL、accession、文件/大小/checksum、配对、坐标、annotation provenance、license、known K、预计资源、是否需要用户人工下载。最终只给出：

- `READY_FRESH_ANNOTATED_CONFIRMATION`
- `READY_LABEL_FREE_REPLICATION`
- `NEEDS_MANUAL_PROVENANCE_REVIEW`
- `NOT_SUITABLE`

本轮不得据 metadata 好坏改变已经锁定的 12 个 consensus 候选。

## 9. E1-EVALUATE：单次开发标签窗口

只有 C1 总锁且 B1/B2 输出已 SHA 锁定后，才启动一个专用 evaluator 进程。

### 9.1 标签来源

优先复用 Night-6C/Night-6D evaluator 已使用并有 hash 的四个 label-vector snapshot；不得重新对原始 H5AD 调用 `anndata.read_h5ad`。逐个核验 label vector SHA、spot ID、顺序、K，与历史 evaluator contract 一致。

fresh external dataset label reads 必须仍为零。

### 9.2 一次性评价

同一 evaluator 一次性读取四个 development label vectors，计算全部 12 candidates 与 G00/H00 reference：

- ARI、NMI、`Q=(ARI+NMI)/2`；
- neighbor agreement、Moran's I、Geary C、boundary disagreement；
- 同 dataset/seed paired deltas；
- 每 dataset 的 mean、median、SD、wins、完整 exact sign-flip 和固定 100,000 次 paired bootstrap；
- 4 个数据集各权重 0.25 的 macro，禁止按 spot 数或 seed 数加权；
- 对 C00 的 secondary paired delta，用于双图复杂度门；
- 所有 candidate×dataset Q 检验的 exploratory Holm 表，但不把它冒充 confirmatory p-value。

使用固定 bootstrap seed `20260818`。所有 lower-is-better 方向单独核对。独立第二实现复算主键、Q、paired delta、wins、空间方向、门槛和最终排序，容差 `1e-12`。

### 9.3 开窗后禁令

标签打开后不得：

- 返回 affinity/cluster 实现；
- 重跑失败 cell；
- 改候选、参数、K、seed、门槛或排序；
- 根据结果添加一个“简单小变体”；
- 查看某 seed 后删除 outlier；
- 把 D1/P22 重新称为 holdout。

## 10. D1-DECIDE：固定选择规则

逐候选严格执行 registry 的 generalization gate：

1. 四数据集 mean ΔQ 均 `>=0.005`；
2. 四数据集 mean ΔNMI 均 `>0`；
3. mean ΔARI 至少 3/4 数据集 `>0`，且任何数据集不得 `<-0.005`；
4. Q wins：A1/tonsil 至少 4/5，D1/P22 至少 7/10；
5. 四数据集 median ΔQ 均 `>0`；
6. 四数据集空间保护门均通过；
7. 无 missing/silent fallback。

所有双图候选还必须补过复杂度门：相对 C00 要有预注册的 macro 或 worst-dataset 材料提升，且不允许用一个数据集的大涨掩盖另一个数据集超过 0.002 的退步。

eligible 候选按以下固定顺序排序：worst-dataset mean ΔQ → dataset-balanced macro ΔQ → 30 个 paired Q wins → future complexity → candidate order。

### 10.1 终态解释

- 无双图候选过材料复杂度门：`KEEP_CONFIRMED_G04_H05_FOR_EXTERNAL_VALIDATION`。这不是失败，而是拒绝为微小开发收益支付双 encoder 成本。
- 某双图候选过全部门并排名首位：`LOCK_NEW_CONSENSUS_FOR_FRESH_EXTERNAL_VALIDATION`。它只是四数据集 development-selected；不得继承 Night-6D confirmatory 地位。
- 源 artifact、语义、标签防火墙或预检存在无法解决的缺口：按对应 blocker/invalid 状态；Night-6D C00 证据仍保留。

输出至少包括：`night7a_decision.json`、`candidate_gate_table.csv`、`per_seed_metrics.csv`、`four_dataset_summary.csv`、`paired_delta_vs_g00h00.csv`、`paired_delta_vs_c00.csv`、`spatial_protection.json`、`exploratory_statistics.json`、`independent_recompute.json`。

## 11. 测试与不变性

至少覆盖：

1. 12 candidate registry parser 与 canonical SHA 唯一；
2. 60 views authority resolution；
3. G00/G04 observation parity；
4. H05 30×2 exact partition parity；
5. C02 six-view mean identity；
6. C03/C04/C05 sparse elementwise公式；
7. C06 zero-row与 symmetrization；
8. C07 Jaccard reliability 手算；
9. C08 support-weighted positive median 手算；
10. C09 simultaneous update、top-k tie-break 和 10 iteration determinism；
11. C10/C11 spatial k6、row-normalize、blend；
12. spectral fixed parameters、重复运行 deterministic；
13. 无 dense N×N 常驻；
14. label firewall 的正向和负向测试；
15. 360 主键唯一、固定顺序、预算；
16. Q、wins、lower-is-better、spatial gate、complexity gate、ranking；
17. external method label-selection scanner 的 synthetic fixture；
18. fresh metadata downloader 的 large-file refusal；
19. delivery index、Git ref、final-tag 单次创建守卫。

所有测试、失败输出和纠正历史进入 compact。不得只汇报最终 green tests。

## 12. Git、交付与空间控制

### 12.1 Git

1. 实现/测试、prelock manifest、evaluation/decision、final delivery index 分阶段 commit；
2. 使用 GitHub SSH 普通 push；禁止 force、force-with-lease、改写历史；
3. final tag 只能在最终 delivery-index commit 后创建一次并普通 push；绝不先打 tag 再移动；
4. 独立核验 remote branch/tag peeled commit；
5. 生成从 Night-6D 到 Night-7A 的增量 bundle，并本地验证。

### 12.2 D 盘 compact

固定目录：

`D:\文档\ChatGPT\博士第一篇科研论文项目\night7a_handoff_20260818\official_compact`

只交付：

- 报告、决定、registry、代码、测试；
- source/transform manifests 与 SHA；
- per-seed metrics、summary、gates、统计与独立复算；
- external source audit、readiness、toy smoke；
- fresh dataset metadata/data-contract drafts；
- failure/retry/budget/access/Git audit；
- planner handoff tar、增量 Git bundle、delivery indexes、shutdown record。

禁止复制：views、affinity matrices、raw runs、checkpoints、cache、external repos/envs、大型新数据。D 盘 compact 超过 20 MB 必须先查明原因；不得为缩小体积删除有用的小型证据。

内部 delivery index 必须从 clean staging root 生成，逐文件 SHA；Windows 端独立复核。报告中写清远端大文件保留路径与对应 manifest，而不是打包到本地。

## 13. 关机

无论成功、混合、失败或 blocker，只要连接过 AutoDL：

1. 先完成最小 compact、Windows SHA 复核、Git 普通 push、remote ref 核验；
2. 保留一条从启动开始就存在的 SSH 控制会话；
3. `/usr/bin/shutdown` 必须是最后一条远端命令，直接派发；
4. 命令后不再 SSH、SFTP、rsync、Git remote check 或状态探测；
5. 只声称“关机命令已派发”；AutoDL 控制台是否关机由用户确认。

不要调用 AutoDL API，不要因为本轮是无卡就省略关机。

## 14. 最终报告必须回答

1. Night-6D authority 和索引是否独立通过；
2. 60/60 views 是否按 manifest exact 定位；
3. H05 30×2 parity 是否 exact；
4. 360/360 cells、失败、纠正、总 attempts；
5. 每候选四数据集 ARI/NMI/Q、wins 与空间门；
6. 是否有候选通过 generalization + complexity gates；
7. 保留 C00 还是锁定新结构，以及为何；
8. 新结构若入选，明确它不是 confirmed，必须 fresh external validation；
9. 各外部方法真实源码 readiness、标签选优风险与许可证；
10. fresh 人类/小鼠数据候选的 annotation provenance 和角色；
11. 下一轮最小 GPU 任务应是什么，估算资源，但不得在本轮启动；
12. Git commit/branch/tag、bundle、compact、索引 SHA、关机派发状态。

## 15. 明确禁止

- 自动开机、AutoDL API、GPU、训练、diffusion、checkpoint forward；
- 重跑 Night-6C/Night-6D encoder；
- 新增/删除候选，改参数、seed、K、threshold 或 selection gate；
- per-dataset winner、best seed、outlier 删除、fallback；
- 开标签前读 per-spot labels，开标签后回到聚类；
- 读取 fresh external labels 或运行正式 benchmark；
- 用不同论文表格直接宣称 SOTA；
- 把代码无许可证写成可自由复用；
- 大文件下载到 D 盘；
- force push、移动 tag、final tag 提前创建；
- 关机派发后重连。

本任务的成功标准不是“必须找到更复杂的新方法”，而是用一次低成本、可审计、全量且不挑结果的 CPU 实验，决定是否值得放弃已确认的 C00；随后把 GPU 预算集中到真正 fresh 数据和同协议现代 baseline。
