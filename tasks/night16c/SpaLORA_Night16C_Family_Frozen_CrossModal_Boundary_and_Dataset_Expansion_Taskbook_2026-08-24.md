# SpaLORA Night-16C 执行任务书

任务：家族冻结的跨模态边界场、可信 prototype 修复与数据扩展

## 0. 本轮真正要回答的问题

Night-16B 已证明公开标签辅助 benchmark HPO 可以继续刷新个别分数，但没有证明 unified decoder 本身有效。Night-16C 要回答：

> 一个共同的跨模态边界机制，能否在不更换整套模型的前提下，用同一组家族参数改善多个独立切片，并产生相对强 start 的独立增益？

分数仍然重要，但本轮不把“调到某个单数据集的峰值”当成方法创新。

## 1. 父级复核与启动

先复核 Night-16B compact 49/49、index/bundle SHA、final commit/tag。读取 report、decision、完整候选 ledger、family default、per-lane 参数、最小贡献表、label flow、failure ledger 和 Methods/HPO 草案。

若父级文件、hash、commit 不一致，立即停止。若仅 GitHub SSH key 缺失，可继续实验并依靠 bundle 交付。

从 parent final commit 新建：

- Branch：`revision/q2-night16c-family-frozen-crossmodal-boundary-field-20260824`
- Final tag：`night16c-final-20260824`

## 2. Stage 0：数据家族与 annotation 协议审计

### 2.1 家族定义

只允许两套 headline family config：

| family | 定义 | 当前物理单元 | 优先新增单元 |
|---|---|---|---|
| RNA_PROTEIN | matched spatial RNA + ADT/protein | A1、D1、tonsil s1/s2/s3 | SPOTS rep1/2、GSE308623 P10S1/S2/S3、Stereo-CITE |
| RNA_CHROMATIN | matched spatial RNA + ATAC/histone mark | P22、MISAR E15.5 | MISAR 其他 stages/replicates、P22 其他 slices/marks、GSE308623 P5、GSE205055、GSE263333 |

同一模型核心、同一 API、同一参数 schema；每家族只能冻结一组 headline 数值配置。morphology 和 batch ID 是有 presence mask 的可选输入，不是另换模型。

### 2.2 必须闭合的 annotation/K 冲突

建立 `annotation_protocol_registry.csv`，逐项记录物理数据、annotation 文件/hash/来源、label 定义、mask、K、论文用途：

- A1/D1：当前 K10 与 SMART A1 K7 的关系；
- tonsil：当前逐切片 K4 与 SMART-MS 联合 K6 的关系；
- P22：K9、SMART K12、18-state 的物理资产与 label 来源；
- MISAR：E15.5 K7/K12 与 E18.5 K10 等 stage-specific annotation。

同一物理样本的不同 label 粒度可以作为独立 protocol，但不得合并计票；公开数字只有在数据、mask、K、endpoint 相同或明确可比时才放入 same-protocol 表。

### 2.3 数据候选 registry

建立 12–20 个物理单元的 `dataset_family_registry.csv`，至少包括：accession/DOI、物种、组织、平台、模态、空间分辨率、N、feature 数、配对 ID、坐标、图像、annotation 来源与可信等级、K、下载状态、文件体量、许可证/引用、family、是否独立 study。

优先审计官方来源：SMART processed Zenodo、GEO、原论文 Data Availability、官方 GitHub reproduce branch。先查已有本地/AutoDL 资产，避免重复下载。只有 exact asset、大小、spot-ID 合约、annotation provenance、磁盘/inode 都闭合后才下载。优先单个 archive/h5ad，避免制造数百万小文件。

本轮目标不是把所有候选都下载；从 registry 中选择 4–8 个最能补足家族转移证据且资源可承受的新单元进入 P0/评价。

## 3. Stage 1：先诊断“错在哪里”

在 development units 上，从强 start、reduced views 和稀疏空间图计算：

- label boundary edge 与预测 boundary edge；
- boundary precision/recall/F1；
- 错误点到 annotation boundary 的图距离；
- split/merge 型错误分解；
- modality-specific edge change、跨模态 agreement/conflict；
- start bank 的 spot-level 稳定性与 prototype margin；
- 每个 family 的错误是否集中在边界、微小区域或域内部。

这些 label-based 项只用于公开 development diagnostics 与 family-level HPO，不进入 producer。先保存 producer outputs/hash，再由 evaluator 追加 labels/metrics。

如果大部分误差不在空间边界附近，或跨模态 edge field 与真实边界完全无关系，不要机械跑大网格；转向更强 start/representation objective，并在结果中登记 `START_GENERATOR_SIGNAL` 或负结论。

## 4. Stage 2：实现共同 CMBF-TPR 核心

### 4.1 跨模态边界场 CMBF

对每条稀疏空间边和每个模态计算局部稳健尺度下的变化。输出三种可解释状态：

1. `support`：各模态都支持域内连续；
2. `boundary`：各模态都支持真实分界；
3. `conflict`：模态之间对该边意见冲突。

三状态必须显式输出并可视化/消融。允许 Codex 根据诊断选择：

- 完全确定性的 robust-rank 公式；
- 小型单调可训练 edge calibrator；
- 两者并行后机械晋级。

若使用可训练 calibrator，训练目标不得用 annotation。可用 masked-view consistency、augmentation stability、互为近邻支持、跨模态可靠边伪目标和稀疏正则。输出转成各向异性 sparse conductance；support 边传播，boundary 边衰减，conflict 边避免错误跨模态传播并保留 self-return。

### 4.2 可信 prototype 修复 TPR

每个点的 trust 至少来自：

- 多起点分区一致性；
- 各模态到 robust prototype 的 margin；
- CMBF 局部 support/boundary；
- 簇内连通性与最小簇保护。

高信任核心锚定；只有低信任、靠近候选边界的点能移动。split/merge 必须有 boundary field 与多模态证据，不能只因簇小或离 prototype 远就发生。

### 4.3 可选的 representation objective

若诊断显示 post-processing 上限不足，优先实现 boundary-aware contrastive representation：support edge 为高置信正对，consensus boundary edge 为负对，conflict edge 不强迫 shared alignment。RNA+protein 与 RNA+chromatin 使用同一 encoder/fusion/core，只允许输入 adapter 处理 feature dimension。

### 4.4 可借鉴但不可冒充原创的组件

- BANKSY：邻域均值与方位/梯度特征；
- stLVG：距离、方向、角度加权；
- ARISE：RNA feature graph 与 spatial graph 交集形成 RNA 锚定 topology；
- PRAGA：prototype-aware graph aggregation；
- SpatialCOC：连续空间映射与跨组学校正；
- SpaMV：shared/private 分解。

必须读论文 Methods 与官方源码的真实入口，固定 commit/许可证，写 `source_code_collision_and_transfer_audit.md`。若 CMBF-TPR 与现有方法高度重合，立即收紧或改变机制，而不是换名字继续。

## 5. Stage 3：真实 P0 与效率漏斗

每个 family 至少一个现有单元、每种新增平台至少一个新单元做真实 P0：

`raw/registered input → preprocessing → sparse graph → model forward/loss → optimizer step（若可训练）→ checkpoint strict reload → CMBF → TPR → clustering endpoint → fresh-process replay`

必须列出每个真实 tensor shape、canonical partition 和有限梯度。先 1 seed smoke；通过后再扩展。

计算漏斗：

1. CPU/单 seed 做诊断与 50–150 个低成本 family config screen；
2. 每家族只晋级少数候选；
3. 真正可训练模块才使用 GPU；
4. 只对晋级配置做多 seed/held-out；
5. 复用 reduced views、sparse graphs、start banks、local compute kit；
6. 禁止 dense N×N。

工程 bug 可直接修复并登记，不设置“一次 correction 后必须自杀”的形式主义；公式或科学选择在 formal freeze 后修改，则完整重跑受影响单元。失败结果不得删除。

## 6. Stage 4：家族级 HPO 与冻结

公开 labels 可以用于 discovery units 的 family-level HPO，但必须如实称为 `label-assisted family-level benchmark HPO`。候选先生成、hash、锁定，再由独立 evaluator 读取 annotation。

若 2026-08-21 长期治理文件中“不得根据标签事后选配置”的旧句与本轮冲突，以用户在 Night-16A/16B 后作出的最新明确决定和本合约为准：公开 discovery annotations 可以用于透明、可复算的跨运行 benchmark HPO；但权限只到 family-level config 冻结为止，不授权标签进入单次 producer，也不授权对 held-out/new unit 再调参。

每个 family 只选一组 headline config。机械排序：

1. exact K、finite、无退化簇；
2. discovery studies 中 ARI/NMI 双升的 study 数最多；
3. 最差 study 的 ΔARI 最大；
4. study-balanced mean ΔARI 最大；
5. study-balanced mean ΔNMI 最大；
6. 复杂度更低。

可计算 per-dataset oracle/tuned best，但只能明确标为 development ceiling，不能作为 family-frozen headline 或新数据部署证据。

建议 discovery/freeze：

- RNA_PROTEIN：用 A1 + tonsil s1 或等价的两个不同 study 做选择；冻结后先运行 D1、tonsil s2/s3，再运行新增 SPOTS/P10 等。
- RNA_CHROMATIN：用 P22 K9 + MISAR E15.5 主协议做选择；冻结后运行新增 MISAR/P22/GSE205055/P5 单元。P22 K18、MISAR K12 只作 sensitivity。

由于现有单元都曾参与历史开发，D1/tonsil/P22/MISAR 不得包装成 pristine blind confirmation。真正新增、未参与 family HPO 的物理单元才可称 frozen transfer；没有 labels 的新增单元做 label-free/biological confirmation。

## 7. Stage 5：评价与贡献归因

有可信 annotation 的 lane：ARI、NMI、AMI、FMI、Homogeneity、V-measure，并给 best/median/mean/min、胜出 seed 数、最小簇、wall/GPU/RSS。

所有 lane：Moran’s I、Geary’s C、neighbor agreement、Silhouette、Davies–Bouldin、Calinski–Harabasz、跨模态 kNN preservation/相关性。无可信 annotation 不计算或制造 ARI/NMI。

每条 primary lane 至少保留：

- strong start only；
- CMBF-TPR full；
- boundary state disabled；
- conflict state disabled；
- trust gate disabled；
- generic repair only；
- directional features only（若使用）。

若 full 没超过同一 strong start，不能归因为方法增益。若只有 directional/BANKSY-style features 或 head 提分，按真实来源分类。

论文主表可以选择覆盖不同 family、platform、species、tissue 且 annotation 最可信的代表数据，但项目总 ledger 和补充材料必须保留所有实际运行的 eligible 单元与排除理由，不能按结果好坏秘密删除数据集。

## 8. 分数方向线

以下是现有 development frontier，不是停止上限：

| protocol | ARI | NMI |
|---|---:|---:|
| A1 K10 | 0.276003 | 0.421740 |
| D1 K10 | 0.365174 | 0.444577 |
| tonsil s1 K4 | 0.236536 | 0.317118 |
| tonsil s2 K4 | 0.258264 | 0.314324 |
| tonsil s3 K4 | 0.350644 | 0.309771 |
| P22 K9 | 0.595552 | 0.717931 |
| MISAR K7 | 0.541424 | 0.666798 |

继续向 same-protocol published context 和内部前沿推进，但不要牺牲 family-frozen 方法证据来只刷一条 lane。

## 9. 终态分类

- `FAMILY_FROZEN_METHOD_SIGNAL`：同一 frozen family config 在至少两个独立物理单元相对同一强 start 有独立增益；共同 core 在两个 family 均不退化或有正信号。
- `BOUNDARY_FIELD_LOCAL_SIGNAL`：CMBF-TPR 只在部分 family/study 有可归因增益。
- `START_GENERATOR_SIGNAL`：表示/start 改进有效，边界修复无独立贡献。
- `SCORE_FRONTIER_ADVANCE`：刷新可信分数，但贡献来源可能是 head/start/HPO。
- `DATASET_EXPANSION_READY`：新增数据 P0/协议闭合但尚无科学结论。
- `NO_ADDED_METHOD_SIGNAL`：实现正确但新机制无独立增益。
- 另保留实现/设施失败。

一次任务可有一个主分类和若干次级事实，但不能用次级分数掩盖主贡献结论。

## 10. 必交付

至少包括：

- `night16c_report.md`、`night16c_plain_summary.md`、`night16c_decision.json`
- `dataset_family_registry.csv`
- `annotation_protocol_registry.csv`
- `dataset_download_and_provenance_audit.json`
- `source_code_collision_and_transfer_audit.md`
- `real_path_p0_audit.json`
- `family_config_search_ledger.csv`
- `family_frozen_config.json`
- `per_dataset_oracle_development_ceiling.csv`
- `boundary_diagnostic_table.csv`
- `absolute_metrics_main_table.csv`
- `minimal_contribution_table.csv`
- `all_runs_and_failures.csv`
- `label_flow_audit.json`
- `resource_audit.json`
- `paper_methods_and_novelty_draft.md`
- `reviewer_risk_register.md`

报告先给“我现在需要知道的三件事”，每个缩写第一次出现即用中文解释；主表在 hash 之前；重要里程碑给 5–8 句导师汇报版。

## 11. Git、compact 与关机

普通 push branch/tag，不 force push。SSH key 若仍缺失，保留失败日志并生成唯一 final tag 的可恢复 bundle。生成 root-relative size/SHA-256 compact index；下载到 Windows 后独立复算 missing/size/SHA/extras。

所有交付与 Windows 验证完成后，最后一条远端命令严格为 `/usr/bin/shutdown`；派发后不重连。
