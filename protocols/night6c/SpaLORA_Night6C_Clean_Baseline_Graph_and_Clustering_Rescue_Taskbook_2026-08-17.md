# SpaLORA Night-6C：clean baseline 图尺度与聚类后端救援任务书

日期：2026-08-17  
任务性质：GPU + CPU 研发实验；A1/tonsil 开发；fresh paired reference；候选冻结  
上游提交：`301f49be2ddd15d2d36823c194df3549874729a4`  
上游标签：`night6b-final-20260817`  
计划分支：`revision/q2-night6c-clean-baseline-graph-rescue-20260817`  
保护标签：`baseline/pre-night6c-clean-baseline-graph-rescue-20260817`  
最终标签：`night6c-final-20260817`

## 0. 唯一目标与本轮修订

本轮继续检验 Night-6B 尚未运行的两个假设：空间/feature 图规则是否限制 encoder，以及只用 fused PCA20+mclust EEE 是否浪费 private-view 信息。

Night-6B 因历史 C04/B01 checkpoint 从未保存而在训练前停止。本任务明确取消“历史 checkpoint replay”硬门，改为在 A1 和 tonsil 上为所有五个固定 seed 训练 fresh C04/B01 `G00/H00` reference。它们是正式科学训练，必须计入预算并保存可加载 checkpoint。

本轮不新增 loss、不做正式 benchmark、不制投稿图、不运行 D1/P22/GSE198353/Night-4B。

## 1. 权威输入与优先级

启动前必须逐字节核验：

1. `SpaLORA_Night6B_Independent_Planner_Audit_and_Night6C_Decision_2026-08-17.md`
2. `SpaLORA_Night6C_Clean_Baseline_Authority_2026-08-17.json`
3. `SpaLORA_Night6B_Candidate_Registry_2026-08-17.json`
4. 本任务书
5. Night-6B compact：`D:\文档\ChatGPT\博士第一篇科研论文项目\night6b_handoff_20260817\official_compact`

冲突优先级：本任务书 > Night-6C authority JSON > SHA 锁定的 Night-6B registry。Night-6B 原任务书仅作历史解释，不再是执行 authority。

原 registry 的 9 个 graph、12 个 head、seeds、指标、空间门、R1/R2 选择阈值全部保留；只覆盖 upstream、reference 获取方式、tonsil K 的已决值和训练预算。执行端必须生成完整 `candidate_registry_resolved_night6c.json`，不得在解析时新增参数。

## 2. 终态枚举

最终状态只能是：

- `NIGHT6C_CANDIDATES_LOCKED_FOR_D1_P22`
- `NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE`
- `BLOCKED_INPUT_INTEGRITY`
- `BLOCKED_BASELINE_SPECIFICATION`
- `BLOCKED_ONTOLOGY_CONTRACT`
- `LABEL_FIREWALL_BREACH`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BUDGET_EXHAUSTED`
- `INFRASTRUCTURE_BLOCKED`

只要连接过服务器，无论何种终态，都要保留证据、完成最小 compact/Git 交付并执行关机流程。

## 3. P0-PROTECT：历史、Git、路径和预算

1. 从已推送的 commit `301f49...` 建立新的 `/root/autodl-fs/SpaLORA-night6c`，不得在 Night-6B worktree 上直接继续。
2. 核验 `night6b-final-20260817`、远端 branch 和 compact Git state 都指向该 commit。
3. 创建计划分支与保护标签并普通 push；同名 ref 若已存在且不一致则停止，禁止移动或 force。
4. 把四份规划 authority 原样复制到 `protocols/night6c/`，记录 SHA。
5. 独立验证 Night-6B `handoff/delivery_index.json` 31/31；相对路径基准必须是 `official_compact/handoff`，`root=repo` 项以 compact 根解析。
6. 新根固定为：
   - raw：`/root/autodl-fs/night6c_raw_runs_20260817`
   - cache：`/root/autodl-fs/night6c_cache_20260817`
   - output：repo 内 `outputs/night6c_handoff`
7. Night-6B 的 raw/cache 不覆盖；只允许只读复用已哈希的 label-free tonsil 数据。
8. 记录 Python、R、mclust、PyTorch、CUDA、GPU、numpy、scipy、sklearn、anndata、h5py、Git 版本。
9. 硬预算：科学训练 66；实现/基础设施纠正 12；总尝试 78；正式 head transforms 552；纠正 transforms 48。
10. 对 D1、P22、GSE198353、Night-4B、Night-5D metric content、Night-6A raw/metric content 建路径访问守卫和正负测试。

通过后记录：`P0-PROTECT PASS; BEGIN P0-DATA-REUSE`。

## 4. P0-DATA-REUSE：只读继承有效 ontology 与 label-free 数据

### 4.1 tonsil

Night-6B 已由授权 data steward 锁定：

- target column：`final_annot`
- known K：4
- label-vector SHA：`a587e8029c83d002f96b72e40bb2fe4ea46867f9a655d01c963d94dd18897c9a`
- RNA label-free SHA：`a53bb5a1d5fcea356db65f8de9f8f6c158d1b96db8dfbef0b85d6d4d45131a0a`
- ADT label-free SHA：`0146aabbd9845b90932c68711c985832895f93082a25a6adabdeccd9795f0ce5`
- paired barcode SHA：`c57d45ada97c49b06dc9a1f2cadc713ece099182a6eaa2554789e848c8fda45f`

本轮 P0 只可：

1. 重算上述两个 label-free 文件的 size/SHA；
2. 对 label-free 副本确认 shape、barcode、spatial 和零 `obs` 列；
3. 重算 compact 内 ontology/manifest SHA。

训练锁定前不得再次打开原始 tonsil H5AD 的 annotation。若 label-free 文件丢失或 SHA 不同，终止 `BLOCKED_INPUT_INTEGRITY`，不得在同轮重建后继续。

### 4.2 A1

A1 trainer 只能读取既有无标签 matrices/cache/coordinates；manual annotation CSV 只对 evaluator 可见。记录 canonical input SHA、ordered observation SHA 和 cache provenance。

### 4.3 三角色隔离

继续使用独立进程/输入根：

- `data_steward`：本轮仅验证已锁定 contract 和 label-free data，不再读取 per-spot labels；
- `trainer_transformer`：只能读 label-free inputs、known K、resolved registry；
- `evaluator`：只能在阶段内所有 cluster outputs 总锁后读取该阶段 A1/tonsil labels。

每个阶段分别记录 `deserialized_into_memory`、`explicitly_indexed_or_observed`、`used_for_training_or_selection`、`authorized_role`。任一 trainer/transformer 对原始 tonsil H5AD 或 label vector 的反序列化立即终止 `LABEL_FIREWALL_BREACH`。

## 5. P0-BASELINE-SPEC：C04/B01 语义锁

缺失的是权重，不是训练定义。正式训练前必须从以下已授权材料独立重建 canonical C04/B01 config：

- Night-5 B01/C04 candidate registry；
- Night-5A 权威 runner 源码；
- A1 C04 seeds 0–4 的 run manifests；
- Night-6B 已复制的 registry 与代码历史。

将非科学字段（seed、输出路径、时间戳、运行资源、结果指标）剔除后，五个历史 seed 的训练语义必须一致。锁定并输出：architecture、input dims、latent dims、active losses、所有 loss coefficients、attention shrink、optimizer、learning rate、weight decay、epochs、scheduler、preprocessing、initialization、determinism flags 和 checkpoint policy。

若这些材料不能唯一确定训练语义，或彼此出现影响数值的冲突，终止 `BLOCKED_BASELINE_SPECIFICATION`；不得猜默认值。dataset、graph、seed 和 content-addressed cache 是仅有的允许变化字段。

fresh reference ID 固定为 `E00C_C04_B01_CLEAN/G00_SP18_F20_CORR_UNION/H00_FUSED_PCA20_MCLUST_EEE`。

## 6. P0-SEMANTIC：实现与 checkpoint 硬门

### 6.1 原 Night-6B 语义测试继续有效

正式训练前必须重新运行并通过原任务书第 5.1、5.3、5.4 节的 graph、head、lower-is-better 和防火墙测试。至少包括：

- 9/9 graph、12/12 head 可解析且 config SHA 唯一；
- kNN tie-break、union/mutual、feature-spatial per-modality intersection、孤点/连通分量；
- H00、BIC、concat、self-tuning sparse affinity、co-association、MRF 的解析测试；
- evaluator 早启、labels 传入 transform、protected paths、单 seed/中间 epoch 改 config 的 fail-closed 测试。

### 6.2 所有 view 必存

每个训练单元必须保存并在 manifest 登记：

- `emb_latent_omics1`
- `emb_latent_omics2`
- `SpaLORA_fused`
- `alpha_omics1`
- `alpha_omics2`
- `alpha_cross`

每个数组保存 shape、dtype、ordered observation SHA 和文件 SHA。

### 6.3 checkpoint round-trip

先用不含科研数据的两步 toy run 验证保存/加载实现。每个正式成功训练单元随后必须原子写入 `model_final.pt`，至少包含：

- `model_state_dict` 与 canonical tensor-state SHA；
- canonical training config；
- dataset、graph、seed；
- input/cache SHA；
- code commit 与 software versions。

训练进程结束后，必须用全新进程从 `model_final.pt` 加载，在同一 deterministic cache 上重新 forward：六类 view 按 `atol=1e-6, rtol=1e-5` 比较，H00 cluster labels 必须逐项一致。记录最大绝对/相对误差、checkpoint file SHA 和 canonical tensor-state SHA。

round-trip 不通过的 cell 无效，禁止进入任何 head transform。修复后是否重试只按第 12 节和预算执行。

原 Night-5 checkpoint replay、fused/ARI/NMI exact parity 不再是本轮硬门。不得把 embedding、state hash 或其他旧权重冒充 checkpoint。

通过后记录：`P0-SEMANTIC PASS; BEGIN R1 FRESH PAIRED TRAINING`。

## 7. 固定科研规则

- seeds 固定 `[0,1,2,3,4]`；禁止 seed search、删除分叉、补跑坏数值。
- fixed final epoch；禁止用标签 early stop、选 checkpoint 或选 epoch。
- encoder、loss、optimizer、epoch、preprocessing 对所有 graph 完全一致。
- 所有 graph/head 规则对 A1 和 tonsil 完全一致，不得 dataset-specific 设置。
- known K 只用于 benchmark cluster count；per-spot labels 不进入训练或 transform。
- `Q=(ARI+NMI)/2`；ARI、NMI 分别报告。
- neighbor agreement、Moran's I、Geary C、boundary disagreement 分别报告；Geary C 与 boundary disagreement 为 lower-is-better。
- 五 seed 是优化重复，不是五个生物学重复，不作虚假生物学显著性主张。

## 8. 固定运行顺序与预算

### 8.1 R1：36 training units

固定顺序：graph registry 顺序 `G00 -> G08`，每个 graph 内 dataset `a1 -> tonsil`，每个 dataset 内 seed `0 -> 1`。

- `G00-G08 × 2 datasets × seeds 0-1 = 36`。
- 不得在看见某个候选的早期数值后跳过其剩余 cell。

全部 36 个有效 training、views、checkpoint round-trip manifests 总锁后，执行：

- `12 heads × 9 graphs × 2 datasets × 2 seeds = 432` transforms。

432 个 outputs 和 manifests 总锁后，evaluator 才可读取 R1 A1/tonsil labels。

### 8.2 R1 晋级

完全沿用 SHA 锁定 registry 的规则：

Balanced graph pool：macro ΔQ ≥0.005；worst-dataset mean ΔQ ≥-0.005；paired wins ≥3/4；两数据集空间门通过，最多 2。

Accuracy graph pool：macro ΔQ ≥0.015；worst-dataset mean ΔQ ≥-0.010；paired wins ≥3/4；空间门可失败但必须标记，最多 2。

去重后最多 4 个非 reference graph。H00 强制保留；按原 marginal、best-valid-combination、family、稳定性和资源规则最多选择 3 个非 reference head。不得用负候选凑名额，不得看到指标后新增 beta、k、model 或组合。

### 8.3 R2：最多 30 training units

- fresh `G00 × 2 datasets × seeds 2-4 = 6`；
- 最多 4 个晋级 graph × 2 datasets × seeds 2-4 = 24。

全部有效 training、views、checkpoints 总锁后，最多执行：

- `4 heads × (G00 + 4 graphs) × 2 datasets × 3 seeds = 120` transforms。

R2 outputs 总锁后 evaluator 才可读取 R2 labels。

科学训练最大 66，head transforms 最大 552。未晋级导致的未用预算不是失败，也不得用于临时增加候选。

## 9. fresh reference 与历史诊断

所有候选 delta 必须相对同 dataset、同 seed 的 Night-6C fresh `G00/H00` 计算。禁止把 Night-5 数值拼入 denominator/reference 表。

在每阶段 outputs 已总锁且 evaluator 已按授权打开 labels 后，可另表计算 fresh G00 与 Night-5 C04 的同 seed ARI/NMI/Q 差异。该表标记 `HISTORICAL_DRIFT_DIAGNOSTIC_ONLY`：

- 不作 hard gate；
- 不触发重跑；
- 不用于选 graph/head；
- 不允许剔除任何 seed；
- 必须解释 input/cache/code/software 差异。

## 10. 五 seed 最终候选

最终比较最多 20 个 `graph × head` 组合，全部相对 fresh G00/H00。

Balanced candidate 必须同时满足：

- A1+tonsil macro ΔQ ≥ +0.020；
- 两数据集 mean ΔQ 均 ≥ 0；
- paired Q wins ≥7/10；
- 两数据集空间门均通过；
- 10/10 cells 完整、无 fallback；
- runtime、GPU、RSS 完整。

满足者按 macro ΔQ 排序，只锁 1 个；分差 ≤0.005 时优先稳定、轻量、机制简单者。

Accuracy-frontier candidate 必须同时满足：

- macro ΔQ ≥ +0.030；
- worst-dataset mean ΔQ ≥ -0.005；
- paired wins ≥7/10；
- 10/10 cells 完整。

空间门可失败，但必须标记 `ACCURACY_FRONTIER_SPATIAL_TRADEOFF`，只锁 1 个，不能冒充 balanced candidate。

两条路线均无候选则为 `NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE`，不得放宽阈值。

## 11. 禁止项

- 不运行、列举、解析、评价 D1/P22/GSE198353/Night-4B。
- 不用 Night-6A checkpoint、embedding 或指标；不以 Night-6B 空 outputs 作科研结果。
- 不替代/伪造历史 checkpoint，不从 SHA 反推权重。
- 不为 A1/tonsil 使用不同候选参数。
- 不根据单 seed、最佳 seed、最佳 epoch、人工图或中间指标改配置。
- 不静默 fallback，不删除失败 attempt，不覆盖历史或本轮 raw run。
- 不把实现失败当科学负结果。
- 不 force push，不移动任何已发布 tag。

## 12. 错误、纠正与预算

只有连接/磁盘/GPU 等基础设施错误，或用最小复现证明的实现语义错误，才可计入最多 12 次纠正。数值低、空间差、loss 分叉、seed 异质性不是 retry 理由。

影响一批 cells 时：先全部标 invalid；保留旧 attempt、stdout/stderr、配置、原因；计算完整纠正预算；只有总尝试不超过 78 才可按原固定顺序完整补齐，否则终止 `BUDGET_EXHAUSTED`。

checkpoint/round-trip 错误不得用“不需要 checkpoint 做本轮指标”为理由忽略，因为保存的 private views 和模型状态是下一轮可审计性的强制产物。

## 13. 必须交付

大文件留在 `/root/autodl-fs`：raw runs、caches、六类 views、affinity、model states。每个文件用 path、size、SHA 和 manifest 定位，不下载至 D 盘。

D 盘只交付 `night6c_handoff_20260817/official_compact`，目标 <100 MB，至少包括：

- `night6c_report.md`
- 四份 authority 原文
- `candidate_registry_resolved_night6c.json`
- `baseline_specification_contract.json`
- `p0_data_reuse_and_firewall_audit.json`
- `p0_semantic_and_checkpoint_contract.json`
- `checkpoint_roundtrip_index.json`
- `graph_cache_manifest_index.json`
- R1/R2 training 与 transform manifests
- R1 decision
- `per_seed_metrics.csv`
- `graph_head_five_seed_summary.csv`
- `balanced_and_accuracy_frontiers.csv`
- `historical_g00_drift_diagnostic.csv`
- `night6c_decision.json`
- tests/invariance/budget/access audit
- 实际代码与测试、失败记录、Git/delivery/shutdown state
- 每个 compact 内部文件的相对 path、size、SHA delivery index
- planner compact tar.gz 与从 Night-6B 到 Night-6C 的增量 Git bundle

禁止下载 raw arrays/checkpoints；也禁止为省空间删除逐 seed 表、失败记录、配置、代码或测试。

## 14. Git

使用已配置 GitHub SSH普通 push。final tag 只能在最终 delivery-index commit 完成后创建一次、push 一次；branch/tag peeled commit 必须一致。禁止 `--force`、`--force-with-lease` 和移动已有 tag。

## 15. 关机

不调用 AutoDL API。用户会在粘贴启动提示前手动以 GPU 模式开机。

预留同一 SSH 控制会话。全部远端计算、本地 compact 同步、Windows SHA 复核、Git push、bundle 和 delivery index 完成后，直接把 `/usr/bin/shutdown` 作为最后一条远端命令派发；随后不重连、不查询。只声称命令已派发，控制台关机由用户确认。

若 P0 阻塞，也必须先形成最小阻塞交付、普通 push，再关机。
