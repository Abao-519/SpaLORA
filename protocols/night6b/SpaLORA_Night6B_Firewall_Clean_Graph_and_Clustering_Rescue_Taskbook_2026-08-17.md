# SpaLORA Night-6B：防火墙清洁的图尺度与聚类后端救援任务书

日期：2026-08-17  
任务性质：GPU + CPU 研发实验；A1/外部 tonsil 开发；候选冻结  
上游提交：`7f204a56690768f22bd06e0dac1b5785c97c4c70`  
计划分支：`revision/q2-night6b-graph-affinity-rescue-20260817`  
保护标签：`baseline/pre-night6b-graph-affinity-rescue-20260817`  
最终标签：`night6b-final-20260817`

## 0. 唯一目标

本轮只做一件事：在同一套全局规则下，判断并利用当前 SpaLORA/C04 encoder 的两个潜在数值瓶颈：

1. A1/tonsil 的空间图 k=18 是否过宽；
2. 只保存 fused embedding 并固定使用 PCA20 + mclust EEE，是否浪费了 modality-private 信息和更合适的 affinity/cluster head。

本轮不新增 loss 模块，不做正式 benchmark，不制投稿图，不写论文结论，不运行 D1/P22/GSE198353/Night-4B。

## 1. 权威输入

必须从 D 盘读取并核验以下文件；README prompt 中给出的 SHA-256 是启动硬门：

1. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night6A_Independent_Planner_Audit_2026-08-17.md`
2. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Post_Night6A_Research_Decision_2026-08-17.md`
3. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night6B_Candidate_Registry_2026-08-17.json`
4. 本任务书
5. Night-6A compact handoff：  
   `D:\文档\ChatGPT\博士第一篇科研论文项目\night6a_handoff_20260814\official_compact`

Night-6A 的唯一权威状态为 `IMPLEMENTATION_SEMANTICS_INVALID`。只允许用它定位代码、协议错误和交付链；禁止用其 checkpoint、embedding 或候选指标作为 Night-6B 正式证据或候选选择输入。

## 2. 终态枚举

最终状态只能是：

- `NIGHT6B_CANDIDATES_LOCKED_FOR_D1_P22`
- `NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE`
- `BLOCKED_INPUT_INTEGRITY`
- `BLOCKED_ONTOLOGY_CONTRACT`
- `LABEL_FIREWALL_BREACH`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BUDGET_EXHAUSTED`
- `INFRASTRUCTURE_BLOCKED`

无论成功或失败，只要连接过服务器，都必须保留已完成证据、完成最小交付并执行关机流程。

## 3. P0-PROTECT：历史、Git 和路径隔离

1. 只从已推送的权威 Git 历史建立 `/root/autodl-fs/SpaLORA-night6b`；不得把未审计目录当基线。
2. 核验上游提交与 `night6a-final-20260814` 可达且精确匹配。
3. 从上游提交建立计划分支，创建保护标签并普通 push。若同名 ref 已存在且不一致，立即停止；不得移动或 force。
4. 将四份规划文件原样复制到 `protocols/night6b/`，保存 SHA。
5. Night-6A compact index 按其 `handoff/` 目录解析相对路径，必须 16/16；不得误用 compact 根目录解析。
6. 记录 Python、R、mclust、PyTorch、CUDA、GPU、scikit-learn、anndata、h5py、Git 版本。
7. 对以下字符串建立路径访问守卫并带正负测试：`P22`、`Human_Lymph_Node_D1`、`GSE198353`、`night5d` 指标目录、`night6a_raw_runs`。
8. P0 后所有 raw runs、cache、checkpoint 写入新的 Night-6B 根，不覆盖任何历史文件。

通过后写入：`P0-PROTECT PASS; BEGIN P0-ONTOLOGY`。

## 4. P0-ONTOLOGY：独立 data-steward 合约

Night-6A 的 K=4 不能沿用。Night-6B 必须先把“已知 K benchmark protocol”与“训练/候选评价”分开。

### 4.1 角色隔离

建立三个逻辑角色，至少用独立进程和独立输入根实现：

- `data_steward`：只负责数据来源、标签 ontology 和 label-free 副本；
- `trainer_transformer`：只能读取 label-free cache、known K 和候选注册表；
- `evaluator`：只能在各阶段所有 cluster outputs 总锁后读取 per-spot labels。

不得再用单一 `label_values_read` 布尔值。每一阶段记录：

- `deserialized_into_memory`
- `explicitly_indexed_or_observed`
- `used_for_training_or_selection`
- `authorized_role`

### 4.2 tonsil ontology contract

目标文件仍是 Night-6A 已核验来源的 section 1：

- `s1_adata_rna.h5ad`
- `s1_adata_adt.h5ad`

`data_steward` 被明确授权仅在本阶段读取 annotation，且必须：

1. 先核验原文件 SHA 与 Night-6A 记录一致：RNA `e1d99b34685805c93a7314f8ef4e07b2f92d244a8c29ac9fced9ee19ff3b96a9`；ADT `f7e7c2b723faf59fbf1d949c131b2f345c268787cd2b72177e0c3a139afa38be`。
2. 使用低层 HDF5 精确读取 `final_annot`、`lab`、`lab_lynn` 和 barcode index；禁止在本阶段运行模型或聚类。
3. 对每列记录 encoding、missing 数、唯一类别名、类别计数、类别序列 SHA；不得根据后续 ARI/NMI 选择列。
4. 注册表已经在读取前锁定目标列为 `final_annot`。known K 定义为该列非空唯一类别数。
5. 验证 RNA/ADT 同 barcode 的 `final_annot` 是否逐项一致；若只有 RNA 有权威列，明确记录而不是伪造 ADT 一致性。
6. 解释 4/6/7 公开口径：仅基于字段和公开来源形成 provenance，不用模型结果倒推。
7. 生成 `tonsil_ontology_contract.json`，包含 source SHA、barcodes SHA、target column、K、categories/counts、per-spot label vector SHA 和公开来源。
8. 若 target column 缺失、全空、RNA/ADT 发生无法解释的冲突，或实际 K 不在注册表记录的公开候选 4/6/7 中，终止为 `BLOCKED_ONTOLOGY_CONTRACT`，不得猜测。

这一步承认 known-K 使用了标签类别数；它是公开 benchmark protocol 输入，不是 label-free model selection。per-spot 值不得交给 trainer。

### 4.3 label-free 副本

ontology contract 完成后，使用低层 HDF5 从原文件复制：

- X、var、barcode index；
- `obsm/spatial`；
- 模态处理真正需要且不含 annotation 的字段。

新 AnnData 的 `obs` 必须零列。禁止对原始 tonsil H5AD 调用 `anndata.read_h5ad`；只允许对新 label-free 文件调用。对原始路径的 `anndata.read_h5ad` 必须有负向测试并 fail closed。

生成 source-to-label-free manifest、每个对象 SHA、shape、barcodes SHA 和零 `obs` 列证明。

通过后写入：`P0-ONTOLOGY PASS; K=<actual>; BEGIN P0-SEMANTIC`。

## 5. P0-SEMANTIC：实现硬门

正式训练前必须实现并通过下列测试。

### 5.1 注册表与图

1. 9/9 graph candidates、12/12 head candidates 可解析；序列化 config SHA 全部唯一。
2. 手工坐标例精确验证 k=3/6/10/18 的非 self 邻居数、距离同值 barcode tie-break、union 与 mutual 的差异。
3. 手工 feature 例验证 correlation/euclidean 与 k=10/20；禁止 silent fallback。
4. `G06/G07` 的 per-modality intersection 必须是相应 feature support 与 spatial support 的交集；不得把两个模态 feature graph 混为同一个。
5. intersection 或 mutual graph 后的孤点只保留 normalization self-loop；不得后设恢复邻居。所有 graph candidate 都必须记录孤点数和连通分量数。
6. 每个 graph candidate 的 ASR/Moran gene scoring 与训练空间图使用同一个候选空间规则。
7. 每个 dataset × graph candidate 建独立 content-addressed cache；manifest 包含所有数组、图、预处理参数和 canonical input SHA。

### 5.2 encoder 与全 view 输出

1. `E00_C04_B01` 与 Night-5 B01/C04 语义精确一致。
2. runner 必须保存 `emb_latent_omics1`、`emb_latent_omics2`、`SpaLORA_fused`、三类 attention，以及各数组 SHA。
3. 用 Night-5 A1 B01/C04 的有效 final checkpoint + 有效 cache 做只读 forward replay：fused cluster 和既有 ARI/NMI 必须精确复现；两个 private view 只作为新增派生 artifact。
4. A1 reference 复用失败即 `IMPLEMENTATION_SEMANTICS_INVALID`；不得偷偷重训后声称复用。
5. Night-6A invalid checkpoint 不得参与此 parity。

### 5.3 cluster heads

1. H00 必须与既有 PCA20 + mclust EEE 精确一致。
2. deterministic PCA、mclust seed、spectral seed/n_init 全部显式固定。
3. H02 的 BIC 只可在注册表 covariance models 中 label-free 选择；记录选中 model 和 BIC，不可用 ARI/NMI 选择。
4. H03/H04 block scaling、neighbor overlap weights 用解析小矩阵验证。
5. H05/H06 self-tuning affinity 在手工距离矩阵上逐元素验证；全程 sparse，禁止 4326×4326 dense affinity 常驻。
6. H07/H08 的 0.05/0.10 必须是分子 affinity 与固定 spatial-k6 affinity 的凸组合，权重不可反向。
7. H09 只在同一 run 的三个 view partitions 内做 co-association；不得跨 seed 偷看哪个 seed 好。
8. H10/H11 必须使用 H00 mclust posterior 作为 unary；无 posterior 时该 cell invalid，禁止换算法。ICM 更新顺序和 tie-break 固定。
9. 所有 head 接口若收到 per-spot label 数组、ARI/NMI 或 evaluator 路径必须抛错。
10. lower-is-better 方向必须覆盖 Geary C 与 boundary disagreement。

### 5.4 防火墙负向测试

至少覆盖：

- trainer 对原始 H5AD 的任何打开；
- trainer/transformer import evaluator；
- candidate transform 接收 label vector；
- evaluator 在 manifest 未总锁时启动；
- 读取 D1/P22/GSE198353；
- 根据中间 epoch/单 seed 指标改 config。

任一语义测试失败时不得开始正式训练。

## 6. 有效 reference 复用与新基线

### A1

- `G00/H00` seeds 0–4 使用 Night-5 B01/C04 的有效 checkpoint 和 cache做只读 forward replay，不计新训练；
- 必须生成新的 multi-view derivative manifest，但不得修改旧文件；
- fused embedding/cluster/指标 parity 是硬门。

### tonsil

- Night-6A 所有 checkpoint/embedding 均不可作为正式 reference；
- 在新 label-free cache、正确 ontology K、`G00` 下新跑 seeds 0–4；
- fixed final epoch，0 retry（除明确基础设施/实现错误）；
- 先锁 embedding、views、clusters 和 manifest，后评价。

## 7. 固定训练与评价规则

- seeds 固定 0–4；禁止 seed search、删除分叉或补跑数值不好的 seed。
- encoder、epoch、optimizer、learning rate、loss、attention、feature preprocessing 除注册表 graph 字段外全部固定。
- 所有候选使用同一条 A1/tonsil 规则；不得 dataset-specific k、metric、head 或 beta。
- fixed final checkpoint；禁止用标签 early stop 或选 checkpoint。
- `Q=(ARI+NMI)/2`；ARI、NMI 必须分别报告。
- 空间指标分别报告 neighbor agreement、Moran's I、Geary C、boundary disagreement；不得合成后掩盖退化。
- 空间门按注册表逐 dataset 对同 seed `G00/H00` 计算。
- 五 seed 是优化重复，不是生物学独立重复；本轮不做虚假的生物学显著性主张。

## 8. 预算与固定运行顺序

### 8.1 R1 graph 训练：34 units

- A1 `G00` seeds 0–1：有效历史 replay，0 新训练；
- tonsil `G00` seeds 0–1：2；
- `G01–G08` × A1/tonsil × seeds 0–1：32。

总计 34。按注册表顺序、dataset 顺序 `a1 -> tonsil`、seed `0 -> 1` 执行；不得按早期数值提前停某候选。

### 8.2 R1 head transforms：432 transforms

在全部 R1 training/embedding manifest 总锁后、读取任何 R1 per-spot label 前：

- 12 heads × 9 graphs × 2 datasets × seeds 0–1 = 432。

所有 cluster outputs 和 transform manifests 总锁后，evaluator 才可打开 A1/tonsil labels。

### 8.3 R2 completion 训练：最多 27 units

- 最多 4 个晋级 graph × A1/tonsil × seeds 2–4：24；
- tonsil `G00` seeds 2–4：3；
- A1 `G00` seeds 2–4：历史 replay，0 新训练。

R1+R2 最大科学训练 61。

### 8.4 R2 head transforms：120 transforms

H00 强制保留，再选最多 3 个非 reference head：

- 4 heads × (`G00` + 4 晋级 graphs) × 2 datasets × seeds 2–4 = 120。

正式 head transforms 最大 552；纠正 transforms 最大 48。科学训练纠正/基础设施 retry 最大 12，总训练尝试最大 73。

## 9. R1 锁定与晋级

### 9.1 graph 选择

只用 H00 比较各 graph 与同 dataset/seed G00/H00。

Balanced pool：

- macro ΔQ ≥ +0.005；
- worst-dataset mean ΔQ ≥ -0.005；
- 4 个 dataset-seed paired Q 至少 3 胜；
- 两数据集空间门均通过。

Accuracy pool：

- macro ΔQ ≥ +0.015；
- worst-dataset mean ΔQ ≥ -0.010；
- paired Q 至少 3/4；
- 空间门可以失败，但必须完整披露。

各 pool 最多 2 个，去重后最多 4 graph。没有满足者不得用负候选凑名额。

### 9.2 head 选择

H00 强制进入 R2。其余 11 个 head 同时计算：

1. `marginal ΔQ`：跨 9 graph、2 dataset、2 seed，相对同 graph H00 的平均；
2. `best-valid-combination ΔQ`：该 head 与任一满足 graph 基本安全条件组合的 macro ΔQ；
3. paired wins、空间 delta、失败 cell 数、运行资源。

最多选 3 个非 reference head：

- 1 个按正向 marginal ΔQ 最高；
- 1 个按正向 best-valid-combination ΔQ 最高；
- 若有正向候选，至少 1 个来自 sparse-affinity/partition-ensemble 家族；
- 余位按 macro Q、跨 graph 稳定性和资源排序；
- 同一 head 若依赖只在一个 seed 上翻转，标记 heterogeneous，不得伪装稳定。

所有规则已经固定；看见指标后不得新增 beta、k、mclust model 或组合。

## 10. R2 五 seed 总锁和最终候选

完成全部 R2 训练与 transform，总锁所有 outputs 后再评价。

最终比较单位是最多 20 个 `graph × head` 组合，相对 `G00/H00`。

### Balanced candidate

必须同时满足：

- A1+tonsil macro ΔQ ≥ +0.020；
- 两数据集 mean ΔQ 均 ≥ 0；
- 10 个 dataset-seed paired Q 至少 7 胜；
- 两数据集空间门均通过；
- 10/10 cells 完整、无语义 fallback；
- runtime 与 GPU/RSS 完整报告。

满足者按 macro ΔQ 排序，只锁 1 个；分差 ≤0.005 时优先更稳定、更轻量、机制更简单者。

### Accuracy-frontier candidate

必须同时满足：

- macro ΔQ ≥ +0.030；
- worst-dataset mean ΔQ ≥ -0.005；
- paired Q 至少 7/10；
- 10/10 cells 完整。

空间门可失败，但必须在候选名和决定文件中标记 `ACCURACY_FRONTIER_SPATIAL_TRADEOFF`，只锁 1 个。它不能冒充 balanced candidate。

若两条路线都没有候选，终态为 `NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE`。不得临时放宽阈值。

## 11. 禁止项

- 不运行、列举内容、解析或评价 D1/P22/GSE198353/Night-4B。
- 不使用 Night-6A invalid checkpoints、embeddings 或指标选候选。
- 不为 A1 与 tonsil 设置不同 graph/head 参数。
- 不用真实标签选择 K 以外的任何参数；target column 已锁为 `final_annot`。
- 不根据单 seed、最佳 seed、最佳 epoch 或人工看图选候选。
- 不删除失败尝试、不覆盖 raw run、不静默 fallback。
- 不把 implementation failure 当科学负结果。
- 不 force push，不移动已发布 tag。

## 12. 错误、重试和预算耗尽

只有以下情况可重试：

- 连接中断、磁盘瞬时错误、GPU 基础设施错误；
- 通过最小复现证明的实现语义错误。

每次都必须保留旧 attempt、failure JSON、stdout/stderr、配置和原因。数值差、空间差、loss 分叉、seed 异质性不是重试理由。

若错误影响一批结果：

1. 先标记全部受影响 cells invalid；
2. 计算纠正所需预算；
3. 在硬预算内完整补齐，否则终止；
4. 不得像 Night-5B 一样让错误语义结果混入候选表。

任何原始 tonsil annotation 被 trainer/transformer 进程反序列化，立即终止 `LABEL_FIREWALL_BREACH`；不得继续跑完再解释。

## 13. 必须交付的证据

大文件留在 `/root/autodl-fs`：

- raw runs、multi-view embeddings、model states、all caches、affinity matrices；
- 用 manifest、大小、SHA 保护，不下载到 D 盘。

D 盘只交付 `night6b_handoff_20260817/official_compact`，目标小于 100 MB，至少包含：

- `night6b_report.md`
- `ontology/tonsil_ontology_contract.json`
- `firewall/data_role_and_access_audit.json`
- `p0_semantic_contract.json`
- `candidate_registry_resolved.json`
- `graph_cache_manifest_index.json`
- `reference_reuse_and_multiview_parity.json`
- `r1_training_manifest.json`
- `r1_transform_manifest.json`
- `r1_decision.json`
- `r2_training_manifest.json`
- `r2_transform_manifest.json`
- `per_seed_metrics.csv`
- `graph_head_five_seed_summary.csv`
- `balanced_and_accuracy_frontiers.csv`
- `night6b_decision.json`
- `tests_and_invariance_audit.json`
- 实际实现代码、测试、四份原始协议文件；
- `delivery_index.json`（每个内部文件相对路径、size、SHA）；
- planner compact tar.gz；
- 从 Night-6A 到 Night-6B 的增量 Git bundle；
- shutdown dispatch status。

不得为节省空间删除配置、测试、逐 seed 指标、失败记录或最终代码；只排除可由服务器 manifest 定位的大矩阵/checkpoint/raw arrays。

## 14. Git 规则

1. 使用已配置的 GitHub SSH，普通 push。
2. 实现 commit、阶段证据 commit、最终交付索引 commit 均可正常提交。
3. final tag 只能在最终交付索引 commit 完成后创建一次、push 一次；不得移动。
4. 禁止 `--force` 和 `--force-with-lease`。
5. 最终核验 GitHub branch/tag peeled commit 一致并记录。

## 15. 关机规则

不使用 AutoDL API。用户会在粘贴启动提示前手动开机。

执行时预留同一远端 SSH 会话用于最后关机；全部远端计算、本地同步、哈希、Git push 和交付索引结束后：

1. `/usr/bin/shutdown` 必须是最后一条远端命令；
2. 记录派发方式、时间和 exit status；
3. 派发后绝不重连、绝不查询服务器；
4. 只声称“命令已派发”，除非用户另行从控制台确认。

若任务在 P0 阻塞，只要已连接远端，也必须先完成最小阻塞交付，再派发关机。
