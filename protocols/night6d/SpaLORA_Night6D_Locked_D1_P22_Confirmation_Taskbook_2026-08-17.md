# SpaLORA Night-6D：锁定 D1 + P22 确认实验任务书

日期：2026-08-17  
任务性质：GPU + CPU；单一锁定候选确认；禁止研发选择  
权威父提交：`172cefba7559b34d2894ffa304553c0068ca23b9`  
权威父标签：`night6c-final-20260817`  
计划分支：`revision/q2-night6d-locked-d1-p22-confirmation-20260817`  
保护标签：`baseline/pre-night6d-locked-d1-p22-confirmation-20260817`  
最终标签：`night6d-final-20260817`

## 0. 唯一目标

本轮只回答一个问题：Night-6C 在 A1+tonsil 锁定的完整 `G04/H05` 组合，能否在 D1 人类淋巴结与 P22 小鼠脑上，相对同轮 fresh `G00/H00` reference 同时提高 ARI、NMI 和 Q，并保持空间结构。

本轮不是新一轮开发：不增加 graph/head/loss，不改参数，不看一个数据集后决定是否运行另一个，不做正式外部方法 benchmark，不制投稿图，不写论文胜利结论。

## 1. 证据地位

### D1

D1 是与 A1 同研究的人类淋巴结独立 section，具有 3359 个配对 RNA+protein spots 与十类 manual ground truth。Night-6C 选择期间对 D1 访问为零；它是本轮主要 held-out within-study confirmation。

### P22

P22 是 9196 spots 的 RNA+ATAC 跨组织/跨模态确认。P22 曾用于 Night-3B 架构分析和 Night-5D 评价，因此不是 pristine holdout；但是新锁定的 G04/H05 graph/head 组合从未在 P22 上运行或选择。本轮只能表述为预锁定的 cross-dataset confirmation。

两个数据集必须在同一标签窗口前全部训练、transform 和总锁。不得把 D1 结果作为是否运行 P22 的 gate，反之亦然。

## 2. 权威输入

启动前逐字节核验：

1. `SpaLORA_Night6C_Independent_Planner_Audit_and_Night6D_Decision_2026-08-17.md`
2. `SpaLORA_Night6D_Locked_D1_P22_Confirmation_Registry_2026-08-17.json`
3. 本任务书
4. Night-6C compact：`D:\文档\ChatGPT\博士第一篇科研论文项目\night6c_handoff_20260817\official_compact`

Night-6C `handoff/delivery_index.json` 必须按 root 规则独立复核 77/77；external index 4/4、local post-dispatch index 3/3。Git branch/tag 必须 peel 到 `172cef...`。

冲突优先级：本任务书 > Night-6D registry > Night-6C authority/registry。只有 registry 明示的数据集适配可以改变；G00/G04/H00/H05 算法与参数不得改变。

## 3. 终态枚举

最终状态只能是：

- `NIGHT6D_D1_P22_BALANCED_CONFIRMED`
- `NIGHT6D_D1_P22_ACCURACY_CONFIRMED_WITH_SPATIAL_TRADEOFF`
- `NIGHT6D_PARTIAL_OR_MIXED_EVIDENCE`
- `NIGHT6D_LOCKED_CANDIDATE_NOT_CONFIRMED`
- `NIGHT6D_CONFIRMATION_INCOMPLETE_NUMERICAL`
- `BLOCKED_INPUT_INTEGRITY`
- `BLOCKED_DATA_CONTRACT`
- `LABEL_FIREWALL_BREACH`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BUDGET_EXHAUSTED`
- `INFRASTRUCTURE_BLOCKED`

无论何种终态，只要连接过服务器，都要完成最小 compact/Git 交付并派发关机命令。

## 4. P0-PROTECT：历史、Git、路径与访问守卫

1. 从权威 commit 新建 `/root/autodl-fs/SpaLORA-night6d`；不得在 Night-6C worktree 上直接写。
2. 创建计划分支与保护标签并普通 push。同名 ref 不一致时立即停止；禁止移动或 force。
3. 三份规划 authority 原样复制到 `protocols/night6d/` 并记录 SHA。
4. 新目录固定为：
   - data：`/root/autodl-fs/night6d_data_20260817`
   - cache：`/root/autodl-fs/night6d_cache_20260817`
   - raw：`/root/autodl-fs/night6d_raw_runs_20260817`
   - repo output：`outputs/night6d_handoff`
5. Night-6C、Night-5D、Night-4A、Night-3AF 目录全部只读；不得覆盖任何历史文件。
6. P0 后建立 fail-closed 守卫：trainer/transformer 禁止访问 ground truth、原始 D1 H5AD、原始 P22 H5AD、A1/tonsil labels、GSE198353、Night-4B 和历史 metric tables。
7. 记录 Python、R/mclust、PyTorch、CUDA/GPU、numpy、scipy、sklearn、anndata、h5py、Git 版本。
8. 科学训练上限 40；实现/基础设施纠正 8；总训练尝试 48；正式 transforms 80；transform corrections 8。

通过后记录：`P0-PROTECT PASS; BEGIN P0-DATA-CONTRACT`。

## 5. P0-DATA-CONTRACT 与标签防火墙

使用三个独立逻辑角色/进程：

- `data_steward`：只做 byte hash、schema、barcode、label-free copy/cache 合约；
- `trainer_transformer`：只能读 label-free inputs/caches、known K、locked registry；
- `evaluator`：只有 40 training + 80 transforms 全部锁定后，才在一个授权窗口读取 D1 与 P22 标签。

每次访问记录 `deserialized_into_memory`、`explicitly_indexed_or_observed`、`used_for_training_or_selection`、`authorized_role`。

### 5.1 D1

先只做字节 SHA：

- RNA `4cf5a125027c8fe91d6b789dda066be703efd3f83a8745adf64a8667cd5bf9af`
- ADT `5249ac7d8b04fde673f8e0e6e663a70e339461fc4156a549de636b7494095abc`
- ground truth `524e2447b743b093afad440fea7cd28e63239fe0660b461f1c404aba68ba84f3`

训练锁定前：

1. 禁止对两个原始 D1 H5AD 调用 `anndata.read_h5ad`；
2. 用低层 HDF5 只复制 X、var/index、obs index/barcode、`obsm/spatial` 及确属建模所需的非 annotation 字段；禁止读取其他 obs values；
3. 新 RNA/ADT H5AD 的 `obs` 必须零列，ordered barcodes、n=3359 和 coordinates 必须精确一致；
4. 对新 label-free 文件保存 size、SHA、shape、barcode SHA、零 obs 证明；
5. 使用 Night-6C A1 的预处理合同建立新的 D1 base cache，不复用含标签对象；
6. known K=10 只作为已锁定 benchmark scalar 传给 head，不传 per-spot label。

ground-truth CSV 在总锁前只允许 byte hash/size，不得 parse header 或内容。任何 trainer/transformer 原始 obs 反序列化立即 `LABEL_FIREWALL_BREACH`。

### 5.2 P22

先只做字节 SHA：

- RNA `7478de043b784a1e1947064c6b3b9413c7c947b02adf6417f573203fa4e78d36`
- ATAC `1ee1b8a60ced6cd16c66f02ee77b8038878fec7c221d375c4b99c0e1aae9c34c`
- ground truth `ca2a702323b42a63588687be29bbffeed7b2d57586386e5a000b0f6135619b32`

trainer 不打开原始 H5AD，必须定位一个只读 Night-3AF deterministic P22 cache，并验证：

- recorded remote manifest SHA `254eda53a7e7a62e30ded97443878045b78af9e573cbb5e6f2c365488ddf8f81`
- canonical cache content SHA `895bfa73c763e1fa1992ddc732fc193c4e12760249b8636c98f1c5f3111dbc40`
- canonical model input SHA `1111f2a7b879a0770e31ca5d98ebb9d4ce407c9f7d3ea9cc6faf02f3c8f63a95`
- observation IDs file SHA `c4381314391f7bf7b05bb55b1c318c58903724a2dc560d1152447b02e704c7eb`
- coordinates file SHA `67e58c064a39ff60903b42e27cb07226b2bd2d42b6fcf363174a6ed8ae572541`
- manifest 中每个文件的 size/SHA，以及 n=9196、HVG=2000。

cache 不一致或缺失则 `BLOCKED_INPUT_INTEGRITY`；本轮禁止重建一个近似 cache 后继续。known K=9 仅作为 scalar。

### 5.3 总锁原则

在所有 40 training、40 checkpoint round-trip、80 transform/cluster manifests 总锁前：

- evaluator 进程不得启动；
- D1/P22 ground truth 不得解析；
- 不得计算 ARI、NMI、Q 或带标签空间指标；
- 不得人工查看 clusters 与 annotation 对齐图。

## 6. P0-SEMANTIC：实现与配置硬门

### 6.1 锁定实现

以 Night-6C commit 中以下源文件为实现母本并核验 SHA：

- `SpaLORA/night6c_pipeline.py`：`b1abd06ab1f5f30d8f3c1d16e89a2ab8c240b61422f8c2b9307adc0619a0e0d1`
- `scripts/night6c_train.py`：`3603b1d857750ff15f9c39b570f19171b7d59b2f53efbcab04aeb10eb820d2f9`
- `scripts/night6c_reload.py`：`32522425a5a56130a67e0b35928f48908e25abdde3b43982dd475a507d3237d9`
- `scripts/night6c_transform.py`：`a33494a19bbfb22768a13e87d0111e3ca663bd970c0268aecbeeadff258b5b0e`

允许新增数据集配置、seeds 5–9、confirmation manifests/statistics。禁止改动图构造、self-tuning affinity、spectral、H00、C04/B01 forward/loss/checkpoint 语义。

### 6.2 精确配置

Graph 只允许：

- G00：spatial k18 union；feature k20 correlation union；无 refinement；config SHA `9e4f637c...f1121`。
- G04：spatial k10 union；feature k10 Euclidean union；无 refinement；config SHA `abd14bfc...c639`。

Head 只允许：

- H00：row-L2 fused、full-SVD PCA20、mclust EEE、seed 2020；config SHA `6a934dde...374`。
- H05：三个 row-L2 views 分别构造 Euclidean k10 self-tuning sparse affinity，三者算术均值，maximum symmetrization，spectral `discretize`、n_init=20、random_state=2020；config SHA `03f542f9...c6f2`。

必须用手工小矩阵复核 kNN tie-break、self-tuning sigma、kernel、三 affinity 平均与 spectral 参数；禁止 dense N×N 常驻或 silent fallback。

### 6.3 Dataset-specific base contract

这不是候选调参。D1 固定继承 A1 已有 C04/B01 contract：HVG3000、latent64、200 epochs、loss factors `[1.9,2.5,1.5,10.0]`。P22 固定继承 Night-5D B01/C04 contract：HVG2000、latent128、1600 epochs、loss factors `[1.5,5.0,1.5,1.0]`。

在同一 dataset 内 G00/G04 除 graph cache 外所有 encoder、active-set IGE、optimizer、learning rate、initialization、fixed-final-epoch 语义完全一致。禁止根据数据标签或中间输出改变 config。

### 6.4 Checkpoint

先运行无科研数据 toy checkpoint round-trip。每个正式 training cell 必须原子保存真实 `model_final.pt`、canonical tensor-state SHA、file SHA、full config、dataset/graph/seed、input/cache/code/version provenance；然后由新进程 reload，在同 cache 重算六 views：`atol=1e-6, rtol=1e-5`，H00 clusters exact。

任何 round-trip 失败的 cell 无效，不得进入 transforms。

### 6.5 负向测试

至少覆盖：原始 D1/P22 H5AD 打开、ground truth parse、label payload、evaluator 早启、protected dataset/result、变更 k/head/config、按中间 epoch/seed metrics 决策。任一失败不得开始科学训练。

通过后记录：`P0-SEMANTIC PASS; BEGIN LOCKED 40-UNIT TRAINING`。

## 7. 固定运行矩阵与顺序

### 7.1 Training：40 units

固定顺序：dataset `d1 -> p22`；每个 dataset 内 graph `G00 -> G04`；每个 graph 内 seed `0 -> 9`。

`2 datasets × 2 graphs × 10 seeds = 40`。

全部训练与 round-trip 完成并总锁前，不运行任何 head，也不读取标签。数值差、loss 分叉或 seed 异质性不是重跑理由。

### 7.2 Transform：80 cells

训练总锁后，固定顺序：dataset `d1 -> p22`；graph `G00 -> G04`；head `H00 -> H05`；seed `0 -> 9`。

`2 datasets × 2 graphs × 2 heads × 10 seeds = 80`。

若 mclust/spectral 在某 cell 产生预注册数值失败，保留 failure 并继续其他固定 cells；禁止换 covariance、k、seed、assign_labels 或 fallback。primary 所需 cell 不完整时不得补造结论。

### 7.3 标签窗口

40+80 总锁 manifest 包含每个 tuple 的 config SHA、input/cache/checkpoint/views/clusters SHA、状态与失败原因。只有 manifest 总锁并通过独立计数后，evaluator 才在一个进程/窗口内解析 D1 与 P22 ground truth。标签打开后禁止返回训练或 transform。

## 8. 指标与 primary 确认统计

对四个 factorial 配置逐 dataset/seed 报告：

- ARI、NMI、`Q=(ARI+NMI)/2`；
- neighbor agreement、Moran I、Geary C、boundary disagreement；
- runtime、peak GPU、RSS；
- 所有 provenance SHA。

唯一 primary contrast：

`G04/H05 - G00/H00`

在 D1 与 P22 分别使用 10 个 paired deltas：

1. 枚举全部 `2^10=1024` 符号翻转，以 mean ΔQ 为统计量，单侧 gain 备择；
2. 两个 dataset-level raw p 做 Holm step-down；
3. 固定 seed `20260817` 做 100,000 次 paired bootstrap，报告 ΔARI/ΔNMI/ΔQ percentile 95% CI；
4. 报告 mean、median、sample SD、10 个逐 seed delta、ARI/NMI/Q wins。

某数据集的 `MATERIAL_ACCURACY_CONFIRMED` 必须全部满足：

- mean ΔARI > 0；
- mean ΔNMI > 0；
- mean ΔQ ≥ +0.010；
- Q wins ≥7/10；
- Holm-adjusted exact p <0.05；
- bootstrap ΔQ 95% CI lower >0。

方向正但门未全过：`POSITIVE_BUT_INCONCLUSIVE`。mean Q 非正或 ARI/NMI 方向不一致：`NOT_CONFIRMED_OR_MIXED_METRICS`。不得用 Q 掩盖 ARI 或 NMI 负方向。

## 9. 空间保护与总体终态

每个数据集相对同 seed G00/H00 计算平均空间 delta。若：

`(Δneighbor < -0.03 and ΔMoran < -0.03)`

或

`(ΔGeary > +0.03 and (Δneighbor < -0.03 or ΔMoran < -0.03))`

则空间门失败；boundary 只报告。

- D1 与 P22 均 material confirmed 且两者空间门通过：`NIGHT6D_D1_P22_BALANCED_CONFIRMED`。
- 两者均 material confirmed、但至少一个空间门失败：`NIGHT6D_D1_P22_ACCURACY_CONFIRMED_WITH_SPATIAL_TRADEOFF`。
- 仅一个 confirmed、一个 positive/inconclusive、两数据集方向冲突，或总体证据不能统一：`NIGHT6D_PARTIAL_OR_MIXED_EVIDENCE`。
- 两者均无 material gain 且无可信一致正向证据：`NIGHT6D_LOCKED_CANDIDATE_NOT_CONFIRMED`。
- primary cells 不完整：`NIGHT6D_CONFIRMATION_INCOMPLETE_NUMERICAL`。

即使两数据集都确认，也只能说该锁定机制得到 D1 held-out 与 P22 cross-dataset 支持；在公平外部 baselines 完成前不得声称 state of the art。

## 10. Secondary factorial 机制诊断

固定三项、不得用于重新选候选：

1. head-only：`G00/H05 - G00/H00`；
2. graph-only：`G04/H00 - G00/H00`；
3. interaction：`(G04/H05-G04/H00) - (G00/H05-G00/H00)`。

每项按 D1/P22 分别报告 paired mean、CI、wins 和 exact sign-flip；六个 secondary p 在同一 family 内 Holm。secondary 结果不能改变 primary 方法身份或阈值。

P22 在标签锁后可另表与 Night-5D B01/C04 历史数值比较，必须标记 `HISTORICAL_DIAGNOSTIC_ONLY`；不得把历史结果拼入 fresh primary reference。

## 11. 禁止项

- 不运行 A1/tonsil、GSE198353、Night-4B 或正式 benchmark。
- 不新增候选，不测试 H07/H08，不更改 G04/H05。
- 不根据 D1/P22 任一中间结果决定另一个是否运行。
- 不用标签选择 preprocessing、HVG、K 以外参数、epoch、checkpoint、seed、graph/head。
- 不删除失败 attempt，不覆盖 raw，不静默 fallback。
- 不把 P22 称 pristine holdout。
- 不把 seeds 当生物学独立重复。
- 不 force push，不移动已发布 tag。

## 12. 错误与预算

只有连接/磁盘/GPU 等基础设施错误或经最小复现证明的实现语义错误可纠正，最多 8 次。保留旧 attempt、stdout/stderr、配置和 failure JSON。

影响一批 cells 时先全体 invalid，再计算完整补齐预算；总尝试超过 48 则 `BUDGET_EXHAUSTED`。mclust/spectral 固定算法返回数值失败不属于 retry。

## 13. 必须交付

大文件留在 `/root/autodl-fs`：label-free D1、caches、raw runs、六 views、affinities、checkpoints。全部由 path、size、SHA manifest 保护。

D 盘只交付 `night6d_handoff_20260817/official_compact`，目标 <100 MB，至少包含：

- `night6d_report.md`
- 三份 authority 原文
- `p0_protect_audit.json`
- `d1_label_free_manifest.json`
- `p22_cache_reuse_audit.json`
- `data_role_and_label_firewall.json`
- `p0_semantic_contract.json`
- `locked_training_manifest.json`
- `checkpoint_roundtrip_index.json`
- `locked_transform_manifest.json`
- `label_window_audit.json`
- `d1_p22_per_seed_metrics.csv`
- `primary_confirmatory_tests.json`
- `secondary_factorial_tests.json`
- `spatial_protection.json`
- `resource_accounting.csv`
- `night6d_decision.json`
- tests/invariance/budget/failure/Git/shutdown evidence
- 实际代码与测试
- root-aware delivery index、planner compact tar.gz、Night-6C→Night-6D 增量 bundle

不下载 raw arrays、affinities、cache 或 checkpoints；也不得为了省空间删除逐 seed、失败、配置或测试证据。

## 14. Git

使用现有 GitHub SSH普通 push。final tag 仅在最终 delivery-index commit 完成后创建和推送一次；branch/tag peeled commit 必须一致。禁止 `--force`、`--force-with-lease`。

## 15. 关机

不调用 AutoDL API。用户会手动以 GPU 模式开机。

从开始保留同一 SSH 控制会话。全部计算、D 盘 compact、Windows SHA、Git push、bundle 和索引完成后，把 `/usr/bin/shutdown` 作为最后一条远端命令直接派发；随后绝不重连或查询服务器。只声称“关机命令已派发”，控制台状态由用户确认。

若 P0 阻塞，只要连接过服务器，也必须先完成最小阻塞交付和 Git，再派发关机。
