# SpaLORA Night-9C 任务书：冻结层次融合在 E18.5 RNA+ATAC 上的外部确认

日期：2026-08-21  
运行模式：AutoDL 有卡模式，GPU 正式实验  
核心问题：Night-9B 的 `N02_HIER_ONLY` 是否是可泛化的 RNA+ATAC 专科组件，而不是 P22 特例。

## 0. 权威起点

### 0.1 Git

- 父提交：`e9bd62e2bb07c58f58c219956b8aaf090527c471`
- 父标签：`night9b-final-20260820`
- 新分支：`revision/q2-night9c-frozen-hierarchy-e18-5-confirmation-20260821`
- 保护标签：`baseline/pre-night9c-frozen-hierarchy-e18-5-confirmation-20260821`
- 最终标签：`night9c-final-20260821`
- 只允许普通 push；禁止 force、移动旧标签或改写历史。

### 0.2 Night-9B 权威交付

- 根目录：`D:\文档\ChatGPT\博士第一篇科研论文项目\night9b_handoff_20260821\official_compact`
- compact index SHA：`105f412ca406888da6b58b3419ce7e27caeef6a1f378a748fbe6896552a0268d`
- Night-9B report SHA：`8683db15871584073462e363a84e96d4a81d61ba646afa914fb77e23a2f294eb`
- N02 实现文件：`SpaLORA/night9b_racf.py`
- N02 实现文件 SHA：`5204e1cd9320336a424ccffd867269e64397c4a6a6210efe09ae614b0a58ab7f`

Night-9B compact 必须独立复核 71/71，才能进入数据预检。

### 0.3 冻结科学假设

- Reference：RNA+ATAC 家族权威 `F00/R02`。
- Frozen candidate：`N02_HIER_ONLY`。
- N02 配置严格为：
  - `common_graph_k = null`
  - `hierarchical_fusion = true`
  - `reliability_gate = false`
  - `dgi = false`
- N02 继续使用 Night-9B 的固定实现：RNA feature/spatial 先融合，再与 auxiliary branch 融合；最终表示保持 `.75 * family_reference + .25 * hierarchical_fused`。
- 禁止改变 `.75/.25`、loss 权重、初始化、latent dimension、epoch、optimizer、学习率、head、seed 或任何模块。
- E18.5 的任何结果不得用于回改 N02；本轮是确认实验，不是开发实验。

## 1. 新外部数据：MISAR-seq E18.5 S1

### 1.1 权威来源

- 原始项目：`https://github.com/gpenglab/MISAR-seq`
- 原始 accession：`OEP003285` / `SRP491963`
- PRESENT 官方教程：`https://bio-present.readthedocs.io/en/latest/Tutorial4_spatial-RNA-ATAC-data_representation_MouseBrain.html`
- 预期文件名：`E18_5_expr.h5ad`、`E18_5_atac.h5ad`
- 预期 spot：2,129
- 预期 RNA features：32,285
- 公开文献锁定 cluster K：10。K=10 来源必须记录为公开文献/教程先验，不得通过读取本地标签唯一值推导。
- 预期 annotation key：`Annotation_for_Combined`。

数据只下载到 `/root/autodl-fs/night9c_data_e18_5_20260821`；禁止下载到 Windows。

### 1.2 标注证据分级

P0 必须在不读取 label values 的前提下完成 provenance 审计：

- `TIER_A_MANUAL_ANATOMICAL`：有原研究、atlas/H&E 或公开 source-data 证据表明是人工解剖标注；
- `TIER_B_PUBLISHED_REFERENCE_CLUSTER`：是原研究公开的无监督 cluster 后再人工命名，不得称为 ground truth；
- `UNRESOLVED`：无法证明标注来源，立即 `BLOCKED_DATA_PROVENANCE`，不得训练。

无论 Tier A 或 B，F00/N02/SMART/PRESENT 都使用同一 reference array，配对比较仍公平；论文措辞必须按 tier 降级。

## 2. P0-DATA：数据、历史访问与标签防火墙

必须先生成并普通 push：

- `e18_5_data_provenance_and_firewall.json`
- `e18_5_source_file_manifest.json`
- `e18_5_label_free_copy_manifest.json`
- `night9c_p0_contract.json`

P0 规则：

1. 只用低层 HDF5 检查 group/key、shape、dtype、稀疏矩阵结构、obs index 和 `obsm/spatial`；禁止 `anndata.read_h5ad` 打开原始文件。
2. 在值层面禁止读取、打印、导出、排序、计数或哈希 `Annotation_for_Combined`。允许对整个原始 h5ad 文件做字节 SHA-256。
3. 低层构造两个新的 zero-obs label-free h5ad：只含 X、var、obs index、spatial、必要 uns；最终用 AnnData 打开副本并证明 `obs.columns == 0`。
4. RNA/ATAC obs index、spot 顺序和 spatial coordinates 必须完全一致；不允许 inner join 丢 spot。
5. 审计历史仓库、raw roots、交付及 shell logs 中是否存在 E18.5 模型结果或 label-value 访问证据。文件曾被下载但未读取标签不自动失去资格；若无法排除训练/选择使用，则将角色降级为 `NON_PRISTINE_EXTERNAL_REFERENCE` 并停止本确认任务等待规划方。
6. GPU/CUDA、磁盘、依赖和数据 SHA 全部锁定。无 CUDA 立即 `INFRASTRUCTURE_BLOCKED`。
7. 正式训练、embedding、partition 和 E18.5 label reads 必须仍为 0。

## 3. P1：冻结复现与现代强基线源码审计

### 3.1 F00/N02 真实语义门

- 从 Night-9B final commit 读取 N02，不复制粘贴重写实现。
- 对代码文件、N02 config canonical SHA、F00 config、family router、H05/K10 endpoint 建立 contract。
- 在 synthetic 和真实 E18.5 label-free 小样本上完成：输入维度、梯度、checkpoint round-trip、fixed endpoint、exact K10、六视图和资源分项测试。
- N02 必须与 Night-9B 的 P22 N02 做一次历史只读 parity，证明代码路径没有漂移；不得重训或覆盖 P22。

### 3.2 SMART 与 PRESENT 源码

正式可执行基线只包括：

- SMART：`https://github.com/Xubin-s-Lab/SMART-main`，优先审计 `SMART-reproduce` branch；
- PRESENT：`https://github.com/lizhen18THU/PRESENT`。

在数据和结果未知时分别 resolve 当前官方 commit，并立即记录 commit、tree SHA、license 和关键文件 SHA，之后不得移动版本。

源码审计必须逐文件回答：

- 是否导入 annotation/label；
- 是否逐 epoch 计算 ARI/NMI/AMI；
- 是否按标签或 best metric 选 checkpoint、resolution 或 cluster；
- 默认 optimizer/epochs/latent/graph/preprocessing；
- embedding 的固定无标签 endpoint 在哪里；
- native clustering 是否需要标签衍生 K。

任何标签选择路径必须禁用；只允许保留官方 fixed final/training-loss endpoint，并把修改做成最小 adapter，不改模型数学。

### 3.3 只作背景、不得伪造复现的来源

- CANDIES 论文报告 E15.5 ARI 0.5597，并在 E18.5 做了附加验证，但论文当前没有提供其自身实现源码，只列出第三方基线代码。因此标记 `SOURCE_UNAVAILABLE_NOT_EXECUTED`。
- soFusion 若不能在官方论文链接中找到可核验源码，同样只作背景。
- 不得根据论文方法描述自行写一个“CANDIES/soFusion”然后冒充官方基线。

输出 `night9c_modern_source_code_audit.md/json`。

### 3.4 P1 修正预算

- 正式科学训练前：最多 6 个不同根因 implementation correction cycles、3 个 environment-only cycles；全部保留。
- 一旦第一个正式训练 unit 启动，模型、adapter、preprocessing、endpoint 和 evaluator SHA 全锁。
- Scientific retry=0；fallback=0。

## 4. 正式实验

注册表：`SpaLORA_Night9C_E18_5_Registry_2026-08-21.json`。

### 4.1 Seeds 与训练单元

- Seeds：`0,1,2,3,4,5,6,7,8,9`。
- `B00_F00_R02_REFERENCE`：10 units。
- `B01_N02_HIER_FROZEN`：10 units。
- `B02_SMART_FIXED`：10 units。
- `B03_PRESENT_FIXED`：10 units。
- 总上限：40 training units。
- 每个 unit 60 分钟硬超时；失败原样保留，不重试、不缩短 epoch、不换 solver。
- F00/N02 20/20 完整是主假设评价的硬门；SMART/PRESENT 不完整只降低 benchmark 层，不得让已完整的主假设失效。

### 4.2 统一公平端点

- 所有方法使用相同2,129 spots、原始 modalities、spatial coordinates、K=10和 seed list。
- F00/N02 使用冻结 H05 K10。
- SMART/PRESENT 每个 seed 只训练一次，保留：
  - `NATIVE_FIXED`：官方无标签固定 endpoint；
  - `COMMON_H05_K10`：官方 embedding 输入相同 H05 K10。
- 直接数值排名只使用 `COMMON_H05_K10`；native lane 只作端点敏感性。
- 禁止 cluster-resolution search、标签选 K、标签选 epoch和 best seed。

### 4.3 预注册 endpoint-only recovery

若训练已成功且 checkpoint/embedding 已锁定，只因 R/依赖/输出序列化导致 endpoint 未生成：

- 每个外部方法最多2个 endpoint-only recovery units；
- 禁止重新训练；必须复用字节一致 embedding；
- 失败和恢复全部保留；
- label reads 必须仍为0。

这不计 scientific retry，但不得用于修复数值失败或 K mismatch。

## 5. 总锁与唯一标签窗口

以下全部完成、SHA 总锁且普通 push 后，才允许打开标签：

- 全部正式训练尝试及失败日志；
- embeddings、checkpoints、partitions、native/common endpoints；
- evaluator、K10 contract、resources 和 raw manifest；
- prelabel lock commit。

唯一 evaluator process：

1. 分别读取 RNA 与 ATAC 原文件中的 `Annotation_for_Combined` 各一次；
2. 检查两数组按 obs index 完全一致；不一致立即 `BLOCKED_INPUT_INTEGRITY`，不计算指标；
3. 验证 observed K=10 仅作完整性检查，不允许返回修改任何输出；
4. 计算 ARI、NMI、Q、AMI、V-measure、homogeneity、completeness、FMI、neighbor agreement、Moran's I、Geary C、boundary disagreement；
5. 标签窗口关闭后禁止返回训练、聚类或源码修改。

## 6. 主确认判定

唯一 primary contrast：`N02 - F00`，十 seed 配对。

必须同时满足：

- mean ΔARI ≥ `+0.010`；
- mean ΔNMI ≥ `+0.010`；
- mean ΔQ ≥ `+0.015`；
- Q wins ≥ `7/10`；
- two-sided exact paired sign-flip p < `0.05`；
- 100,000 次 paired bootstrap ΔQ 95% CI lower > `0`，bootstrap seed 固定为 `20260821`；
- 复用 Night-6D 权威 spatial-protection gate 并通过；
- worst-seed ΔQ ≥ `-0.010`。

若通过，N02 才升级为 `RNA_ATAC_HIERARCHY_EXTERNALLY_CONFIRMED`，并成为 RNA+ATAC family candidate；不自动替代 RNA+蛋白 C00。

若只部分通过，终态为 `NIGHT9C_HIERARCHY_EXTERNAL_MIXED_EVIDENCE`；若 mean ΔQ≤0 或空间门失败，终态为 `NIGHT9C_HIERARCHY_P22_SPECIFIC_KEEP_F00`。

## 7. 强基线层

SMART/PRESENT 比较是 secondary，不影响 N02-vs-F00 主确认的显著性。

报告：

- F00、N02、SMART、PRESENT 的十 seed mean/SD/CI、wins、runtime、peak GPU；
- N02 相对每个完整 common-head baseline 的 paired deltas；
- 若 N02 比所有完整强基线 mean Q 至少高 `+0.010`，且 ARI/NMI 均不低于对方 `-0.005`，可称 `STRONG_BENCHMARK_FRONTIER`；
- 不得称 SOTA，因为 CANDIES/soFusion 等方法未必有可执行源码或同协议结果。

## 8. 合法终态

- `NIGHT9C_RNA_ATAC_HIERARCHY_EXTERNALLY_CONFIRMED`
- `NIGHT9C_HIERARCHY_EXTERNAL_MIXED_EVIDENCE`
- `NIGHT9C_HIERARCHY_P22_SPECIFIC_KEEP_F00`
- `BLOCKED_DATA_PROVENANCE`
- `BLOCKED_INPUT_INTEGRITY`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `INFRASTRUCTURE_BLOCKED`
- `BUDGET_EXHAUSTED`

## 9. Git、交付与关机

- raw/checkpoint/embedding/affinity/data 全留在 `/root/autodl-fs`；不下载到 Windows。
- D 盘只交报告、逐 seed CSV、决策 JSON、source audit、测试、关键源码、增量 bundle和索引；compact目标 <10 MB。
- final tag 只创建一次，必须在所有需跟随标签的 tracked commit 完成后创建并普通 push。
- 最终回答先用通俗中文说明：新数据是否真正支持 N02、绝对分数多少、与 SMART/PRESENT 差多少、是否可升级为 RNA+ATAC 专科组件。
- 保留同一 SSH 会话，以 `/usr/bin/shutdown` 作为最后一条远端命令；派发后不重连，不虚报控制台关机状态。

