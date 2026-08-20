# SpaLORA Night-9B 任务书：公平 SOTA 差距校准与 RNA 锚定协同层次融合

日期：2026-08-20  
任务性质：GPU 有卡模式；先审计、后实现、再做锁定开发实验  
目标：把“是否真正接近强方法”与“怎样继续提高 A1/P22 分数”放在同一套公平、无标签调参的协议下回答。

## 0. 权威起点与本轮禁止事项

### 0.1 Git 起点

- 父提交：`aa933b8fc11fc05470287a21f80facd03d0acfb9`
- 父标签：`night9a-final-20260820`
- 新分支：`revision/q2-night9b-rna-anchor-cooperative-fusion-rnd-20260820`
- 保护标签：`baseline/pre-night9b-rna-anchor-cooperative-fusion-20260820`
- 最终标签：`night9b-final-20260820`
- 只允许普通 push；禁止 force、force-with-lease、移动旧标签或改写旧历史。

### 0.2 本地权威交付

- Night-9A：`D:\文档\ChatGPT\博士第一篇科研论文项目\night9a_handoff_20260820\official_compact`
- Night-8B cardinality-safe：`D:\文档\ChatGPT\博士第一篇科研论文项目\night8b_cardinality_safe_eval_handoff_20260820\official_compact`
- Night-7A：`D:\文档\ChatGPT\博士第一篇科研论文项目\night7a_handoff_20260818\official_compact`
- Night-6D：`D:\文档\ChatGPT\博士第一篇科研论文项目\night6d_handoff_20260817\official_compact`

### 0.3 当前科学基线

- RNA+蛋白家族：`C00_G04_H05_CONFIRMED`。
- RNA+ATAC 家族：保留 `F00/R02`；MISAR 上已确认准确性提升，但运行时间约为 U00 的 2.26 倍。
- Night-9A 已证明：现有轻量 topology-transfer 候选不能在保持分数与 fidelity 的同时把运行时间压至 1.50 倍以内。因此本轮不再继续 E01-E09 的小变体搜索。
- 允许按模态家族路由；不得为单个数据集、单个 seed 或标签类别另设参数。

### 0.4 永久禁止

- 禁止用标签选 epoch、超参数、seed、cluster resolution、候选或失败重跑。
- 禁止训练期间计算 ARI/NMI/Q，禁止 `best_ari`、`best_nmi`、`best_metric` 或同义逻辑。
- 禁止在锁定前用 `anndata.read_h5ad` 打开含标签的原始 h5ad；元数据审计必须用低层 HDF5，并排除 `obs` 值。
- MISAR `Y` 已达到该 lineage 的最终读取次数；Night-9B 禁止再次读取 MISAR `Y`，也不得重新评价 MISAR。
- 禁止把 ARISE 官方源码中逐 epoch 使用真实标签选最佳 embedding 的路径带入任何正式实验。
- 禁止改变历史产物、补写历史 checkpoint、覆盖失败单元或删除负结果。
- 禁止调用 AutoDL 开关机 API。

## 1. P0：只读权威审计与资源口径纠正

完成并落盘 `p0_authority_and_resource_contract.json` 后，才能回复“开始 P1”。

1. 核验父 commit/tag、远端分支及保护标签语义。
2. 独立复核 Night-9A compact 35/35，以及本任务实际使用的 Night-7A、Night-8B、Night-6D 文件和索引。
3. 复算 Night-9A 54 个正式 chain 的候选表与所有 gate；不得改写终态。
4. 新增 `night9a_resource_semantics_audit.json`：把以下时间分开报告：
   - source backbone train；
   - adapter train；
   - checkpoint serialization/reload；
   - topology/head transform；
   - total wall time。
5. 明确说明 Night-9A 的 adapter subprocess 时间含 reload，和历史 training-only 口径并不完全一致；再做一次去除 reload 的诊断性敏感性。该敏感性只用于资源解释，不得推翻或回填 Night-9A。
6. 核验当前 GPU、CUDA、依赖、磁盘和远端持久盘空间。有卡模式下若 CUDA 不可用，立即 `INFRASTRUCTURE_BLOCKED`。
7. 建立标签防火墙和访问计数器。A1/P22 标签在所有训练、embedding、affinity、partition 和 benchmark 输出总锁前读取次数必须为 0。

任一权威 SHA、commit、配置语义或输入顺序不能对齐，立即停止，不得“近似继续”。

## 2. P1：源码审计、实现和真实语义测试

### 2.1 外部源码只用于两件事

1. 公平校准强基线；
2. 提取可解释的结构思想并重新实现，不复制带标签选择的实验控制逻辑。

至少锁定并记录：

- COSMOS：`https://github.com/Lin-Xu-lab/COSMOS`，优先复用 Night-7A 已审计提交 `56ea355be51e64d9253e2871b8bd447fdfd0d230`；若对象不可获取则停止并报告，不得静默换版本。
- ARISE：`https://github.com/XiangxiangWang-code/ARISE`，只做源码语义审计和结构溯源。必须在报告中指出其公开脚本读取真实标签、逐 epoch 评价并按 ARI 保存最佳 embedding 的路径；Night-9B 不运行该选择路径，也不得把由此产生的数值作为公平 SOTA 基线。
- Night-7A 已锁定的 SpatialGlue/COSMOS/source readiness 表作为既有证据，不重复大规模下载。

输出 `external_source_semantics_lock.json`，含 URL、commit、license、关键文件 SHA、标签命中位置、endpoint 语义及是否允许进入公平 benchmark。

### 2.2 本轮核心结构：MF-RACF

暂名 `MF-RACF`（Modality-Family-aware RNA-Anchored Cooperative Fusion），仅为代码阶段代号，不是最终论文命名。

结构组成：

1. **RNA 锚定公共图**：分别构造 RNA feature-kNN 与 RNA spatial-kNN，取交集并对称化；加自环。公共图只承载第二模态的协同传播，不能用标签修边。
2. **层次融合**：先融合 RNA 的 feature/spatial 两个分支，再把第二模态的 common-graph 表示融入 RNA 表示。不得一次性把所有表示直接拼接后用标签挑最佳层。
3. **spot 可靠性门**：从每个 spot 的标准化重构残差生成 RNA/辅助模态权重；残差与尺度统计量 stop-gradient，温度固定为 1.0，权重固定截断到 `[0.10, 0.90]`。
4. **可选 DGI 约束**：只在注册候选中启用，权重固定为 0.05；负样本由 seed 派生的固定 spot permutation 生成。不得用评价结果改变权重。
5. **模态家族策略**：RNA+蛋白以 C00 为 reference，RNA+ATAC 以 F00/R02 为 reference；新增模块参数和训练规则相同，不允许按数据集单独设权重。

### 2.3 必须通过的真实语义测试

- 每个候选的 config SHA 唯一，解析后的真实模型结构与注册表一致。
- RNA common graph 的边必须真的是 feature/spatial 交集；构造负测例，确保 union 或误用 auxiliary feature graph 会失败。
- self-loop 后无零度节点；不得 fallback 到 union。
- 层次融合的梯度同时到达 RNA feature、RNA spatial 和 auxiliary common-graph 分支。
- reliability 权重逐 spot、两模态和为 1、范围为 `[0.10,0.90]`，且标签不在计算图中。
- DGI corruption permutation 对同一 seed 完全确定，不同 seed 不相同。
- checkpoint round-trip、final embedding、六视图和 H05 partition exact parity。
- 对 A1/P22 各做一个真实 smoke；smoke 不能读取标签，不能进入科学结果。

任何真实构造或梯度测试失败，终态为 `IMPLEMENTATION_SEMANTICS_INVALID`，不得开始正式训练。

## 3. P2：公平 COSMOS 差距校准

目标不是追求一个漂亮数字，而是回答论文中 `.63 ARI` 与本项目 P22 数字是否处在真正相同的协议下。

1. 只使用 P22；固定 seeds `0,1,2,3,4`。
2. 运行 COSMOS 的无标签 fixed endpoint：
   - 允许使用训练损失型 early stopping，但规则必须在标签打开前锁定；
   - 禁止任何真实标签评价回调、best-metric selection 或按标签选 resolution；
   - cluster K 使用项目既有权威 registry 中已经锁定的 P22 K，不得从标签类别数重新推导。
3. 同时保留两个只读输出：
   - `COSMOS_NATIVE_FIXED`：官方预处理和官方无标签 endpoint；
   - `COSMOS_COMMON_HEAD`：同一 embedding 输入项目锁定的 common H05 评价端点。
4. 报告 preprocessing、spot intersection、feature selection、K、endpoint 和空间图的全部差异。只有完全同协议的 common-head lane 才允许与 F00 做直接数值比较。
5. 单 unit 45 分钟硬超时；超时原样保留，不重试、不减 epoch、不换 solver。

## 4. R1：两数据集锁定开发窗口

注册表：`SpaLORA_Night9B_RACF_Registry_2026-08-20.json`。不得追加候选。

- 数据集：A1（RNA+蛋白）与 P22（RNA+ATAC）。
- seeds：`0,1,2`。
- reference 直接复用并核验历史结果；不得为制造同环境 parity 而覆盖历史结果。
- 正式新增训练：7 candidates × 2 datasets × 3 seeds = 42 units。
- 科学重试：0；fallback：0。
- 所有候选使用当前 family reference 的 optimizer、epoch、batch、latent dimension、preprocessing 与 fixed endpoint。除注册模块外不得同时改训练配方。
- 每个训练单元必须保存 final checkpoint、optimizer/config/input SHA、embedding、affinity、partition、六视图、runtime 分项和 peak GPU。
- 运行时间只作次要指标；本轮不再用 `≤1.50×` 淘汰准确性候选。只有 total runtime 超过对应 reference 3 倍时，才以资源安全门停止该配置后续 seed。

## 5. 总锁、一次标签窗口与 shortlist

在以下内容全部完成、SHA 锁定并普通 push 前，A1/P22 标签读取必须为 0：

- 42 个正式单元；
- 5 个 COSMOS training units 与 10 个 endpoint outputs；两个 lane 必须共用同一 seed 的一次训练，不能把 endpoint 误记成额外训练；
- 全部 embedding、affinity、partition、资源记录和 failure logs；
- 独立 evaluation 脚本及其 SHA。

锁定后只开一次授权标签窗口；计算 ARI、NMI、Q、neighbor agreement、Moran’s I、Geary C 和 boundary disagreement。之后禁止返回训练、聚类、resolution 或候选修改。

空间保护门必须从 Night-6D 的权威实现逐字复用并记录 source SHA，不得重新发明阈值。

候选可进入以下任一 slot；不强求一个模型统治所有平台：

- `BALANCED`：A1 和 P22 各自 mean ΔQ ≥ `+0.010`，各自 Q wins ≥ 2/3，并通过两数据集空间门。
- `LYMPH_SPECIALIST`：A1 mean ΔQ ≥ `+0.020`，P22 mean ΔQ ≥ `-0.005`，且两数据集空间门通过。
- `BRAIN_SPECIALIST`：P22 mean ΔQ ≥ `+0.030`，A1 mean ΔQ ≥ `-0.005`，且两数据集空间门通过。

每个 slot 最多保留两个候选；排序依次为该 slot 主数据集 mean ΔQ、最差 seed ΔQ、paired wins、较低运行时间。禁止事后加权。

## 6. R2：扩展 seed

只有 shortlist 非空才进入 R2。

- 对入选候选补 seeds `3,4`，并对 A1/P22 同时运行。
- 最多 4 个去重候选，因此最多 16 个新增 training units。
- 总体最多 58 个正式新训练单元；COSMOS 独立计数。
- 重新按五 seed 计算同一套 gates，锁定 `BALANCED`、`LYMPH_SPECIALIST`、`BRAIN_SPECIALIST` 候选。
- 本轮不运行 D1、tonsil、MISAR、GSE198353 或正式多方法 benchmark；这些留给候选冻结后的确认轮。

## 7. 合法终态

只能使用：

- `NIGHT9B_RACF_CANDIDATES_LOCKED`
- `NIGHT9B_COSMOS_GAP_CALIBRATED_NO_RACF_CANDIDATE`
- `NIGHT9B_NO_SCORE_GAIN_KEEP_FAMILY_BASELINES`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `INFRASTRUCTURE_BLOCKED`
- `BUDGET_EXHAUSTED`

不得宣称 SOTA，除非同协议强基线、数据、K、preprocessing、endpoint、seeds 和统计均完成公平验证；Night-9B 本身不授权 SOTA 声明。

## 8. Git、交付和关机

1. 测试、独立复算、标签访问审计、Git clean/ancestor/fsck、branch/tag peel 全部落盘。
2. 普通 push 分支；final commit 完成后只创建一次 annotated final tag 并普通 push。标签创建后不得再追加需要它跟随的提交。
3. raw runs、checkpoints、embeddings 和 affinity 留在 `/root/autodl-fs`，不下载到 Windows。
4. D 盘只交付报告、CSV/JSON 摘要、任务注册表、关键源码/测试、图、增量 bundle 和 delivery index；目标 compact < 10 MB。
5. 最终回答必须先用通俗中文说明：这轮做了什么、分数有没有提高、和强方法差多少、哪条结构有用、下一步是什么；随后再列技术审计。
6. 保留同一 SSH 会话，把 `/usr/bin/shutdown` 作为最后一条远端命令。派发后不得重连；只能声称“命令已派发”，不能虚报控制台状态。
