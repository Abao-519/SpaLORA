# SpaLORA Night-7B：GPU 自适应关系融合提分任务书

日期：2026-08-18  
任务性质：有卡 GPU；当前四数据集统一提分开发；固定候选漏斗；不跑正式外部 benchmark  
权威父提交：`8b67e4bc09196f6c44b20e7afcfd0c3f0345e88b`  
权威父标签：`night7a-final-20260818`  
计划分支：`revision/q2-night7b-adaptive-relational-score-rnd-20260818`  
保护标签：`baseline/pre-night7b-adaptive-relational-score-rnd-20260818`  
最终标签：`night7b-final-20260818`

## 0. 这一步要做什么

用通俗话说，当前最好方案 `C00` 已经稳定超过旧基线，但 `C06` 又在 P22 小鼠脑上明显更强、在 A1/tonsil 上略弱。Night-7B 要解决的不是“再随便加一个 loss”，而是让模型从数据本身判断：在每个 spot 上应更相信 G00 还是 G04，同时把近年公开方法中确有实现证据的关系对齐、互近邻、遮挡重建和伪簇一致性拆开试验。

本轮分两大部分：

1. **H 阶段：** 不训练模型，完整比较 18 个 affinity/cluster head，寻找低成本提分；
2. **A 阶段：** 用 GPU 训练 10 个轻量适配器 recipe。R1 在四数据集 seeds 0–1 全量筛选；固定规则晋级最多 4 个配置后，R2 扩展到全部剩余 seeds。

目标不是靠标签把每个数据集调成不同参数，而是锁一个统一配置，在 A1、D1、P22 尽量都提高，并控制 tonsil 不显著下降。当前四数据集都属于 development；即使成功也不能称 SOTA，后续仍需新数据确认和同协议现代基线。

## 1. 权威文件与优先级

启动前必须完整读取并按 bootstrap prompt 给出的 SHA-256 核验：

1. `SpaLORA_Night7A_Independent_Planner_Audit_and_Night7B_Decision_2026-08-18.md`
2. `SpaLORA_Night7B_Source_Code_Transfer_Audit_2026-08-18.md`
3. `SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json`
4. 本任务书
5. Night-7A compact：`D:\文档\ChatGPT\博士第一篇科研论文项目\night7a_handoff_20260818\official_compact`

冲突优先级：本任务书 > registry > planner audit > source audit > Night-7A 报告。registry 中的候选、参数、顺序、seed、预算、ranking 和 gates 不得根据结果修改。

Night-7A compact 必须独立验证 internal `97/97`、external `4/4`、post-dispatch `3/3`。不要只相信报告摘要。Git branch 与 annotated tag 必须 peel 到 `8b67e4...`。

## 2. 允许的终态

- `NIGHT7B_UNIFIED_SCORE_CANDIDATE_LOCKED`
- `NIGHT7B_P22_FRONTIER_ONLY_NO_UNIFIED_WINNER`
- `NIGHT7B_NO_NEW_SCORE_CANDIDATE`
- `BLOCKED_INPUT_INTEGRITY`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `LABEL_FIREWALL_BREACH`
- `BUDGET_EXHAUSTED`
- `INFRASTRUCTURE_BLOCKED`

无论 Night-7B 成败，Night-6D 的既有 D1/P22 confirmation 不被倒写。新候选若未通过统一门，仍保留 C00；C06 的 P22 accuracy frontier 也必须如实保留。

## 3. P0-AUTHORITY：GPU、Git、目录、空间和预算

### 3.1 实例要求

1. 用户已在 AutoDL 控制台用**有卡模式**开机；执行者不得调用 AutoDL API。
2. 登录后立即记录 `nvidia-smi`、CUDA、GPU 型号/显存、CPU 核数、RAM、持久盘空间、Python/PyTorch/R/mclust 版本。
3. `torch.cuda.is_available()` 必须为 true，并运行一个小型 tensor forward/backward smoke test；否则 `INFRASTRUCTURE_BLOCKED`。
4. 不得退回 0.5 CPU/2 GB 无卡模式。若 GPU 暂时不可用，不要在 CPU 上慢跑一夜。
5. 设定合理线程数，不得让 OpenMP/MKL/BLAS 线程总数远高于实际 CPU；H 阶段可多进程，但进程数必须由 RAM/CPU 实测上限决定。

### 3.2 Git 与目录

1. 从 `8b67e4...` 新建 `/root/autodl-fs/SpaLORA-night7b` worktree 和计划分支；不得直接修改 Night-7A worktree。
2. 保护标签只创建一次并普通 push；若同名 ref 已存在且不一致立即停。禁止 force、force-with-lease、删 tag、移动 tag。
3. 四份规划 authority 原样放入 `protocols/night7b/`，保存 SHA。
4. 路径固定：
   - raw：`/root/autodl-fs/night7b_score_rnd_20260818`
   - immutable input links/index：`/root/autodl-fs/night7b_score_rnd_20260818/source`
   - H outputs：`/root/autodl-fs/night7b_score_rnd_20260818/head_stage`
   - adapters/checkpoints：`/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage`
   - repo handoff：`outputs/night7b_handoff`
5. Night-6C/Night-6D/Night-7A raw、compact、cache 和 refs 全部只读，禁止覆盖。

### 3.3 磁盘与交付

- 运行前记录 `/root/autodl-fs` free space；预估每个 checkpoint 后再决定是否可以保留 optimizer state，但 final model state、config、RNG state、embedding、clusters、audit 和 SHA 不得省略。
- raw affinity、runs、checkpoints、cache 留在远端持久盘。
- D 盘只交付 official_compact，目标 `<25 MB`；不得把 checkpoint、raw affinity、external repo、conda env 或大型数据复制到 D 盘。
- 不向 C 盘写交付；Windows 临时登录脚本用后删除。

### 3.4 硬预算

- H 正式 transforms：`540/540`
- R1 scientific training：`80/80`
- R2 scientific training：最多 `88`
- scientific training 总上限：`168`
- scientific same-cell retry：`0`
- pre-evaluation 全局实现/基础设施 corrections：最多 `12`
- training attempts 总上限：`180`
- R1 partition transforms：最多 `320`
- R2 partition transforms：最多 `88`
- 正式外部 benchmark：`0`
- fresh dataset download/label read：`0`

P0 通过后写 `p0_authority_environment_budget.json`，明确输出：`P0-AUTHORITY PASS; BEGIN P0-SOURCE`。

## 4. P0-SOURCE：只读复用与 30 个单元

### 4.1 固定输入

数据与 seeds：

| dataset | K | seeds | role |
|---|---:|---|---|
| A1 | 10 | 0–4 | development |
| tonsil | 4 | 0–4 | development |
| D1 | 10 | 0–9 | development after prior confirmation |
| P22 | 9 | 0–9 | development after prior confirmation |

共 30 个 dataset-seed 单元。每个单元复用 Night-6C/Night-6D 已锁定的 G00/G04 六 views、coordinates、C00 clusters/affinity 与 C06 clusters/affinity。只能通过 Night-7A `source_views_index.csv`、`source_prediction_index.csv` 和 verified raw manifests 的 absolute path 定位，不得按 mtime、最新目录或猜文件名。

逐项验证：size、SHA、dataset、graph、seed、attempt、obs IDs/order、array keys/shape/dtype/finite、coordinates order、C00/C06 partition SHA。60/60 views 与所需历史输出缺一不可；缺失时 `BLOCKED_INPUT_INTEGRITY`，不能重训历史结果补齐。

### 4.2 训练副本

训练进程只接收：

- 六个 embedding 数组；
- coordinates；
- obs IDs；
- fixed K；
- registry config。

不得接收 original H5AD、ground-truth path、历史 metric CSV、dataset/tissue name或可反推出身份的文件名。创建 canonical dataset index `0..3` 只用于固定顺序，但不得输入模型/gate。训练用 AnnData 若存在必须 `obs` 零列。

### 4.3 来源锁

生成：

- `source_reuse_manifest.json`
- `source_unit_index.csv`
- `training_input_firewall_manifest.json`
- `historical_reference_partition_index.csv`

全部锁定并 commit 后进入 P0-SEMANTIC。

## 5. P0-SEMANTIC：先证明代码语义，再看任何新结果

### 5.1 H05 与历史 parity

必须调用/抽取父提交中已经验证的 H05 affinity 语义，不能按描述重写一个近似版本。对 30 单元重新构建 S_G00、S_G04、C06，并要求：

- G04 spectral partition exact equivalent to C00；
- C06 affinity canonical CSR 数值误差 `<=1e-12`，partition exact equivalent to Night-7A C06；
- observation order、tie-break、self removal、local sigma、symmetry、diagonal 均通过；
- 重复运行 SHA 完全一致。

parity 不足 30/30 时停止 `IMPLEMENTATION_SEMANTICS_INVALID`；不得调 tolerance、换 solver、重装版本或用 ARI 接近 1 代替 exact partition。

### 5.2 合成语义测试

至少覆盖：

1. 四个固定 row-weight 候选的手算小矩阵；
2. WNN local/global own-vs-cross prediction error、median scale、clip、softmax、directed mix 和 symmetry；
3. zero degree、disconnected graph、ties、NaN/Inf；
4. spectral、eigen-mclust EEE/VVV、eigen-kmeans100、Leiden exact-K；
5. 六 projector、residual、equal/MoE fusion、decoder shape；
6. RELKL、CLIP、MNN、MASK、SEMANTIC、MOE_BALANCE、DCCA 每项单独 forward/backward finite；
7. loss 关闭时梯度为零、开启时相关参数有梯度；
8. checkpoint save/reload、RNG restore 与 exact forward parity；
9. training worker 无 label path、无 metric import、无法打开 original H5AD；
10. registry 10 recipes、2 endpoints、stage seed split 和预算计数。

### 5.3 真实无标签 smoke

只在一个固定 source unit 上跑每个 recipe 2 epochs，验证显存、runtime、checkpoint reload、无标签访问；全部输出标记 `smoke_invalid_for_science`，不得混进正式结果。

通过后写 `p0_semantic_contract.json`，运行专项 tests，做 pre-science implementation commit 并普通 push。此后 H registry 不得改。

## 6. H 阶段：540 个低成本 head transforms

### 6.1 固定矩阵

严格按 dataset `[a1, tonsil, d1, p22]` → seed 升序 → registry `H00..H17` 执行 18×30=`540` 个正式 transforms。全部候选都必须尝试，不能因为前几个数据看起来差就跳过。

每个 cell 保存：

- `clusters.csv` 与 canonical partition SHA；
- canonical sparse affinity SHA；
- config SHA、input SHA、obs order SHA；
- zero degree、components、symmetry、diagonal、finite、nnz；
- runtime、peak RSS、GPU allocation（H 阶段可以为 0）；
- status、warnings、failure type、fallback=false、retry=false。

科学数值失败原样保留，不 fallback、不加 epsilon、不改 solver、不重跑该 cell，继续其余 cells。

### 6.2 H 总锁与第一次评价

540 cells 全部 success 或 preserved failure 后生成 `locked_head_transform_manifest.json` 并 push。随后独立 evaluator 才能打开当前四数据集标签，一次性算 ARI/NMI/Q 与空间指标。

评价必须：

- 逐 dataset 先平均 seed，再按权重算 priority Q；
- 不把 D1/P22 的 10 seeds 当成双倍数据集权重；
- 所有失败仍在 denominator，不能删；
- 报每个候选绝对 ARI/NMI/Q、相对 C00 delta、wins、空间变化；
- 依 registry ranking 选两个 distinct partition-head templates；H00 永远保留为 fallback comparator，但除非排名 top2，不占晋级名额。

evaluator 只把晋级 head IDs、aggregate table SHA 和固定 pseudo-partition path 交给后续 orchestrator；训练 worker 不接收真实 label 文件或逐 spot label。

## 7. A 阶段共同实现

### 7.1 架构与训练终点

严格实现 registry 的六 view projector、G00/G04 summary、residual adapter、equal/MoE fusion 和六 decoders。所有 recipe：

- float32；
- AdamW lr `1e-3`、weight decay `1e-5`；
- exactly `160` epochs；
- cosine decay；
- no early stopping、no best epoch；
- gradient clip norm 5；
- final epoch 才是权威 endpoint。

若某 recipe 不含 MoE，必须使用 equal fusion；不能根据 dataset 选择。任何 dataset ID、tissue name、label/K 以外的 annotation 都不得进入 gate。

### 7.2 损失

完全按 registry 权重实现 RECON、RELKL、CLIP、MNN、MASK、SEMANTIC、MOE_BALANCE、DCCA。特别注意：

- `SEMANTIC` 的 target 只能来自 H 阶段锁定的预测 partition，不是真实标签；
- MNN 与 negative indices 在训练前锁定，不能每 epoch 依据损失重采；
- MASK indices 由 dataset-unit seed 和 recipe ID 的 canonical hash 确定并保存；
- RELKL 用 sparse union top20，不允许常驻 dense N×N；
- 所有 loss components 每 epoch 记录，不能只保存 total loss；
- 不允许动态改 loss weight 或数据集特定 recipe。

### 7.3 每个成功 training 必须交付

1. final checkpoint（state_dict、optimizer、scheduler、config、RNG）；
2. checkpoint byte SHA 与 state tensor canonical SHA；
3. final fused embedding、loss curve、gate weights/quantiles（若有）、MNN/mask/pseudo-target index SHA；
4. E0/E1 affinity 与两个晋级 head 的 partitions；
5. 即刻用新进程 reload checkpoint，forward 与 embeddings/affinities/partitions exact parity；
6. runtime、peak GPU、peak RSS、GPU model；
7. no-label/no-retry/fallback audit。

checkpoint 缺失或 reload 不等价，该 cell 不是 success。不得只保留 CSV 分数。

## 8. R1：10 recipes × 8 units

R1 固定使用 A1/tonsil/D1/P22 的 seeds 0、1，共 8 units；10 recipes=`80` trainings。每次 training 的两个 endpoints × 两个晋级 heads 最多生成 4 个 partitions，因此最多 320 transforms。

执行顺序：recipe order → dataset order → seed order。可以在不改变 RNG/语义的前提下做 GPU job batching，但每个 unit 必须有独立 config、checkpoint 和 audit。

80 trainings 和全部 partitions 锁定、checkpoint round-trip 全通过后，生成 `locked_R1_manifest.json` 并 push，才允许 R1 evaluator 打开 labels。

R1 evaluator 根据 registry ranking 自动晋级 top4 `recipe × endpoint × head` configs。禁止人工把“看起来有潜力”的第五名塞回；也禁止因一个失败 cell 删除候选后重算有利均值。输出的 R2 contract 只能含 config IDs、固定参数 SHA、remaining seed list 和 aggregate ranking SHA。

## 9. R2：晋级配置扩展到剩余 seeds

剩余 seeds：

- A1/tonsil：2、3、4；
- D1/P22：2–9。

每个晋级配置共 22 remaining units；最多四个配置，训练上限 88。若多个配置共享同一 recipe，只训练一次并复用 checkpoint/embedding，不能重复训练制造额外 seed 结果，也不能因此把省下的预算用于新候选。

R2 不得修改 R1 recipe、loss、head、endpoint、epoch 或 threshold。每个成功 cell同样必须 checkpoint round-trip。全部锁定后生成 `locked_R2_manifest.json` 并普通 push，最后 evaluator 才打开 labels。

## 10. 最终评价与解释

### 10.1 指标

逐 seed 报：ARI、NMI、Q、neighbor agreement、Moran I、Geary C、boundary disagreement。逐 dataset 报 mean/std、paired deltas/wins；再报 priority-weighted Q 和 four-dataset balanced macro Q。

新统一候选必须通过 registry `final_lock_gate`。门槛是硬门，不允许在看结果后解释性放宽。还要给：

- 与 C00、C06 的绝对/相对表；
- 每个 loss 家族的增益/伤害；
- MoE gate 是否发生塌缩、是否和无标签质量指标相关；
- runtime 与 peak GPU；
- A1/D1 人类淋巴 aggregate 与 P22 单独结论；
- tonsil 的降幅是否在容许范围；
- negative/failure candidates 全表。

### 10.2 终态规则

- 至少一个配置通过全部统一门：`NIGHT7B_UNIFIED_SCORE_CANDIDATE_LOCKED`。
- 无统一候选，但有配置在 P22 比 C06 继续提高且其余数据无法通过统一门：`NIGHT7B_P22_FRONTIER_ONLY_NO_UNIFIED_WINNER`。
- 没有新配置达到门：`NIGHT7B_NO_NEW_SCORE_CANDIDATE`。

不能把 R1 小样本赢家称为最终赢家；只有 R2 全 seeds 结果可触发 final lock。不能声称 SOTA或外部 confirmation。

## 11. 失败、纠正与“不放弃每个尝试”

1. 科学数值失败：保留 checkpoint/log/trace/partial artifacts，no retry/no fallback，继续矩阵。
2. 在相关 stage 开标签前发现全局实现 bug：保存旧 attempt 为 invalid，统一修复，所有受影响 cells 一起重做，并计入最多 12 次 correction；不能只补好看的结果。
3. stage 已开标签后发现实现 bug：不得选择性修补；整个受影响 candidate-stage comparison 标为 invalid，并报告还需要多少预算才能重做。
4. 训练预算到 180 attempts 时停止；不能私自扩大。
5. negative candidates、failed attempts、数值分叉、gate collapse 全部保留。这就是“不轻易放弃尝试”的正确方式：先排除实现错误，再把真实负结果留下，而不是悄悄重跑到满意。

## 12. 测试、Git 与交付

### 12.1 测试

至少包含：

- P0 synthetic semantic tests；
- 30× C00/C06 real parity；
- candidate/config SHA uniqueness；
- loss gradient routing；
- zero-label worker；
- checkpoint round-trip；
- metric recompute；
- seed/dataset weighting；
- budget and attempt accounting；
- no retry/fallback；
- delivery index and Git evidence。

最终必须从 clean checkout 运行专项 tests，不能只在脏 worktree 通过。

### 12.2 Git

建议里程碑 commits：

1. authority/source verification；
2. implementation + semantic tests；
3. H lock；
4. H evaluation + R1 contract；
5. R1 lock/evaluation；
6. R2 lock/evaluation；
7. report + delivery index final commit。

全程普通 push。final tag 只能在 final delivery-index commit 后创建和推送一次，绝不移动。

### 12.3 D 盘 compact

交付到：

`D:\文档\ChatGPT\博士第一篇科研论文项目\night7b_handoff_20260818\official_compact`

至少保留：

- 完整报告与通俗摘要；
- authority/registry/source audit；
- source/firewall/P0 contracts；
- H/R1/R2 manifests、candidate tables、逐 seed metrics、spatial gates；
- checkpoint round-trip index（不含 checkpoint 本体）；
- loss/gate diagnostics 的小型表；
- tests、budget、failure、label、Git audits；
- code/tests；
- incremental Git bundle；
- delivery indexes；
- shutdown dispatch evidence。

Windows 端逐文件 SHA 验证。不要为了体积删除小型关键证据；不要下载 raw runs、affinities 或 checkpoint。

## 13. 关机与最终回复

只要连接过服务器，无论成功、失败或 blocker，都必须先完成可完成的 Git/compact/Windows verification。保留同一 SSH 会话，把 `/usr/bin/shutdown` 作为最后一条远端命令直接派发，之后绝不重连、绝不调用 AutoDL API。

只能声称“关机命令已派发”；用户在控制台确认最终电源状态。

最终回复必须先给一段任何非代码人员都能理解的说明：

1. 这一轮实际试了什么；
2. 新方案有没有同时提高人类淋巴和小鼠脑；
3. 最终 ARI/NMI/Q 比 C00/C06 高多少；
4. 哪个机制最有用、哪个没用；
5. 是否锁定统一候选，下一步是什么；
6. 是否已派发关机命令。

之后再给训练数、失败数、测试、commit/tag、文件路径和 SHA。不要只扔一个终态代号和几十行技术数字。

