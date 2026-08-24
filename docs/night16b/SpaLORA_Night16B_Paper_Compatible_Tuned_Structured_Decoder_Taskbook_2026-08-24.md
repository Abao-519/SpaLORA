# SpaLORA Night-16B：论文可解释的调参与统一结构化解码器任务书

## 任务目标

Night-16B 只解决一个核心问题：现有高分能否在一条统一、可复算、可在 Methods 中完整解释的模型路径中重现并继续提高。

本轮不再修补 Night-16A 已失败的无标签 Ridge selector，也不再随意叠加一个全新神经网络。主对象是 Night-15F/15G/16A 中真正产生高分的结构化空间解码部分。工作名 `Unified Reliability-Structured Decoder`，中文为“统一可靠性结构化解码器”：它把多视图初始化、跨模态可靠性、稀疏多尺度空间能量、自返回稳定项和通用簇结构修复合成一个计算图。

它可以成为方法创新论文的候选核心，但只有在统一重跑和贡献对照成立后才能下结论。

## 执行原则

- Codex 2 对具体实现、缓存、并行、搜索算法、参数范围和停止点有充分自主权；不要因轻微工程问题停下等待用户。
- 允许公开 benchmark annotation 用于 known K、跨运行超参数选择和评价。必须把它写成 `label-assisted benchmark HPO`，不能伪装成无标签自动校准。
- annotation 不得进入 feature、graph、prototype、energy、clustering fit、gradient、move acceptance 或单次运行 checkpoint 选择。
- 所有数据调用同一个 candidate producer API、同一个 operator bank 和同一个 evaluator；差异用外部 config 表达，不在源代码中写 dataset-name 结构分支。
- 允许每条数据选择不同数值参数、初始化和 clustering head；这属于同一搜索空间中的 HPO。所有候选、失败、预算和选择顺序必须保留。
- BEST 是合法的公开 benchmark 开发结果；同时报告参数敏感性、次优区间和 family-default，不能只留一个幸运点。
- 不跑完整外部方法，不扩展新数据集。主要算力用于自身方法和分数。
- 工程错误可以修复并重跑，不设机械 correction 次数。开发期改变公式要递增 revision 并留痕；formal freeze 后若改变科学公式，完整重跑受影响 lane。

## Stage 0：父级恢复与依赖谱系

1. 复核 Night-16A compact 64/64、index 和 bundle SHA、final commit/tag。
2. 若 AutoDL repo 已含 `7bcc7696c3eb170e7191d9266691271a2ec225b6`，直接从该 commit 新建 Night-16B branch；否则从 `night16a_incremental.bundle` 恢复。远端缺 SSH 私钥不是科学阻塞。
3. 读取 Night-15F/15G/16A 的 runner、完整 ledger、frontier registry、method contract 和所有 headline partition 来源。
4. 生成 `frontier_dependency_graph.csv/json`：对每个 headline row 列出 raw/reduced view、embedding、start partition、energy、repair、HPO stage、标签接触点和最终 hash。
5. 若任何 headline 的真实父级 artifact 找不到或 hash 不一致，不删除该结果；标为 `historical_only_not_cleanly_replayable`，继续对其他 lane 工作。

## Stage 1：冻结一个共同模型对象

### 1.1 共同输入接口

每条 lane 统一接收：

- ordered observation IDs；
- RNA reduced view；
- second-modality reduced view；
- retained/fused view；
- coordinates；
- fine/registered/broad sparse graphs；
- optional morphology view 和 presence mask；
- K；
- JSON config。

标签只交给独立 evaluator，不得包含在 producer 的对象或序列化输入中。

### 1.2 共同初始化银行

对每条 primary lane 机械运行同一候选超集，至少包含：

- RNA-only、second-only、equal fusion 和连续权重 fusion；
- raw PCA、whitening、coordinate basis、可选 morphology basis；
- KMeans 和 GMM diag/tied 的固定 seed bank；
- no-filter、稀疏 low-pass、bilateral/support-aware 变体；
- Night-15F 结构化能量可接受的 registered start。

如果某个 optional view 不存在，用 presence mask 关闭，不以数据集名称跳转。

### 1.3 统一结构化解码器

保留并整理为单一公式：

1. retained/RNA/second-modality prototype unaries；
2. fine/registered/broad 三尺度稀疏图；
3. cross-modal agreement/conflict edge conductance；
4. rejected conductance mass 的 current-state self-return unary；
5. 只依赖预测簇数量、N 和 K 的 size/degeneracy cost；
6. sparse Potts disagreement；
7. alpha-expansion 或经实测更有效、仍可清楚说明的稀疏结构优化。

Codex 2 可以提出更好的通用 unary、edge reliability 或 move proposal，但必须应用到所有 lane 的共同超集，并留下从旧公式到新公式的精确差异。

### 1.4 通用 split/merge/repair

把 D1 的结构修复改造成只依赖预测对象的通用算子，不允许 `if D1`：

- 识别过小簇、低 margin 簇、空间碎裂簇；
- 候选 merge 由 prototype/空间边界/跨模态支持决定；
- 候选 split 由同一 feature bank 和 quantile bank 生成；
- 所有 lane 都能选择 no-op 或相同 repair 候选。

若该算子最终只帮助 D1，可以如实报告局部有效；不能事后从其他 lane 的共同搜索空间中删除。

## Stage 2：建立可审稿的 HPO 协议

### 2.1 两阶段搜索

1. `coarse`：固定共同参数 schema、共同预算规则和固定 seed bank；一个 batch 的 partitions 全部保存并 hash 后，再由 evaluator 追加 ARI/NMI。
2. `refine`：围绕 coarse leaders 按机械邻域规则生成下一批；先锁 candidate descriptor/partition，再追加指标。

允许上一阶段指标决定下一阶段邻域，这是公开 benchmark HPO；必须在 ledger 中保留 parent candidate 和 refinement rule。

### 2.2 Headline 选择规则

Primary headline 统一使用：

1. 只保留 exact K、finite、最小簇 `>= max(5, ceil(0.01*N/K))` 的非退化候选；
2. 最大化 ARI；
3. ARI 数值并列时最大化 NMI；
4. 再并列时选择自由参数更少者；
5. 最后选择最小簇更大者。

ARI 是主 endpoint，NMI、AMI、FMI、Moran、Geary 独立报告。不要用不可解释的加权总分把 ARI/NMI 合成一个“神分数”。

Supplement 同时保留：max-NMI、ARI/NMI Pareto frontier、无 guard、以及 guard 为预期均衡簇大小 1%/2%/5% 的敏感性。P22 K18 与 MISAR K12 单独作为 secondary protocol，不与 primary study 重复计票。

### 2.3 参数敏感性

- 对 headline 配置的每个主要连续参数至少做相邻一档上下扰动；
- 报告 ARI/NMI 曲线或热图、稳定高分区宽度、最小簇和空间指标；
- 标出是否只有单个尖峰；
- 报告每条 lane 的 candidate 数、CPU/GPU 时间和总 HPO 预算。

### 2.4 新数据可用性

主表允许 dataset-level label-assisted HPO。另做一张不读取当前 lane 标签的 `family-default` 辅助表：

- lymph-node：A1 与 D1 互相留出；
- tonsil：三切片按整切片留出；
- RNA+ATAC：P22 与 MISAR 互相留出，K18/K12 不当作独立训练 study；
- family default 只能由其他 study 的 tuned configs 取 medoid/低复杂度共识，不能读取 held-out lane 的 ARI/NMI。

family-default 低于 tuned profile 不会否定 tuned 高分；它回答“无 annotation 新数据怎么给一个合理起点”，必须如实报告。

## Stage 3：分数冲刺

Primary 当前可信 frontier：

| lane | ARI | NMI |
|---|---:|---:|
| A1 K10 | 0.276003 | 0.421740 |
| D1 K10 | 0.338775 | 0.435972 |
| tonsil s1 K4 | 0.236536 | 0.317118 |
| tonsil s2 K4 | 0.258264 | 0.314324 |
| tonsil s3 K4 | 0.350644 | 0.309771 |
| P22 K9 | 0.593963 | 0.714517 |
| MISAR K7 | 0.541424 | 0.666798 |

工作优先级：

1. 用 clean unified runner 重现上述 frontier；
2. D1 非退化 ARI 尝试超过公开 context 约 0.3427；
3. P22 K9 向 COSMOS context ARI 0.63 推进；
4. MISAR K7 先突破 ARI 0.55；
5. A1 和 tonsil 争取取得实质而非第四位小数的提升。

这些是方向线，不是失败门。不同 annotation、K 或 mask 的公开数字只写 `context`，不能假装直接公平胜负。

允许 Codex 2 在共同方法内部大胆尝试以下通用改进：

- robust prototype/unary（Huber、trimmed、local density corrected）；
- edge reliability 的 margin calibration、mutual-kNN support、boundary uncertainty；
- multi-scale schedule 或 continuation；
- label-free split/merge proposals；
- sparse Leiden/graph-cut/GMM start 扩充；
- morphology 作为 optional view，而不是独立模型分流。

不要求把所有点子都运行。先用 A1、D1、P22、MISAR 做高信息量 screen，找到有效算子后再扩到 tonsil 三切片。不要用幸运 training seed 冒充方法；本轮主要是确定性/endpoint HPO时，重复 seed 的意义按真实随机源报告。

## Stage 4：最小贡献证据

在每条 primary lane 的同一 headline config/initialization 下至少报告：

1. start partition；
2. full unified structured decoder；
3. cross-modal reliability disabled；
4. self-return disabled；
5. generic repair disabled。

同时给出从 start 到 full 的 ΔARI/ΔNMI、簇大小和空间指标。若 full 没有独立增益，就把结果分类为 `HEAD_ONLY_OR_INITIALIZATION_SIGNAL`，保留高分但不把 structured decoder 说成提分来源。

本轮不要求完成最终投稿所需的全部外部 baseline、全部生物学解释或大规模资源曲线；只建立能够支持下一阶段决策的最小真实贡献证据。

## Stage 5：真实路径、效率与复现

### P0

至少选择 RNA+protein 的 A1 或 D1、RNA+ATAC 的 P22 或 MISAR 各一条真实路径，覆盖：

`preprocessing/reduced artifact load → common producer → candidate serialization/reload → structured decoder → partition → independent evaluator`

列出两家族真实 tensor shape、graph shape/nnz、optional view shape/presence。确认父级基线究竟是锁定 artifact 复用还是重新构造。

### 效率

- 优先复用 Night-15B local compute kit、reduced views、sparse graph 和 PCA cache；
- embedding/head/energy HPO 优先在 Windows CPU 或 AutoDL CPU 执行；
- 只有 raw fragments/peaks、真正 GPU 模型或本地内存不安全的步骤使用 GPU；
- 单进程按 lane 串行缓存，避免重复读取 raw；
- 不生成 dense N×N；
- 普通 screen 只保存 config、parent、partition hash、metrics 和资源；finalist 保存完整 partition/replay object。

### 重放

- 每条 primary headline partition 至少两次 fresh-process exact replay；
- evaluator 对保存 partition 独立重算 ARI/NMI/AMI/FMI/Moran/Geary；
- config registry、candidate ledger、selection rule 和最终 profile 可从空输出目录重建；
- 工程 bug 修复和 superseded 输出均入 ledger，不隐藏。

## Stage 6：论文方法与审稿材料草案

交付一份 `paper_methods_hpo_draft.md`，至少含：

- 方法计算图与公式；
- 公开 benchmark HPO 的真实流程；
- 共同搜索空间、共同预算和两阶段 refinement；
- 参数选择规则；
- family-default 的生成；
- 参数敏感性；
- tuned profile 与 deployment/default profile 的区别；
- baseline 公平比较在最终论文阶段必须使用相称调参预算的声明。

交付 `reviewer_risk_register.md`，逐条回答：

- 参数为何不是任意 factor=6；
- 无 annotation 新数据如何初始化；
- 是否按数据集切换模型；
- 是否把 labels 用进训练；
- 是否只展示幸运配置；
- 是否给本方法比 baselines 更多优化。

这里不是要求写辩解话术，而是确保代码、ledger 和论文叙述能互相对上。

## 结果分类

1. `PAPER_COMPATIBLE_TUNED_SCOREBOARD`：统一重跑、HPO、敏感性、family-default 和审稿材料闭合，至少重现当前 frontier。
2. `SCORE_FRONTIER_ADVANCE`：至少一个 primary lane 在非退化约束下刷新 ARI。
3. `UNIFIED_STRUCTURED_DECODER_SIGNAL`：full decoder 在多个 study 上相对同一 start 有独立增益。
4. `HEAD_ONLY_OR_INITIALIZATION_SIGNAL`：高分成立，但 full decoder 不是来源。
5. `NO_CLEAN_REPRODUCTION`：历史高分无法在 clean unified path 重现。
6. `IMPLEMENTATION_FAILURE` / `INFRASTRUCTURE_FAILURE`：真实路径或资源失败。

可同时登记第 1–4 类中的多个事实，但报告必须明确主分类。不得仅凭公开开发 BEST 宣称 SOTA、盲测或 paper-ready。

## 主报告与交付

正文先写“我现在需要知道的三件事”：问题、实际动作/流水线层、论文含义。第一次出现 URSD、HPO、family-default、Pareto、self-return 时给通俗中文解释。

主表至少含：lane、N/eval、K、旧 frontier、headline config、绝对 ARI/NMI、Δ、AMI/FMI、Moran/Geary、min cluster、candidate budget、BEST/邻域 median、wall/GPU/RSS。另给：

- 全部参数表和搜索 ledger；
- max-ARI/max-NMI/Pareto 表；
- 参数敏感性；
- family-default 表；
- 五行最小贡献对照；
- 高分依赖谱系；
- 5–8 句导师汇报版；
- 失败/修正 ledger；
- 技术附录中的 commit/tag/bundle/index/hash。

Git 使用普通 branch/tag push，不 force push。若 GitHub SSH 私钥仍缺失，记录基础设施事实并生成可恢复 bundle，不影响科学交付。生成 root-relative size/SHA-256 compact index，并在 Windows 独立复算 missing/size/SHA/extras。

所有交付下载并验证后，最后一条远端命令严格为 `/usr/bin/shutdown`；派发后不重连，不把它表述为控制面板电源状态证明。
