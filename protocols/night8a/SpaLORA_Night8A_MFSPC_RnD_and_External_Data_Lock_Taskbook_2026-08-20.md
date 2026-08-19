# SpaLORA Night-8A：MF-SPC 模型研发与外部数据锁定任务书

版本：2026-08-20 REV0  
任务性质：GPU 研发、固定候选消融、开发集筛选、外部数据只读预检  
前置终态：Night-7C `IMPLEMENTATION_SEMANTICS_INVALID`，正式训练为 0；Night-7C 不提供科学胜负结论  
本轮核心：以 human lymph 与 P22 mouse brain 的可泛化提分为最高优先级，同时形成能支撑二区、向一区方法设计看齐的清晰创新主线

## 0. 执行口径

收到本任务书后，第一条回复必须是：

> 已读取 Night-8A 任务书，开始 P0-AUTHORITY。服务器应使用有卡模式；我不会自动调用 AutoDL API，不会在标签或语义门失败后继续训练。

面向用户的阶段更新必须使用通俗语言说明“正在做什么、为什么、已完成多少、是否异常”。不要只输出内部缩写。

本轮不是正式论文 benchmark，不允许宣称 SOTA。本轮是把已经观察到的分数规律固化成一个可解释、可消融、可在新数据复现的方法候选。

## 1. 不可变科研边界

### 1.1 允许大胆优化，但禁止伪造优势

允许：

- 大幅修改模型结构、图构建、融合方式与无监督损失；
- 阅读并吸收高水平论文和开源源码的思想；
- 使用当前 A1、D1、tonsil、P22 作为明确标注的开发数据；
- 固定候选后按本任务书的开发集规则选出最多三个 frontier；
- 对不同模态家族使用预注册、可由 assay metadata 自动决定的配方。

禁止：

- 标签进入训练、early stopping、checkpoint 选择、图构建、family selector、聚类分辨率或任何超参数计算；
- 按数据集名称写分支；
- seed search、删掉坏 seed、重跑负结果、best epoch/ best seed、事后换阈值；
- 看过评价结果后新增配置、改损失权重或改候选代码；
- 用同一研究中的新 section 冒充完全独立外部验证；
- 未经许可证检查复制第三方实现；
- 把源码中按真实 ARI 选模型的做法迁入本项目。

所谓“尽最大可能提分”，在本任务书中解释为：尽可能广泛而系统地探索合法的模型空间、保留所有失败证据、提高真正能迁移到新数据的分数，而不是数据泄漏或结果选择。

### 1.2 当前数据角色

- A1、D1、tonsil、P22：开发集。它们不再被称为 pristine holdout。
- A1 与 D1：同属 human lymph 来源，在总排名中先取二者数据集均值，不得当成两个独立生物研究重复加权。
- P22：重要的 `RNA_EPIGENOME` 开发集，seed 异质性较强。
- tonsil：保留为 RNA+protein 泛化保护数据，但总权重低于 human lymph 和 P22。
- MISAR 等新数据：Night-8A 仅作 provenance 与输入完整性预检，标签保持密封；正式外部验证留给后续 Night-8B。

## 2. 权威输入与启动门

必须读取并 SHA-256 校验：

1. `SpaLORA_Night8A_Literature_Code_and_Dataset_Scout_2026-08-20.md`
2. `SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json`
3. 本任务书
4. Night-6D 报告：`333192ce979a02ce8cbc785828e58fa7b9314906ce653c7d8cae016b92e8a053`
5. Night-7A 报告：`274cf68d82634ab935b894a101e439904f0ce3f4e0d16f6d146b2d7ee95e4de3`
6. Night-7B 报告：`48e16b12934fb13c668ea08e966e26e1c2454f5aad59e8510e7b391a4d2cd0a9`
7. Night-7B candidate lock：`74d76ca45ca7d9d632bdfdd694bfccbe635318b3cdfaad409ca21100805ddd97`
8. Night-7C 报告：`af7ae61a747e865bace5fce70da72085280c7147ddaf12c9df82362a4c77a840`
9. Night-7C compact index：`1797d4f1583d3dea0fedca93aa53ad1ade9417d1b08f0bc79fb03729c472e734`

Git 起点必须为：

- base commit：`e34567db5ace4f0fcdd2526cfb94a84fc9148020`
- base tag：`night7c-final-20260818`
- 新分支：`revision/q2-night8a-mfspc-rnd-20260820`
- 保护 tag：`baseline/pre-night8a-mfspc-rnd-20260820`
- final tag：`night8a-final-20260820`

若 base/tag/远端不一致、任何 authority SHA 不符、注册表 JSON 无法解析，终止为 `BLOCKED_AUTHORITY_MISMATCH`。不得自行找“差不多的版本”替代。

## 3. P0-AUTHORITY：历史结果与 family reference

### 3.1 独立复算

从 Night-6D、Night-7A、Night-7B 的 per-seed 结果重新计算：

- C00 对 G00/H00 的 A1、tonsil、D1、P22 ΔARI/ΔNMI/ΔQ；
- R02 对 C00 的四数据集 ΔARI/ΔNMI/ΔQ；
- A1 与 D1 合并后的 `Q_HLN`；
- `priority_macro_Q = 0.45*Q_HLN + 0.45*Q_P22 + 0.10*Q_tonsil`。

误差容限 `1e-12`。若原始逐 seed 表缺失或复算不一致，终止 `BLOCKED_HISTORICAL_RECOMPUTE`。

### 3.2 family reference 定义

- `RNA_PROTEIN` 使用 C00 的 G04/H05 语义；
- `RNA_EPIGENOME` 使用 R02 的锁定语义；
- R02 只是 development reference，尚未获得独立 RNA+ATAC 外部确认；
- selector 只能读取规范化 assay metadata。

必须添加以下负向测试：

1. 把 A1、D1、tonsil、P22 随机重命名，family 输出完全不变；
2. 删除 assay metadata 后必须 hard error，不能根据文件名猜测；
3. 在 selector 调用栈中放置禁止访问的 labels sentinel，任何读取都立即失败；
4. `RNA+ADT` 与 `RNA+protein` 归入 `RNA_PROTEIN`；`RNA+ATAC`、`RNA+histone` 归入 `RNA_EPIGENOME`；未知组合 hard error。

若 R02 无法从已锁定代码、配置、视图和 endpoint 明确重建，终止 `BLOCKED_REFERENCE_SEMANTICS`，不得把相近配置冒充 R02。

## 4. P0-SOURCE：第三方源码审计与独立实现边界

在远端使用只读/隔离目录固定以下官方仓库的 commit、许可证、核心文件 SHA。允许补充更新提交，但不得让上游漂移改变本轮注册配置：

- SMART：https://github.com/Xubin-s-Lab/SMART-main
- SpaMosaic：https://github.com/JinmiaoChenLab/SpaMosaic
- MultiGATE：https://github.com/cuhklinlab/MultiGATE
- COSMOS：以论文官方仓库为准
- ARISE：以论文官方仓库为准
- scMultiBench：https://github.com/PYangLab/scMultiBench

输出 `third_party_source_audit.json`，至少含 URL、commit、license、读取文件、可迁移思想、禁止直接复制原因。

特别核验：

- ARISE 官方训练是否把真实标签传入 epoch loop 并按 best ARI 选择；若是，明确标红，任何相关实现必须去除此语义。
- COSMOS accelerated spatial regularization 是否存在两个坐标都引用同一索引的问题；只记录，不修改第三方仓库。
- SMART 教程的平台/数据集参数差异；把它解释为 family-conditioned 的先例，而不是按名字调参的许可证。

本轮项目代码采用独立实现。若复制任何 MIT/Apache 代码片段，必须在文件头与 `THIRD_PARTY_NOTICES.md` 记录；GPL/AGPL 或未确认许可证代码不得复制进不兼容目标。

## 5. P0-IMPLEMENTATION：MF-SPC 语义

工作名：`MF-SPC`，不作为最终论文名。

### 5.1 SP：shared/private

每个模态编码输出 shared/private 两部分：

- shared 维度为现有 latent 维度的 75%；
- private 为 25%；
- 解码器从 shared+本模态 private 重构；
- shared/private 使用 centered cross-covariance Frobenius penalty；
- shared 用于融合与聚类，private 不直接进入最终聚类 embedding。

必须测试梯度都能到达 encoder/decoder；module off 时不得偷偷改变 baseline 维度或初始化。

### 5.2 RR：same-spot redundancy reduction

实现 VICReg-style shared alignment：

- invariance 25、variance 25、covariance 1；
- 外部归一化权重分 `RR10=0.10`、`RR30=0.30`；
- 不需要负样本；
- 严禁密集 spot×spot similarity matrix。

### 5.3 PROTO：软原型共识

- prototype 数量只读取现有锁定 benchmark K 常量，不能现场读取标签再统计；
- student temperature 0.10、teacher EMA 0.99；
- 对各模态 shared 与 fused embedding 的软分配做一致性；
- 仅使用弱 occupancy regularizer，不把各簇强制等大；
- 不通过 ARI/NMI/H&E 选择温度、K 或权重。

### 5.4 RNA_ANCHOR

仅 `RNA_EPIGENOME` 激活：空间边与 RNA feature-neighbor 边做稀疏交集并加 self-loop，用作共享拓扑。RNA_PROTEIN 中严格 no-op。

若 B05 在 RNA_PROTEIN 中解析后的 active runtime config 与 B04 完全相同，可用 content-addressed alias 引用同一 checkpoint/output，但必须：

- 记录两个注册 config SHA；
- 记录同一个 resolved runtime SHA；
- 验证输出/模型 SHA 完全相同；
- 不把 alias 计为一次新训练。

### 5.5 DGI 与 SMART_TRIPLET

- DGI 必须是 sparse global-local objective；禁止任何 dense N×N 路径。
- SMART_TRIPLET：MNN k=3、farthest fraction=0.60、margin=0.50；独立实现。
- 两者是对照模块，不能因单数据集好看而绕过统一/专科门。

### 5.6 辅助损失尺度

严格使用注册表的 EMA scale normalization：

`L_base + λ_i * clip(EMA(|L_base|)/(EMA(|L_i|)+1e-8), 0.1, 10.0) * L_i`

- EMA beta 0.99；
- warmup 10 epochs；
- scale ratio stop-gradient；
- 每 epoch 记录原始 loss、归一化因子、加权 loss 与梯度范数；
- 不能根据评价分数再调 λ。

## 6. P0-SEMANTIC 与 P0-RUNTIME 硬门

正式科学运行前必须通过：

1. registry 25/25 以上结构/类型/互斥规则测试；
2. family selector 正向与负向测试；
3. labels sentinel 测试；
4. module-off baseline forward parity；
5. 每个模块单独 forward/backward，所有预期参数有 finite gradient；
6. deterministic seed replay；
7. checkpoint save/load round-trip，embedding、clusters、所有模块 state SHA 一致；
8. 模型 state、optimizer state、config、input cache、embedding、clusters、affinity、metrics placeholder 都进入 manifest；
9. one-step CUDA smoke：模型参数、输入、主要 loss 与 backward 必须在 CUDA；
10. full-size P22 label-free runtime smoke，证明没有 dense N×N 分配。

运行门：

- 服务器必须是有卡模式；记录 GPU 型号、CUDA、torch、PyG、R/mclust。
- P22 full-size 单次固定 endpoint transform 上限 45 分钟；超时必须终止子进程并记录 `RUNTIME_TIMEOUT`，不能再等一天。
- CPU transform 最多 3 个并行 worker，每 worker 最多 3 个 BLAS/OpenMP threads；先测峰值内存再并行。
- 若训练阶段 GPU utilization 长期为 0，先检查 tensor/device 与 CPU preprocessing；若进入 clustering/ARPACK/mclust 等 CPU 阶段，GPU 为 0 属正常，但必须在 runtime timeline 标注。
- 允许 1 次纯基础设施纠正：只修等价的设备放置、线程/进程调度、稀疏化或 I/O；算法输出必须用小例子 exact/容差证明等价。
- 科学 retry=0，fallback=0。

任一语义测试失败：终止 `IMPLEMENTATION_SEMANTICS_INVALID`。任一非等价加速：终止 `RUNTIME_SEMANTICS_INVALID`。

## 7. 标签防火墙与开发阶段

训练进程只能读取零 obs/零 annotation 副本或注册过的无标签 cache。对原始 `.h5ad` 的预锁审计只用低层 HDF5 元数据读取，不能调用 `anndata.read_h5ad` 反序列化原始 obs。

本轮允许两个明确的开发评价窗口：

1. `DEV_WINDOW_1`：R1+R2 全部输出、checkpoint、clusters、manifest 与 SHA 锁定后打开；只产生 R2 shortlist，不允许改配置或代码。
2. `DEV_WINDOW_2`：R3 扩展全部锁定后打开；只产生最终开发结论。

窗口之间必须：

- evaluator 仅输出候选 ID、固定比较表与 SHA；
- 再次关闭/隔离 labels；
- R3 trainer 只读取 shortlist ID，不得读取评价表或 labels；
- R3 仅补固定 seeds，不得改任何超参数。

这仍然是开发集选择，因此任何 Night-8A 赢家都必须在 Night-8B 新 RNA+ATAC/新 section 上冻结验证。

## 8. R1：单机制探针

按注册表运行：

- A00_FAMILY_REFERENCE
- A01_SP
- A02_RR10
- A03_RR30
- A04_PROTO
- A05_RNA_ANCHOR
- A06_DGI
- A07_SMART_TRIPLET

固定 A1/tonsil/D1/P22 seed 0。 nominal 32 cells；允许符合第 5.4 节的 no-op alias。

R1 不决定是否临时新增候选。即使某模块数值失败，也保留 failure manifest，R2 注册配置不变；若失败属于模块语义错误而非正常数值失败，整轮 fail-closed。

## 9. R2：固定组合 pilot

按注册表运行 B00–B07，固定每个数据集 seeds 0,1,2。不得根据 R1 结果删改配置。

全部运行结束后，先锁：

- checkpoint 与 optimizer state；
- config/resolved-runtime SHA；
- shared/private/fused embeddings；
- affinity、clusters、loss trace；
- runtime、peak RAM、peak GPU；
- failed units；
- 逐文件 SHA manifest。

然后才打开 `DEV_WINDOW_1`。

### 9.1 排名

`Q=(ARI+NMI)/2`  
`Q_HLN=mean(mean_Q_A1, mean_Q_D1)`  
`priority_macro_Q=0.45*Q_HLN + 0.45*Q_P22 + 0.10*Q_tonsil`

最多锁定三个不同 R3 finalist：

1. `unified_balanced`：ΔQ_HLN≥0、ΔQ_P22≥0、ΔQ_tonsil≥-0.01、空间保护通过；按 priority_macro ΔQ 排第一。
2. `human_lymph_frontier`：Q_HLN 最大，P22 ΔQ≥-0.01，空间保护通过。
3. `P22_frontier`：Q_P22 最大，Q_HLN ΔQ≥-0.01，空间保护通过。

同一 config 可占多个 slot，但只算一个 finalist。并列时按：priority ΔQ、最差重要数据集 ΔQ、paired Q wins、较低复杂度、config ID 字典序。

若某 slot 无合格候选，留空；不得放宽阈值。所有 B00–B07 仍进入 Pareto 表，不因未晋级而删除。

## 10. R3：固定 finalist 的剩余 seeds

对最多三个 finalist 只补：

- A1：3,4
- tonsil：3,4
- D1：3–9
- P22：3–9

每 finalist 18 个新 cells；不得重训 seeds 0–2。总科学训练硬上限 180，基础设施 retry 上限 12，科学 retry 0。

R3 全部锁定后打开 `DEV_WINDOW_2`，计算：

- 全 seeds 的 ARI/NMI/Q、paired deltas、wins；
- 100000 次 paired bootstrap CI，固定 bootstrap seed；
- exact sign test 与 Wilcoxon（零差按预先固定规则处理）；
- Holm correction，family 与数据集层级明确记录；
- AMI/FMI/homogeneity/completeness/V-measure；
- 空间保护指标；
- runtime/显存/参数量/模型大小；
- R1/R2/R3 全部配置的 Pareto frontier。

正式 balanced material gate：

- priority_macro ΔQ ≥ +0.010；
- Q_HLN 不下降；
- Q_P22 不下降；
- 空间保护通过。

专科 frontier 即使不满足 balanced gate也必须保留并报告，但不能叫统一赢家。

## 11. 外部数据只读预检与锁定

Night-8A 不跑外部正式 benchmark，不打开外部标签值。

### 11.1 第一优先

MISAR-seq mouse brain：

- 原始项目 OEP003285；
- 可检查 SMART Zenodo 17093158 的 processed data；
- 优先选择一个 E15.5 或 E18.5 section，要求 RNA、ATAC、坐标同 spot 对齐且规模适中；
- 审计 annotation 是专家手工、原论文聚类、图谱转移还是其他来源。

若 provenance 可接受，生成 `external_dataset_lock.json`，记录 accession、文件名、大小、SHA、spot 数、feature 数、坐标、模态、annotation 来源、family=`RNA_EPIGENOME`，但不读取/输出标签取值或分布。

### 11.2 次级

- GSE205055 mouse embryo E13：同研究 replication；
- Zenodo 12654113 tonsil section 2/3：同组织 section replication；
- GSE198353 SPOTS spleen：label-free replication。

### 11.3 存储

- Windows 本轮新增下载总量上限 100 MB；
- 大数据只进 `/root/autodl-fs/` 持久盘；
- D 盘只交付 manifest、脚本、报告和小表；
- 不下载 raw runs/checkpoints 到 D 盘；
- STARmap/RIBOmap 约 5.8 万 spots，只登记，不下载。

## 12. 结果终态

按顺序判定：

1. 权威/语义/标签/运行时违规：相应 `BLOCKED_*` 或 `*_SEMANTICS_INVALID`，无科学结论。
2. 至少一个 finalist 通过 balanced material gate：`NIGHT8A_MFSPC_BALANCED_CANDIDATE_LOCKED`。
3. 没有新 MF-SPC balanced，但 family policy 明显提高 priority score且有可锁外部 RNA+ATAC：`NIGHT8A_FAMILY_POLICY_ONLY_LOCKED`。
4. 只有 human lymph 或 P22 frontier：`NIGHT8A_FRONTIER_ONLY_NO_UNIFIED_WINNER`。
5. 所有新配置均无有效增益：`NIGHT8A_NO_VALID_GAIN`。

不能把“跑完”自动等同于成功，也不能因没有统一赢家隐藏明显的专科突破。

## 13. Git 与交付

### 13.1 Git

- 从指定 base 建 branch 与 protection tag；
- 普通 commit、普通 push；禁止 force/force-with-lease；
- final tag 只创建一次，必须在最终科学报告、测试、manifest 与 delivery index 都进入最终 commit 后创建；
- branch、final tag 与 GitHub peeled commit 必须一致；
- 若 GitHub 凭据失败，只尝试一次并保存 bundle，不盲目重试。

### 13.2 远端必须保留

- 所有 raw runs、失败 runs、checkpoints、optimizer states；
- per-seed embeddings/clusters/affinity/loss trace；
- source pins、environment、tests、manifests；
- 外部数据（若下载）及 SHA。

### 13.3 D 盘 compact 交付

交付根：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8a_handoff_20260820/official_compact/`

至少包含：

- `night8a_report.md`
- `plain_language_summary.md`
- `night8a_decision.json`
- `p0_authority_and_semantic_contract.json`
- `third_party_source_audit.json`
- `resolved_family_policy.json`
- `config_registry_frozen.json`
- `training_manifest.json`
- `checkpoint_and_roundtrip_audit.json`
- `per_seed_metrics.csv`
- `paired_deltas.csv`
- `pareto_frontier.csv`
- `module_ablation_summary.csv`
- `runtime_and_gpu_timeline.csv`
- `label_firewall_audit.json`
- `external_dataset_preflight.json`
- `external_dataset_lock.json`（若可锁）
- 测试日志、Git audit、delivery index
- planner handoff tar.gz
- 从 Night-7C 到 Night-8A 的 incremental Git bundle

compact 目标小于 20 MB。delivery index 对全部内部文件做 SHA-256；Windows 再独立复核。

### 13.4 关机

无论成功、硬门停止或没有候选，只要本轮已经连接服务器，在所有远端写入、Git、D 盘复制和校验结束后：

1. 保留同一 SSH 会话；
2. 最后一条远端命令必须直接是 `/usr/bin/shutdown`；
3. 命令后不重连、不调用 AutoDL API；
4. 只陈述“关机命令已派发”和客户端状态，不虚报控制台状态。

## 14. 最终用户汇报格式

先用通俗语言回答四件事：

1. 这轮到底改了模型的什么；
2. human lymph、P22、tonsil 分数分别怎么变；
3. 是否找到统一候选，若没有，哪个专科方向最强；
4. 下一步外部数据是否已经锁好。

然后再给：运行完整性、失败单元、Git commit/tag、D 盘路径、SHA、关机派发状态。不要只贴内部状态码。
