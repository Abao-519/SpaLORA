# SpaLORA Night-14B：RNA+ATAC 提分与边状态模型研发任务书

日期：2026-08-23  
任务性质：高自由度方法研发与性能冲刺。  
工作模块代号：TSPR（Tri-State Propagation with Rejection，三态拒绝式传播；不是最终论文名）。

## 我现在需要知道的三件事

1. 本轮只回答一个实际问题：能否在统一 RNA+ATAC 大架构下，通过新的 preprocessing、表示学习、融合、空间传播和聚类模块，把 P22 与 MISAR 分数推向已登记高水位。
2. 主要资源用于自己的方法，外部方法本轮只提供“公开最高分目标板”和源码灵感，不要求复现整张 benchmark。
3. 本轮允许单最佳 seed 先形成开发突破；只有成功候选才补少量重复，因此 Night-14B 的成功不等同于论文已经确认。

## 1. 核心研发问题

当前 W02 的主要收益来自空间低通，而不是已被证明的冲突过滤。Night-14B 要研制一套真正有独立增量的统一 RNA+ATAC 方法：

> 在不按数据集名称切换整套模型的前提下，如何同时利用 RNA、ATAC 和空间几何，学习哪些局部结构应聚合、哪些边界应保留、哪些跨模态冲突应弃权，并使这种判断进入表示学习或可验证的图滤波过程？

统一的含义是：共享方法逻辑、模块类型、信息流和优化目标。允许 RNA 与 ATAC 使用各自合理的 preprocessing/adapter，也允许不同平台或数据集使用透明的数值超参数、聚类 K 和 endpoint 参数。禁止的是 `if dataset == ...` 后换成另一套 backbone 或另一篇方法。

## 2. 执行风格与资源分配

这是目标与预算授权，不是逐行操作清单。执行 Codex 应像研发工程师一样主动阅读论文和源码、提出假设、快速试验、淘汰失败方案并重分配算力。

建议资源分配：

- 75%–85%：自己的 preprocessing、模型、模块、loss、优化和聚类 head；
- 5%–10%：公开高水位及其 label/K/mask/endpoint 最小登记；
- 10%–15%：只对领先候选做必要消融、真实回放、资源检查和失败原因诊断。

执行 Codex 有权：

- 从空间多组学之外的图信号处理、图像分割、点云、异配图学习、多视图聚类、专家混合和不确定性估计中借用思想；
- 改写或替换 TSPR 的具体公式；
- 改 preprocessing、encoder、fusion、graph filter、loss、optimizer、训练步数和聚类 head；
- 使用 Optuna、successive halving 或自写的分层搜索；
- 允许 dataset/platform-specific numeric HPO，只要统一大框架不变且完整记录；
- 使用公开 annotation 做开发评价、HPO 和候选选择；
- 修复依赖、显存、源码和 API 问题，不设一次 correction 上限；
- 在 8 GB GPU 内存允许时使用经过预估的 dense 小模块，也可以分块/近似实现；不再因“出现 dense”自动封死可用思想。

不要把 token 和墙钟花在反复核对同一批历史 hash、完整复现大量外部方法或为每个失败候选跑多 seed。

## 3. 权威输入

先完整读取：

1. `night14a_delivery_20260823/official_compact/compact_delivery_index.json`
2. `night14a_delivery_20260823/official_compact/outputs/night14a_handoff/night14a_report.md`
3. `night14a_delivery_20260823/official_compact/outputs/night14a_handoff/night14a_decision.json`
4. `night14a_delivery_20260823/official_compact/outputs/night14a_handoff/absolute_metrics.csv`
5. `night14a_delivery_20260823/official_compact/outputs/night14a_handoff/development_leaderboard.csv`
6. `night14a_delivery_20260823/official_compact/outputs/night14a_handoff/key_ablation_summary.csv`
7. `night14a_delivery_20260823/official_compact/outputs/night14a_handoff/mechanism_diagnostics.csv`
8. `night14a_delivery_20260823/official_compact/SpaLORA/night14a_tcf.py`
9. 本规划目录的 Worker 2 审计、JSON 合约和 planning index。

Night-14A compact 应为 77/77，index SHA-256：

`031d31581413c639c6ae5dea459f62eaea1939937d244bd93bcc78696c0b2dc8`

独立定位并复算一次；不匹配就停止报告，匹配后不再反复做整套审计。

## 4. Stage 0：建立可追赶的高水位目标板（短）

这一步不是外部复现。只需为每个目标固定：来源、数据版本、annotation、评价 mask、K、cluster endpoint、公开 ARI/NMI 和源码/论文链接。

至少建立四条 lane：

1. `P22_PROJECT_K9`：当前项目 coarse annotation，K=9；内部 native N02 高水位 0.5063/0.6562。
2. `P22_PAPER_K18`：取得 3d-OT/COSMOS 使用的 manual annotation，K=18；公开 3d-OT ARI 约 0.390，最终以论文图表/源码复核值为准。
3. `MISAR_PROJECT_K7`：当前项目 annotation，K=7；当前 W02 0.2706/0.4488，support-only 0.3137/0.4924。
4. `MISAR_PAPER_K12`：取得 SEPAR 使用的 original-MISAR ground truth，K=12；公开 SEPAR integrated ARI 0.644。

同一 embedding 可以在多个 lane 评价。目标板数字只作为研发航标；不要把不同 K/annotation 的数值直接相减后宣称胜负。

## 5. Stage 1：低成本表示与滤波模块冲刺

优先复用 Night-14A checkpoints、N02/RNA-only 等现有 embedding，快速判断什么结构有提分潜力。建议探索 40–100 个便宜配置，初筛只跑一个固定 training/embedding seed 和少量 endpoint seeds；找到明显信号再扩大。

可自由组合的模块池如下，不要求全部实现：

### 5.1 多尺度、边界保持的空间滤波

- identity + 多个半径/k 的 low-pass filter bank；
- bilateral/guided filter（用分子内容引导的双边滤波）；
- anisotropic diffusion（各向异性扩散，允许组织内部扩散、在边界减弱）；
- personalized PageRank、GPR/Jacobi/Bernstein polynomial graph filters；
- MixHop/ACM-GCN 类 identity、aggregation、diversification 通道。

### 5.2 TSPR：三态拒绝式传播

对每条空间边推断：

- `INTERIOR_AGREE`：两个模态都支持为内部边，允许传播；
- `BOUNDARY_AGREE`：两个模态都认为差异大，保留边界；
- `MODALITY_CONFLICT`：一个支持、一个反对，允许弃权或低权重。

可使用同图内 rank/quantile、按空间距离和节点度匹配的 permutation null、局部不确定性、masked-feature reconstruction gain 或小型可训练 gate。不要只直接比较两个未经校准的 cosine 值。允许软三态、连续 mixture 或显式 abstention；最终以分数和配对消融决定。

### 5.3 点云与图像分割启发

- EdgeConv/DGCNN 或简化 PointNet++ 局部几何编码；
- CRF/total variation/level-set 类边界正则；
- U-Net 式多尺度 skip/residual 思想，但在稀疏坐标图上实现；
- 形态/空间坐标编码与分子表示的轻量 cross-attention。

### 5.4 聚类 head

除了 KMeans，可尝试 mclust、Leiden、Gaussian mixture、DEC/IDEC、SwAV/DeepCluster 式无监督 prototype head。公开标签可以选择 head 与数值参数，但若标签进入训练 loss，必须把该 lane 明确标成监督或半监督；不要伪称无监督。

Stage 1 目标不是证明论文，而是尽快找到能超过 identity、固定低通、support-only 和当前 W02 的结构。

## 6. Stage 2：统一可训练 RNA+ATAC 核心

把 Stage 1 的最优 2–4 个模块接入真正可训练的同一架构。建议一开始每个配置只跑一个 training seed；总计约 8–20 个完整训练配置，可根据早期饱和自由增减。

优先可改的层：

1. **Preprocessing**  
   RNA：library-size/log1p、HVG、PCA/可训练稀疏 encoder；ATAC：TF-IDF、LSI、peak selection、gene activity 或二者联合。允许 P22/MISAR 分别调 feature 数和维度。
2. **双视图 encoder**  
   保留 modality adapters，但升级共享 core；可选 GraphSAGE/GATv2/EdgeConv、multi-scale residual、shared/private representation。
3. **融合**  
   content-conditioned mixture-of-experts、cross-attention、product-of-experts、uncertainty-weighted fusion；gate 读取内容，不读取 dataset name。
4. **自监督目标**  
   self/cross reconstruction、masked modality modeling、VICReg/Barlow Twins 风格去坍塌对齐、modality dropout、graph contrastive、prototype consistency、edge-state auxiliary loss。
5. **传播模块**  
   TSPR 或更优替代模块应尽量端到端训练；若离线模块分数更高，也可先作为论文候选保留，但必须说清楚它位于 encoder 后处理层。
6. **优化**  
   允许 600–3000 steps、warmup、cosine decay、EMA、gradient clipping、不同 optimizer、mixed precision 和 early stopping；标签不可用于“无监督 lane”的单次训练 checkpoint loss，但可用于选择一次训练结束后的配置。

不要再次用弱 simple PCA anchor 强行把 learned representation 拉回起点。每个真实训练必须确认参数更新、表示变化和 checkpoint reload；成功后再做 fresh-process 回放，不要求每个失败配置都做完整交付审计。

## 7. Stage 3：按数据集追分

同一架构稳定后，允许为 P22 与 MISAR 分别搜索数值超参数，包括 feature 数、latent dim、graph k/radius、filter order、loss weights、学习率、训练长度、cluster head 和其参数。

开发优先级：

1. `P22_PROJECT_K9`：先超过 N02 native 0.5063/0.6562；达到后设 stretch target ARI 0.55。
2. `MISAR_PAPER_K12`：先闭合标准协议，再把 ARI 从当前项目水平推向 0.50、0.60 和公开 0.644 三个台阶。
3. `MISAR_PROJECT_K7`：至少超过 support-only 0.3137/0.4924，证明新模型不是只换评价协议。
4. `P22_PAPER_K18`：在相同 manual annotation/K 下追公开高水位。

允许单个最好 seed/配置先触发开发突破。不要为了让均值好看隐藏其他运行；leaderboard 同时保留 `BEST_RUN`、`MEDIAN_RUN` 和所有失败记录。达到目标后，再对固定配置补 2–3 个 training seeds；若算力有限，可放到 Night-15。

## 8. Stage 4：只对领先候选做必要检查

仅当候选达到或明显接近目标时执行：

- 相同强度的 fixed low-pass、support-only、无 conflict、无 TSPR 配对消融；
- 把同样的后处理施加到强参考 embedding，避免“只有候选得到平滑”的假优势；
- 从 raw second modality 开始的 0%、25%、50%、75%、100% corruption，完整重跑 preprocessing→forward→fusion→filter→endpoint；
- 边界泄漏、区域内连通和 marker/peak-gene 一致性；
- 真实资源和规模检查。

RNA+ATAC 六个 Night-14A corruption checkpoints 已经 6/6 exact identity；A1 seed 2 的失败不阻塞 RNA+ATAC 主论文，除非将来重新声称 universal cross-family safety。

## 9. 数据扩展触发条件

先把 P22/MISAR 主板做高。只有出现领先候选后，再投入：

- MISAR E11/E13.5/E18.5 其他阶段；
- GSE205055 的 P21/ME13 或 P22 其他 epigenome-transcriptome 切片；
- P5S1/P5S2/P5S3，用于无标签规模、稳定性和生物一致性。

新公开数据、annotation 和官方源码可以下载到新根，登记 accession、URL、license、size 和 SHA；不要修改历史 raw。

## 10. 结果分类

本轮允许如下终态：

- `SCORE_BREAKTHROUGH`：至少一个统一架构配置/seed 在同协议 lane 达到登记高水位或超过内部 native 高水位；这是开发突破，不等于论文确认。
- `MULTI_DATASET_SCORE_SIGNAL`：同一大架构、分别透明调参后，在 P22 和 MISAR 都超过各自强内部参考。
- `MODULE_SIGNAL`：新模块相对 identity/fixed-low/support-only 有可复现的独立增量，但尚未到高水位。
- `BACKBONE_OR_HEAD_SIGNAL`：分数提高来自 backbone/preprocessing/cluster head，新边模块没有贡献；如实保留并转主线。
- `NO_SCORE_SIGNAL`：实现正确但未超过现有高水位。
- `IMPLEMENTATION_FAILURE` / `INFRASTRUCTURE_FAILURE`：真实训练或环境未成立。

不要要求所有数据集和所有种子同时上涨，也不要因一个最好 seed 达标就声称 SOTA、confirmed milestone 或 paper-ready evidence。

## 11. 最小科研与工程边界

只保留五条不可越过的边界：

1. 不伪造、删除或隐藏失败 run；最佳分数必须能对应到真实配置、checkpoint 和 seed。
2. 不按数据集名称切换整套 backbone/flow；可以有 assay adapter 和透明 numeric HPO。
3. 公开标签可用于评价与 HPO；若进入 loss/gradient，就明确重分类监督属性。
4. 不修改历史 raw、不覆盖历史 tag、不 force push；第三方代码保留许可证和 attribution。
5. 不把不同 annotation/K/endpoint 的数字伪装成公平胜负。

其余工程修复、重构、试错、重新训练和合理资源使用均授权。

## 12. 交付

建议交付 `outputs/night14b_handoff/`，至少包括：

- `night14b_report.md` 与 `night14b_plain_summary.md`
- `night14b_decision.json`
- `reported_high_water_target_board.csv`
- `protocol_registry.csv`
- `all_run_ledger.csv`
- `best_run_board.csv`
- `absolute_metrics.csv`
- `module_ablation.csv`
- `architecture_and_formula_notes.md`
- `source_inspiration_and_collision_audit.md`
- 领先候选的配置、checkpoint hash、fresh-process 回放和关键源码
- resource、label-use、raw-immutability、Git/compact 审计

主报告必须先给“我现在需要知道的三件事”、明确结果分类和绝对 ARI/NMI；最佳单 seed 与均值/中位数分列。技术 hash 放最后。完成新 branch、普通 push、新 annotated tag、compact 和 Windows 独立复算后，再将 `/usr/bin/shutdown` 作为最后一条远端命令派发。

## 13. 导师汇报需要回答的七句话

1. Night-14A 的提分为什么主要是空间低通而不是冲突过滤？
2. Night-14B 新增了哪个真实模块，它来自什么问题？
3. 它与 ARISE 的 RNA 锚定硬交集有什么实质差异？
4. P22 与 MISAR 各自最好的绝对 ARI/NMI 是多少，对应什么协议？
5. 单最佳 run 是否达到目标，固定配置的其他 run 怎样？
6. 提分来自 preprocessing、backbone、传播模块还是 clustering head？
7. 当前是开发突破、跨数据信号、模块信号，还是仍未找到有效方法？

