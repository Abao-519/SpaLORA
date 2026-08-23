# SpaLORA Night-15B 稳定性锚定原型与分数冲刺任务书

## 任务目标

Night-15B 不再继续 MCDF。它要完成两件互相支持的工作：

1. 用已有最强表示做更广但成本低的聚类 head/表示组合搜索，把 P22、MISAR 和 protein 数据的绝对 ARI/NMI 尽量推高；
2. 在强表示之上实现一个统一、轻量、聚类感知的有界残差，检验它能否独立增加分数，而不是再次由 preprocessing 或 head 冒充新模型贡献。

工作名 SAPR 指 `Stability-Anchored Prototype Residual`，中文为“稳定性锚定的原型残差”。它不是冻结论文名，也不是预先成立的创新声明。

## 执行风格

- Codex 2 对模块、公式、参数网格、缓存方式和实验顺序有充分自主权。若源码或真实 smoke 表明更好的统一结构更有希望，可替换 SAPR，但要说明替换理由。
- 允许大胆试错、公开标签 HPO、每个数据集不同的数值超参数、单个最佳 seed/配置作为开发突破。所有运行行必须保留，BEST、median、mean 分开写。
- 不要求所有数据集同时提高。优先取得 2–4 个强数据集上的实质绝对分数突破，再判断论文范围。
- 不复现外部整张 benchmark；只查清每个同协议数据集的公开高水位、K、mask、annotation 和评价口径。
- 不设置机械的一次 correction 限额。工程问题直接修；若改变科学公式或数据协议，登记 revision boundary。
- 审计保持够用即可：输入/标签来源、配置、seed、绝对指标、失败行、finalist checkpoint、代码版本和交付哈希。不要为每个普通 screen 生成繁复证明。

## Stage 0：接续与一次真实 P0

1. 复核 Night-15A compact 64/64、index SHA、final commit/tag、主结果表和 all-run ledger。
2. 阅读 `night15a_mcdf.py`、训练 runner、head search runner、candidate grid 和 unseen freeze；明确 MCDF 不再晋级。
3. 每个模态家族只做一个完整真实 P0：
   - RNA+ATAC：P22 或 MISAR；
   - RNA+protein：A1 或 D1。
4. P0 必须覆盖 preprocessing → trainable forward/loss/gradient → checkpoint reload → fusion → clustering endpoint；其余数据复用同一已验证路径，不重复做几十次形式化 round-trip。
5. 说明每条基线是复用锁定 artifact，还是重新构造。

## Stage 1：建立权威分数背景板

只读取论文、补充材料与官方源码，不运行完整外部方法。对每条 primary/secondary lane 登记：

- 数据/切片与 observation 数；
- 专家标签来源、K、mask；
- 论文报告的 ARI/NMI 和配置；
- 是否与项目协议直接可比；
- 当前项目 BEST、差距和可追目标。

不把不同 K、不同 annotation、不同 observation mask 的数字写成直接胜负。背景板的作用是确定冲刺方向，不占用主要 GPU 预算。

## Stage 2：一次生成可移植的 local compute kit

AutoDL 完成重型预处理后，为 P22 K=9/K=18、MISAR K=7/K=12、A1、D1、tonsil s1/s2/s3 以及已闭合标签的 GSE213264 导出轻量包。每个数据集至少包含：

- ordered observation IDs；
- coordinates、public labels、evaluation mask、K；
- 经过标准化/降维的 RNA 与第二模态 view；
- Night-14B/Night-15A 和历史 C00/F00/N02/C15/W02 中实际存在的强表示；
- 稀疏图 CSR；
- 当前最优 partitions 与配置；
- dtype、shape、ID hash、来源 commit。

推荐 float32 或经过数值回放确认的 float16；压缩目标不超过 2 GiB。不要把 raw fragments、dense N×N affinity 或无界 checkpoint 仓库塞入包中。

同时提供 Windows 可执行 runner：给定 kit 和 JSON 配置，能完成 PCA/whitening、view 加权、坐标特征、KMeans/GMM、稀疏 refinement、metrics、partition alignment、medoid/consensus 和 CSV ledger。这样 `/usr/bin/shutdown` 后，head HPO、指标复算、报表与最终打包继续在本机运行。

## Stage 3：低成本绝对分数冲刺

先在 reduced views 上广泛搜索，不训练新网络。允许用公开标签选择开发最优配置，但不删除失败行。

### 表示组合

- RNA、第二模态、base fused、历史强 fused；
- 连续权重混合，而非只比较 0/0.5/1；
- view whitening、variance normalization、CCA/PLS 或轻量 multi-view PCA；
- 对 P22 K=18 特别保留强 ATAC latent；对 MISAR 不强迫融合优于单模态。

### 空间与滤波

- 坐标线性/二次/RBF basis；
- 小范围 coordinate weight；
- no filter、low-pass、bilateral、support-only；
- 只用稀疏图，不生成 dense N×N。

### 聚类与稳定化

- KMeans 多初始化；
- GMM 的 covariance_type、reg_covar 与初始化；
- 若环境已具备且协议闭合，可试 mclust、Leiden/Walktrap 或 Ward；
- 多分区 Hungarian 对齐、partition medoid、轻量 consensus；
- 允许标签选 BEST head，也要同时登记不使用标签的 medoid。

搜索应缓存数据与 PCA，批量处理 candidate/seed，避免每行重载原始数据。先 coarse search，再在 top neighborhoods 做细网格。目标不是形式上的固定网格，而是用合理资源逼近每个数据集的可达上限。

### 第一阶段冲刺线

- P22 K=9：先过 ARI 0.60；
- P22 K=18：先过 ARI 0.65；
- MISAR K=7：先过 ARI 0.55；
- MISAR K=12：先过 ARI 0.50，再向 0.644 的 SEPAR 背景线靠近；
- A1、D1、tonsil s1/s2/s3、GSE213264：各自建立同协议历史 BEST，并争取至少 3 个 protein 切片提升。

未过线不等于实现失败；按真实 BEST 与差距继续决策。

## Stage 4：SAPR 或更优替代模型

### 4.1 teacher 与锚点

1. 取 Stage 3 中 5–15 个分数较高且结构不同的 partitions；teacher 的生成不直接使用 ground-truth class ID。
2. 用 Hungarian matching 对齐 cluster ID。
3. 对每个 observation 计算跨 partition 一致率、局部邻域一致率和多模态一致率。
4. 高一致率点作为 interior anchors；低一致率、空间邻域冲突或多模态冲突点作为 uncertain boundary。

### 4.2 统一残差核心

同一核心接收：强 frozen embedding、两个模态 view 的差异/一致性摘要、稀疏空间邻域摘要和 anchor confidence。输出：

- 小幅 embedding residual；
- 每个点的 residual trust/gate；
- K 个 prototype assignment。

RNA+ATAC 与 RNA+protein 允许不同输入 adapter 和数值超参数，但 residual/prototype 主体相同，不允许 `if dataset == P22` 之类的结构分流。

### 4.3 候选 loss

Codex 2 可从下面选择简洁组合，不要求全上：

- anchor-weighted prototype sharpening；
- balanced assignment/Sinkhorn，防止 cluster 塌缩；
- cross-view prototype consistency；
- boundary-focused sparse graph consistency；
- interior trust-region，限制已稳定内部点漂移；
- residual norm penalty，保证新模块是增量而不是暗中替换 backbone。

优先控制在一个小 MLP/稀疏 GNN、64–128 latent、几百步训练。先 1 个 training seed 做 10–20 个有针对性的候选；选 top 2–3，再增加两个 training seeds。不要在没有信号时直接跑 30 个正式模型。

### 4.4 必要但极简的贡献对照

只要求同一 head 下三行：

1. retained teacher/strong embedding；
2. SAPR full；
3. SAPR residual disabled。

若 full 没有超过 retained teacher，就登记 `HEAD_ONLY_SCORE_GAIN`，不得把 SAPR 写成提分来源；但 head 高分仍保留为阶段性成果。

## Stage 5：跨家族和新数据扩展

Primary lanes 先形成候选，再迁移到 protein：A1、D1、tonsil s1/s2/s3 和 GSE213264 Spatial-CITE-seq。GSE213264 必须先闭合真实标签、mask 与 observation IDs；若没有可审计标签，可只进入无标签空间/稳定性结果，不伪造 ARI。

“统一”指模型思想和主核心一致，不指所有数据必须用同一数值参数。论文可以合理使用：

- RNA adapter；
- ATAC adapter；
- protein adapter；
- 平台/数据集级学习率、图邻居数、坐标权重、PCA 维数和聚类 head 参数。

不能使用的是：按数据集名称选择完全不同的模型结构，再把它包装成一个自适应模型。

## 资源与效率

### AutoDL

- 单个 Python 进程复用 raw tensors、graphs 和 SVD/PCA caches；
- mixed precision 仅在回放误差可接受时启用；
- GPU 训练采用 coarse-to-fine funnel；
- 只给 finalist 保存完整 checkpoint；普通 screen 保存 config、seed、metrics 和必要 embedding hash；
- GPU 阶段完成、kit 下载并验证后，最后一条远端命令 `/usr/bin/shutdown`，不再重连。

### Windows 本地

- 所有 embedding-level head HPO；
- all-run ledger、metrics、plot、表格；
- compact SHA 与 bundle 复核；
- 本地失败可修复重跑，不消耗 AutoDL。

若本地 15 GiB RAM 不足，runner 必须支持 memmap、按数据集串行和 float32；不要为了并行速度导致交换内存或崩溃。

## 结果分类

1. `HEAD_ONLY_SCORE_GAIN`：绝对 BEST 提升，但 SAPR/替代模型没有独立增益。
2. `CLUSTER_AWARE_RESIDUAL_LOCAL_SIGNAL`：SAPR 在至少部分真实数据集独立提高，尚未形成跨家族证据。
3. `CROSS_DATASET_METHOD_SIGNAL`：同一残差核心在多个数据集、至少两个模态家族产生可复算增益；这是下一阶段可发展论文故事的信号，不自动等于 SOTA。
4. `NO_ADDED_SCORE_SIGNAL`：head 与新模型均没有超过已有高水位。
5. `IMPLEMENTATION_FAILURE` 或 `INFRASTRUCTURE_FAILURE`：真实路径或资源未闭合，不能形成科学结论。

## 主报告要求

正文先写“我现在需要知道的三件事”，并解释 SAPR、prototype、anchor、teacher、partition medoid 第一次出现时的中文含义。

主表至少包括：数据集、K、总数/评价数、历史强参考、BEST candidate、绝对 ARI/NMI、ΔARI/ΔNMI、AMI/FMI、Moran/Geary、training seed、endpoint seed、BEST/median/mean、wall/GPU/RAM。另给：

- SOTA/公开高水位与协议可比性板；
- teacher/full/residual-disabled 三行最小对照；
- local compute kit 清单、体积与本地回放结果；
- GPU 时间与移到本地的 CPU 时间；
- 5–8 句导师汇报版；
- 技术审计附录中的 commit/tag/index/hash。

不要宣称审稿人不会运行代码，也不要用“以后补消融”解释当前无效模块。我们可以把审计压缩到必要程度，但论文故事必须与已观测的分数来源一致。
