# SpaLORA Night-13A：数据扩展、统一 runner 与开发基线板

## 1. 本轮只解决什么

本轮不再设计远距离机制。它只把历史有效资产和新增真实数据放入同一个可复算接口，得到一张绝对 ARI/NMI 与运行资源齐全的开发基线板，作为 Night-13B 大规模组件搜索的起点。

用户已明确决定：历史标签是否读过，不影响数据进入开发、调参和比较。A1、D1、tonsil、P22、MISAR 可作为普通公开 benchmark 使用。报告只需如实说明它们是开发/复现实验，不必再用 pristine 门阻塞执行。

## 2. 数据任务

### 2.1 物理切片去重

建立 `canonical_dataset_registry.csv`，至少含 study、donor、physical section、accession、modality、platform、shape、coordinate source、label source、prior use、canonical representation、overlap status、raw path、size、SHA。

tonsil 只采用 Zenodo 12654113 三切片作为分析表示。GSE263617 tonsil A1/D1 仅用于来源核对；必须用 ordered spot ID、坐标和共享 count 签名判断它们是否对应 Zenodo slice1/2 的过滤版本。若无法闭合，标记 `AMBIGUOUS_OVERLAP_FAIL_CLOSED`，不得重复计数。

SPOTS 的 GEX/ADT GSM 是同一物理切片的两种模态，不得算两个样本。P22 的多种处理版本和 P5 的不同分辨率也不得扩充成虚假的独立样本。

### 2.2 下载

优先复用远端历史原件并核对 size/SHA。缺失或不匹配时，只允许 contract 的 Zenodo `data_imputation.zip` 和 SPOTS 四个 processed 文件。不得下载 SRA/FASTQ、整包 raw、E18.5、SPOTS 乳腺癌或其他新数据。

新下载逐文件保存 URL、时间、size、SHA。raw 只读，不原位修改。压缩包只提取白名单成员。

## 3. 源码审计与基线范围

亲自读取正式论文、Methods/Supplement 和官方源码；不能只读 README。对 SpatialGlue、SMART、ARISE 固定 commit 和 license，记录真实模型、loss、默认配置、数据集特例、聚类 endpoint、手工 cluster 编辑和支持模态。

执行基线：

1. 标准化模态预处理后直接拼接的简单锚点；
2. SpatialGlue；
3. SMART；
4. ARISE。

项目内部 C00（通用 G04+H05）、F00/R02（RNA+ATAC 参考）和 N02（P22 层级融合）优先复用锁定 artifact；不得用同名重构结果冒充复用。Night-9B COSMOS 只复用已有结果作背景，不重新训练。SpaMV、SpaMode、SpaBalance、CANDIES 本轮只完成迁移审计，留给 Night-13B。

外部代码放在独立 source root；除非许可证明确允许，不复制进项目仓库或 compact。wrapper 只做输入、输出、seed、checkpoint 和资源记录适配。若必须改变模型、图或 loss 才能运行，该方法单元记为 unsupported/blocked，而不是伪称复现成功。

## 4. 统一 runner

所有方法共享一个 CLI 和输出 schema：dataset manifest、modality adapters、method、seed、resolved config、embedding path/hash、partition path/hash、checkpoint path/hash、runtime、peak GPU/RAM、status、exception summary。

允许 RNA、ADT、ATAC 使用各自标准预处理；允许输入维度决定 PCA/LSI 上限，允许为避免 OOM 机械缩小 batch。禁止读取 dataset 名称后切换模型、loss、epoch、权重、聚类算法或后处理。

common endpoint 是主比较，native endpoint 只作背景。任何人工合并、拆分、重命名 cluster 的步骤全部跳过并登记。

## 5. 真实端到端 P0

### RNA+protein

首个 smoke 使用 SPOTS spleen replicate 1。10x H5 内 RNA 和 21 ADT 按 `feature_type` 拆分；4992 行原始位置表与 2653 个 barcodes 按 barcode 和 tissue 状态机械对齐，最终必须为 2653 个唯一 paired spots。

### RNA+ATAC

首个 smoke 使用 P5S1：真实 RNA 加 83,593,412 fragment rows，复用 Night-12A 冻结的 Ensembl79 clean-room gene-score 合约，不能声称与 ArchR GeneScoreMatrix 数值等价。

两条都执行真实 `load → preprocessing → forward/loss → embedding → checkpoint strict reload → fresh-process round-trip → fusion → sparse clustering endpoint`。seed=0，optimizer steps=0。禁止用合成 adapter 替代真实混合路径，禁止 dense N×N。

P0 通过后，简单锚点和每个外部强基线至少在 A1 与 P22 各跑一次真实 round-trip。单个 baseline 阻塞不影响其他独立 lane。

## 6. 开发基线板

P0 通过后才读公开 benchmark 标签并生成 `development_baseline_board.csv`。本轮只有 seed 0，不做结果驱动的超参数搜索：外部方法用官方默认，项目方法用锁定配置。

主列至少包括 dataset、method、endpoint、K、absolute ARI、absolute NMI、AMI、FMI、homogeneity、V-measure、Moran's I、wall time、GPU time、peak GPU、peak RSS、status。失败和不支持的单元必须保留在同一表中。

已知 K 从该数据的 canonical public annotation 读取一次，所有方法共用。不得看分数后选 seed、epoch、resolution、聚类算法或合并 clusters。Night-13A 的一颗 seed 结果只用于建立开发起点，不构成 SOTA、论文成功或失败。

## 7. 工程修正政策

取消“全局只能修一次”的规则。纯 API、依赖、I/O、shape、checkpoint 兼容问题可以正常调试；每次增加最小回归测试并在统一 changelog 写一行。修复后从受影响的 method×family P0 起点完整重跑，不能只补最后一步。

任何模型公式、图、loss 或科学超参数语义变更必须另存 resolved config，不能伪装成工程修复。某基线连续出现三个不同且仍无法闭合的 blocker 后，停止该 lane、保留日志，继续其他方法。

## 8. 成功与停止

成功状态 `NIGHT13A_UNIFIED_BENCHMARK_BOARD_READY` 需要：数据去重表闭合；两家族真实 P0 通过；简单锚点加至少两个强外部基线在两家族真实路径通过；开发基线板形成；失败单元未隐藏。

若只完成部分，使用 `NIGHT13A_PARTIAL_BENCHMARK_BOARD`。数据 provenance、实现或基础设施失败分别明确分类。无论成功与否，本轮不训练新候选、不声称 SOTA。

## 9. 精简交付

只需一个简短报告、一个 decision JSON、一个数据表、一个源码迁移审计、一个 run manifest、一个 P0 JSON、一个基线表、一个 engineering changelog、测试摘要、增量 bundle 和 root-relative compact index。完整命令日志留远端，不在聊天逐条复述。

聊天只在四个节点更新：本地 authority 完成、数据与源码 P0、真实两家族 P0、最终基线板与交付。最终先用通俗中文给三件事，再给主表；Git/hash 放技术附录。

