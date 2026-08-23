# SpaLORA Night-15A 执行任务书

## 任务目标

把 Night-14B 的高分拆成三部分：空间几何、分子多模态、聚类 head；保留真实有效部分，研发一个在 strongest head 之上仍有独立贡献的 RNA+ATAC 模块。允许大胆试错和平台级数值调参，不要求所有 seed 或所有数据集同时成功。

## 执行风格

- Codex 2 对模块、参数网格和运行顺序有自主权；若发现更有希望的结构，可直接替代本文候选，但要留下理由与完整 ledger。
- 公开专家标签可用于固定 K、开发 HPO、候选排序和评价；必须区分开发峰值与未参与选择的验证结果。
- 不按数据集名称切换整套模型。允许按公开平台/模态统计量配置参数，也允许每个平台有不同数值超参数。
- 工程问题可修复并重跑受影响单元，不人为设置一次 correction 限额；科学公式或数据协议改变时另记 revision boundary。
- 不追求形式化审计的繁复。真正必须保留的是：输入/标签来源、配置、seed、绝对指标、失败行、checkpoint、代码版本和交付哈希。

## Stage 0：接续与真实路径

1. 校验 Night-14B compact 45/45、commit/tag、四条 frozen formal config 与 all-run ledger。
2. 列出 P22、MISAR 的真实 RNA、ATAC、坐标、label shape 与稀疏度。
3. 各取一个真实样本完成 preprocessing → forward → loss/gradient（若可训练）→ checkpoint reload → fusion → cluster endpoint。
4. 说明基线是锁定 artifact 复用还是重新构造。

## Stage A：分数来源闭合（先做，成本低）

对 P22 K=9、MISAR K=7、MISAR K=12，在相同 K、mask、seed 和 head 下至少比较：

1. coordinate-only；
2. RNA-only；
3. ATAC-only；
4. RNA+ATAC fused；
5. fused without coordinates；
6. fused without graph filtering；
7. fused without spatial refinement；
8. Night-14B frozen best chain。

主表必须同时给 BEST、median、mean、min ARI/NMI，并给 Moran's I、Geary's C、AMI、FMI、wall time、GPU time、peak memory。特别报告：fused−coordinate-only、fused−best-unimodal、with-coordinate−without-coordinate。

Stage A 后立即分流：

- 若 coordinate-only 解释 ≥90% 的 ARI 增益，当前高分登记为 `GEOMETRY_ONLY_SIGNAL`，但继续 raw-feature 模型研发，不包装成多组学成果。
- 若 fused 明显高于 coordinate-only 和两条 unimodal，进入 MCDF 增益验证。

## Stage B：降低 seed 方差（工程设施）

1. 把 backbone seed 与 endpoint seed 方差分开。
2. 比较多起点 KMeans/GMM、label-free medoid partition、稀疏 affinity 平均与轻量 consensus。
3. 允许开发标签选择一套配置；冻结后用未参与 HPO 的新 backbone seeds 3–7 与 endpoint seeds 验证。
4. BEST_RUN 仍是合法开发指标，但不得删除其他结果；单独报告“开发最佳”和“新种子复现”。
5. consensus/ensemble 只作为稳分设施，不作为唯一论文创新。

## Stage C：raw-feature backbone

优先解决 MISAR K=12，兼顾 P22：

- RNA：比较标准 HVG+PCA、轻量 masked encoder、图平滑前后表示；
- ATAC：比较 TF-IDF/LSI、可追溯 gene activity、稀疏 peak encoder；
- 融合：shared/private 表示、cross-view reconstruction、modality dropout；
- 先单 seed 小网格，出现实质增益后再扩展；无效路线及时停。

不得把已有 C15 压缩表示视为唯一入口。优先阅读 SEPAR、COSMOS、SpatialGlue、SMART、PRAGA、ARISE、3d-OT 的真实源码，以迁移可解释结构；不要求完整复现它们的整张 benchmark。

## Stage D：MCDF 候选模块

工作定义：受模态贡献约束的多尺度扩散融合。

最小可用版本：

1. 稀疏 geometry、RNA-guided、ATAC-guided、joint bilateral 四个专家；
2. 每个专家 2–3 个尺度，不做 dense N×N；
3. 内容门输出非负归一化权重；
4. 用 masked reconstruction、cross-view prediction、modality-dropout consistency 和 anti-collapse 训练；
5. 增加一项防止 coordinate-only 支配的约束，具体公式由 Codex 2 依据真实梯度/数值选择；
6. 与 strongest fixed preprocessing+head 做等口径消融。

若此结构无效，允许 Codex 2 从图信号处理、计算机视觉、多视图学习、聚类稳定性或最优传输中引入一个结构清晰的替代模块。要求只有两个：同一大框架服务所有 RNA+ATAC lane；新模块必须能做独立消融。

## Stage E：协议与数据扩展

### P22

- K=9：沿用项目 exact groundtruth，主开发 lane。
- K=18：从 3d-OT 官方 Zenodo/教程或对应论文资产取得 exact h5ad、annotation、mask、IDs；先做 hash/provenance，再跑。不可由 K=9 标签人工拆成 18 类。

### SEPAR 与其他数据

建立数据队列：

| 优先级 | 数据 | 用途 |
|---|---|---|
| P0 | MISAR E15.5 exact SEPAR artifact | K=12 主追分 |
| P0 | P22 K=18 exact artifact | 新公开协议 |
| P1 | GSE213264 Spatial-CITE-seq | 与现有 tonsil 去重后决定是否新增 |
| P1 | A1、D1、tonsil s1/s2/s3 | 统一模型旁路验证 |
| P2 | DLPFC 12 切片 | 单模态 head 稳定性辅助，不作多模态主证据 |
| P2 | Stereo-seq、osmFISH、MERFISH、CRC | 仅在低成本直接兼容时加入 |

新增下载必须来自论文、作者仓库、GEO/Zenodo 等权威入口并记录 URL、size、SHA；可下载 exact benchmark artifact，本轮不是“零下载”任务。

## 自适应算力策略

1. 便宜对照先跑；明显无效的模块不扩 seed。
2. 单 seed 达到目标即记阶段性成功，并保存配置。
3. 每类保留最多 3–5 个有希望候选进入新种子验证。
4. 大网格优先用于 raw-feature/MCDF，不再消耗在 TSPR 或外部整表复现。
5. 标签 HPO 允许，但 all-run ledger 不删行；新种子结果另表。

## 必须交付的科学表

1. `score_source_ablation.csv`：坐标、单模态、融合、滤波、refinement 对照。
2. `seed_variance_decomposition.csv`：backbone/endpoint 方差。
3. `raw_feature_and_mcdf_results.csv`：全部候选绝对指标。
4. `protocol_and_dataset_registry.csv`：P22 K9/K18、MISAR K7/K12、SEPAR 数据。
5. `main_results_table.csv`：BEST/median/mean/min、绝对 ARI/NMI、空间指标、runtime。
6. `all_run_ledger.csv`：包括失败与被停止路线。

## 终态措辞

只能从下列最贴切者选择，并可组合一个主终态加一个次级信号：

- `GEOMETRY_ONLY_SIGNAL`
- `STABLE_HEAD_SIGNAL`
- `MULTIMODAL_CONTRIBUTION_SIGNAL`
- `RAW_FEATURE_BACKBONE_SIGNAL`
- `METHOD_SIGNAL`
- `NO_ADDED_METHOD_SIGNAL`
- `IMPLEMENTATION_FAILURE`
- `INFRASTRUCTURE_FAILURE`

报告先用通俗中文说明问题、做了什么、论文含义，再给技术审计。最后一条远端命令仍为 `/usr/bin/shutdown`，但无需展开讨论控制面板状态。
