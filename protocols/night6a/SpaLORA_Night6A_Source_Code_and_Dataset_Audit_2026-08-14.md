# Night-6A 源码与数据集审计

日期：2026-08-14

## 1. 审计目的

本审计不是按论文摘要罗列模块，而是区分三件事：

1. 哪些思想值得 clean-room 迁移；
2. 哪些公开高分依赖标签、数据集专用配置或不可比流程；
3. 哪些独立数据适合进入下一轮开发或验证。

## 2. 近期方法的源码级结论

| 方法 | 源码中实际做法 | 可迁移思想 | 不能照搬的证据问题 |
|---|---|---|---|
| ARISE | 构造空间图与 RNA 特征图交集的 shared-edge graph；分层融合。官方训练代码周期性聚类并计算真标签 ARI，以最高 ARI 保存 `best_embeddings` | 共享边/空间边去噪、分层图融合 | 真标签参与 checkpoint 选择，报告数值不能作为公平 label-free 目标；仓库未见清晰许可证，限 clean-room |
| SpaBalance | 对多任务梯度计算冲突；冲突时用 MinNorm 权重，否则等权；同时含 Barlow Twins 和 DGI 类局部-全局损失 | 训练全过程的梯度方向协调、跨模态冗余约束 | 官方代码按数据类型使用不同 task weight，且默认随机 seed，不应复制为统一公平配置 |
| SpaMode | 图 VAE、shared/private experts、modality discriminator、逐 spot MoE/PoE；训练函数本身不读 GT | 共享/私有分解、软模态路由 | A1、D1、MISAR 等教程使用不同 epoch、优化器、学习率和损失权重；高复杂度模块不宜第一轮整套移植 |
| PRESENT | Bayesian GAT、跨模态对齐、ZINB/ZIP/MSE 模态似然、固定无监督终止 | 模态适配重构、轻量跨模态对齐 | 整套移植变量过多，先做单模块验证 |
| CoMo | 分阶段重构与对比学习，包含 cluster-aware 和 neighbor contrastive 目标 | 延迟启用的邻域/聚类一致性 | 仓库规模小、默认运行配置与论文设置映射不完全透明；未见清晰许可证，限 clean-room |
| MultiGATE | 两层 GAT 与 CLIP 风格对齐 | 对称跨模态对比 | 论文明确对部分无 GT 数据使用 ICC；不同工作的 P22 标签口径并不统一，不能假设所有 P22 accuracy 都直接可比 |
| PRAGA | 原型/邻域对比与可学习邻接 | 原型稳定性、邻域一致性 | 稠密 N×N 邻接有显存和过拟合风险；部分设置依赖已知 cluster 数 |

## 3. 对 Night-6A 的取舍

### 立即进入候选注册表

- RNA 支持的空间边软剪枝；
- 固定顺序、确定性的 PCGrad；
- MinNorm 梯度权重对照；
- 标准化 Barlow Twins；
- 延迟启用的跨模态邻域对比；
- 由已验证单模块组成的少量组合。

### 本轮暂缓

- 稠密 N×N 可学习邻接：计算和统计混杂太大；
- 全套 VAE + adversarial + MoE/PoE：一次引入变量太多；
- 标签选 checkpoint 或按数据集分别手工调参：禁止；
- 大规模 prototype/pseudo-label 自训练：在当前基线不够强时容易放大错误；
- 直接照搬没有许可证或许可证不清晰的实现：禁止。

## 4. 新数据集审计

### 4.1 人类扁桃体（优先新增）

SpaMosaic/SpaMode 公开材料描述了三个 Stereo-CITE-seq 人类扁桃体切片，具备配对 RNA 与 ADT；section 1 含四类人工注释区域。它与 A1/D1 淋巴结相关但独立，适合作为 Night-6A 的第二核心开发集。

执行前硬门：

- 数据必须来自官方仓库或论文链接的 Zenodo 记录；
- RNA、ADT、坐标和 annotation 条码一一对齐；
- annotation 来源写入 provenance，不得是本轮模型生成；
- 标签文件在每个 round 的训练、embedding 和聚类锁定前不可打开；
- 保存原始文件、处理缓存和标签文件的 SHA-256。

### 4.2 MISAR E15.5（条件新增）

该数据含配对 RNA/ATAC 与空间坐标，多个近期方法使用 E11/E13/E15/E18。E15.5 可作为将来独立多组学类型验证，但其“ground truth”来源必须先审计；若是由模态聚类或参考映射派生，只能明确称 reference annotation，不能称人工组织学真值。

### 4.3 D1 与 GSE198353

- D1 有人工区域标签，但与 A1 同研究，只计 within-study cross-section；
- GSE198353 两个脾脏 replicate 可做 label-free 稳健性，不进入 ARI/NMI 主表。

## 5. 文献数值的正确用途

公开 A1 数值只用来设置研发方向和竞争力标志。只有在数据 section、预处理、特征选择、cluster 数、聚类器、seed、标签版本和 checkpoint 规则全部对齐后，才能作公平 method-vs-method 声明。Night-6A 不以“高于某论文单个数值”自动宣称 SOTA。

## 6. 主要来源

- ARISE paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC13360277/
- ARISE source: https://github.com/XiangxiangWang-code/ARISE
- SpaBalance paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12752582/
- SpaBalance source: https://github.com/nudt-bioinfo/SpaBalance
- SpaMode paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC13335753/
- SpaMode source: https://github.com/bridge1924/SpaMode
- PRESENT source: https://github.com/lizhen18THU/PRESENT
- CoMo paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC13092272/
- CoMo source: https://github.com/Lab-Xu/CoMo
- MultiGATE paper: https://www.nature.com/articles/s41467-025-63418-x
- Human tonsil / SpaMosaic data documentation: https://spamosaic.readthedocs.io/
- Human tonsil Zenodo record: https://zenodo.org/records/12654113

所有借鉴均限于公开方法思想和 clean-room 重新实现；具体许可证需在真正复用任何代码前再次核验。
