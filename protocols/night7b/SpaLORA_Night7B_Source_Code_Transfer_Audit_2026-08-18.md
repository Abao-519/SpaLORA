# SpaLORA Night-7B 真实源码迁移审计

日期：2026-08-18  
用途：只记录可转化为 SpaLORA 数值研发候选的真实实现机制；不是论文表格比较，也不是 SOTA 声明。  
审计边界：本地与官方 GitHub 只读检查；未连接 AutoDL、未运行训练、未读取任何新数据标签、未复制无许可证仓库代码。

## 1. 结论

Night-7A 暴露的主要矛盾是：`C06_DUAL_ROW_STOCHASTIC_MEAN` 相对已确认的 `C00_G04_H05`，在 P22/D1 的 mean Q 分别再提高约 `+0.04505/+0.00545`，但在 A1/tonsil 分别下降约 `-0.00320/-0.00333`。这说明双图并非整体无效，而是不同组织、不同 spot 对两张图的可靠性不同。

因此，Night-7B 不再继续堆固定平均公式，而是优先检验三类可迁移机制：

1. **自适应可靠性路由**：依据无标签的邻域可预测性，让每个 spot 或每个数据集在 G00/G04 之间分配权重；
2. **关系级对齐**：对齐邻域相似度分布，而不是只要求配对 spot 的两个向量相等；
3. **聚类友好的表示约束**：互近邻三元组、遮挡重建和高置信伪簇一致性。

这些机制将在锁定候选、固定 seed、固定终点的开发流程中分别测试，不能用真实标签参与梯度、epoch 选择、重试或 per-dataset 参数选择。

## 2. 逐仓库源码证据

| 方法 | 解析提交 | 真实源码中与本项目相关的机制 | Night-7B 采用方式 | 不能照搬的部分 |
|---|---|---|---|---|
| COSMOS | `56ea355be51e64d9253e2871b8bd447fdfd0d230` | `pyWNN.py` 计算每个 spot 在本模态邻域与跨模态邻域下的预测误差/亲和比，再用 logistic 权重融合距离；`cosmos.py` 还使用 DGI 与空间正则 | 独立实现 G00/G04 的 local/global reliability gate | 不复制实现；不沿用其数据集特定预处理或训练终点 |
| SpaMode | `d8d8e2b70c6ad47ef12aa1a5d9a65cf4fb226c00` | `model.py` 为每个 spot 生成 noisy top-k mixture-of-experts gate，并加入 expert load/importance 平衡；另有 product-of-experts 融合 | 两专家 G00/G04 gate 与负载平衡候选 | 仓库无清晰许可证，思想重实现，源码不复制；不使用 dataset ID 作 gate 输入 |
| PRESENT | `c88a609b34aae9b84c2c8a7ffb6824bae5c78f23` | `Layers.py::IOA_loss` 将融合表示的 pairwise similarity softmax 分布与各模态分布做 KL；RNA/ATAC/ADT 分别配 ZINB/ZIP/MSE decoder | 稀疏 top-k relational KL 与重建候选 | 不采用随机未登记采样；不使用标签或 best metric endpoint |
| SMART | `75676546a66d48a7ebfd7c5f3a2758a4c538fcfc` | `MNN.py` 构造 mutual-nearest-neighbor 正样本并从远端样本中固定采负样本；`train.py` 使用 reconstruction + TripletMarginLoss，可选 Laplacian | 固定初始表示上的 MNN triplet 与难负样本候选 | 不使用结果驱动的早停；Night-7B 固定 epoch |
| MultiGATE | `7e846af5c5f5cf6a8f318dd432b68fc0752030d3` | 双图注意力自编码器；`model_MultiGATE.py` 使用两模态同 spot 的对称 CLIP loss，加各模态重建；官方 P22 notebook 训练 3000 epoch | 同 spot symmetric InfoNCE/CLIP 候选 | 不直接照搬 TensorFlow 1 实现或 3000-epoch recipe；不把图注意力解释性当性能证据 |
| SpaMCA | `33319c63350821ae701436c20753a05e87f754f6` | spatial/feature 两视图、masked node reconstruction、instance alignment、cluster distribution alignment；A1 等数据集有不同权重/epoch | 固定遮挡率重建与高置信伪簇一致性候选 | 无许可证，仅重实现思想；禁止沿用按数据集写死的权重、K 以外的标签信息或 seed 覆盖 |
| SpatialCOC | `40612e6c82368f3c6bae7f61230d78ff9fd3703e` | paired autoencoder reconstruction + deep CCA correlation objective，随后 linear CCA | DCCA/covariance alignment 作为单独候选 | 不直接把连续坐标 INR 扩展进本轮，避免同时改变过多因素 |
| MultiSP | `add9b5876cf185723c6b39d897cec3914a5605f6` | modality-specific variational graph autoencoders；融合阶段用 reconstruction、KL 与 modality discriminator/adversarial alignment | 只提取轻量 modality-confusion 对照候选的思想，不照搬整套模型 | 官方代码包含多套 dataset-type loss weights；本轮禁止 dataset-specific recipe |
| ARISE | `fefdd849494c0d08e755052a7a31b20169945e40` | RNA feature-similarity graph、spatial-distance graph、common-edge graph三支路，再与 ADT 融合；空间邻接正负约束 | dual feature/spatial/common-edge 只作为后续 graph 结构灵感 | `train.py` 每个 epoch 用真实标签算 ARI/NMI并保留 best ARI embedding；该行为是标签泄漏，Night-7B 严禁采用 |

## 3. 本轮候选映射

### 3.1 低成本 affinity/cluster head

- 固定 G00/G04 row-stochastic 权重梯度：检验 C06 的 P22 收益能否在更偏向 G04 时保留，同时修复 A1/tonsil。
- WNN-local 与 WNN-global：用同一张图的邻域是否更能预测该图的表示，形成 spot-level 或 dataset-level 权重。
- 统一 affinity 后分别走 spectral-discretize、spectral-eigen mclust EEE/VVV、eigen-kmeans 与 exact-K Leiden；K 只作为任务定义，不允许看标签内容。

### 3.2 GPU 轻量关系适配器

- `RELKL`：融合表示的稀疏邻域分布对齐各输入 view/teacher 分布；
- `CLIP`：同 spot 跨 view 对称 InfoNCE；
- `MNN`：固定 MNN 正样本和远端难负样本三元组；
- `MASK`：15% 节点遮挡后重建输入表示；
- `SEMANTIC`：仅用锁定基准 partition 生成的高置信伪簇目标，不接触真实标签；
- `MOE`：两专家 G00/G04 gate，并加固定负载平衡；
- `DCCA`：跨 view correlation alignment。

每个机制有单独候选，组合候选只在相同基础架构上相加。这样即使组合失败，也能知道是哪个模块带来收益或伤害，而不是把所有失败归因于一个不可解释的大模型。

## 4. 许可证与科研边界

1. MIT/Apache/GPL 仓库可用于理解和外部运行，但任何代码复用都必须保留许可证与归属；GPL 代码不能被静默拷入不兼容代码库。
2. SpaMode、SpaMCA、ARISE 当前解析快照未提供可依赖的兼容许可证；Night-7B 只根据公开算法思想独立重实现，不复制源码片段。
3. 允许激进工程搜索，但不允许伪造数据、读取标签训练、best-ARI epoch、seed 搜索、失败结果删除、per-dataset 隐藏分支或用文件名识别数据集。这些做法会在未来新增数据上直接失效，也无法形成可信论文结论。
4. 当前四套切片全部作为 score-development 数据；Night-7B 的任何赢家都必须在后续真正新数据上重新确认，不能称为 SOTA。

## 5. 官方源码入口

- COSMOS: https://github.com/Lin-Xu-lab/COSMOS
- SMART: https://github.com/Xubin-s-Lab/SMART-main
- PRESENT: https://github.com/lizhen18THU/PRESENT
- MultiGATE: https://github.com/cuhklinlab/MultiGATE
- SpatialCOC: https://github.com/xjtu-omics/SpatialCOC
- SpaMode: https://github.com/bridge1924/SpaMode
- SpaMCA: https://github.com/wenwenmin/SpaMCA
- ARISE: https://github.com/XiangxiangWang-code/ARISE
- MultiSP: https://github.com/jinworks/MultiSP

