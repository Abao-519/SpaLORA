# SpaLORA Night-7C 定向源码迁移审计

日期：2026-08-18  
目标：只提取与 Night-7B“P22 强增益、人类轻微过纠正”直接相关的公开源码机制

## 1. 审计对象与固定版本

| 项目 | 官方仓库 | 本次读取 commit | 直接相关机制 |
|---|---|---|---|
| MaxFuse | `https://github.com/shuxiaoc/maxfuse` | `2900b260b697deec678060ea653015291c4dc5cb` | 图平滑、迭代匹配、按匹配距离过滤低质量 pairs |
| ARISE | `https://github.com/XiangxiangWang-code/ARISE` | `fefdd849494c0d08e755052a7a31b20169945e40` | RNA 相似图、空间图及其 shared-edge intersection；分层融合 |
| PRAGA | `https://github.com/Xubin-s-Lab/PRAGA` | `4adb11c96fc7ddad800fa1787eadcc8b91b42784` | warm-up 后动态原型、split/merge、prototype contrast |
| PRESENT | `https://github.com/lizhen18THU/PRESENT` | `c88a609b34aae9b84c2c8a7ffb6824bae5c78f23` | 模态特异 Bayesian GAT encoder 与融合 MLP |
| SMART | `https://github.com/Xubin-s-Lab/SMART-main` | `75676546a66d48a7ebfd7c5f3a2758a4c538fcfc` | reciprocal-MNN triplet、远端负样本、GraphSAGE、重构与度量学习 |
| CoMo | `https://github.com/Lab-Xu/CoMo` | `70965cba0ca7df33e426f6a1dd5626347978cdae` | 分阶段重构、neighbor-aware contrast、cluster-aware contrast、cross-attention/WNN |
| 3d-OT | `https://github.com/dbjzs/3d-OT` | `39a7cb02748d83299cd471f172f3b972896e61d8` | PointNet++ 几何编码与 soft-communication optimal transport |
| CuPy/cuGraph | 官方 NVIDIA/CuPy 仓库与文档 | 2026-08-18 在线状态 | GPU sparse linear algebra / spectral clustering 可行性 |

本次只是临时浅克隆读取；没有把这些仓库长期复制到项目交付，也不直接复制第三方实现。Night-7C 只按概念重新实现少量、可测试的机制。

## 1.1 SpatialGlue 官方权重语义复核：用户的记忆正确

额外核验了 SpatialGlue 官方论文、官方源码仓库与复现 notebook 仓库：

- 主源码固定到 commit `7c976d811d27ace51ce47ae0ad94a068a7d222fa`；
- notebook 仓库固定到 commit `2609427ad0ca0ce1c168d29166e19c15c3e7b592`；
- 论文明确说明 loss weight factors 会随 spatial multi-omics technology 改变，但同一 technology 的数据固定；学习率统一为 `1e-4`，不同技术的 epochs 也分别设为 600、200、1500、1600；
- 主源码 `SpatialGlue/SpatialGlue_pyG.py` 第 83–98 行按 `datatype` 硬编码：SPOTS `[1,5,1,1]`/600 epochs、Stereo-CITE-seq `[1,10,1,10]`/1500、10x `[1,5,1,10]`/200、Spatial-epigenome-transcriptome `[1,5,1,1]`/1600；第 117 行用四个数分别加权两个 reconstruction loss 和两个 correspondence loss。

这不是“同一套已学习网络权重跨数据集复用”。SpatialGlue 仍对每个数据集分别训练；跨技术变化的是训练配方中的固定 loss weights 与 epochs。其官方 benchmarking notebooks 主要载入和绘制既有结果，并未提供一个跨所有平台直接推理的统一 checkpoint。

对 Night-7C 的直接结论：

1. 不应把“跨数据集统一”定义成一个 frozen checkpoint 或一个固定数值权重适配所有数据；这既不是 SpatialGlue 的做法，也不是本轮必要的创新门槛。
2. 可以接受按技术家族预设参数，但它的泛化主张较弱，而且新平台需要人工选择 preset。
3. Night-7C 采用更有价值、也更可检验的中间路线：每个 dataset × seed 独立训练，但 architecture、候选公式和 label-free router 完全共享；同一公式只根据当前单元自身的无标签统计自适应产生权重，禁止读取 dataset/platform/modality 名称。
4. 因此 A1/D1、tonsil、P22 是共同用于检验和锁定同一规则的开发面板，不是共同反向传播的联合训练集。A1 与 D1 计为两个数据实例、一个人类淋巴结数据家族。

SMART 2026 也支持这个判断：论文实验中各模态 `alpha` 一致设为 1，但明确说其他 training parameters 按数据集调整，并采用 loss-slope early stopping。也就是说，当前高水平方法通常追求同一架构/方法跨平台，而不是要求所有数据共享一个已训练 checkpoint 或每个训练超参数完全相同。

## 2. MaxFuse：可以迁移什么

实际源码位置：

- `maxfuse/match_utils.py` 的 refined matching 循环；
- `maxfuse/utils.py::filter_bad_matches`；
- `maxfuse/model.py::find_initial_pivots/refine_pivots`。

源码不是简单把所有匹配同等对待。`filter_bad_matches` 按距离分位数丢弃最差的一部分匹配，refined matching 又结合 CCA 与图/centroid smoothing 迭代更新。这正对应 Night-7B 的潜在缺陷：R02 对每个固定 MNN triplet 使用相同权重，低置信度 pair 也能产生同等梯度。

Night-7C 的迁移：

1. `FILTER75`：按固定 MNN 正匹配的 margin confidence 保留最高 75%，其余权重严格为 0。
2. `QUALITY_SOFT`：用最近正匹配和次优候选的相对距离 margin 形成连续权重。
3. 所有过滤规则在训练前由 embedding 和固定索引计算，不能读取标签。

不迁移：MaxFuse 的完整 CCA/跨样本 matching pipeline，因为当前数据是同 spot 配对，问题不是寻找跨样本一一对应关系。

## 3. ARISE：可以迁移什么，也必须拒绝什么

实际源码的 `build_dual_graph`/mouse brain 脚本构建 RNA feature KNN 与 spatial KNN，再取 edge intersection 作为第二模态的 common graph。这个“只有多种无标签证据共同支持时才强化边”的思想，适合抑制错误 MNN 纠正。

Night-7C 不打开原始 `.h5ad`，而是在 Night-7B 已保存的两套 canonical graph summaries 上计算 top-10 邻居集合的 Jaccard shared-edge support。它只改变 MNN loss 或 affinity gate 的权重，不改变 K、seed、标签或 ground truth。

必须拒绝的实现细节：ARISE 当前公开训练脚本把 `true_labels` 传入训练循环，每 epoch 计算 ARI/NMI，并保存/报告 `best_ari` 对应结果。Night-7C 绝不复制这种 ground-truth best-epoch 选择；所有模型固定 160 epochs，标签只在所有输出总锁后打开一次。

## 4. PRAGA 与 PRESENT：保留为后续候选，不塞入本轮主路径

PRAGA 的 prototype 模块在 warm-up 后用 split/merge 得到 centroids，再做 prototype contrast。这可能增强聚类边界，但它引入额外的伪标签动力学和聚类数稳定性风险；Night-7C 只登记为下一阶段备选，不与当前冲突门控同时叠加。

PRESENT 的源码采用模态特异 encoder、Bayesian GAT 和融合 MLP。SpaLORA 当前已经具有模态/图特异投影与融合，直接照搬不会针对 Night-7B 的“是否应该启用纠正”问题，因此本轮不迁移。

## 5. 2026 新方法源码复核：哪些现在用，哪些留作下一条结构路线

### 5.1 SMART

`smart/MNN.py` 实际先计算 pairwise distance，找 reciprocal nearest neighbors，再从远距离候选中随机抽负样本；`smart/train.py` 把 features、edges 和模型放到 GPU，但标准 `TripletMarginLoss(reduction='mean')` 对构造出的 triplet 等权求均值，并带有 loss-slope early stopping。Night-7B 的 R02 已经证明这一 loss family 对高冲突 P22 很有效，同时也暴露了低冲突数据被过纠正的问题。

因此本轮不是再照搬一份 SMART，而是对现有 R02 做更精确的延伸：固定 epoch、固定负样本和 checkpoint 语义不变，只给 reciprocal-MNN triplet 加训练前冻结的置信度/共享边权重。这样能直接检验“坏匹配是否在拖累人类数据”，而不会把 GraphSAGE、early stopping 和新 preprocessing 一次性混进来。

### 5.2 CoMo

`CoMo/contrast.py` 的真实实现包含两类值得后续考虑的机制：一类以邻接矩阵乘 embedding 形成 neighbor-aware positive，另一类对 cluster-head 输出做 hard-negative reweighting；`CoMo/CoMo_pyG.py` 又把 reconstruction 与 contrast 分阶段训练，并叠加 cross-attention/WNN。源码没有在训练循环直接计算 ARI，但需要固定 K、伪聚类分布和多个 stage weights。

这是一条有潜力的下一阶段结构路线，尤其适合在 Night-7C 证明简单 MNN 置信度仍不足后再独立测试。本轮不把它塞入主路径，因为伪聚类动力学、dense pairwise contrast 和 stage schedule 会同时改变太多因素，无法判断收益究竟来自冲突门控还是新训练目标。

### 5.3 3d-OT

3d-OT 的公开实现以 PointNet++ 提取坐标几何结构，并用 soft-communication OT 处理切片/模态对应。它在 2026 年 Nature Methods 的 P22 分析中强调了细粒度脑层边界，是与“小鼠大脑必须继续冲高”最相关的结构储备之一。

但它不是一个可插拔的小 loss：引入它等于更换几何 encoder、环境和相当一部分训练流程。Night-7C 先验证无需重训或只需 48 个 pilot 的高期望方案；若没有安全统一候选，后续应把“PointNet++ 几何 encoder”和“CoMo neighbor/cluster contrast”分成互斥的独立结构分支，而不是同时堆叠。

### 5.4 当前分数标尺

COSMOS 论文在其 P22 流程中报告 ARI `0.63`，SpatialGlue 为 `0.43`；当前 R02 的 mean ARI 约 `0.504`。CoMo 2026 论文报告同类 Human Lymph 的 ARI `0.2989`，而当前 C00 A1 为约 `0.2692`；它在 P22 报告 ARI/NMI `0.3896/0.5804`，低于当前 R02 的约 `0.5036/0.6504`。这些横向数值说明：P22 已进入有竞争力区间但仍低于 COSMOS 的已报告 ARI，人类 A1 仍有约 0.03 ARI 的明显追赶空间。

这些都只是公开论文标尺，不是已经完成的公平 benchmark。不同方法的 preprocessing、annotation、K、单次/多 seed 和后处理口径仍需正式 fixed-endpoint adapter 后才能作可投稿的直接比较。

## 6. GPU/CPU 加速源码判断

Night-7B 的 CPU 瓶颈来自 SciPy sparse eigensolver 与后续 partition，不是 CUDA 训练缺失。CuPy 提供 SciPy-compatible sparse linear algebra，cuGraph/cuML 也提供 GPU graph/spectral 路径；但不同 eigensolver 的浮点顺序可能改变 eigenvector rotation 乃至 partition。

因此 Night-7C 的加速顺序固定为：

1. 首选不改变算法的多进程调度 + 每进程固定 BLAS threads；
2. 只有在 30 个历史 reference units 上 canonical partition SHA 全部一致时，才允许 GPU spectral backend；
3. 任意不一致都只作为性能诊断保留，正式科学结果继续用原 CPU backend。

## 7. 许可证与实现策略

- 不把第三方源码整体复制进 SpaLORA。
- 新实现只保留通用数学思想，并写独立单元测试、来源注释和 commit pin。
- 若未来需要直接复用具体函数，必须先单独核对对应仓库许可证；本轮不依赖直接代码复制。

## 8. 最终迁移清单

本轮进入正式 registry：

- MaxFuse-inspired fixed 25% bad-match filtering；
- soft MNN confidence weighting；
- ARISE-inspired shared-neighbor support；
- initial-conflict global gate；
- phase-aware NVML/CPU profiling；
- exact-output parallel transform scheduling。

本轮明确不进入：

- ground-truth best epoch；
- 数据集名称/组织/模态类型分支；
- PRAGA prototype split/merge；
- PRESENT 全模型替换；
- 未通过 exact parity 的 GPU spectral output。

保留为下一条独立结构路线、但不进入本轮：

- SMART GraphSAGE 全主干与 slope early stopping；
- CoMo neighbor-aware/cluster-aware contrast；
- 3d-OT PointNet++ 几何 encoder 与 OT；
- 这些路线只在各自固定 registry 中单独评价，不能与 Night-7C 结果事后拼装。
