# Night-10A 本地证据、指标与创新路线审计

日期：2026-08-21  
范围：只读审计 D 盘交付、现有公开源码快照和 Night-9B 实现；没有连接 AutoDL、没有训练、没有打开新的标签文件。

## 1. 结论

下一阶段不继续耗费主要资源寻找新数据集，也不大规模运行第三方 benchmark。工作重心改为：

1. 只读复用 AutoDL 持久盘的既有 clusters、views 和 checkpoints，补齐当前最终方案的指标面板；
2. 优先修复 A1/D1 的 RNA+protein 表现，同时保留 P22 的强准确率前沿；
3. 研发“质量校准的跨模态残差去噪”（Quality-Calibrated Cross-modal Residual Denoising, QCRD）作为下一条结构主线；
4. 所有候选同时保留准确率前沿、跨数据平衡前沿和空间前沿，不再因一项轻微偏差清空整个 shortlist；
5. 真正的新外部数据留到模型冻结后确认。Night-9C 的 E18.5 数据不删除，改为待规范重建的数据资产。

## 2. 本地证据可用性

### 2.1 可以直接在 Windows 完成的工作

- Night-7A、Night-7B、Night-8B、Night-9B 的逐 seed 或汇总指标已经在 compact 中；
- 已建立 `authoritative_scoreboard_20260821.csv`，显式区分 5-seed 与 10-seed 结果，避免将不同 seed 范围当作配对结果；
- 已建立 `metric_coverage_matrix_20260821.csv`；
- 已实现独立的 `metric_expansion_reference.py`，覆盖：
  - ARI、NMI、AMI、MI、FMI、Homogeneity、Completeness、V-measure；
  - Silhouette、Davies-Bouldin、Calinski-Harabasz；
  - 对称 FOSCTTM、跨模态 Recall@K 与 paired rank；
- Windows 最小 Python 环境没有 scikit-learn，因此本机完成了语法检查和不依赖 sklearn 的跨模态测试；服务器环境必须补跑完整测试。

### 2.2 仍只能从 AutoDL 持久盘读取的证据

近期 compact 为节省 D 盘空间，没有下载当前最终候选的逐 spot clusters、views、affinity 和 checkpoint。相关权威 raw roots 包括：

- `/root/autodl-fs/night6c_raw_runs_20260817`
- `/root/autodl-fs/night6d_raw_runs_20260817`
- `/root/autodl-fs/night7b_score_rnd_20260818`
- `/root/autodl-fs/night8b_raw_runs_20260820`
- `/root/autodl-fs/night9b_racf_20260820`

指标补算必须复用这些文件；不得为了补指标重新训练，不得把 raw/checkpoint 下载到 D 盘。

## 3. 当前分数证据的一个关键修正

P22 的“当前分数”必须同时报告 seed 范围：

- Night-7B R02 十 seed：ARI/NMI/Q = `0.503580 / 0.650423 / 0.577001`；
- Night-9B F00/R02 seeds 0-4：`0.467740 / 0.633373 / 0.550556`；
- Night-9B N02 seeds 0-4：`0.506257 / 0.656187 / 0.581222`。

这不是文件矛盾，而是 P22 seed 异质性。后续不允许只挑更好的一组均值；正式结果应使用注册的完整 seed 集，并保留分布、置信区间和最差 seed。

## 4. 指标缺口

四个核心切片目前系统拥有 ARI、NMI、Q、neighbor agreement、Moran's I、Geary's C 和 boundary disagreement。MISAR 额外拥有 AMI、FMI、Homogeneity、Completeness 和 V-measure。

下一次服务器工作首先补齐：

- 核心准确性：MI、AMI、FMI、V-measure、Homogeneity、Completeness；
- embedding 几何：Silhouette、Davies-Bouldin、Calinski-Harabasz；
- 跨模态对齐：FOSCTTM、Recall@1/5/10、paired median rank；
- 空间指标：保留现有四项；CHAOS/PAS 只有在复现公开定义并通过小型 parity test 后才加入；
- 生物学 marker/peak-gene 分析暂不进入 Night-10A，留到候选冻结后。

Silhouette/DB/CH 不能替代 ARI/NMI，也不能单独用于挑选最终模型；它们容易奖励过度压缩或过度分离，只作为辅助证据。

## 5. 现有 RACF 为什么没有解决跨平台问题

Night-9B `SpaLORA/night9b_racf.py` 的关键语义：

- `robust_reliability` 位于第 142 行；
- 第 210-218 行用模型当前重构残差生成 per-spot reliability，再在融合末端加权；
- 第 224 行又将结果以 `0.75 reference + 0.25 fused` 拉回固定 reference；
- 第 237-241 行同时优化 reconstruction、cooperation 和 reference loss。

由此产生三个问题：

1. **质量信号不独立。** 重构残差来自正在训练的同一个模型，模型可以通过改变 decoder 误差改变自己的门控；残差不等同于真实数据质量。
2. **纠正发生太晚。** reliability 只在末端融合，低质量模态在 encoder/common graph 中产生的错误已经传播。
3. **reference 正则过强。** `0.75` 的固定 reference 混合加上 reference MSE，会限制新模块从旧表示中真正跳出。

这解释了为什么 N02 的固定层次融合在 P22 有效，而 reliability/DGI 组合没有形成 A1+P22 的统一收益。

## 6. 官方源码的可迁移机制

### 6.1 SMART

源码快照：commit `75676546a66d48a7ebfd7c5f3a2758a4c538fcfc`。

- `smart/MNN.py:57` 开始构造 reciprocal MNN；
- 第 98-129 行显式计算 pairwise distances、选择 mutual neighbor，并从远端样本抽 negative；
- `smart/train.py:129,265` 使用等权 `TripletMarginLoss`。

Night-7B 已证明 MNN loss 对 P22 有积极信号，同时会对低冲突数据过纠正。Night-10A 只保留“固定、稀疏、置信度加权的 MNN”作为可选正则；严禁再次运行稠密 `N×N pairwise_distances`。

### 6.2 SpaDDM

官方仓库：`https://github.com/WHY-17/SpaDDM`  
审计 commit：`5939e7d6921c7da2d895856f340896101ee1dd91`，MIT license。

真实实现不是轻量的跨模态条件去噪：

- 每个模态独立运行图 diffusion denoiser；
- 再用跨模态 attention 融合；
- decoder 中把邻接转为 dense；
- `Train_SpatialDDM` 按 datatype 写死 epochs 和 loss weights。

这些代码说明 diffusion 可以提供分数灵感，但整套照搬会带来高运行成本、dense adjacency 和大量平台参数。Night-10A 不复制整套 SpaDDM，只吸收“对低质量表示做显式去噪”的目标。

### 6.3 CANDIES

论文公开的方法语义包括：用 SI、DBI、Moran's I、CHI 识别较低质量模态，再以高质量模态和空间信息进行条件 diffusion 去噪。到本审计时没有定位到可核验的官方 CANDIES 源码仓库，因此这一部分只能作为论文机制灵感，不能声称为源码移植。

完整 diffusion 需要长预训练、数百 diffusion steps 和多个阶段，不适合直接放入本轮高效率漏斗。

### 6.4 SpatialCOC

官方源码使用坐标 INR/SIREN、空间 kernel 和 DCCA：

- `SpatialCOC/INR.py:72` 构造空间 kernel；
- 第 113 行构造完整 kernel；
- `model.py` 提供 CCA loss。

坐标连续先验值得借鉴，但完整 dense kernel 不适合大数据。Night-10A 只测试低维 Fourier 坐标特征作为小型 residual adapter 的输入，不构造 dense spatial kernel。

### 6.5 ARISE、PRESENT、SpaMCA

- ARISE 的 RNA shared-edge topology 与 hierarchical fusion 已公开，不能作为本项目独立核心创新；
- PRESENT 的模态似然和关系 KL 已在 Night-7B 测试相近机制，没有统一收益；
- SpaMCA 的 masked reconstruction 在 Night-7B/Night-8A 相近候选中没有形成稳定收益。

因此下一轮不再重复这些已经失败或高度重合的完整机制。

## 7. 新主线：QCRD

### 7.1 核心结构

QCRD 的基本形式：

1. 从冻结的基础 views 计算与训练无关的全局质量证据：Silhouette、DB、CH、Moran，以及图局部残差；
2. 计算每个 spot 的局部置信度：邻域残差、邻域熵、跨模态 cosine disagreement、MNN support；
3. 质量较高的模态作为软 teacher，不使用 hard dataset name 路由；
4. 对较低质量模态执行小型 conditional residual denoising：
   `z_low_corrected = normalize(z_low + gate * delta(z_low, stopgrad(z_high), spatial_context))`；
5. gate 和 quality features 在训练开始前冻结；adapter 不能通过改变 decoder residual 操纵自己的权重；
6. correction magnitude、局部边界和原始表示均有约束，防止把准确率换成过度空间平滑；
7. 对 RNA+protein 与 RNA+ATAC 使用同一结构，只允许由模态家族决定输入 preprocessing，不允许按 A1/D1/P22 名称写死参数。

### 7.2 为什么它比当前零散模块更适合作为论文创新

- 它直接回答空间多组学的真实问题：不同模态、不同 spot 的质量不一致；
- 它不是 ARISE 的 RNA shared-edge/hierarchy，也不是 SpatialGlue 的全局 attention；
- 它能解释为什么人类淋巴需要少纠正，而 P22 需要更强纠正；
- 它可以适配未来数据，不依赖某一套 ground-truth 标签；
- adapter 可以先在冻结 views 上快速筛选，再把成功候选并入端到端模型，成本可控。

## 8. 研发决策原则

- A1、D1、tonsil、P22 已经多次用于开发，不再伪装成 pristine holdout；允许阶段性评价后继续研发，但必须明确写成 development evidence。
- 任何训练仍不得把标签输入 loss、epoch、seed 或 cluster endpoint 选择。
- R1 后至少保留三条前沿：accuracy、balanced、spatial；一个候选可同时占多个前沿。
- 只有实现语义错误、标签泄漏、输出损坏或非有限数值才硬停止；普通负结果和轻微数据集损失不触发全轮清空。
- 不运行新外部数据、不运行新第三方 benchmark、不重新训练旧基线来补指标。

