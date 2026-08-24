# SpaLORA Night-15G 可选形态视图分数冲刺报告

## 我现在需要知道的三件事

1. 本轮想判断：在 RNA+protein 的分子表示和空间坐标之外，精确配准的 H&E 组织图像能否提供可复算的额外分区信息。
2. 实际完成了 A1、D1、tonsil slice 3 的真实 patch→手工/ResNet18 特征→可选视图能量或融合 head→分区路径，并保留缺失图像、打乱图像、关坐标和仅分子对照；它发生在表示融合/聚类 head 层，没有训练新的分子或图像 encoder。
3. 终态为 `NIGHT15G_OPTIONAL_MORPHOLOGY_LOCAL_SIGNAL`，分类 `LOCAL SIGNAL`。D1 和 tonsil slice 3 有明确局部信号，A1 只有很小双升；结果来自公开标签跨运行 HPO，尚不是统一可训练方法、盲测、SOTA 或论文级证据。

## 绝对指标主表

| 数据/配置 | K | Night-15F ARI/NMI | Night-15G ARI/NMI | ΔARI/ΔNMI | AMI/FMI | Moran/Geary | 最小簇 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 平衡 optional energy | 10 | 0.275543/0.420138 | 0.276003/0.421740 | +0.000460/+0.001602 | 0.417970/0.416275 | 0.562052/0.441094 | 114 |
| D1 无微小簇 morphology head | 10 | 0.255044/0.389356 | 0.288876/0.413762 | +0.033832/+0.024406 | 0.409788/0.428783 | 0.518540/0.484108 | 86 |
| D1 max-ARI morphology head | 10 | 0.255044/0.389356 | 0.350729/0.416032 | +0.095685/+0.026677 | 0.412060/0.504699 | 0.431672/0.555986 | 1 |
| D1 Stage2 平衡 | 10 | 0.255044/0.389356 | 0.315365/0.397100 | +0.060322/+0.007745 | 0.393130/0.485661 | 0.530663/0.489695 | 1 |
| tonsil s3 平衡 optional energy | 4 | 0.341107/0.300966 | 0.349881/0.309267 | +0.008774/+0.008300 | 0.308624/0.614030 | 0.698586/0.308226 | 295 |

D1 主结论采用无微小簇解；两个含单样本簇的更高分只作为开发 ceiling 原样保留。P22、MISAR、tonsil s1/s2 没有闭合形态图像，presence mask 精确返回 Night-15F 分区，不计为 Night-15G 提分。

## 机制证据

D1 无微小簇配置的固定配对消融为：完整输入 0.288876/0.413762；缺失图像 0.246261/0.356071；打乱图像 0.268477/0.377298；关闭坐标 0.283602/0.393674；仅分子 0.246530/0.369177。完整输入严格高于这些对照，说明 D1 增益不是单纯换一个 GMM head。

A1 完整结果为 0.276003/0.421740，固定可靠度降至 0.273043/0.418548，打乱图像为 0.273669/0.417868，缺失图像精确回退。tonsil s3 完整结果为 0.349881/0.309267，固定可靠度为 0.330678/0.303592，打乱图像为 0.310132/0.277032，缺失图像精确回退。这支持“局部一致/冲突可靠度有用”的开发解释，但 unary-only 在 A1/tonsil s3 与 full 相同，不能宣称形态 edge 项已经得到独立普适支持。

## 真实资产与工程边界

A1、D1 分别用 GEO GSM8195494/GSM8195496 的 hires H&E 和 tissue positions，barcode 经冻结纯格式前缀剥离后 3484/3484、3359/3359 对齐。GEO 未提供 scalefactors_json.json；hires scale 0.1 是由 deposited full-resolution coordinates 与图像尺寸机械闭合，必须视为限制。tonsil s3 使用 h5ad 内嵌图像及官方 hires scale 0.95283467。三条路径的 patch 半径均为 8/16/32，所有 patch 有界；ResNet18 权重 SHA 为 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec。

宽 D1 morphology-head 网格计划 5280 行，在 3432 行时 exit code 1 且无捕获异常文本。部分 ledger 完整保留、没有重跑；其中 6 个 profile 在原 Worker 环境完成精确算法重放。

## 复现边界

最终冻结分区在两个新进程中完成 18/18 artifact SHA 与指标复算，11/11 targeted tests 通过。与此同时，当前 Windows 环境从 PCA/GMM 或 PCA/alpha-expansion 重新执行算法时，微小 BLAS 数值差异会跨越离散聚类边界，partition SHA 不再等于原冻结值；1/2/4/6/8/12/16/32 线程均已审计。因而本轮可声称“冻结工件与指标可独立复算”，不能声称“跨 BLAS 环境的完整算法 byte-exact portability 已闭合”。

## 与已有工作的碰撞和论文含义

MISO、Proust、STESH、stGCL、SpatialEx/COSIE 已覆盖把 H&E 作为通用额外模态；optional third view 本身不是创新。Night-15G 目前最多支持一个待继续验证的组合：在同一稀疏空间能量/融合消费者中使用局部形态一致、冲突和 presence mask，并在缺失时 exact fallback。D1 的主信号来自 PCA+tied-GMM head，A1/tonsil s3 来自 optional energy，消费者尚未统一；也没有新的 trainable representation objective。因此证据只能归为公开 benchmark 开发 `LOCAL SIGNAL`。

## 导师汇报版

Night-15G 检验了 H&E 图像能否为已有分子和空间聚类提供额外证据。我们在三个真实 RNA+protein 单元上完成了精确 spot-image 对齐和多尺度图像特征路径。A1 只有小幅双升，tonsil s3 的 ARI/NMI 同时提高约 0.0088/0.0083。D1 的无微小簇结果由 0.2550/0.3894 提高到 0.2889/0.4138，缺失或打乱图像时明显回落。更高的 D1 分数含单样本簇，只作为开发高位保留。结果说明配准形态确有局部价值，但还没有形成一个统一的可训练表示方法。冻结工件和指标可两次新进程精确复算，算法跨 BLAS 环境的 byte-exact 重算仍未闭合。结论是 `LOCAL SIGNAL`，不是 SOTA、确认里程碑或论文完成。

## 技术附录

- 测试：11/11, 0 failed。
- 冻结 artifact replay：两次各 18/18。
- 训练标签/表示标签/能量标签/head fit 标签读取：0；公开标签跨运行 HPO 与评价：1。
- dense N×N：0；历史 raw 修改：0；第三方完整 benchmark：0。
- AutoDL shutdown_dispatched=false；状态指令 `KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS`。
