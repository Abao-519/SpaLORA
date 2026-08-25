# Night-18C report

## 我现在需要知道的三件事

1. **本轮问了什么：** RNA 与 ATAC 能否先分成一个共享的组织空间趋势和两个模态私有残差，再让共享趋势改善聚类。这里“图总变差（graph total variation, graph-TV）”指允许区域内部平稳、又尽量保留边界的稀疏图正则。
2. **实际改了哪一层：** 我实现的是确定性表示层分解，不是 selector，也不是新 head。两个模态先分别降维，再用同位点、无标签的正交 Procrustes 协调坐标；随后同一稳健目标求共享趋势与组稀疏私有残差，最后所有 arm 接完全相同的 KMeans endpoint。
3. **结论与分类：** `SCIENTIFIC_NEGATIVE`。Full 在 0/3 lane 同时胜过关键 matched controls；P22 被普通 L2 低通解释，MISAR 未胜过 identity/private，人海马被 L2 或置换残差解释。因此没有启动可训练展开，也不主张新方法成立。

## 绝对指标主表

| lane | Night-16H 结构高位 ARI/NMI | retained+同 head | best full | 同配置/预算最强解释对照 | full 对最强对照 ΔARI/ΔNMI | 独立双升 |
|---|---:|---:|---:|---:|---:|---|
| P22 K9 | .587533/.708974 | .470897/.599897 | .475339/.604288 | L2 .476312/.604843 | -0.000973/-0.000556 | 否 |
| MISAR K7 | .534624/.656772 | .357936/.528705 | .363376/.535039 | private .363446/.535744 | -0.000141/-0.000705 | 否 |
| Human hippocampus K7 | .596178/.585490 | .152934/.213902 | .173062/.239081 | L2 .174856/.249512 | -0.001794/-0.010431 | 否 |

`absolute_metrics_and_controls.csv` 保留 51/51 个真实候选，包括 AMI、FMI、Moran/Geary、邻居一致率、完整簇大小、wall/RAM/GPU。这里 Night-16H 数字只作当前结构化 consumer 高位背景；它与本轮 common KMeans endpoint 不同，不能把差距全归因于表示。

## 贡献归因

- P22：full 相对 retained 略升，但同参数 L2 同时更高，故属于普通平滑解释，不是 shared/private 残差贡献。
- MISAR：full 的最佳 ARI/NMI 被 private/identity 端点双指标覆盖；没有独立增量。
- 人海马：full 相对 retained 有提升，但最佳 L2 双指标更高；一个置换私有残差 arm 也能达到相近区间，位置特异的 private residual 证据不足。
- 三 lane 的私有残差数值非零、目标逐块单调、exact K 均成立；这说明工程实现工作，并不等于科学假设成立。
- 同位点 Procrustes 协调修复了“PCA 只有 shape 相同却坐标系不一致”的语义风险；三条 rotation hash 与 ordered-ID hash 已冻结。

## 为什么不进入可训练展开

图趋势过滤及其 ADMM/神经展开已有明确先例。Graph Unrolling Networks 已覆盖无监督可训练的 graph trend-filtering unrolling，2025 的 nonconvex graph-TV 工作也覆盖 Huber 相关图正则与展开。因此展开 solver 本身不新；预注册门又是 0/3，继续训练只会扩大成本而没有可归因信号。

## 导师汇报版（7句）

我们测试了一个更具物理解释的 RNA+ATAC 表示层：共同组织趋势负责跨模态共享结构，spot 级私有残差吸收模态特有变化。为避免两个独立 PCA 轴直接相加的错误，我们先用同位点无标签 Procrustes 把坐标系协调。三条数据都用同一公式、同一参数网格和同一 KMeans endpoint，候选先锁哈希后才读公开标签。数值求解是稳定且可复算的，三 lane fresh-process partition 均 byte-exact。科学上 full 没有一条 lane 同时胜过全部关键对照：P22 与人海马主要由普通低通解释，MISAR 没有独立提升。现有 Night-16H 高位也没有被追平，因此本轮分类为 `SCIENTIFIC_NEGATIVE`，不启动 trainable unrolling。论文方向上应停止把图-TV/共享私有/展开本身当创新，下一步若继续表示研发，需要换到能先在 matched endpoint 上形成明确增量的对象。

## 失败、限制与审计边界

- Stage A 是 direct optimization，不含可训练网络参数；Stage B 未获授权。
- retained anchor + trend block 的 common endpoint 是严格贡献对照，但不是 Night-16H alpha-expansion/selector；跨 endpoint 比较只作背景。
- 公开标签用于锁定候选后的开发评价；没有盲测、SOTA 或 paper-ready 结论。
- 本轮未下载数据、未写 `/autodl-fs/data`、未派发 shutdown。

## 技术附录

- Targeted tests: 6/6 PASS；覆盖目标单调、私有残差非零、permutation 确定性、exact K、常数列、Procrustes 旋转/符号不变性。
- Fresh-process replay: 3/3 exact partitions and arrays.
- Candidate rows: 51；producer label arrays accessed: 0；evaluator label reads: 3。
- 资源：GPU training 0 秒；working 1642442 bytes；根盘剩余 2355720192 bytes；持久盘新增文件 0。
- `shutdown_dispatched=false`，AutoDL 保持有卡开机。
