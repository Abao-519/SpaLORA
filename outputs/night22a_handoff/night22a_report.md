# Night-22A geometry-aware partition junction report

## 我现在需要知道的三件事

1. **问题**：Night-21C 显示锁定表示里仍有标签后置 probe 可分信息，但普通 KMeans/GMM 无法稳定把它变成空间分区。本轮先测非球形几何上限，再直接优化 `N×K` 分区变量；它不是从旧候选中挑一个答案。
2. **实际动作与结果**：feature-Ncut/Leiden 在 A1、P22 相对 Night-21C endpoint 双升；随后 clean-room junction 在同一 retained carrier 上联合椭圆发射项与五张稀疏图。`FULL` 在 P22 和 placenta 严格胜过同起点全部原子臂，但 A1 与 tonsil s1 被更简单的 shared/additive control 解释，跨家族冻结门失败。
3. **论文意义与分类**：终态为 **LOCAL_PARTITION_JUNCTION_SIGNAL_WITHOUT_FAMILY_CONFIRMATION**，并有独立的 `HEAD_ONLY_GEOMETRY_SIGNAL`。Placenta ARI 刷新到 0.551001，但 NMI 未同步刷新；不能称盲测、SOTA、统一方法里程碑或已经冻结的方法名。

## 绝对指标与匹配贡献

| lane | FULL ARI/NMI | Δ vs strongest atomic | Δ vs Night-21C endpoint best | min cluster | strict independent pass |
|---|---:|---:|---:|---:|---|
| A1_K10 | 0.269838/0.399981 | -0.002120/-0.005149 | +0.023931/+0.029558 | 114 | False |
| P22_K9 | 0.504496/0.643214 | +0.000062/+0.000158 | +0.028806/+0.026211 | 44 | True |
| PLACENTA_K10 | 0.551001/0.625391 | +0.008517/+0.003692 | +0.077363/+0.018534 | 43 | True |
| TONSIL_S1_K4 | 0.198037/0.257534 | -0.008402/-0.011920 | -0.002340/-0.018260 | 676 | False |

完整 AMI、FMI、homogeneity、V-measure、Moran/Geary、簇大小和 partition SHA 见 `absolute_main_table.csv`。固定输入下优化器是确定性的；门失败后没有用多 seed 扩张制造“最好一次”。

## 几何 ceiling

- A1 retained + feature-Ncut k24：0.268066/0.395941，比 Night-21C endpoint best +0.022159/+0.025518。
- P22 retained + feature-Leiden：0.503401/0.641908，+0.027710/+0.024905。
- Placenta feature-Leiden：0.535146/0.592084，ARI 上升但 NMI 下降。
- Tonsil s1 最佳新几何 0.194390/0.259507，低于 Night-21C endpoint best。

这说明 non-spherical feature geometry 确实补回了一部分 endpoint gap，但仍没有达到 A1 0.276172/0.421937、P22 0.596390/0.718243 等历史高位。

## junction 的真实归因

`Q` 是直接训练的软分区；每一步在稀疏图上计算 cluster-conditioned normalized association，并与锁定表示上的对角椭圆发射项联合。`FULL` 相对 matched atom 的净增益在 P22 极小（约 +0.000062/+0.000158），在 placenta 较明确（约 +0.008517/+0.003692）。Protein 两条均由 `SHARED_GRAPH_ONLY` 或 `ADDITIVE_SHARED` 支配，因此不能把“组合后高一点”推广为跨家族方法。

Cycle 1 是有效开发结果，源码保存在 commit `1ac9231d577f22db3651312f6cbac846eed85499`。Cycle 2 只依据已锁定 Stage-A 证据向所有 lane 同时补上 retained k24 图，其他目标和预算未变；仍未越过跨家族门，因此停止，不运行 D1/tonsil s2/s3/MISAR。

## 新颖性与可继续性

DEC、SwAV、P²OT、DeepCut、普通多视图图融合、BANKSY、spaMGCN、S3RL、SEPAR 与 CRCT 已占据软分配、平衡、normalized-cut、邻域几何、图融合和原型等组件。Night-22A 没有复制第三方源码。当前仅“cluster-conditioned 多图 junction + 椭圆发射的直接分区联合体”可作为工作对象，但证据只有 chromatin 本地信号，不足以冻结论文名称。下一轮若继续，应该先做外部 chromatin frozen confirmation；不应继续在这四条 discovery lane 扩网格。

## 外部数据旁路

GSE205055 ME13 50 µm 是同一 GSE205055 study family 的新物理样本，官方 raw 总包约 7.6 GB，canonical K/mask 尚未闭合。Stereo-CITE thymus 是独立 RNA+protein 来源且有四张切片，但原论文中的 cortex/medulla 解释不能自动等同独立全域 ground truth。两者都只完成 provenance/体量审计，没有下载或制造标签。

## 导师汇报版

我们把 Night-21C 的“表示里有信息但 KMeans 接不出来”拆成了非球形 head 和直接分区优化两层。第一层很明确：feature-Ncut 或 Leiden 在 A1 和 P22 都比 Night-21C 的八个 endpoint 双升，说明几何确实是瓶颈之一。第二层我们不是挑旧候选，而是直接训练 N×K 分区，并把椭圆簇似然与多尺度、多模态、空间稀疏图放进同一目标。这个 FULL 在 P22 和 placenta 都严格超过同预算原子臂，placenta 的 ARI 到 0.551，但 protein 两条没有独立贡献。因此本轮是 chromatin 局部的 partition-junction 信号，不是跨家族里程碑，也没有启动冻结迁移。已有工作已经覆盖 normalized cut、原型分配、图融合和邻域特征，我们只保留联合对象的窄边界。最合理的下一步是用冻结 chromatin 公式做真正外部样本确认，而不是继续在开发集调参。

## 技术状态

- 真实 P0：4/4，实际 backward、参数改变、strict checkpoint reload 均通过。
- Fresh-process replay：4 条 headline bank 的全部 15 个 trainable candidates 均 exact；另一个 start 的完整 bank 也已重放，完整表在 ledger。
- Stage-A candidates 先锁/hash，Stage-B candidates 与 checkpoints 先锁/hash，标签随后由独立 evaluator 打开。
- 根盘保持约 22.6 GiB 可用，超过 15 GiB 安全线；未下载新 raw。
- GitHub 认证预检在本轮开始时一次通过；普通 push 只在 final commit 后尝试一次。
