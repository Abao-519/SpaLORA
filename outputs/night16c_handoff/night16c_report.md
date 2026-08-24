# SpaLORA Night-16C report

## 我现在需要知道的三件事

1. 本轮没有再把 Night-16B 的高分当作统一解码器证据，而是在同一强起点上新增了跨模态边界场（CMBF：把稀疏空间边分为域内支持、共同边界和模态冲突）以及可信 prototype 修复（TPR：只有不稳定、prototype 余量低、且位于边界的点才允许移动）。RNA+protein 与 RNA+chromatin 共用同一代码、公式和参数字段，每个家族只冻结一组数值配置。
2. 主分类是 `FAMILY_FROZEN_METHOD_SIGNAL`。蛋白家族配置由 A1+tonsil s1 选择后，在未参与选择的 D1、tonsil s2、tonsil s3 全部实现 ARI/NMI 双升；染色质家族配置在 P22、MISAR 两个 discovery study 双升。D1 刷新到 **0.366478/0.448158**，但所有增益都很小，且消融并不支持每个子项普遍必要。
3. 新增 SPOTS rep1、P10S1/S2/S3、P5S1/S2/S3 共 7 个真实单元完成 feature-level preprocessing、稀疏图、checkpoint 严格回放、CMBF-TPR 和新进程回放；7/7 通过、标签读取 0、dense N×N 0、下载 0。冻结配置在这 7 个单元都选择不移动任何 spot，所以这里只能说明数据扩展与保守拒绝路径可用，不能说明科学提分。

## 绝对主结果

| family/unit | role | strong start ARI/NMI | family-frozen ARI/NMI | delta vs start | moved | min cluster |
|---|---|---:|---:|---:|---:|---:|
| RNA_PROTEIN/A1 | discovery | 0.276003/0.421740 | 0.276004/0.421690 | +0.000001/-0.000050 | 3 | 114 |
| RNA_PROTEIN/D1 | frozen_transfer | 0.365174/0.444577 | 0.366478/0.448158 | +0.001304/+0.003581 | 5 | 58 |
| RNA_PROTEIN/tonsil_s1 | discovery | 0.236536/0.317118 | 0.236683/0.317365 | +0.000147/+0.000247 | 3 | 478 |
| RNA_PROTEIN/tonsil_s2 | frozen_transfer | 0.258264/0.314324 | 0.258785/0.316546 | +0.000521/+0.002222 | 8 | 445 |
| RNA_PROTEIN/tonsil_s3 | frozen_transfer | 0.350644/0.309771 | 0.351283/0.310249 | +0.000639/+0.000478 | 3 | 294 |
| RNA_CHROMATIN/P22 | discovery | 0.595552/0.717931 | 0.595958/0.718006 | +0.000406/+0.000075 | 10 | 163 |
| RNA_CHROMATIN/MISAR_E15_5_S1 | discovery | 0.541424/0.666798 | 0.541637/0.666949 | +0.000213/+0.000151 | 1 | 125 |

D1 相比 Night-16B 可信 frontier 的增量为 **+0.001304 ARI / +0.003581 NMI**，并保留 58 个 spot 的最小簇；P22 相比 Night-16B 为 **+0.000406/+0.000075**。A1 的 ARI 仅增加 0.000001、NMI 下降 0.000050，不能称双指标胜。以上均为公开 benchmark 的 development/frozen-transfer 结果，不是 pristine blind confirmation。

## 边界诊断、贡献与含义

CMBF 对真实边界的 AUC 为 0.538--0.606。蛋白数据中，起始分区的错误有 89.4%--96.6% 位于真实边界一跳内，说明边界修复对象是合理的；P22/MISAR 只有 46.4%/58.4%，提示染色质家族的剩余误差不主要是局部边界错误，后续更应改善表示或 start generator。

Matched ablation 显示 full 相对同一 strong start 在 D1、tonsil s1/s2/s3、P22、MISAR 有正增量；但 D1/s1 的 boundary/conflict disabled 可与 full 相同，s3/P22/MISAR 的某些 disabled 变体还略好。因此本轮只支持“同一组合机制在家族冻结条件下有可复算的局部增益”，不支持每个组件都普适有效，更不支持把普通 prototype、方向图或 Potts 类平滑单独写成创新。

## 新增真实数据 P0

| unit | family | N | reduced views | graph nnz | spatial agreement | kNN overlap | wall s / peak RSS MiB |
|---|---|---:|---|---:|---:|---:|---:|
| P10S1 | RNA_PROTEIN | 7447 | [7447, 30] + [7447, 30] | 49100 | 0.3813 | 0.002175 | 11.3 / 784.7 |
| P10S2 | RNA_PROTEIN | 41289 | [41289, 30] + [41289, 30] | 273000 | 0.3635 | 0.000354 | 201.6 / 1795.0 |
| P10S3 | RNA_PROTEIN | 5845 | [5845, 30] + [5845, 30] | 39026 | 0.3540 | 0.002310 | 8.1 / 1795.0 |
| P5S1 | RNA_CHROMATIN | 7794 | [7794, 30] + [7794, 30] | 50396 | 0.1799 | 0.001578 | 10.8 / 1795.0 |
| P5S2 | RNA_CHROMATIN | 28545 | [28545, 30] + [28545, 30] | 190320 | 0.2714 | 0.000403 | 94.2 / 1795.0 |
| P5S3 | RNA_CHROMATIN | 9426 | [9426, 30] + [9426, 30] | 63958 | 0.4251 | 0.001209 | 14.0 / 1795.0 |
| SPOTS_SPLEEN_REP1 | RNA_PROTEIN | 2653 | [2653, 30] + [2653, 21] | 16246 | 0.4424 | 0.012288 | 4.7 / 201.8 |


第一次 P5S1 P0 因 label-free centrality 选中了含 singleton 的 fused start 而 fail-closed；失败目录保留。修复是共同 start selector 的 exact-K 最小簇 guard，不改 family config/K；随后 7 个单元整体重跑并通过。

## 对论文意味着什么

这是一条比 Night-16B 更干净的方法证据：贡献相对同一强 start 计算，且 protein family 有跨切片冻结转移。但效应量仍很小、所有输入都依赖历史强表示/authority start，新增无标签单元没有产生移动，chromatin 也没有未参与 HPO 的带标签 transfer unit。因此它适合进入方法候选与消融章节，不足以成为 SOTA、confirmed milestone 或 paper-ready evidence。下一阶段最关键的是闭合更多可信 annotation 的物理单元，并把边界场用于真正的表示学习，而不是扩大 post-processing 网格。

## 导师汇报版

我们把 RNA 与第二模态在每条空间边上的变化显式分成域内支持、共同边界和模态冲突。只有多起点不稳定、两个模态的 prototype 证据都弱、同时边界证据高的点才允许修改。两个模态家族共用同一计算图，每个家族只冻结一组数值配置。蛋白家族配置从 A1 和 tonsil s1 选出后，在 D1、tonsil s2/s3 三个未参与选择的切片都双指标提高，D1 达到 0.3665/0.4482。P22 和 MISAR 也用另一组 family config 双升，但尚缺独立带标签 transfer 单元。七个新增真实单元完整工程路径全部通过，却都没有触发修复，说明算法很保守而不是已经证明新数据有效。消融表明并非每个子模块都普遍必要，所以当前应表述为家族冻结方法信号，而不是论文已经成立。

## 技术审计摘要

- Parent: `80e584ebfc82544f1e37f7eed81942d55878a31e` / `night16b-final-20260824`。
- Formal unique family HPO: 831 rows；全部 materialize/hash 后由独立 evaluator 读公开 discovery annotations。
- Final replay: 7/7 partitions 两次新进程 exact；targeted tests 13/13。
- Producer label reads 0；new-unit label reads 0；dense N×N 0；GPU peak 0 MiB；new downloads 0。
- Final commit/tag、bundle 与 Windows compact hash 在 Git 封口后写入 delivery metadata。
