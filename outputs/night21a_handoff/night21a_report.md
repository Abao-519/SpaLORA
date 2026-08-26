# Night-21A AMCF 报告

## 我现在需要知道的三件事

1. **解决的问题**：本轮不是再选择旧候选，而是训练一个直接输出新表示的模型；它把三尺度邻域均值/图高通纹理、节点级模态内/模态间权重和强载体零起步有界残差放在同一条计算路径。
2. **真实结果**：工程路径成立，但组合协同被否定。Full 仅在 placenta 相对 carrier 双升；A1、tonsil s1、P22 都双降，且 4/4 lane 均未同时超过全部匹配原子臂。Stage C 和多 seed 因此没有授权。
3. **论文含义**：BANKSY/SpatialGlue/spaMGCN 等底层组件均有明确先例；剩余的窄联合对象只有在匹配协同成立时才值得写成贡献。本轮结果不支持它，分类为 `NO_COMPOSITIONAL_METHOD_SIGNAL`。

## 绝对指标与归因

| lane | carrier ARI/NMI | full ARI/NMI | Δ vs carrier | strongest atomic evidence | strict synergy |
|---|---:|---:|---:|---|---|
| A1_K10 | 0.236432/0.388764 | 0.208563/0.360388 | -0.027869/-0.028377 | ARI: LOWPASS_CARRIER_CONTROL 0.236265; NMI: LOWPASS_CARRIER_CONTROL 0.392540 | False |
| TONSIL_S1_K4 | 0.152816/0.207225 | 0.138822/0.189195 | -0.013994/-0.018031 | ARI: ANCHOR_WITHOUT_GRADIENT 0.180865; NMI: MEAN_ONLY_ANCHORED 0.253364 | False |
| P22_K9 | 0.470817/0.599877 | 0.375038/0.559182 | -0.095779/-0.040695 | ARI: LOWPASS_CARRIER_CONTROL 0.480940; NMI: LOWPASS_CARRIER_CONTROL 0.610378 | False |
| PLACENTA_K10 | 0.345179/0.519646 | 0.394816/0.533318 | +0.049637/+0.013672 | ARI: ANCHOR_WITHOUT_HIERARCHICAL_FUSION 0.416573; NMI: ANCHOR_WITHOUT_HIERARCHICAL_FUSION 0.528449 | False |

Placenta full 的 0.394816/0.533318 是局部信号，但 `ANCHOR_WITHOUT_HIERARCHICAL_FUSION` 的 ARI 为 0.416573，因此不能把该提升归因于完整分层组合。P22 的固定低通为 0.480940/0.610378，mean-only 为 0.465021/0.600332，也说明简单算子比 full 更稳。所有 Night-21A 结果都未刷新 Post-Night-19C 登记的绝对 frontier。

## 数学与工程性质

- 初始化时 anchored arms 的 `Z_out-Z0` 最大绝对误差为 0；残差投影零初始化，同时后续训练具有非零梯度和参数变化。
- 三尺度运算保持 CSR 稀疏；复杂度为 `O(sum_s nnz(P_s)*(d1+d2) + N*C*h)`，没有 dense N×N。
- 节点置换等价、注册图固定下的坐标旋转/缩放不变语义、exact K、strict checkpoint load 和 fresh-process byte-exact replay通过。
- A1/P22 真实 P0 的参数量分别为 31873 和 34069；formal 28 个 producer artifact 均有独立 replay。

## 贡献边界与停止理由

Full 的失败不是 endpoint 偷换：全部 7 arms 使用同一 KMeans endpoint、同一 seed、同一 60-step budget（非训练 controls 除外）和同一 carrier。P0 的 4-step结果只用于接口验证，已明确 superseded。因为 discovery 已使跨家族协同门不可达，继续 D1/tonsil s2/s3/MISAR/human 的 frozen confirmation 只会消耗算力且无法修复主张，所以 fail-closed 停止。

## 导师汇报版

我们这轮把之前分散的多尺度空间纹理、分层跨模态融合和强载体保护真正写成了一个统一可训练模型，而不是候选选择器。代码在 A1 和 P22 上完成真实 GPU forward/backward、checkpoint 严格加载和 fresh-process 重放，工程对象是成立的。来源审计确认邻域均值、梯度、双层注意力、多阶图卷积和残差都有成熟先例，所以只有它们的联合协同可能成为贡献。正式 discovery 结果却显示，full 在 A1、tonsil s1 和 P22 上都低于强 carrier；只有 placenta 相对 carrier 上升，但又不能胜过最强简化臂。最有价值的正信号其实来自 P22 的固定低通和 mean-only，而不是完整模型。因而没有进入 confirmation 或多 seed，也没有刷新现有绝对 frontier。结论是该组合架构工程完整但科学负，不能作为论文主方法；代码中的稀疏多尺度库、零起步有界残差、可复现 CLI/checkpoint/replay 仍可作为计算机求职作品资产。

## 技术状态

分类：`NO_COMPOSITIONAL_METHOD_SIGNAL`。确认运行：4 discovery lanes × 7 arms；未运行：5 confirmation lanes、多 seed、score-HPO。关机字段在封口构建时为 false，最终远端动作由交付流程另行登记。
