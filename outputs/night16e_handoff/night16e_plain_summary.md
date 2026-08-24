# Night-16E report — Tri-State Relation Energy and independent chromatin transfer

## 我现在需要知道的三件事

1. 本轮把两模态空间边分成 support（两模态都支持域内传播）、boundary（两模态共同提示边界）和 conflict（两模态意见冲突），并分别进入非负平滑、边界排斥 unary、私有模态 unary；被拒绝的邻域质量回到当前状态，避免弱边被强行归一化。
2. 历史七条 lane 上，冻结 family profile 的结果并不统一：protein transfer 有负值；chromatin 的 P22 仅微升、MISAR 为 ARI-only。真正新增证据来自未参与 family HPO 的人海马：固定无标签 start 从 0.165734/0.263190 提到 TSRE full 的 0.515565/0.509679。
3. 同起点、同 base 的归因对照闭合：Night-15F direct 为 0.167510/0.266756，support-only 为 0.544877/0.557795，净增量 +0.377368/+0.291039。这支持 RNA+chromatin 的 family-frozen support-weighted operator signal；但它同时改变边位置和总 Potts 质量，尚未与全局衰减分离，relation stay 也未单独识别。分类因此是 **FAMILY_FROZEN_METHOD_SIGNAL**，并伴随 **SCORE_FRONTIER_ADVANCE**；不是完整三态机制、跨两家族统一成功、SOTA 或论文封口。

## 结果分类

- 主分类：`FAMILY_FROZEN_METHOD_SIGNAL`
- 次分类：`SCORE_FRONTIER_ADVANCE`
- 状态：`NIGHT16E_CHROMATIN_FROZEN_SUPPORT_ENERGY_SIGNAL`
- 不升级：protein family 迁移未成立；boundary/private 独立贡献未成立；只有一个真正独立的新 transfer study。

## 透明逐 lane 分数前沿（balanced profile）

| 数据集 | ARI | NMI | ΔARI vs Night-16B | ΔNMI vs Night-16B | 最小簇 |
|---|---:|---:|---:|---:|---:|
| A1 | 0.276172 | 0.421937 | +0.000169 | +0.000197 | 114 |
| tonsil_s1 | 0.236683 | 0.317365 | +0.000147 | +0.000247 | 478 |
| D1 | 0.367750 | 0.449171 | +0.002576 | +0.004594 | 58 |
| tonsil_s2 | 0.258785 | 0.316546 | +0.000521 | +0.002222 | 445 |
| tonsil_s3 | 0.352788 | 0.311275 | +0.002144 | +0.001504 | 297 |
| P22 | 0.596181 | 0.718063 | +0.000629 | +0.000132 | 165 |
| MISAR_E15_5_S1 | 0.541637 | 0.666949 | +0.000213 | +0.000151 | 125 |

这张表是公开 annotation 驱动的逐 lane benchmark HPO，不进入 family-frozen profile。tonsil s1/s2 与 MISAR balanced 选择为注册 no-op；这不是内容自动 gate。
