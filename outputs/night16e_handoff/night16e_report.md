# Night-16E report — Tri-State Relation Energy and independent chromatin transfer

## 我现在需要知道的三件事

1. 本轮把两模态空间边分成 support（两模态都支持域内传播）、boundary（两模态共同提示边界）和 conflict（两模态意见冲突），并分别进入非负平滑、边界排斥 unary、私有模态 unary；被拒绝的邻域质量回到当前状态，避免弱边被强行归一化。
2. 历史七条 lane 上，冻结 family profile 的结果并不统一：protein transfer 有负值；chromatin 的 P22 仅微升、MISAR 为 ARI-only。真正新增证据来自未参与 family HPO 的人海马：固定无标签 start 从 0.165734/0.263190 提到 TSRE full 的 0.515565/0.509679。
3. 机制对照把结论进一步收窄：support modulation-only 达到 0.544877/0.557795，高于 full；boundary/private off 与 full 相同。分类因此是 **FAMILY_FROZEN_METHOD_SIGNAL（RNA+chromatin 的 support/质量保持局部信号）**，并伴随 **SCORE_FRONTIER_ADVANCE**，不是完整三态机制、跨两家族统一成功、SOTA 或论文封口。

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

## Family-frozen 主方法板

RNA+protein 的 profile 用 A1+tonsil s1 冻结。A1 仅 +0.000282/+0.000140；D1 为 -0.000365/-0.000576，tonsil s2 为 -0.001897/-0.003353，s1/s3 为 exact no-op。因此 protein family 不支持方法成功。

RNA+chromatin 在人海马揭盲前修订为 P22+MISAR discovery，因为候选 base `C_GEO` 已含两者历史 HPO 信息。冻结 full 在 P22 为 +0.000223/+0.000058，MISAR 为 +0.000296/-0.001082；它随后不读取人海马标签，从固定无标签 producer start 得到 +0.349831/+0.246490。这只证明 operator/profile 在一个独立研究上的迁移；历史 start 已逐 lane HPO，不能把旧 lane 写成完整 blind pipeline transfer。

## 人海马真实路径与机制拆分

- 输入：RNA 2500×7666，ATAC 2500×28270；两个 H5AD 的 2500 spot IDs 集合和顺序闭合，坐标完全一致。
- 无标签 producer：HVG/稀疏 SVD → 三尺度稀疏图（nnz 10200/20212/48770）→ partition-consensus ARI medoid start → 冻结 chromatin profile → 保存/重载。
- 独立 evaluator：official result carrier 的 `true_label` 非缺失 2500/2500，K=7，类别计数与 ordered-label SHA 均锁定后再算指标。
- matched：input 0.165734/0.263190；Night-15F direct 0.167510/0.266756；full 0.515565/0.509679；support-only 0.544877/0.557795；stay-off 0.167510/0.266756。
- 解释：主要信号来自 residual-base Potts 上的 support 调制与 rejected-mass self-return；不是 boundary/private unary。`support_mix<1` 的 full 必须准确称为 base Potts + tri-state modulation，而非“只有 support 才平滑”。

## 次级协议与数据扩展

- P22 K18：label-assisted sensitivity balanced 为 0.737787/0.755348；相对本轮 input +0.009480/+0.005233，刷新项目 NMI context，但 ARI 仍低于 Night-15F 0.739685。
- MISAR K12：可用 carrier 只有 K7 reference；本轮仅能作为 K12 endpoint-against-K7 sensitivity，不能冒充独立 K12 annotation，balanced 为 no-op。
- Zenodo 268.2 MB 小包 MD5 闭合，但六个 accession 属于 GSE205055，而非 MISAR E11/E13/E18；本轮不猜 stage 或 reference，MISAR 多阶段 P0 如实停在 provenance insufficient。
- Slide-tags melanoma 未下载；原始 K=2 compartment 若要评价，下一轮必须从 SCP2176 对齐 tumor-cell mask 与原研究字段，不能把 MultiGATE notebook 的 Louvain/WNN 当 ground truth。

## 失败、限制与下一步

1. boundary/private relation carrier 继承 base-edge suppression，可能使边界/冲突边在进入 unary 前已被削弱；正式 screen 开始后未据分数改公式，登记为下一 revision 的预先机制修订。
2. 人海马 support-only 优于 full，说明三态全组件不是当前主故事。下一轮应先独立重写 relation carrier，再在第二个独立 chromatin study 上冻结确认。
3. MultiGATE 论文报告人海马 ARI 0.60；本轮 full/support-only 均未达到该背景线，且 protocol/model 不同，只作 context。
4. 所有正式 producer 均为 sparse，dense N×N=0；GPU 训练=0；两次 fresh-process 重放 family 7/7、frontier 7/7、人海马 8/8 exact。

## 导师汇报版

我们把直接聚类能量改造成了有明确角色的跨模态关系场，而不是继续训练一个浅层残差网络。历史数据上的 family-frozen 增量很小，protein transfer 还出现负值，所以不能说两类模态都成功。关键的新证据来自独立人海马：不看标签生成起点并套用冻结 chromatin profile 后，ARI 从 0.166 提到 0.516。更重要的是，预注册消融显示 support 调制单独达到 0.545，boundary 和 conflict-private 项没有增加分数。这个结果支持 RNA+ATAC 的“可靠域内传播 + 拒绝质量保持”方向，但不支持完整三态故事。逐数据集公开 HPO 还刷新了 D1、tonsil s3、P22 等开发分数，不过它们只能作为 score frontier。下一步需要在第二个独立 chromatin study 上冻结复验，并重写 boundary/private 的 carrier，才可能升级为更稳的论文方法证据。

## 技术附录摘要

- final exact replay：family 7/7×2；frontier 7/7×2；human 8/8×2。
- targeted tests：10/10。
- historical raw file-content modifications：0。
- shutdown：未派发；按夜间联动要求保持 AutoDL 在线。
