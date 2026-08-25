# Night-17G report: Cross-modal Signed Boundary Objective

## 我现在需要知道的三件事

1. **问题**：我们检验了真正可训练的统一 RNA+ATAC 深度聚类核心，而不是继续做 selector 或候选共识。核心把空间边分为共同同域吸引、共同边界排斥和模态冲突弃权。
2. **实际动作所在层**：两模态 adapter、低秩逐元素交互、重构/DEC scaffold 和 retained residual 都真实训练；三态 CSBO 直接进入 embedding loss。三条真实 lane 均完成梯度、参数更新、严格 checkpoint reload 和两次新进程精确重放。
3. **论文意义与分类**：结果为 **SCIENTIFIC_NEGATIVE**。full 只在人海马相对 matched backbone 双升，但没有同时胜过 permuted/unsigned 控制；P22 与 MISAR 均下降。预注册门为 0/3，因此不进入多 seed，也不把三态组合写成论文贡献。

## 绝对指标主结论

| 数据 | Night-16H 输入 authority ARI/NMI | 本轮 matched backbone | 本轮 full CSBO | full - backbone | 最强匹配控制 | 门 |
|---|---:|---:|---:|---:|---:|---:|
| P22_K9 | 0.587533/0.708974 | 0.359261/0.557097 | 0.350929/0.545491 | -0.008332/-0.011606 | 0.391699/0.582426 (UNSIGNED_ONLY) | FAIL |
| MISAR_K7 | 0.534624/0.656772 | 0.361367/0.537500 | 0.358794/0.534029 | -0.002573/-0.003471 | 0.361028/0.536720 (PERMUTED_EDGE_STATES) | FAIL |
| HUMAN_HIPPOCAMPUS_K7 | 0.596178/0.585490 | 0.199666/0.245123 | 0.211936/0.251796 | +0.012270/+0.006672 | 0.204486/0.253270 (CONFLICT_AS_POSITIVE) | FAIL |

`INPUT_STRONG_START` 是 Night-16H 已锁定 partition authority；`FROZEN_RETAINED_KMEANS_ENDPOINT` 是本轮在 frozen retained 上采用相同 prototype-initialized KMeans 端点的显示别名。两者的差距同时包含端点变化，不能归因于表示。CSBO 的独立判断只使用 full 与同一配置、同一端点、同一训练预算的 backbone/结构控制。

## 机制归因

边状态本身数值可识别：三条 lane 的冲突质量均约 0.43--0.47，吸引与边界各约 0.26--0.28，单纯形误差小于 5e-8。失败不是全零状态或实现未训练。P22 和 MISAR 中，unsigned 或 boundary-to-abstain 比 full 更好，说明显式共同边界排斥没有带来独立收益；人海马的 full 相对 backbone 有局部增益，但 NMI 被 permuted/unsigned 控制超过，关系位置特异性不足。

Permuted 控制逐通道匹配 base-weighted 质量。训练中 attraction 与 boundary loss 各自再除以有效质量，因此移除了全局通道强度；仍保留并检验的是边位置对应的相对权重。

## 最重要失败与限制

- 三态由输入模态的局部相似秩构造，不使用标签，但“共同低相似”并不可靠等价于真实域边界。
- 可训练 residual 的绝对结果远低于 Night-16H 强 partition；安全 anchor 未能同时保留强起点与获得新边界信息。
- retained representation 和输入 authority 来自历史公开 benchmark 开发链，本轮是 transparent public benchmark development，不是盲测。
- 本轮只跑 seed 0，因为预注册 Stage-A 门失败后明确禁止多 seed。

## 导师汇报版

我们这轮第一次把跨模态三态边直接放进可训练表示，而不是继续修聚类后端。三条真实 RNA+ATAC 数据都完成了真实梯度、参数更新、checkpoint 回放和独立标签评价。三态权重不是数值退化：吸引、边界和冲突都有足够质量。但 full CSBO 只在人海马相对普通 backbone 上涨，且未同时胜过置换和 unsigned 对照；P22、MISAR 都更差。按预注册规则，方法门是 0/3，因此结论是科学负结果，不补 seed、不扩参数网格。这个结果说明“共同低相似边直接作为排斥”过于粗糙，不能作为论文核心。成熟 backbone 零件和正负图已有充分先例，本轮组合也没有获得独立数值支持。下一方向应换信息来源或更强原生表示，而不是继续修同一三态损失。

## 技术审计

- targeted tests: 7/7 PASS；包含 midrank tie、三态闭合、edge-order invariance、base-weighted permutation mass、真实微型 train/reload。
- final producer: 3 lanes × 14 rows；全部 finite、exact K、真实 optimizer steps。
- fresh-process replay: 2 × 3 lanes，全部 checkpoint representation/partition SHA exact。
- label flow: producer 0 label reads；三个 partition bank 锁定后由 evaluator 各读取一次公开 reference。
- preformal ordinal-rank 工件保留为 superseded，不进入科学主表。
- AutoDL 保持开机；`shutdown_dispatched=false`。
