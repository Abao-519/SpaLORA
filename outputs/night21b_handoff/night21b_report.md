# Night-21B final report

## 我现在需要知道的三件事

1. **想解决什么。** Night-21A 的低起点不是历史最高分分区本身变差，而是“最高分 partition”通常经过逐 lane 标签辅助 head/HPO，不能直接当作一个可重放的 pre-clustering embedding。`frontier_carrier_bridge.csv` 显示，八条 lane 的 retained carrier 在统一 KMeans 下均明显低于历史分数前沿。
2. **实际做了什么。** 我完整审阅 spaMGCN 的 model/train/utils/config/notebook，固定官方 commit 和 MIT License；由于官方训练路径含 dense N×N 和 notebook 内标签监控，本轮实现的是明确标注的稀疏 source-faithful port。唯一自研层 MSRD 在训练表示时，对 carrier/双模态共同支持的空间邻边做吸引，对共同低支持的空间边界做 margin 排斥，其余边 abstain。
3. **结果和论文意义。** A1/P22/tonsil s1/placenta 的 700-step 匹配板中，FULL 只在 placenta 1/4 lane 严格双胜 T0、B0 和三个原子臂；一次正关系主导修订仍只有 1/4。预注册的两-lane 门失败，因此未开展 transfer 或多 seed，终态是 **NO_RELATIONAL_METHOD_SIGNAL**。公开骨干和我方模块都没有刷新历史前沿，不能包装成新方法成功。

## 绝对结果与匹配贡献

| lane | T0 ARI/NMI | B0 ARI/NMI | FULL ARI/NMI | FULL-T0 | FULL-B0 | 严格胜全部对照 |
|---|---:|---:|---:|---:|---:|---|
| A1_K10 | 0.226780/0.385512 | 0.180175/0.330330 | 0.239712/0.384709 | +0.012932/-0.000803 | +0.059537/+0.054378 | FAIL |
| TONSIL_S1_K4 | 0.137744/0.213706 | 0.103424/0.205584 | 0.111786/0.225677 | -0.025958/+0.011970 | +0.008362/+0.020093 | FAIL |
| P22_K9 | 0.471088/0.600489 | 0.339874/0.472858 | 0.268551/0.427002 | -0.202537/-0.173487 | -0.071323/-0.045856 | FAIL |
| PLACENTA_K10 | 0.421949/0.548313 | 0.332160/0.548139 | 0.472844/0.591089 | +0.050896/+0.042777 | +0.140684/+0.042951 | PASS |

完整 ARI/NMI/AMI/FMI、categorical Moran/Geary、cluster sizes、资源与全部 40 个 post-lock candidates 见 CSV。A1 V1 FULL 相对 B0 明显提高，但 NMI 比 T0 低 0.000803；P22 FULL 明显低于 T0/B0；tonsil s1 被 T0 或正关系原子臂解释；placenta 是唯一严格独立局部信号。

## Backbone、模块与 score frontier 分离

- `BACKBONE`: B0 是公开设计的 clean-room 稀疏端口，不是官方数值复现；它在四条 discovery lane 都未超过 T0。
- `OUR_MODULE`: 只看相同 carrier、endpoint、seed、700 steps 的 B0/B1/B2/B3/FULL；1/4 严格通过，不授权家族冻结。
- `SCORE_FRONTIER`: Post-lock HPO 的 best rows 单独列在 `score_frontier_board.csv`；0/4 刷新历史可信前沿。

## 训练与复现

A1、P22 真实 P0 覆盖 feature-level inputs、稀疏图、forward/loss/backward、optimizer update、strict checkpoint reload、表示与 KMeans partition。随后扩展 tonsil s1 与 placenta。主循环固定 700 steps，参数实际变化；所有 24 个稳定主板 artifacts 在独立 Python process 中表示数值重放通过、partition exact 24/24，最大 absolute representation deviation 见 replay 表。没有 dense N×N 分配。GPU 峰值未单独 profiler 采样，因此不补造数字；wall/RSS 和参数量在资源表中。

## 新颖性与来源边界

spaMGCN 的 AE、多阶图传播、融合均是公开先例；关系蒸馏、度量学习、正负图损失亦有广泛先例。只把“强 carrier 支持邻接 + 共同低支持空间边界”作为窄模块接受匹配证伪。本轮该对象未建立跨 lane 独立贡献，故不进入原创/论文主张。

## 数据扩展短名单

优先级最高的是 GSE205055 mouse embryo RNA+epigenome（先闭合 annotation/K/mask）和 Stereo-CITE mouse thymus RNA+protein（先闭合物理切片与标签）。GSE213264 目前只有作者计算型 K8/K7/GC biology，不能冒充专家全域 ground truth；SPOTS mouse spleen 尚无权威全域标签。Melanoma K2 仅保留二元压力测试。详见 `dataset_expansion_shortlist.csv`。

## 局限

- source-faithful sparse port 改写了官方 dense objective，不能把当前 B0 分数称作 spaMGCN 官方复现分数。
- retained carrier 与历史最高 partition 的 head/HPO 依赖分离后，统一 common endpoint 显著掉分；关系正则无法弥合该差距。
- 发现门失败后没有运行 family transfer、3 training seeds 或 endpoint seed 分布；这不是缺失的正证据，而是预注册停止。
- 只有 placenta 单 lane signal，不足以支撑关系模块或跨家族结论。

## 导师汇报版

Night-21B 先厘清了一个关键误区：过去最高分是分区和 head 的结果，不等于背后还有一个同样强、可复用的 embedding。我们基于 spaMGCN 的公开结构做了稀疏 source-faithful port，并只增加一个保护 carrier 邻域和边界关系的训练正则。真实 A1、P22、tonsil s1、placenta 都完成了 700-step 同预算六臂对照。模块只在 placenta 同时超过 carrier、backbone 和所有原子臂；A1 是混合信号，P22 和 tonsil 未成立。一次有机制理由的正关系主导修订仍未达到两条 lane 门，因此严格停止 transfer 和多 seed。结论是工程路线闭合、但关系模块没有稳定科学信号，且没有任何 score frontier 刷新。这个结果支持下一步先解决强数值 carrier/endpoint 的可迁移性，而不是继续叠关系损失。

## 技术审计

- Taskbook SHA-256: `593707fb8836220842e73cc13e114ff5ef42d117f8a9299cbe60d4f709be7aa0`
- spaMGCN fixed commit: `77dfe67d4fd80c124722e68a0f71af36d10fa5fa`, MIT.
- Candidate rows: 40; stable matched rows: 24; exact fresh-process replays: 24/24.
- Producer label reads: 0; evaluator label access happens after artifact/hash lock.
- Disk gate prevented clone/install/download; no writes to `/autodl-fs/data`.
- Shutdown is intentionally deferred until Windows compact verification; science seal itself records `shutdown_dispatched=false`.
