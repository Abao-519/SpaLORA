# SpaLORA Night-15B 稳定性锚定原型与分数冲刺报告

## 我现在需要知道的三件事

1. **问题**：本轮先问已有强表示还能否靠统一的 head/consensus 提高绝对分数，再问稳定性锚定的可训练原型残差 SAPR 是否在同一 strongest representation/head 上有独立增益。
2. **实际做了什么**：Windows 对 7 个真实数据资产运行 12,377 行 embedding-level coarse-to-fine head HPO；AutoDL 用同一 SAPR core 完成 2-family P0、12 配置 screen、top-3 多 seed 与冻结后的 D1/tonsil s2/s3 确认；最后把 41 份 checkpoint embedding 下载到 Windows，在与 head HPO 完全相同的 sklearn 环境中重放 123 行三行贡献对照。
3. **对论文的意义**：终态是 `HEAD_ONLY_SCORE_GAIN`。P22 K=18 的 label-free partition consensus 把 ARI 从 0.6122 提到 0.6878，但 SAPR 在 8 个 lane 中 **0/8** 的 mean ARI 超过 exact retained teacher；所以这不是新方法成功，更不是 SOTA 或 paper-ready 证据。

## 结果分类

- 终态：`HEAD_ONLY_SCORE_GAIN`
- 解释：有可复现的 head/partition-ensemble 分数收益，但没有独立的 trainable-core 收益。
- 明确否定：`CROSS_DATASET_METHOD_SIGNAL` 与 `CLUSTER_AWARE_RESIDUAL_LOCAL_SIGNAL` 均不成立。
- 标签角色：公开标签仅用于 known K、跨运行 HPO 与最终评价；没有进入模型输入、无监督 loss、梯度或单次 checkpoint selection。

## 绝对分数主表

| 数据/协议 | 历史 BEST ARI/NMI | Night-15B BEST ARI/NMI | BEST ΔARI/ΔNMI | median / mean / min ARI | 结论 |
|---|---:|---:|---:|---:|---|
| P22 K=9 | 0.5691/0.6851 | 0.5205/0.6523 | -0.0486/-0.0328 | 0.4470/0.4470/0.3736 | 未过 0.60，也未超过旧高位 |
| P22 K=18 author assignment | 0.6122/0.7233 | 0.6878/0.7164 | +0.0756/-0.0070 | 0.6878/0.6878/0.6878 | ARI 过 0.65；NMI 未超过旧高位；不是独立 GT |
| MISAR K=7 | 0.5099/0.6290 | 0.4303/0.5736 | -0.0796/-0.0554 | 0.4303/0.4303/0.4303 | 未过 0.55 |
| MISAR K=12 | 0.4143/0.5329 | 0.3742/0.5906 | -0.0401/+0.0577 | 0.3742/0.3742/0.3742 | 未过 0.50；未接近 0.644 context |

Protein head HPO 的 BEST ARI 相对 Night-13B simple anchor：A1、D1、tonsil s1、tonsil s2 提升，tonsil s3 回撤，即 4/5 ARI lane 为正；NMI 是 3/5 为正。逐行绝对 ARI/NMI、AMI/FMI、Moran、Geary、seed 和资源见 `absolute_metrics_main_table.csv` 与 `head_score_board.csv`。

## SAPR 最小三行贡献对照

Windows exact-endpoint 回放显示：冻结 finalist S22 的 `SAPR_FULL` 在 A1、D1、P22、MISAR K=7/K=12、tonsil s1/s2/s3 共 8 个 lane 中，没有一个 lane 的 mean ARI 超过 `RETAINED_TEACHER`。个别 seed（MISAR K=7）有较高 BEST，但 mean 仍略低于 retained；D1 与 tonsil s2/s3 的冻结确认明确翻转。`SAPR_FULL - SAPR_RESIDUAL_DISABLED` 只在少数 lane 为正，且不足以超过 strongest retained control，因此不能把普通 adapter/prototype sharpening 包装为独立新模块贡献。三行逐 seed 与汇总见 `sapr_local_endpoint_ledger.csv`、`sapr_minimal_contribution_table.csv`。

## 工程与资源

- 真实 P0：RNA+protein A1 与 RNA+ATAC P22 2/2 通过 finite forward/loss/backward。
- checkpoint：P0 2 + finalist 30 + confirmation 9 = 41/41 strict fresh-process 数值回放通过。
- AutoDL：101 次训练调用、13,880 optimizer steps，累计 GPU 训练时间 114.3s，peak GPU 83.6 MiB，peak RSS 947.6 MiB。
- Windows：最终 head HPO 235.0s；SAPR 41-export/123-row CPU 回放 17.2s；含保留的失败尝试，总 endpoint 工作时间下界 50.2 分钟。
- local compute kit 小于 2 GiB，不含 raw fragments 或 dense N×N；GSE213264 因 canonical label/mask/ID 协议未闭合而未伪造 ARI。
- 历史 raw 的 count/size/mtime/fingerprint 前后 5/5 相同。

## 最重要的失败

1. SAPR 的远端 screen 曾显示局部增益，但 Windows 同版本 exact-endpoint 回放未保持，说明 trainable representation 与最终 cluster separability 仍错位。
2. P22 K=9、MISAR K=7/K=12 均未超过 Night-14B/15A 的绝对高位。
3. 冻结确认中 D1 与 tonsil s3 整体下降；tonsil s2 的提升来自 residual-disabled 路径而非 SAPR residual。
4. P22 K=18 的提升来自 label-free partition consensus/head，不是 SAPR，也不能因使用 author 18-state assignment 而称独立外部确认。

## 5–8 句导师汇报版

Night-15B 把计算拆成远端训练与本地 endpoint 两部分，完整保留了 head 搜索、失败和所有 seed。已有表示经过更广的无标签 head/consensus 搜索后，P22 K=18 的 ARI 从 0.612 提高到 0.688，但 NMI 没超过旧高位。P22 K=9 和 MISAR 两个协议都没有突破已有最好结果。我们还实现了统一的 SAPR 小型可训练核心，并完成两家族 P0、三候选多 seed 和冻结后的 D1/tonsil 确认。远端初筛的局部增益在 Windows 同端点回放中没有保持，8 个 lane 的 mean ARI 都未超过 retained teacher。因而本轮只能定为 head-only score gain，不能把 SAPR 写成方法贡献。下一步若继续，应更换真正改变可分性的 representation objective，而不是继续调 prototype residual 或 consensus。当前结果不是 SOTA、不是 confirmed milestone，也不是 paper-ready evidence。

## 技术附录说明

Git commit/tag、bundle、compact index 与 Windows 独立复算在最终 delivery audit 中登记。完整 run ledger、source/license/novelty audit、工程修正、label/routing firewall、local compute kit 与 GPU asset SHA 均随交付提供；compact 不包含 raw、checkpoint、embedding、partition 大数组或 vendor 环境。
