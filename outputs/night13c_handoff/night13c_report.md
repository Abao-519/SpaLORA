# SpaLORA Night-13C endpoint 稳健性与可训练统一核心报告

## 负责人现在需要知道的三件事

1. Night-13B 的 B10 提分并不是训练所得：它是 simple-concat PCA 后的确定性图残差；Night-13C 首先检验这部分增益是否跨 KMeans endpoint 和强表示仍成立。
2. Stage A 证明 B10 只在 P22 的 common endpoint 上稳定，在 A1 共识上略降，加入 C00/F00/N02 强表示后分区不变；Stage B 随后真正训练了三类共用核心，完成 6/6 真实 P0 和 seed 生效审计。
3. 所有可训练路线都被强内部参考支配，零散 tonsil 改善不能抵消 P22/MISAR 退化；本轮终态是科学负结果，B10 只能保留为辅助 head/消融，不能作为论文核心。

## 结果分类

- 终态：`NIGHT13C_B10_ENDPOINT_FRAGILE`
- 分类：`SCIENTIFIC_NEGATIVE`
- 没有冻结 final candidate，也没有进入 D1/tonsil s2/s3 confirmation；这是预注册的提前停止，不是隐藏失败。

## Stage A：B10 到底稳不稳

- A1：30 个 endpoint seeds 的 B10 同时胜 ARI/NMI 比例为 0.50；共识 ΔARI=-0.0003，ΔNMI=-0.0059。
- P22：对应胜率为 1.00；共识 ΔARI=0.1740，ΔNMI=0.1709。
- MISAR 的单侧 P4/P18 图残差明显强于 B10 混合，说明效果依赖具体图残差方向，而不是一个稳定的统一混合机制。
- C00(A1/D1/tonsil)、F00(P22)、N02(P22) 上，B10 residual 的共识分区与 identity 完全一致；它没有把历史强表示推到更高水位。

## Stage B：真正可训练核心

真实 P0：6/6 通过；每行均有两个真实模态输入、有限梯度、80 optimizer steps、严格 checkpoint、fresh-process 数值与分区 round-trip。seed 0/1 在 2/2 个真实数据上产生不同参数和表示。

下面列每个数据上 seed 0 最好的已跑候选；Δ 是相对该数据冻结的最强 common-endpoint 内部参考：

| dataset | candidate_id | ari_mean | nmi_mean | delta_ari_mean | delta_nmi_mean | consensus_ari | consensus_nmi | optimizer_steps | peak_gpu_mib | peak_rss_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tonsil_s1 | X10_MODALITY_DROPOUT_CROSS_RECON | 0.1878 | 0.2844 | 0.0460 | 0.0002 | 0.1936 | 0.2871 | 80.0000 | 43.1196 | 1408.9141 |
| P22 | E10_EDGE_RELIABILITY | 0.3166 | 0.4594 | -0.1037 | -0.1178 | 0.3185 | 0.4612 | 80.0000 | 182.4053 | 2507.0977 |
| MISAR_E15_5_S1 | X10_MODALITY_DROPOUT_CROSS_RECON | 0.1767 | 0.3174 | -0.0255 | -0.0388 | 0.1537 | 0.3049 | 80.0000 | 29.2012 | 1408.9141 |
| A1 | E11_EDGE_RELIABILITY_ANCHORED_R10 | 0.2245 | 0.3579 | -0.0060 | -0.0181 | 0.2313 | 0.3596 | 160.0000 | 76.6807 | 1117.6758 |

三种机制（edge reliability、modality dropout cross-reconstruction、shared/private orthogonal）加一次公开的 E10 邻域均未形成两个家族同时为正的 ΔARI/ΔNMI。训练标签没有进入 loss、gradient 或单次 run 的 checkpoint selection；公开标签只在统一 evaluator 和跨运行 HPO 中使用。

## 对论文意味着什么

Night-13C 排除了两个容易误导的故事：一是把 B10 的确定性后处理当作端到端模型，二是把某个 endpoint 的局部提升当作统一机制成功。当前资产证明工程路径可训练、可复现，但尚未证明训练所得表示优于强内部参考。下一轮若继续，应重新审视训练目标与表示锚点，而不是继续调 B10 阈值或在同一 edge-reliability 邻域追分。论文阶段仍缺真正跨家族稳定的模型增益、外部强基线、消融与冻结后确认。

## 5–8 句导师汇报版

Night-13B 的主要提分来自一个确定性图残差，不是训练模型。Night-13C 用 30 个 KMeans seeds 和共识检验后发现，它在 P22 稳定，但在 A1 共识上不成立。把同一 residual 加到 C00、F00 和 N02 的强表示上，分区基本没有变化。我们随后实现了三个真正可训练、两家族共用的模型核心，6/6 真实 P0、checkpoint reload 和 seed 生效都通过。科学结果却一致偏弱：只有 tonsil s1 出现零散改善，P22 与 MISAR 明显退化。按照预先约定的淘汰规则，没有进入多 seed 晋级和 confirmation。结论是科学负结果，但它明确指出下一步必须改训练目标，而不是继续包装或微调 B10。

## 技术附录

- Stage A wall time：5106.1s；全轮记录 peak GPU：182.4 MiB；peak RSS：2709.4 MiB。
- 历史 raw metadata：7/7 roots 不变，changed=0。
- 新下载 0；dataset-name routing 0；dense N×N 0；force push 0；完整外部方法复现 0。
- Git commit/tag、compact 与 bundle SHA 在封口后写入 delivery verification。
