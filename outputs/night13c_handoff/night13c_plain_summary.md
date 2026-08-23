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
