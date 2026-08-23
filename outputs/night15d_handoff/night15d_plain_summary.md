Night-15D 没有重训大模型，而是修正了 Night-15C 聚类 head 的可靠性语义：弱边不再被强行归一化，被拒绝的邻边质量可回到自身，同时允许两个模态按局部 prototype margin 连续贡献，并加入稀疏多尺度图特征。

在公开标签参与逐数据集 HPO 的开发口径下，9/9 lane 相对 Night-15C 稳定线都实现 ARI 和 NMI 同时上升，且两个最终 fresh-process replay 的分区 SHA 和指标完全一致。这是跨 RNA+ATAC 与 RNA+protein 的开发里程碑，但不是盲测、SOTA、CONFIRMED_MILESTONE 或 paper-ready。每条 lane 从同一个组件超集选择离散模块与数值参数，选择由公开标签 HPO 完成，不能称为模型自动 gate。
