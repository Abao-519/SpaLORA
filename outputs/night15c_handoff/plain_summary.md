# Night-15C 通俗摘要

我们没有再训练一个大模型，而是改进“最后怎样把 spot 分成 K 类”。每个 spot 先根据分子表示得到它属于各簇的代价，再让空间邻边按 RNA 和第二模态的一致程度提供有限支持；强边界不会被无差别抹平。这个统一规则在 P22、MISAR 和 tonsil 上都出现了可复算增益；A1/D1 是公开标签 development HPO 从相同数值网格中选择 `steps=0` registered no-op，并不是模型自动识别后拒绝更新。

最关键的是，tonsil s3 的最高分会随 PCA 线程环境变化。我们没有隐藏这个问题：线程敏感值只作为开发峰值，正式稳定值改用不做 PCA 的输入，并通过两次独立进程逐字节回放。当前结论是 `DIRECT_CLUSTER_ENERGY_SIGNAL`，只说明这个 clustering head 值得继续验证，不说明方法已是 SOTA 或论文已经成立。
