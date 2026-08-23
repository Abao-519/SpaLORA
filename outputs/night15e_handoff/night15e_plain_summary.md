# Night-15E 通俗摘要

Night-15E 把上一轮“从多个离散模块里逐数据集挑一个”的做法，收拢成一条连续的稀疏聚类能量公式。公开标签参与逐 lane 调参后，九条 lane 的 ARI/NMI 都比 Night-15D 略高；但真正不看 held-out study 标签的增量策略没有形成可迁移收益，所以结果仍是 `LOCAL SIGNAL`，不是自动方法或盲测确认。

最重要的风险是 D1 出现一个仅 1 个 observation 的簇，说明 endpoint 峰值很脆弱。消融也表明 trust 并非普遍有效，P22 某些简化配置甚至略优 full。因此可讲的是“连续可靠性能量组合在公开 benchmark 上有开发信号”，不能说每个零件都已验证或方法新颖性已经成立。

本轮还接入了 GSE213264 Human tonsil：2492 个 RNA/蛋白 spot 的集合相同但原始顺序不同，已按字符串 ID 显式对齐并完成稀疏真实路径与两次新进程回放。K=8 只来自作者公开报告的 RNA cluster 数，没有逐 spot 人工参考，本轮不报 ARI/NMI。
