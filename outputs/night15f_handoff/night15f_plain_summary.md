# Night-15F 通俗摘要

本轮把“每个 spot 属于哪个 cluster 的分子代价”和“相邻 spot 分开要付多少空间代价”放进同一个连续能量，再用一次可以移动整批 spot 的 alpha-expansion 求解。9 条公开开发 lane 都比 Night-15E 的 ARI/NMI 略高或明显更高，P22 K18 与 tonsil s3 最突出。

不过，大邻域求解本身只在 5/9 lane 比相同能量的单点更新更好；另外 4 条完全相同。三尺度图混合和 rejected-mass stay cost 的证据更一致，簇大小正则和显式 pairwise 只在部分 lane 有帮助。

这仍是逐 lane 看公开标签做 HPO 的开发上限，不是自动部署、盲测、SOTA 或论文已经成立。AutoDL 按夜间联动要求保持开机，没有派发 shutdown。
