# Night-16F 冻结公式与归因合约

本轮不修改 Night-16E 的 RNA+chromatin 家族数值配置，只把它的支持边机制拆成质量严格匹配的对照。每组对照使用同一起点、同一 prototype unary、同一稀疏图尺度、同一 Potts 系数、同一继承的 base self-return 和同一 alpha-expansion 端点。

`BIMODAL_SUPPORT` 使用 RNA 与 ATAC 共同产生的 edge support；`UNIFORM_MASS_MATCHED` 只保留相同总平滑容量；`PERMUTED_SUPPORT` 保留权重分布但打乱边位置；RNA-only 与 ATAC-only 分别检验单模态是否足够。每个图尺度都按 `sum(base_edge * factor)` 精确匹配，不用普通算术均值。

SCP2176 melanoma 在公式与代码冻结后才由 evaluator 打开标签。公开 annotation 只机械定义 tumour_1/tumour_2 的 833-cell 协议成员；正式 producer 只读去掉 annotation 的数值 carrier。所有候选先 materialize、保存并哈希，再由独立 evaluator 计算指标。

只有 human hippocampus 和至少一个冻结 melanoma 协议同时显示 bimodal support 超过 direct、uniform、permuted 及两个 single-view，才把净增益归因于跨模态边位置。否则按实际结果归为全局平滑容量、单模态或局部信号。
