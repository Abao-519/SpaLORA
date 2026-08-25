# Night-17D 方法与选择合约

对每个 Night-16H 结构可行候选，分别在 Night-17C Z01 seeds 0/1/2 的冻结表示上计算 explained variance、Calinski-Harabasz separation、q10 centroid margin 和 relation alignment。四轴先在 lane/seed 内转 percentile 后等权平均；三 seed 均值为 learned evidence，标准差为 learned uncertainty。

若候选参与 UNBIASED relation posterior，relation alignment 使用候选特异的 leave-one-candidate-out posterior；完整 posterior alignment 仅作为敏感性列。PERMUTED control 使用同一分层置换顺序同步变换 candidate contribution。

固定选择分数为 `molecular_rank + topology_rank + learned_evidence - 0.5*uncertainty_rank`。透明 HPO 只枚举冻结的 81 个 rank-weight config。严格 LOSO 在两个训练 study 上机械排序配置，held-out producer 不读取评价，最后独立 evaluator 才合并指标。
