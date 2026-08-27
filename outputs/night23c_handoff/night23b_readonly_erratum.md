# Night-23B 决策语义只读勘误

日期：2026-08-27  
父级 commit：`8cd67b652df145d6f618dcb33ad01bcff18b6183`  
父级 tag：`night23b-final-20260827`

本勘误不修改 Night-23B final commit、tag、候选 bank、partition、replay、指标或 compact。

Night-23B 的决策程序先按 primary studies 的最差 `min(ARI,NMI)` 选择全局 consumer，再对该单一 consumer 统计预冻结 per-lane gates。这个组合与原任务的“至少两个 primary 可显著逼近 teacher”目标不一致，且会忽略通过 study 数更多的候选。

对锁定 `oracle_all_candidates.csv` 的独立复算为：

- signed scale=1：1/3 primary pass；
- signed scale=2：2/3 primary pass；
- scale=2 P22：ARI/NMI `0.9654644770570312 / 0.9640659105024072`，PASS；
- scale=2 MISAR：`0.7441480211926709 / 0.8060876410865845`，FAIL；
- scale=2 human hippocampus：`0.9435390773632971 / 0.9149151305283488`，PASS。

修正后的只读决策顺序是：先最大化满足冻结 per-lane gate 的 primary pass 数，再最大化 worst recovery、mean recovery，最后优先较低复杂度/scale。它在既有 formal bank 上唯一选择 signed scale=2。

因此：

- Night-23B 的数值、SHA、fresh-process replay 和交付继续有效；
- 原 `SCIENTIFIC_NEGATIVE / RELATION_CONSUMER_NOT_IDENTIFIABLE` 不再作为下游终止依据；
- Night-23B 决策层正确分类为 `IMPLEMENTATION_FAILURE / DECISION_SEMANTICS_INVALID`；
- oracle 证据分类为 `ORACLE_CONSUMER_IDENTIFIABLE_2_OF_3`；
- 该结论仅授权 Night-23C learned Stage B，并不是 learned method 成绩。

