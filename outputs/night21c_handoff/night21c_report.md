# Night-21C official spaMGCN math and endpoint junction report

## 我现在需要知道的三件事

1. **问题**：Night-21B 的稀疏目标不是 spaMGCN 官方 dense 数学，因此它的负结果不能回答成熟骨干本身是否有效；历史高分 partition 也不能被误当成同分的可复用 embedding。
2. **实际动作**：本轮固定官方 commit，在四条真实双模态 lane 上恢复 dense 图重构/全对相似度训练，并把 retained、Night-21B sparse port、official-math embedding 放入完全相同的八种 endpoint bank。全部 embedding、checkpoint、候选分区先锁定，标签随后才由 evaluator 和诊断 probe 打开。
3. **论文意义**：方向分类为 **MIXED_REPRESENTATION_ENDPOINT_JUNCTION**。这是一项成熟骨干和聚类出口的归因结论，不是新的自研方法贡献；本轮另有 1 个 metric-specific `SCORE_FRONTIER_ADVANCE`，均按透明 label-assisted profile 单列。

## 绝对分数与 junction 主表

| lane | retained + common KMeans | sparse port + common KMeans | official default700 + common KMeans | 本轮透明 HPO 最佳组合 | 最佳 ARI/NMI | 历史可信 frontier |
|---|---:|---:|---:|---|---:|---:|
| A1_K10 | 0.226780/0.385512 | 0.180175/0.330330 | 0.176666/0.330082 | RETAINED_CARRIER + HISTORICAL_CONCAT_KNN_S02 | 0.245907/0.370423 | 0.276172/0.421937 |
| TONSIL_S1_K4 | 0.137744/0.213706 | 0.103424/0.205584 | 0.090958/0.198313 | RETAINED_CARRIER + HISTORICAL_CONCAT_KNN_S02 | 0.200377/0.275794 | 0.236683/0.317365 |
| P22_K9 | 0.471088/0.600489 | 0.339874/0.472858 | 0.383164/0.532373 | RETAINED_CARRIER + GMM_FULL_N3_S0 | 0.475691/0.617003 | 0.596390/0.718243 |
| PLACENTA_K10 | 0.421949/0.548313 | 0.332160/0.548139 | 0.250106/0.420406 | NIGHT21B_SPARSE_PORT_STABLE700 + HISTORICAL_CONCAT_KNN_S02 | 0.473637/0.606857 | 0.499999/0.631180 |

Official dense objective 相对 Night-21B sparse port 在 1/4 lane 的 common KMeans 上实现 ARI/NMI 双升。完整 endpoint、AMI/FMI、簇大小和空间指标见 `all_locked_endpoint_evaluations.csv`；每个 embedding 的 endpoint gap 与历史 gap 见 `representation_endpoint_decomposition.csv`。

P22 default700 的三个训练 seed 在 common KMeans 下 ARI mean/median/min 为 0.394565/0.383164/0.367904，NMI 为 0.540324/0.532373/0.523338；3/3 均双指标高于 Night-21B sparse port，但都没有超过 retained carrier。标签后置 probe 的最高 balanced accuracy 分别为 A1 0.752、tonsil s1 0.803、P22 0.941、placenta 0.874，明显高于无监督 endpoint 的类别恢复，说明“可分信息存在”和“聚类几何可直接恢复”不是一回事。

具体地，A1 与 tonsil s1 的最好透明组合仍是 retained carrier 加历史稀疏 head；P22 的 official dense 目标稳定优于 sparse port，但 retained carrier 仍更强；placenta 则由 sparse-port embedding 加历史 head 得到本轮最高 ARI。因而单一“官方表示恢复”或单一“换 head 即解决”都与四条 lane 不一致，`MIXED_REPRESENTATION_ENDPOINT_JUNCTION` 是对当前证据最窄的表述。Placenta 的 sparse-port + full GMM 刷新 max-NMI 到 0.641761（ARI 0.469949），它是 endpoint/HPO 的 metric-specific frontier，不是新方法贡献。

## 贡献边界

- `OFFICIAL_SOURCE_SNAPSHOT` 是未修改的固定官方源码；`OFFICIAL_MATH_COMPATIBILITY_RUNNER` 只负责数据/API/标签隔离；`NIGHT21B_SPARSE_PORT` 改了目标，三者没有混称。
- 透明最优 head 使用公开标签做候选锁定后的 benchmark HPO，不是 blind、自动 selector 或新算法。
- 线性、kNN、nearest-centroid probe 是 embedding 锁定后的监督可分性诊断，绝不进入训练、checkpoint 或无监督主表。
- 历史 frontier 仍是 partition/head 层面的 development ceiling；只有本轮锁定 embedding 能被可复算 head 达到的分数才算 endpoint recovery。

## 导师汇报版

我们先把上一轮最关键的混淆拆开了：Night-21B 并没有复现 spaMGCN 的官方 dense 目标，而是资源受限的稀疏替代。本轮固定官方源码，把相同 carrier 分别送入 retained、稀疏 port 和官方 dense backbone，再用同一组聚类出口比较。训练程序完全不读标签，所有表示和候选分区锁定后才评价。官方目标恢复计数为 1/4；P22 三个 seed 稳定恢复 sparse port，但 A1、tonsil 与 placenta 没有恢复，所以最终是 MIXED_REPRESENTATION_ENDPOINT_JUNCTION。监督 probe 显示四条表示都含有较强可分信息，但普通无监督 head 仍无法系统恢复历史分数，下一轮应研究直接可训练且结构约束的 clustering junction，而不是继续给表示堆正则。本轮没有新增论文模块，公开骨干或 label-assisted 最优 head 的高分都不能算成我们的贡献。所有源码、官方快照、checkpoint、候选 bank 与 fresh-process replay 都进入可复算 compact。

## GitHub 与资源

扩容后根盘为 50 GiB 级，清理仅限 conda 可再生包缓存；未删除项目、环境、raw、checkpoint、compact、bundle、Git 或失败证据。旧公钥对应私钥在主机上缺失；已生成隔离的新公钥（见去敏 SSH JSON），尚未在 GitHub 注册，因此没有尝试 push。

## 技术附录

- taskbook SHA-256: `dbd11881dbe13d578317984034d0f218034dc0ab0bff1151dd3e79527fb6614a`
- upstream commit: `77dfe67d4fd80c124722e68a0f71af36d10fa5fa`
- parent commit: `3f94b1296d26016263a9b6b3efc42b827a806ed1`
- shutdown at report build: `false`（最终 compact 和 Windows 验证后才派发）
