# Night-16A 独立复核与 Night-16B 决策

## 我现在需要知道的三件事

1. Night-16A 的新分数是真实可重放的公开 benchmark 开发结果，但无标签自校准器在 9/9 协议上没有追上这些分数；两类证据不能混写。
2. 现有高分的标签只参与已生成候选之间的事后超参数选择和评价，没有进入分子特征、图、能量、聚类拟合、梯度或图割接受规则。因此这些结果可以保留为“公开 benchmark 的 label-assisted HPO”，前提是论文完整披露搜索空间、选择规则、搜索预算和敏感性。
3. Night-16B 不再追求一个目前已证伪的自动 selector，而是把真正有效的候选生成、跨模态可靠性空间能量、可选形态视图和通用结构修复收敛为一个统一 structured decoder，并用一套正式、可审稿的 HPO 协议重跑。

## 独立文件复核

权威交付根：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night16a_delivery_20260824/official_compact`

- compact：64/64；index SHA-256 `cdc0bbf6a1134d7c79ab4086143edce2f459347692c1c9e2ebee773d894e6ded`
- incremental bundle SHA-256：`55a46e6babaaedaaeb08c5ee2b564f7f413c23055a8c7992990e1dce21079df4`
- final commit：`7bcc7696c3eb170e7191d9266691271a2ec225b6`
- final tag：`night16a-final-20260824`
- 普通 push 因 AutoDL 环境缺 GitHub SSH 私钥失败；bundle 已闭合恢复路径，未 force push。

关键文件 SHA-256：

- `night16a_report.md`：`8ba7507bb1a0a8e3452dc0f4e1ea5c02df8cb8db4f2858fa6960f5d355fb2995`
- `night16a_decision.json`：`b90c3314b0ecd8a63e1f566a1064c49a130408836a466f888562225cdaa13984`
- `internal_frontier_registry.json`：`368ed1efbe14e95e0646b62dc207cf9550b5d967c909186d663f1386d3f33a41`
- `night16a_execution_contract.json`：`6727f5710677c98edd204098128772dc65d278e0ead2cc9db0a403b7bf05ea03`
- `internal_development_search_ledger_corrected.csv`：`f5720e7d746157256da8788e8d463120a6379ec20f55dc0c27649faf2f2bf3a2`
- `label_flow_audit.json`：`e1761565d067978185d59ea7921270c0f4c919c5221247b94b5068d21bf368b5`

## 对 Night-16A 的科学判定

Night-16A 有两条必须分开的结论：

1. `SCORE_FRONTIER_ADVANCE`：D1 的可信非微小簇结果达到 ARI/NMI `0.338775/0.435972`；其余多个协议也有小幅刷新。
2. `SCIENTIFIC_NEGATIVE`：由数据统计量自动生成参数、再由整组留出 Ridge selector 选择 partition 的路径，在 9/9 协议都低于已知开发高位。

因此失败的是“自动选择高分候选”的假设，不是现有结构化空间能量，也不是公开 benchmark HPO 的合法性。

## 标签用于调参的边界审查

### 可以保留的做法

- 人工 annotation 用于确定已知 K；
- 候选 partition 生成后，由 annotation 计算 ARI/NMI；
- 在公开 benchmark 上用 ARI 选择超参数、聚类 resolution 或 clustering head；
- 不同数据集在同一模型和同一参数空间中选择不同数值；
- 把这种结果明确称为 development/HPO，而不是盲测或自动部署。

COSMOS 公开比较不同 Leiden resolution 的 ARI 并展示最佳分区；SpatialCOC 也按实际 clustering performance 选择不同 clustering algorithm。这说明“公开标签参与事后 benchmark 优化”并非天然不可发表。

### 会成为审稿雷点的做法

- 标签进入表示学习 loss、gradient、prototype target、图边、能量或单次运行 checkpoint 选择，却仍声称无监督；
- 只展示最优行，不披露搜索空间、预算、失败配置和参数敏感性；
- 对本方法大规模 HPO，却只给 baseline 默认参数，然后宣称公平 superiority；
- 用不同模型主干按数据集名称切换，再包装为一个统一模型；
- 把公开 benchmark 的开发上限称为盲测、自动泛化或严格 SOTA；
- 在论文中隐去真实发生过的 label-assisted HPO。

第一版审稿意见没有直接否决标签调参，但 Reviewer 3 明确追问固定 factor=6 在无标注新数据上如何确定；Reviewer 2 明确质疑 SpaLORA 是否比 baselines 获得更多优化。因此 Night-16B 必须同时给出：公开 benchmark 的 tuned profile、同一搜索预算、参数敏感性，以及不读取当前数据标签的 family-default profile。

## 现有高分能否继续使用

可以继续使用，但用途分层：

### 主开发结果候选

- A1 K10：`0.276003/0.421740`
- D1 K10 非微小簇：`0.338775/0.435972`
- tonsil s1/s2/s3 K4：`0.236536/0.317118`、`0.258264/0.314324`、`0.350644/0.309771`
- P22 K9：`0.593963/0.714517`
- MISAR K7：`0.541424/0.666798`

这些必须在 Night-16B 的统一正式 runner 中重新生成或至少从完整父级依赖精确回放，才能进入论文候选主表。

### 仅作次级敏感性

- P22 K18：作者 18-state assignment，不是独立 ground truth；`0.741061/0.754336` 可作敏感性结果。
- MISAR K12：当前项目仍用 primary K7 annotation 评价 12-cluster partition，不能直接等同 SEPAR 的 K12 协议。
- D1 singleton `0.350729/0.416032`：病理性极小簇，不能作为 headline。

## Night-16B 的模型决策

工作对象暂称 `Unified Reliability-Structured Decoder`，中文为“统一可靠性结构化解码器”。造这个工作名是为了把目前散落在 Night-15F/15G/16A 的有效真实对象合成一个计算图，不是预先宣称原创或最终论文名。

统一计算图为：

1. 从 RNA、第二模态、融合表示、空间坐标和可选 morphology 生成同一套初始化候选；
2. 用 modality prototype margin 和跨模态邻域一致/冲突计算 unary 与 edge reliability；
3. 在稀疏多尺度空间图上执行含 self-return 的连续能量优化；
4. 用同一个、只依赖 N/K/预测簇结构的 split/merge/repair 算子库处理退化簇；
5. 用 alpha-expansion 或经验证更有效的同类稀疏结构优化得到最终 partition。

所有 lane 调用同一 API 和算子超集；缺失 morphology 由 presence mask 归零。允许在公开的共同搜索空间里选择数据集级数值参数和 head 类型，但不在源代码中按数据集名称改换整套模型。

## Night-16B 的成功定义

- `PAPER_COMPATIBLE_TUNED_SCOREBOARD`：统一 producer、HPO ledger、参数表、敏感性和 family-default 均闭合，且至少重现当前可信 frontier。
- `SCORE_FRONTIER_ADVANCE`：至少一个 primary lane 在非退化约束下刷新 ARI；NMI 如实独立报告。
- `UNIFIED_STRUCTURED_DECODER_SIGNAL`：full decoder 在多个 primary lane、至少两个数据 study 上超过相同初始化和相同评价 head，且贡献不是只来自初始 partition。
- `HEAD_ONLY_OR_INITIALIZATION_SIGNAL`：高分来自 start/head，full decoder 无独立增益。
- `IMPLEMENTATION_FAILURE` / `INFRASTRUCTURE_FAILURE`：真实路径或资源未闭合。

任何一项都不自动等于 SOTA 或 paper-ready。Night-16B 的目标是形成一张审稿时能完整解释的高分开发表和一个诚实的统一方法对象。
