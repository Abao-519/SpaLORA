# SpaLORA Night-6D 独立规划审计与 Night-7A 决策

日期：2026-08-18  
审计性质：本地只读、独立复算；未连接 AutoDL、未运行训练、未改动 Night-6D 结果  
Night-6D 权威终态：`NIGHT6D_D1_P22_BALANCED_CONFIRMED`  
权威提交：`e8a49fb874209b2bd4474691ee5d03ee7639c0c7`

## 1. 结论先行

Night-6D 是当前项目最重要的一次阳性结果：预先锁定的完整 `G04/H05` 在 D1 人类淋巴结和 P22 小鼠脑上都相对同轮 fresh `G00/H00` reference 获得正向、达到材料阈值且通过多重校正的 Q 提升，并且两个数据集的空间保护门均通过。

因此，Night-6D 的确认性结论可以保留，不得被后续探索倒写或淡化。

但机制证据也非常明确：跨数据集收益主要来自 `H05_EQUAL3_AFFINITY_SPECTRAL`；`G04_SP10_F10_EUC_UNION` 不是稳定的独立贡献。若直接把最终论文方法叙述为“G04 图结构带来跨数据集优势”，会与现有数据冲突。

下一步不应立即消耗 GPU 跑正式 benchmark，也不应继续扩大 encoder/loss 搜索。先执行一次无 GPU、零训练的 Night-7A：使用已经保存的 G00/G04 六视图和 H05 affinity，进行有上限的跨图 consensus-head 归并，同时完成现代基线真实源代码和新数据集的可复现性预检。Night-7A 之后只锁一个最终结构，再进入新数据外部验证和公平 benchmark。

## 2. 本地完整性复核

以下权威文件 SHA-256 与交付声明一致：

| 文件 | SHA-256 |
|---|---|
| `night6d_report.md` | `333192ce979a02ce8cbc785828e58fa7b9314906ce653c7d8cae016b92e8a053` |
| `night6d_decision.json` | `555f088913776976c91db02538e06f44b9c1b4f9b560dc75660c90df8e98f313` |
| `d1_p22_per_seed_metrics.csv` | `97beb1adca50f753ccb35aa95a35d3a54a5f6758d3599844eaca2ac6b7cb1a1e` |
| `primary_confirmatory_tests.json` | `45f01923735e96b1ea7e10f382937e19e5290e976ca882a1f84e1fae44664d9d` |
| `secondary_factorial_tests.json` | `8b8c0067464cf3cadae8a6f34314a1d2e1a2790982a22d6fe76acf562d381d15` |
| internal `delivery_index.json` | `337c93cefbeb0e08b83820ded4e3ec7ac94e5248a0af1431849c03b58f16c4a6` |
| `external_delivery_index.json` | `55d6736ad46d143b1fe00c1040e32595fb57be4340ba3eaba9caee27650b9458` |
| `local_post_dispatch_index.json` | `78de25032f233fde3ce23b9162dd39ab6c37431c9d33c32c025ed295f961cd24` |

按索引声明的双根解析规则复核：internal `68/68`、external `5/5`、post-dispatch `3/3` 全部通过。最初若只把所有 `tests/` 路径解释为单一根，会得到错误的 `64/68`；按索引中代码 tests 与 handoff tests 的各自根目录解释后为 `68/68`。这只是路径解析歧义，不是交付缺失。

Git bundle 只读列头显示 Night-6D 分支指向 `e8a49f...`，annotated final tag peel 到同一提交；Git 证据记录 final tag 只创建并普通推送一次，未 force，保护标签保留 Night-6C 父提交。

## 3. 独立逐 seed 复算

从 80 行原始指标表独立检查：

- 80 个 `(dataset, graph, head, seed)` 主键全部唯一；
- `Q=(ARI+NMI)/2` 每行精确成立；
- `boundary_disagreement=1-neighbor_agreement` 每行精确成立；
- 完整 `2^10=1024` 单侧 sign-flip、Holm 和独立 100,000 次 paired bootstrap 与正式结果一致；
- 独立实现与正式实现的 bootstrap 端点只有 Monte-Carlo 采样级差异，不改变任何门槛或结论。

| 数据集 | mean ΔARI | mean ΔNMI | mean ΔQ | ARI/NMI/Q wins | exact p | Holm p | 独立 ΔQ 95% CI | 空间门 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D1 | +0.031699599 | +0.028964883 | +0.030332241 | 10/10, 10/10, 10/10 | 0.0009765625 | 0.001953125 | 约 [0.02476, 0.03682] | PASS |
| P22 | +0.016361491 | +0.046474915 | +0.031418203 | 6/10, 9/10, 8/10 | 0.0087890625 | 0.0087890625 | 约 [0.01239, 0.04914] | PASS |

P22 的 ΔARI 独立 95% CI 约为 `[-0.00532, 0.03725]`，与正式 CI 跨零的限定一致。故 P22 可以说 Q/NMI 和综合门确认，但不能说 ARI 对 seed 稳定。

空间指标均值复算：

- D1：Δneighbor `+0.003383`、ΔMoran `-0.007309`、ΔGeary `+0.011529`、Δboundary `-0.003383`；
- P22：Δneighbor `+0.084777`、ΔMoran `+0.074936`、ΔGeary `-0.079118`、Δboundary `-0.084777`。

两者均通过预注册空间保护门。

## 4. 绝对指标位置

| 数据集 | 配置 | mean ARI | mean NMI | mean Q |
|---|---|---:|---:|---:|
| D1 | G00/H00 | 0.209501 | 0.348691 | 0.279096 |
| D1 | G00/H05 | 0.257631 | 0.383706 | 0.320669 |
| D1 | G04/H00 | 0.221574 | 0.361494 | 0.291534 |
| D1 | G04/H05 | 0.241201 | 0.377656 | 0.309428 |
| P22 | G00/H00 | 0.403912 | 0.543831 | 0.473872 |
| P22 | G00/H05 | 0.446145 | 0.614224 | 0.530184 |
| P22 | G04/H00 | 0.385668 | 0.532479 | 0.459074 |
| P22 | G04/H05 | 0.420274 | 0.590306 | 0.505290 |

这说明两个事实同时成立：

1. 预锁定完整 G04/H05 相对 G00/H00 确实确认成功；
2. 已预注册的 secondary cell `G00/H05` 在 D1 与 P22 的绝对 Q 都高于完整 G04/H05。

第二点不能回头改写 Night-6D primary，但必须影响最终研发方向。

## 5. 机制分解

相对 G00/H00 的 mean ΔQ：

| 因子 | D1 | P22 | 解释 |
|---|---:|---:|---|
| G04-only | +0.012438 | -0.014798 | 跨数据集不稳定 |
| H05-only | +0.041572 | +0.056313 | 两数据集稳定且幅度最大 |
| G04×H05 interaction | -0.023678 | -0.010096 | 两数据集均为负 |

在 Night-6C 的 A1/tonsil development evidence 中，H05-only 的 macro ΔQ 约 `+0.03638`，G04-only 约 `+0.00405`，完整组合约 `+0.04575`；A1 对 G04/H05 的正交互较重要，而 D1/P22 的交互为负。综合四个数据集，最合理的推断是：

- H05 的三个表示 affinity 共识是目前最可信的可泛化组件；
- 图构造选择存在组织/模态依赖；
- 最终结构应考虑 G00/G04 的标签无关 consensus，而不是宣称单个 G04 图普适更优。

## 6. 执行语义审计

Night-6D 的 40 training、40 fresh-process checkpoint/六视图/H00 round-trip、80 transforms 均有唯一主键和完整 SHA；正式 retry 为零，fallback 为零。训练与 transform 固定顺序、同 seed paired initial state、checkpoint state/file 和 metric-to-cluster 交叉引用均一致。

标签窗口审计通过：D1 与 P22 只有在 40 training、40 round-trip、80 transforms 全部锁定后才由一个 evaluator 窗口打开；之后没有返回训练、transform 或聚类。原始 D1 H5AD 的 `anndata.read_h5ad` 访问为零；训练副本 `obs` 为零列；P22 immutable cache 逐项通过。

P0 有三项有记录、未改变科研选择的纠正：

1. shell `nounset` 与 conda activation 的基础设施纠正；
2. D1 RNA 与 ADT 源坐标方向相反，零-obs ADT 副本用精确符号反射对齐 RNA，且全部 pairwise distance 不变；源文件未修改；
3. synthetic spatial-gate test fixture 的预期值写错，开标签前只修正测试数据，使其与既定公式一致；阈值、evaluator 和正式输出未改。

这些都应在最终论文的可复现材料中保留，但不构成 Night-6D 失效理由。

## 7. 与近期方法真实代码的关系

本轮只把代码作为研发证据，不把不同数据/标签协议的论文表格直接横比：

- MultiGATE 的公开实现最终把两个 L2-normalized modality embeddings 做等权平均；这与 H05“先在各 view 建 affinity、再等权融合”的成功方向一致。
- PRESENT 有直接 RNA+ADT / RNA+ATAC CLI、固定训练入口和基于目标 K 的 Leiden 输出，是后续公平 benchmark 的高优先级方法。
- COSMOS 的 pyWNN 源码按 spot 计算 within-vs-cross prediction reliability，再形成局部 modality weights；这支持在 Night-7A 中测试无标签的局部可靠性加权，但不能复制其代码而不遵守许可证。
- SpaMode 明确构建 spatial graph 与 correlation feature graph，且训练代码采用多模态图融合；但其仓库未提供清晰机器可识别许可证，Night-7A 只可独立重实现思想，不复制源码。
- SpaMCA 公开代码包含 dataset-specific epochs、K、mask rate 和 loss weights；可作为 multi-view masked graph 灵感和候选 benchmark，但必须把官方数据集特定 recipe 与统一公平 protocol 分开。
- GROVER 需要第三模态 histology 与无法公开获取的 Omiclip 权重，而且论文表格使用 RNA/ADT 各自派生标签、跨 6–10 个 cluster settings 的均值；其高分不能作为我们当前 D1/P22 数值的直接 SOTA 标尺。
- ARISE 的现有实现含逐轮读取真实标签并保留 best ARI 的路径；正式公平表只能使用固定终点、标签无关适配版，官方标签选择结果必须另表披露。
- 2026 年新增的 MultiSP 与 histology-anchored SpatialEx 也有公开代码，应进入源码与输入可行性审计；SpatialEx 若必须依赖当前数据没有的 histology，只能列为任务不匹配，而不能把第三模态优势与双模态结果直接比较。

因此，目前不能诚实声称 SOTA。Night-6D 证明“相对自己的严格 fresh reference 有可泛化提升”，但是否超过 2025–2026 方法必须通过同输入、同 K、同 evaluator、无标签选优的 benchmark 才能回答。

2026-08-18 复核的主要官方入口：

| 方法/数据 | 官方入口 | 本轮用途 |
|---|---|---|
| SpatialGlue | https://github.com/JinmiaoChenLab/SpatialGlue | 经典必要 baseline |
| Seurat/WNN | https://github.com/satijalab/seurat | 经典必要 baseline |
| COSMOS | https://github.com/Lin-Xu-lab/COSMOS | 局部模态可靠性与现代 baseline |
| SMART | https://github.com/Xubin-s-Lab/SMART-main | 2026 现代 baseline |
| PRESENT | https://github.com/lizhen18THU/PRESENT | 直接 RNA+ADT / RNA+ATAC endpoint |
| MultiGATE | https://github.com/cuhklinlab/MultiGATE | 双层图注意力与 CLIP 对照 |
| SpatialCOC | https://github.com/xjtu-omics/SpatialCOC | 2026 连续空间建模 baseline |
| SpaMode | https://github.com/bridge1924/SpaMode | 多图/多专家源码审计 |
| SpaMCA | https://github.com/wenwenmin/SpaMCA | masked multi-view graph 审计 |
| ARISE | https://github.com/XiangxiangWang-code/ARISE | fixed-endpoint adapter 必要性审计 |
| GROVER | https://github.com/Xubin-s-Lab/GROVER | 第三模态/private weight 可行性 |
| MultiSP | https://github.com/jinworks/MultiSP | 2026 frontier 源码可行性 |
| SpatialEx | https://github.com/KEAML-JLU/SpatialEx | histology-anchored 任务匹配性 |
| Human tonsil | https://zenodo.org/records/12654113 | fresh RNA+protein metadata 预检 |
| GSE205055 / OEP003285 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055 / https://www.biosino.org/node/project/detail/OEP003285 | fresh mouse RNA+ATAC metadata 预检 |

## 8. Night-7A 决策

Night-7A 定位为：`CPU_ONLY_ZERO_TRAINING_FINAL_CONSENSUS_AND_BENCHMARK_PREFLIGHT`。

其科研部分只允许：

1. 复用 Night-6C A1/tonsil 与 Night-6D D1/P22 已锁定的 G00/G04 六视图、H05 affinity 和 G00/H00 reference；
2. 在任何标签读取前，一次性生成并锁定固定 registry 中的跨图 consensus heads；
3. 四数据集全部视为 development，等权选择一个最终结构；
4. 若新结构没有达到预注册的泛化与复杂度优势，保留已确认的 G04/H05，不为“新颖”强行换方法；
5. 任何 Night-7A 新选结构都不能继承 Night-6D 的确认性地位，必须在新数据上重新确认。

其工程预检部分只允许下载/检视源码、锁 commit、建 CPU 环境、做 toy/input-adapter smoke test；不运行正式 benchmark，不打开 fresh dataset labels，不把大型数据下载到 D 盘。

## 9. 下一阶段判定

Night-7A 结束后只允许三种推进路径：

- `KEEP_CONFIRMED_G04_H05_FOR_EXTERNAL_VALIDATION`：无新 consensus 以足够幅度胜出；保留 Night-6D confirmation。
- `LOCK_NEW_CONSENSUS_FOR_FRESH_EXTERNAL_VALIDATION`：新 consensus 通过全部门；D1/P22 降为 development evidence，新数据负责确认。
- `NO_FINAL_STRUCTURE_READY`：artifact/语义/公平 benchmark preflight 存在不可消除缺口；停止并报告。

无论哪种路径，下一次 GPU 轮次都应优先做 fresh annotated dataset + common-protocol modern baselines，而不是继续在 A1/D1/P22 上调参。
