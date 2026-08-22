# Night-13A 数据资产、统一 runner 与一种子开发基线报告

## 我现在需要知道的三件事

1. 本轮解决的是公开开发 benchmark 的数据去重、统一工程入口和绝对指标起点，不是训练新方法。
2. 实际完成了 RNA+protein 与 RNA+ATAC 各一条 feature-level 零步真实路径、严格 checkpoint/reload，以及 6 个有可靠公开标签的数据集上的同一 simple anchor/common endpoint。
3. 结果是 **部分完成**：数据表、2/2 真实路径和一种子 simple baseline 板已闭合，但三个外部强基线的官方实现都违反本轮 sparse/identity-blind/common-endpoint 边界，因此不能诚实地写成完整统一 benchmark board，更不能声称 SOTA。

## 结果分类

`NIGHT13A_PARTIAL_BENCHMARK_BOARD`

## 一种子开发基线主表

以下均为 seed 0、simple standardized concatenation、同一 `COMMON_KMEANS_SEED0` endpoint。K 来自每个 canonical public annotation 的唯一类别数，只读取一次并对方法一致；本轮没有根据分数调 seed、epoch、resolution 或 loss weight。

| 数据集 | K | ARI | NMI | AMI | FMI | Moran's I | 墙钟(s) | GPU(s) | 峰值GPU(MiB) | 峰值RSS(MiB) | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 lymph node | 10 | 0.2212 | 0.3556 | 0.3517 | 0.3665 | 0.4772 | 4.67 | 0.00 | 0.0 | 686.0 | PASS |
| D1 lymph node | 10 | 0.2428 | 0.3630 | 0.3590 | 0.3895 | 0.4758 | 2.92 | 0.00 | 0.0 | 637.6 | PASS |
| tonsil slice 1 | 4 | 0.1418 | 0.2842 | 0.2836 | 0.4513 | 0.6879 | 8.99 | 0.00 | 0.0 | 720.2 | PASS |
| tonsil slice 2 | 4 | 0.1517 | 0.2690 | 0.2683 | 0.4655 | 0.6661 | 9.11 | 0.00 | 0.0 | 862.1 | PASS |
| tonsil slice 3 | 4 | 0.1875 | 0.2476 | 0.2470 | 0.4745 | 0.7355 | 14.49 | 0.00 | 0.0 | 1529.6 | PASS |
| P22 mouse brain | 9 | 0.2754 | 0.4175 | 0.4165 | 0.3936 | 0.5868 | 9.26 | 0.00 | 0.0 | 1272.1 | PASS |

完整的 32-cell 板保存在 `development_baseline_board.csv`：6 个 simple 单元通过；MISAR E15.5 因没有可靠 canonical spatial-domain label 而 unsupported；legacy placenta 的第二模态只闭合到 1662x63 Gene Expression 对象且有 2 个零 library observations，不能冒充原始 ATAC/LSI；所有外部失败/未运行单元均保留。

## 数据与真实工程路径

- canonical registry 共 15 行；Zenodo tonsil slice 1/2/3 是 canonical，GSE263617 tonsil A1/D1 因 count signature 不同而只保留为歧义来源审计对象，未重复计数。
- 本轮只有 1 个新下载：Zenodo `data_imputation.zip`，486,654,827 bytes；SPOTS 四个 processed 文件和 P5S1 均复用已有权威原件。
- SPOTS spleen rep1：原始 `2653x32285 RNA + 2653x21 ADT`，2,653 个 byte-exact barcode+tissue paired spots，fresh-process 数值与 partition round-trip 通过。
- P5S1：`7794x32285 RNA + 83,593,412x5 fragment rows`，按 Night-12A Ensembl79 clean-room gene-score 合约得到 `7794x256 + 7794x256`，fresh-process 数值与 partition round-trip 通过。
- 两条 P0 都是 seed 0、optimizer steps 0、finite reconstruction loss、sparse graph、dense N x N 计数 0；P0 峰值 GPU 68.80 MiB，峰值 RSS 858.06 MiB。

## 外部源码与失败 lane

- SpatialGlue：官方 preprocessing 将邻接矩阵 `toarray()`，并按 dataset key 解析配置；A1/P22 lane fail-closed。
- SMART：官方 MNN 路径使用 `pairwise_distances(X)` 构造 dense N x N，tutorial 另有 dataset-specific 和手工 cluster 编辑；A1/P22 lane fail-closed。
- ARISE：官方路径构造 dense cosine/adjacency，并在训练期间读取真标签计算 ARI/NMI 来选 best；A1/P22 lane fail-closed。
- 这些不是把 wrapper 调通就能修的 API 问题；若改变图、loss 或 label-driven selection，就不再是官方复现。因此三个 baseline 没有伪造通过。SpaMV、SpaMode、SpaBalance、CANDIES 仅做 source-transfer audit，不执行完整训练。

## 公开标签与研究边界

本轮按用户新授权把 A1、D1、canonical tonsil 和 P22 明确作为公开 benchmark 的开发/复现实验；不声称 pristine blind test。没有为 SPOTS 或 GSE308623 P5 制造标签。没有结果驱动 HPO，没有新方法候选，没有 SOTA 结论。失败单元和 4 次失败尝试保留。历史 raw 的 7/7 metadata fingerprints 在审计前后完全一致：True。 targeted tests：6 passed, 2 warnings in 2.76s。

## 对论文意味着什么

Night-13A 给后续性能优先路线提供了可审计的数据版本表、统一输出格式、绝对指标和资源基线。它也暴露了一个现实问题：若坚持 sparse、无 dataset routing、统一 endpoint，当前三个外部官方实现不能直接纳入同一公平 runner。下一阶段可以把 simple 板作为 Night-13B 的起点，但必须先决定是接受每个方法的 native engineering boundary，还是另设“数学语义改变后的重实现”组；二者不能混称官方复现。本轮本身不支持新方法有效、聚类提升或 SOTA 的结论。

## 导师汇报版

我们先把重复和来源歧义的数据资产去重，固定了真正计入的物理切片。RNA+protein 与 RNA+ATAC 两条真实零步路径都完成了 checkpoint 和 fresh-process round-trip。随后用同一个 simple concatenation 加 common KMeans endpoint，在 6 个有可靠公开标签的数据集上给出了 seed 0 的绝对 ARI/NMI 与资源基线。这个表是公开 benchmark 的开发结果，不是盲测。SpatialGlue、SMART 和 ARISE 的官方代码分别涉及 dense N x N、dataset-specific 设置或标签驱动选择，与本轮的公平边界冲突，所以没有伪造复现通过。legacy placenta 和 MISAR E15.5 的缺口也被透明保留。最终状态是部分 benchmark board，足够作为 Night-13B 的工程起点，但还不是新方法证据，也不构成 SOTA。

## 技术附录

- parent: `c2ed63d460521c198cea79b4f64c2b126672e493` / `night12b-final-20260822`
- branch: `revision/q2-night13a-benchmark-expansion-unified-runner-20260822`
- final tag: `night13a-final-20260822`
- engineering corrections: 5；preserved failed attempts: 4
- new downloads: 1 file / 486,654,827 bytes；outside whitelist: 0
- historical raw: 7/7 unchanged；changed roots: 0
- final commit、compact index SHA 和 bundle SHA 由最终 tag/Windows delivery manifest 给出，避免在 commit 内容内制造循环 hash。
