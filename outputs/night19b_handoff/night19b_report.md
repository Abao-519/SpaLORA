# Night-19B 跨模态空间交替扩散报告

## 我现在需要知道的三件事

1. **问题**：RNA、ATAC 与空间图按有序交替扩散并在冲突处回流 self，能否产生比普通扩散/图平均更可分的空间域？答案是否定的。
2. **实际动作所在层**：本轮实现了真实三稀疏算子表示生成器，不是 selector 或关系蒸馏；所有稀疏乘积立即 top-k，最终用同一谱嵌入和 KMeans endpoint。
3. **论文含义**：P0 工程 4/4 闭合，但 Stage A 独立方法门为 0/3；FULL 在三条主 lane 都明显输给同预算简单控制。因此分类 `SCIENTIFIC_NEGATIVE`，停止 refine、family seeds 与可训练展开。

## 绝对指标与 matched attribution

| lane | best FULL ARI/NMI | gate 坐标强控制 ARI/NMI | gate FULL 差值 | best FULL 最小簇 |
|---|---:|---:|---:|---:|
| P22_K9 | 0.269346/0.343784 | 0.412139/0.540694 | -0.146535/-0.194784 | 376 |
| MISAR_K7 | 0.204343/0.297283 | 0.283144/0.446877 | -0.078801/-0.149594 | 128 |
| HUMAN_HIPPOCAMPUS_K7 | 0.065695/0.123978 | 0.283430/0.404234 | -0.217735/-0.280256 | 157 |
| PLACENTA_K10 | 0.054022/0.128717 | 0.467106/0.635772 | -0.413083/-0.507055 | 60 |

## 次级 score frontier（不归因于 CSAD）

Placenta K10 的 matched control `CONCATENATED_FEATURE_KNN` 在 S01 达到 `0.499999/0.631180`，S02 为 `0.467106/0.635772`，高于 Night-18D matched-control context 约 `0.370597/0.533451`。这是可复算的 **backbone/head control score-frontier**，不是 CSAD_FULL 的方法贡献。Human 的 simple average 最高 ARI 为 `0.373640`、spatial-only 最高 NMI 为 `0.404234`，仍明显低于 Night-16H 高位，故不登记 human frontier。

Night-16H strict LOSO 的历史强结果为 P22 `0.596390/0.718243`、MISAR `0.535306/0.658265`、human `0.596178/0.585490`。它依赖已锁 89-candidate selector，不是本轮 same-endpoint control；因此本轮绝对差距不能全归因于 CSAD。但 FULL 对本轮 exact matched controls 仍是 0/3，足以否定独立信号。

## 机制诊断

四条真实 P0 的 RNA/ATAC/retained shapes、ordered ID、三尺度稀疏图与 source SHA 均重新登记；active core `addcdf944...` 下 P0 4/4 和 Stage-A 4/4 fresh-process partition/eigenvalue replay exact。冲突可靠性连续且非退化，但 median 很低；self-return 使约三至五成传播质量留在本节点。结果表明该组合更像过度局部化，而不是稳定提取共同组织几何。

## 标签、失败与资源

四个 20-candidate artifact 先锁 SHA 并新进程重放，之后独立 evaluator 才打开公开 annotations；producer label read 为 0。最初 global-median/self-tuned 语义不一致、P0 replay padding 和 spatial-anchor 控制流错误都在 formal label evaluation 前修正并保留 superseded ledger。全程未下载数据、未新建环境、未向 `/autodl-fs/data` 写文件，working 远低于 150 MiB。

## 导师汇报版

这轮换了一个真正的新表示层对象，不再修 selector 或关系后验。我们分别构建 RNA、ATAC 和空间的稀疏扩散算子，并测试双向有序传播与冲突质量回流。四条真实路径、稀疏性、谱分解和新进程重放全部闭合。科学结果却很明确：FULL 在 P22、MISAR 和人海马均显著低于简单图平均、空间单轴或其他 matched control。冲突门没有提供独立增益，反而可能让传播过度局部化。因此本轮分类是 `SCIENTIFIC_NEGATIVE`，不追加 refine、多 seed 或神经展开。这个结果也排除了把普通 alternating diffusion 加空间门包装成论文核心的路线。

## 技术附录

- Taskbook SHA-256: `4083713ea11280a9cfeea2882380da4352f0d76687a4cdd3ae795fea2ebdf096`
- Formula contract SHA-256: `a3ce750a4845f4b28e1ed7d607f30c5ecf5c7b6bddf416063ba1f71138ee1b67`
- Active core SHA-256: `addcdf944beda0a225497b3d9da0b82d3ec62ca26fec88bd78f3b00c7871a152`
- Parent commit: `e58da0eb307235553fa8359f75e82b527cb735a3`
- shutdown_dispatched at science seal: `false`
