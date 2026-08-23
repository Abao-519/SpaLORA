# SpaLORA Night-14B：RNA+ATAC 分数加速与 edge-state 审计

## 我现在需要知道的三件事

1. 本轮要解决的是：在不按数据集切换整套模型的前提下，P22 与 MISAR 的真实 RNA+ATAC 表示还能否通过更合理的图滤波、模态融合和聚类 head 显著提分。
2. 实际完成了 56 个冻结滤波配置、8 个真实可训练 edge-state 配置、3171 行统一 head 搜索、1173 行 MISAR K=12 定向搜索，以及冻结后的 36 行多 seed 正式重放。标签只用于公开 benchmark 的跨运行 HPO 和评价，没有进入无监督 loss、gradient 或单次 checkpoint 选择。
3. 结果应归类为 **`BACKBONE_OR_HEAD_SIGNAL`**：P22 单次达到 `0.5683/0.6845`，MISAR K=7 达到 `0.5099/0.6290`，但优势来自 bilateral/low-pass preprocessing 与聚类 head；TSPR 没有独立增量，多 seed 中位数也明显低于最佳值，所以不是 SOTA、confirmed milestone 或 paper-ready evidence。

## 结果分类

- 终态：`BACKBONE_OR_HEAD_SIGNAL`
- 单次分数门：已满足 P22 K=9 的 `.55` ARI 冲刺目标，也满足 MISAR K=7 的 `.50` ARI 阶段目标。
- 稳定性边界：P22 主配置 9 行中位数 `0.4959/0.6348`；MISAR K=7 中位数 `0.4004/0.5657`。这支持“值得继续验证的开发峰值”，不支持“稳定多 seed 突破”。
- MISAR K=12：最佳 `0.4143/0.5329`，相对公开 ARI `.644` 仍差 `0.2297`；没有伪装成同协议胜利。
- P22 K=18：准确的 18-class annotation artifact 未闭合，本轮只保留协议高水位登记，没有用 K=9 标签替代。

## 绝对指标主表

| lane | BEST_RUN ARI/NMI | median ARI/NMI | mean ARI/NMI | 目标 | BEST gap |
|---|---:|---:|---:|---:|---:|
| P22 K=9 max-ARI | 0.5683/0.6845 | 0.4959/0.6348 | 0.5075/0.6345 | .5063/.6562 | +0.0620/+0.0283 |
| P22 K=9 high-NMI | 0.5657/0.6983 | 0.4591/0.6306 | 0.4808/0.6425 | .5063/.6562 | +0.0594/+0.0421 |
| MISAR K=7 | 0.5099/0.6290 | 0.4004/0.5657 | 0.4097/0.5551 | .3137/.4924 | +0.1962/+0.1366 |
| MISAR K=12 | 0.4143/0.5329 | 0.2766/0.4807 | 0.2933/0.4821 | ARI .644 | -0.2297/NA |

完整 AMI、FMI、homogeneity、V-measure、Moran's I、Geary's C、seed 与 hash 在 `formal_absolute_metrics.csv`。上表每条均为 3 个 backbone seeds × 3 个 endpoint seeds；没有删除坏 seed。

## 提分来自哪一层

- P22 最佳链：C15 真实 checkpoint 表示 → 双模态内容调制的 bilateral 空间算子 → fused PCA32 + 透明坐标权重 → KMeans → 稀疏空间 refinement。它把项目 K=9 高水位从 `.5063/.6562` 推到单次 `0.5683/0.6845`。
- P22 high-NMI 链只将 KMeans 换成 diagonal GMM 与匹配 refinement，单次为 `0.5657/0.6983`。
- MISAR K=7 最佳链：三步低通 → equal3 PCA32 + 二次坐标基 → diagonal GMM → 稀疏 refinement，单次为 `0.5099/0.6290`。
- 8 个可训练配置均有 300–500 optimizer steps、非零有限梯度、参数变化和 fresh-process round-trip；但 fixed-low 优于 support/TSPR，full-core finetune 还略退化，因此不能把提分归因于新 edge-state 模块。

## 关键消融与失败

P22 的 full-train fixed/support/TSPR/TSPR-finetune ARI 依次为 `0.5162`、`0.5134`、`0.5151`、`0.5103`。MISAR K=7 对应为 `0.4252`、`0.4247`、`0.4247`、`0.4219`。因此 TSPR 没有可辨认的独立增量。

保留了两类工程失败：过度 BLAS 并行导致的 stage1 全局中止，以及 CUDA 惰性初始化顺序导致的 8/8 pre-forward 失败。两者修复后受影响 lane 均整体重跑；失败目录未删除。长时间 R/mclust 输出曾使 SSH transport reset，但远端进程继续完成，未局部补跑。

## 资源与完整性

- full-train peak GPU：`994.1 MiB`；peak RSS：`2029.8 MiB`。
- full-train 8/8 通过；训练 fresh-process 8/8 通过；formal partition fresh-process 36/36 byte-exact。
- 搜索评价行：`4701`；训练标签读取 0；labels in loss/gradient/checkpoint selection 0；dataset-name backbone routing 0；dense N×N 0；新下载 0。
- 历史 raw metadata：7/7 roots 匹配，changed roots=0。

## 对论文意味着什么

本轮把 RNA+ATAC 的近期主线从“继续微调 TCF/TSPR sigmoid”转成“成熟表示 + 可解释的内容调制空间 preprocessing + 稳健 head”。P22 的单 seed 已超过项目 native 高水位并越过 `.55` ARI，MISAR K=7 也越过 `.50`，说明这条工程主线值得在 Night-15 做真正的多 seed 稳定化。反过来，MISAR K=12 与 `.644` 仍有大缺口，且正式中位数回落明显；因此还不能写成统一模型已经稳定领先。下一步应把有效 bilateral/low-pass 与 head 选择收敛成一个固定、可复现的 RNA+ATAC pipeline，并在 exact P22 K=18 annotation 与更强外部基线上验证。

## 导师汇报版

1. Night-14B 没有继续包装 TCF，而是先检查提分究竟来自 edge module 还是表示与聚类头。
2. 新的可训练 TSPR 真实跑了 300–500 步，但相对 fixed-low 没有独立收益，full-core 微调还略退化。
3. 真正有效的是内容调制的 bilateral 图滤波、低通、多视图 PCA、GMM/KMeans 与稀疏空间 refinement 的组合。
4. P22 K=9 单次达到 `0.5683/0.6845`，超过 `.5063/.6562`，并跨过 `.55` ARI 目标。
5. MISAR K=7 单次达到 `0.5099/0.6290`，但 K=12 只有 `0.4143/0.5329`，没有追平 `.644`。
6. 多 seed 中位数回落明显，所以结论是 head/preprocessing 的开发信号，不是稳定里程碑、更不是 SOTA。
7. 下一轮应固定有效 pipeline、降低 endpoint seed 敏感性，并闭合 P22 K=18 的准确 annotation 协议。

## 技术附录

Git commit/tag、compact index 与 bundle SHA 在交付后的 `git_and_delivery_audit.json`。本报告所用冻结配置在 `frozen_formal_configs.json`，全部结果在 root-relative index 中可独立复算。
