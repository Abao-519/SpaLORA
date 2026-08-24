# Night-16F report

## 我现在需要知道的三件事

1. 本轮要判断 Night-16E 的大幅提分是不是来自“正确的双模态 support 边位置”，而不只是总体平滑变弱、权重被打乱后仍有效，或单一模态已经足够。
2. 我们冻结同一起点、同一 unary、同一 base 能量、同一 self-return 和同一 alpha-expansion，只替换 support 因子；每个图尺度都精确匹配 `sum(base_edge × factor)`。随后把同一冻结路径一次性移到独立的 SCP2176 melanoma tumour-only K=2 协议。
3. 结论分类是 **LOCAL SIGNAL**：人海马开发单元中，双模态 support 为 0.544877/0.557795，双指标超过所有总质量匹配对照；但冻结到独立 melanoma 后，bimodal 0.294092/0.197580 低于 uniform/单模态对照，因而没有独立迁移证据。它不是 confirmed milestone，更不是 SOTA。

## 绝对结果主表

| 数据/协议 | N/eval/K | 输入起点 ARI/NMI | Direct ARI/NMI | 双模态 support ARI/NMI | 最强质量匹配对照 ARI/NMI |
|---|---:|---:|---:|---:|---:|
| P22_K9 | 9196/9196/9 | 0.595958/0.718006 | 0.594837/0.715742 | 0.595791/0.717336 | RNA_ONLY_SUPPORT 0.596377/0.717926 |
| MISAR_K7 | 1949/1949/7 | 0.541637/0.666949 | 0.535306/0.658265 | 0.538320/0.661201 | RNA_ONLY_SUPPORT 0.540680/0.663674 |
| HUMAN_HIPPOCAMPUS_K7 | 2500/2500/7 | 0.165734/0.263190 | 0.167510/0.266756 | 0.544877/0.557795 | ATAC_ONLY_SUPPORT 0.537152/0.553334 |
| MELANOMA_TUMOR_K2 | 833/833/2 | 0.119094/0.056992 | 0.293643/0.195757 | 0.294092/0.197580 | UNIFORM_MASS_MATCHED 0.304385/0.204494 |

完整 4 lane × 47 rows 在 `matched_attribution_table.csv`；每个 primary arm 的 5 个 KMeans robustness starts 给出 min/mean/median/max，另与 authority/medoid 分开，见 `attribution_distribution_summary.csv`。

## Slide-tags melanoma 数据闭合

- SCP2176 公开 annotation 与 spatial endpoint 均为 2535 cells，类别含 pDC=6；小类是数据事实，不用任意 microcluster 阈值删除。
- 指定的两个官方 Figshare H5AD 经官方 MultiGATE reproduce carrier 逐矩阵核对后均为 **833 tumour cells**，不是 2529。它们恰好对齐 tumour_1=561、tumour_2=272，形成 K=2 协议。
- Figshare 导出将 observation/feature IDs 替换成位置编号；只有在 shape、row/column numeric X 全等后，才从官方 reproduce carrier 恢复 biological IDs/features，并再与 SCP ID/coordinates 对齐。
- all-cell K=10 只有 2535 条公开标签/坐标，匿名接口的完整 RNA/ATAC matrices 需要认证，因此状态为 `DATA_INPUT_BLOCKED_AUTH_REQUIRED`，没有伪造或用 tumor matrix 冒充。

## 机制归因

- `UNIFORM_MASS_MATCHED` 保留相同总 Potts 容量；`PERMUTED_SUPPORT` 保留 support 分布但打乱边位置；RNA-only/ATAC-only 保留单视图位置并精确质量匹配。
- 人海马 authority start 下，bimodal 为 0.544877/0.557795；最强质量匹配对照是 ATAC_ONLY_SUPPORT，为 0.537152/0.553334。这支持“边位置有用”的开发期局部信号。
- melanoma authority medoid 下，bimodal 仅略高于 direct，却低于 UNIFORM_MASS_MATCHED 0.304385/0.204494；uniform/permuted/single-view 也不弱。因此该局部信号没有跨研究迁移。
- `RELATION_STAY_OFF_BASE_STAY_ON` 与 bimodal support 相同，说明本轮数值不支持把增益归给新增 relation stay；base self-return 是继承项。

## Score frontier 与正式输出必须分开

Melanoma 的五个 KMeans starts 中，公开标签事后看见的最高行是 `KMEANS_RETAINED_S3__INPUT_START`，ARI/NMI=0.975766/0.942756。它只是 **public benchmark label-assisted BEST**，没有替换正式无标签 medoid。常规无标签诊断中 S3 赢 inertia 与 Calinski–Harabasz，但 silhouette、Davies–Bouldin 和 partition centrality 选择别的 starts；当前没有单一机械准则稳健锁定该峰值。

## 重放、资源和局限

- 工程审计定位到 Night-16F 首轮漏掉 Night-16E final replay 已冻结的 `OMP/MKL/OPENBLAS=1`。旧 human 连续表示仅有约 1e-4–3e-4 RMS 漂移，却令离散最小割从移动 5 点放大到 1779 点；所有首轮 carrier/结果已整体标为 superseded。
- 只把旧图从 float32 恢复为 float64并不能改变旧 DIRECT 分区，排除了“graph dtype 是根因”。修复线程边界后，human DIRECT 分区 SHA 与 Night-16E byte-exact，mean retained/view1/view2 weights 和移动 5 点也恢复；carrier 同时改为不再无谓降精度。
- 修复后两个 fresh processes 对 4/4 lanes、188/188 candidate partitions 的 SHA、cluster sizes 与 config exact。P22/MISAR 的 carrier、图、47/47 分区在修复前后本来就 byte-exact；human/melanoma 被完整重建。
- producer 标签读取 0；评价器在 partitions 锁定后分别读取 4 个公开 reference；dense N×N=0；GPU time=0。
- 人海马最终同时恢复 byte-exact Night-16E parent start 和父级单线程数值边界；DIRECT 分区 SHA=`6c6c334d...`，与父级一致。
- 独立迁移门失败后没有补 selector、换公式或用 melanoma 标签重试。

## 导师汇报版

这轮专门拆解了 Night-16E 人海马的大提分来源。我们把双模态 support 与统一衰减、位置置换、RNA-only、ATAC-only 做成总平滑质量完全匹配的对照。人海马上，bimodal support 确实双指标超过全部对照，说明边级双模态位置有开发期局部证据。独立 melanoma K=2 上，同一冻结算子却没有超过 uniform/单模态对照，所以还不能写成可迁移方法。工程上还发现并修复了 BLAS 线程边界遗漏；修复后 human DIRECT 与 Night-16E byte-exact，旧首轮数值全部作废。Melanoma 五个公开 benchmark starts 中有一个达到 0.975766/0.942756，但它是标签事后 BEST；无标签准则没有一致地唯一选择它。下一步应优先解决跨研究校准和初始化选择，而不是把人海马单点信号升格。

## 技术附录摘要

- preformal freeze commit: `1e33dc8ccfb1b9fc5f13de0b7160c502511abe1e`
- exact replay: 4/4
- SCP2176 K2 ordered reference SHA: `c016143bf46f03c01b566f490dc7154a79861922215d2e065f749883a6a2dfe1`
- shutdown dispatched: `false`；实例按连续协作要求保持在线。
