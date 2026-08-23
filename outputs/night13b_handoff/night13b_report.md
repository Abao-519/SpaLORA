# Night-13B 统一模型性能冲刺报告

## 负责人现在需要知道的三件事

1. 本轮问的不是“能否超过简单拼接”，而是同一个统一模型核心能否同时超过 RNA+protein 的 C00/G04 与 RNA+ATAC 的 F00/N02 强内部参考。
2. 实际工作位于三层：先修 Night-13A 数据与评价语义，再把历史工件桥接到同一 observations/known-K/common evaluator，最后用公开标签做跨运行配置选择、用无标签损失保持单次运行训练语义。
3. 结果是 **LOCAL SIGNAL**：A1 与 P22 提升清楚，D1 小幅保持，tonsil 基本不变，但冻结后的 MISAR confirmation 明显回落；因此不能归类 CONFIRMED MILESTONE，更不能宣称 SOTA 或论文已成立。

## 结果分类

- 终态：`NIGHT13B_UNIFIED_MODEL_LOCAL_SIGNAL`
- 分类：`LOCAL SIGNAL`
- 候选：`B10_T66_S40_M70`
- 统一性：RNA+protein 与 RNA+ATAC 使用同一个 core、融合公式、图语义和全局参数；family 只影响正常输入预处理。
- 标签边界：公开标签进入 evaluator 和跨运行 candidate/config 选择；不进入无监督 loss、gradient 或单次 run checkpoint selection。本轮不是 pristine blind evaluation。

## 绝对指标主表（五个登记 seed；固定 common endpoint）

| 数据集 | 候选 | total obs | eval obs | K | seeds | ARI | NMI | ΔARI | ΔNMI | 双指标胜 | wall(s) | GPU forward(s) | peak GPU MiB | peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | B10 | 3484 | 3484 | 10 | 5 | 0.2693 | 0.3817 | +0.0388 | +0.0056 | 5/5 | 4.11 | 0.1120 | 8.7 | 886.6 |
| tonsil_s1 | B10 | 4326 | 4326 | 4 | 5 | 0.1418 | 0.2842 | +0.0000 | +0.0000 | 0/5 | 2.54 | 0.0264 | 10.9 | 1728.2 |
| P22 | B10 | 9196 | 9196 | 9 | 5 | 0.4619 | 0.6042 | +0.0416 | +0.0270 | 5/5 | 4.91 | 0.0369 | 24.4 | 2792.2 |
| D1 | B10 | 3359 | 3359 | 10 | 5 | 0.2508 | 0.3643 | +0.0081 | +0.0013 | 5/5 | 3.40 | 0.0207 | 8.4 | 1078.9 |
| tonsil_s2 | B10 | 4519 | 4518 | 4 | 5 | 0.1491 | 0.2679 | -0.0327 | +0.0059 | 0/5 | 2.59 | 0.0283 | 11.7 | 1867.6 |
| tonsil_s3 | B10 | 4521 | 4460 | 4 | 5 | 0.1969 | 0.2498 | +0.0000 | +0.0000 | 0/5 | 2.64 | 0.0240 | 11.6 | 2792.2 |
| MISAR_E15_5_S1 | B10 | 1949 | 1949 | 7 | 5 | 0.1592 | 0.2966 | -0.0430 | -0.0596 | 0/5 | 1.90 | 0.0157 | 5.2 | 2792.2 |

完整逐 seed 表见 `absolute_metrics.csv`，包含 AMI、FMI、homogeneity、V-measure、Moran's I、Geary's C、embedding/partition SHA。候选本身是确定性的单参数连续稀疏图残差；登记 seed 0–4 全部保留，固定 evaluator 下每个数据集五行数值一致，不是挑最好 seed。

## 模型与搜索

公式为 `g=sigmoid(40*(mean(1-cos(z,P4z))-0.66)); beta=0.70*g; z_out=(1-beta)z+beta*((1-g)P4z+g*P18z)`。A1 的 beta 为 0.0692、tonsil s1 近似 0、P22 为 0.6518，因此它是内容自适应连续残差，而不是 dataset-name 分流。15 个预登记候选在 A1、tonsil s1、P22 做 seed-0 discovery；公开标签用于跨运行排序，并已明确披露。冻结 B10 后，7 个数据集 × 5 seeds 全量执行；confirmation 后没有返回改公式。

## 对论文的意义

这说明“同一 core 按表示本身的图不一致程度调节残差”值得继续，但当前证据只支持局部机制信号。P22 已超过同口径 N02 common bridge，A1 也超过 C00；tonsil s1 与 s3 持平，s2 相对最强 RNA-only 参考呈 ARI 下降、NMI 小升的混合结果；MISAR 则双指标回落。因而该机制尚未形成跨 RNA+ATAC 数据的稳定优势。下一轮应把 MISAR 失败作为设计约束，先做消融和失败分析，再决定是否新开 revision；不能在本轮结果上追加公式并覆盖终态。论文前仍缺外部强基线公平复现、消融、生物解释和真正冻结后的外部确认。

## SOTA context 边界

COSMOS 论文中的 P22 ARI 约 0.63 只作为不同 endpoint/protocol 的 stretch context；本轮不把 0.4619 与其做公平胜负判断。SpatialGlue、SMART、ARISE 在 Night-13A board 的语义已纠正为 `NOT_RUN_POLICY_BLOCKED`，不是实测失败。本轮未完整运行外部方法。

## 真实 P0、修正与失败保留

- A1/P22 真实链 2/2 通过：finite gradient、checkpoint strict reload、fresh-process numerical round-trip 和 partition exact。
- 工程修正 4 条；P0 byte-exact 失败、search 写目录失败、formal evaluator seed 错误均登记，能保留的工件均未删除。
- 历史 raw metadata 复核：0 个 root 改变，passed=True。
- 新下载 0；外部方法训练 0；dataset-name routing 0；dense N×N 0。

## 导师汇报版

我们先把 Night-13A 的数据登记、MISAR 标签和 tonsil 评价 mask 错误全部纠正，并没有篡改旧交付。随后把 C00、F00 和 N02 的真实 checkpoint、embedding、partition、manifest 在同一 observation、known K 和 common evaluator 下重新桥接。我们设计了一个只有同一核心公式的内容自适应稀疏图残差，不按数据集名称切模型。它在 A1 和 P22 上分别优于当前最强同口径内部参考，tonsil s1 基本保持。冻结后在 D1 仍有小幅正向，tonsil s2/s3 近似不变，但 MISAR 出现明确回落。因而本轮只能定为 LOCAL SIGNAL，不能称为 CONFIRMED MILESTONE。公开标签只用于 evaluator 和跨运行选择，没有进入无监督损失或 checkpoint 选择。下一步必须围绕 MISAR 的失败做消融与机制约束，再考虑外部强基线和论文级验证。

## 技术附录

- formal rows：35/35；search rows：45/45；P0：2/2。
- formal GPU 时间使用 CUDA event，仅覆盖模型 forward；wall time 覆盖该单元的完整加载后运行与评价路径。
- peak GPU：24.382 MiB；peak RSS：2792.234 MiB。
- Git、bundle 和 compact SHA 在最终封口后写入 `delivery_manifest.json`。
