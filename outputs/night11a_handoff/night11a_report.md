# SpaLORA Night-11A 选择性跨模态传递无标签可识别性报告

## 负责人现在需要知道的三件事

1. 终态是 `NIGHT11A_NO_IDENTIFIABLE_TRANSFER_UTILITY`，分类为 `SCIENTIFIC NEGATIVE`。实现、复现、资源和零标签防火墙均通过，但预注册机制门没有全部通过。
2. gate 对“传递有益/有害”的排序有明显局部信号：RNA+PROTEIN AUROC 0.8605、Spearman 0.6907；RNA+ATAC AUROC 0.8767、Spearman 0.7428。然而两家族 gate separation 分别只有 0.2220 和 0.1837，均低于冻结门 0.25。
3. B3 在平均 MSE 和 regret 上通常比 B0/B1/B2 更温和，但没有达到冻结的 target-damage 1% 优势门，也没有达到 auxiliary-conflict 相对 B1 的 10% 强安全门。因此不能派生近邻候选，不能声称聚类分数提高、SOTA 或论文方法已经成立。

## 状态分类与问题—动作—论文含义

- **问题**：检验无标签的交叉拟合证据是否不仅能排序局部传递效用，还能在损伤和冲突条件下可靠决定“传递或拒绝”。
- **实际动作**：固定两种模态家族各一个真实 seed，完成 2 units × 3 conditions × 3 replicate seeds × 2 directions × 4 arms = 144 行；使用真实 G00/G04 权威视图、稀疏 20% patch、固定 ridge α=1、固定 folds/seed/公式和 H05 endpoint；没有标签或科学指标读取。
- **结果**：排序识别通过多数相关门，但 gate separation、target-damage usefulness 和 auxiliary-conflict safety 三类关键门未同时通过。
- **论文含义**：这是一个工程正确的科学负结果。它说明当前最小 B3 gate 捕捉到局部效用次序，却还不足以成为受冻结条件约束的稳健选择性传递机制。

## 可识别性主表

| 家族 | AUROC | AUPRC | Spearman | gate separation | 正/负非模糊事件 | gate std |
|---|---:|---:|---:|---:|---:|---:|
| RNA+PROTEIN | 0.8605 | 0.8407 | 0.6907 | 0.2220 | 26318/35748 | 0.1713 |
| RNA+ATAC | 0.8767 | 0.8691 | 0.7428 | 0.1837 | 74305/89568 | 0.1517 |
| POOLED | 0.8717 | 0.8613 | 0.7277 | 0.1932 | 100630/125311 | 0.1578 |

## 三条件四 arm 的 MSE / regret

> 每格为 `mean MSE / mean absolute regret`；regret 相对逐 spot 的 min(B0, B1)。

### RNA+PROTEIN

| 条件 | B0 SELF | B1 ALWAYS | B2 UNCERTAINTY | B3 SELECTIVE |
|---|---:|---:|---:|---:|
| CLEAN_HOLDOUT | 0.00110530935 / 1.7e-05 | 0.00111432708 / 2.6e-05 | 0.00110354592 / 1.52e-05 | 0.0011001478 / 1.18e-05 |
| LOCAL_TARGET_DAMAGE | 0.00125812618 / 2.7e-05 | 0.00125941035 / 2.83e-05 | 0.0012534651 / 2.24e-05 | 0.00124671031 / 1.56e-05 |
| LOCAL_AUXILIARY_CONFLICT | 0.00110530935 / 1.38e-05 | 0.00111780231 / 2.63e-05 | 0.00110492998 / 1.34e-05 | 0.00110200998 / 1.05e-05 |

### RNA+ATAC

| 条件 | B0 SELF | B1 ALWAYS | B2 UNCERTAINTY | B3 SELECTIVE |
|---|---:|---:|---:|---:|
| CLEAN_HOLDOUT | 0.00199319349 / 1.65e-05 | 0.00200036663 / 2.37e-05 | 0.00199018945 / 1.35e-05 | 0.00198821248 / 1.16e-05 |
| LOCAL_TARGET_DAMAGE | 0.00216608097 / 2.37e-05 | 0.00216807441 / 2.57e-05 | 0.00216029752 / 1.79e-05 | 0.00215611255 / 1.37e-05 |
| LOCAL_AUXILIARY_CONFLICT | 0.00199319349 / 1.4e-05 | 0.00200396298 / 2.47e-05 | 0.0019918416 / 1.26e-05 | 0.00198996233 / 1.07e-05 |

## 冻结门结果

- auxiliary_conflict_safety: `FAIL`
- clean_safety: `PASS`
- event_counts: `PASS`
- firewall: `PASS`
- no_gate_collapse: `PASS`
- per_family_auroc: `PASS`
- per_family_gate_separation: `FAIL`
- per_family_spearman: `PASS`
- pooled_auroc: `PASS`
- pooled_spearman: `PASS`
- resource: `PASS`
- roundtrip: `PASS`
- target_damage_usefulness: `FAIL`
- uncertainty_ablation: `PASS`

通过项不能抵消失败项；合约要求全部门同时通过。正式 correction cycle 为 0；没有按行补跑、换 seed、换阈值或 scientific retry。

## 复现、资源与防火墙

- 正式矩阵：144/144 唯一行，18/18 case manifests。
- 两家族 smoke：2/2；fresh-process checkpoint/endpoint round-trip：2/2，四个 arm 的 canonical partition 均 exact。
- 独立审计：PASS；冻结实现与 config SHA 均保持不变。
- 资源：远端墙钟 4116.4 s；formal CPU 汇总 3476.7 s；peak RSS 1172764 KiB；peak GPU 0 MiB。
- 所有硬禁区计数均为 0：训练/评价/总标签、ARI/NMI/Q/AMI/FMI、annotation 空间指标、MISAR Y、E18.5、新数据、第三方 benchmark、QCRD、dataset-name routing、dense N×N、scientific retry/fallback。

## 导师汇报版（7句）

Night-11A 完成了两种模态家族上的无标签选择性传递机制检验。
整个实验没有读取组织标签，也没有计算 ARI、NMI 或 Q。
当前 gate 对传递效用的排序有清楚的局部信号，两家族 AUROC 约为 0.86 和 0.88。
但 gate 对正负效用的幅度分离不够，两个家族都没有达到预注册的 0.25 门。
在 target damage 与 auxiliary conflict 条件下，B3 也没有同时达到冻结的实用性和安全性优势门。
因此终态是科学负结果，而不是实现失败，也不是新的性能提升。
按照预注册规则，本轮不派生相邻候选，结果用于收缩论文方向并保留可复现的负证据。

## 技术附录

- parent commit/tag: `10f16eab93d9dbc4c33c3aaff2b84ee782694a80` / `night10b-final-20260821`
- branch: `revision/q2-night11a-selective-transfer-identifiability-20260822`
- protection tag: `baseline/pre-night11a-selective-transfer-identifiability-20260822`
- intended final tag: `night11a-final-20260822`
- freeze manifest SHA-256: `dbbff7e3f9ec83f056cff529967399526c0ce7c9b60516e6a68815c8ad7f7611`
- decision SHA-256: `cc02fee040f76535c10db3b61097910bb4f990c6f52632317eb77db7b82eb638`
- independent audit SHA-256: `ae7b0921fee94d0f9ba6c7f3081196d974179e569982b76215e8be8acb639205`
- final commit/tag peel、bundle 与 Windows compact 独立复算记录见 `final_delivery_audit.json`。
