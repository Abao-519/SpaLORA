# SpaLORA Night-14A：拓扑冲突过滤性能冲刺报告

## 我现在需要知道的三件事

1. **问题**：本轮检验一套真正更新参数的统一双模态图模型，配合 TCF（Topology-Conflict Filter），能否在两个模态家族中识别“空间传播可用”与“应退回原表示”的情形，而不是继续调整 Night-13C 的 B10/E11/E12。
2. **实际做了什么**：从两种真实模态输入出发，用同一套稀疏双视图图自编码核心训练 600 步；训练后在同一冻结 checkpoint 上离线计算 identity、固定低通和 TCF。候选与强参考统一使用 20 个 KMeans endpoint seeds、`n_init=10`、相同 observation/mask/known K；3 个真实 training seeds 都进入了参数初始化和优化。开发板为 A1、tonsil s1、P22、MISAR，冻结后才运行 D1、tonsil s2/s3。
3. **对论文的意义**：结果分类为 **`ATAC_FOCUSED_SIGNAL`**。P22 与 MISAR 在 common-head 口径下都稳定超过 matched strong reference，但 RNA+protein 的 study-balanced 结果仍为负，尤其 tonsil 明显失败。因此这支持继续收窄 RNA+ATAC 机制线，不支持“跨家族统一成功”、SOTA、confirmed milestone 或论文已成立。

## 明确终态

- 终态：`NIGHT14A_TCF_ATAC_FOCUSED_SIGNAL`
- 分类：`ATAC_FOCUSED_SIGNAL`
- 主候选：`C15_BAL_XREC_600_WEAK_ALIGN + W02_TCF_FINAL`
- 证据语义：公开 benchmark 开发 + 冻结后内部确认；不是 pristine blind evaluation。
- 模型训练无监督：标签没有进入 loss、gradient 或单次训练 checkpoint selection；公开标签只用于跨运行 HPO 与 evaluator。

## 模型位于流水线哪一层

选定核心是独立 clean-room 实现的稀疏双视图图自编码器：模态允许各自输入 projection，但之后共享 graph residual blocks、fusion、self/cross reconstruction 和同一组 loss 规则。latent 为 64，空间图 `k=8`，训练 600 步。TCF 位于训练后的表示层：它从两个模态对同一稀疏空间边的支持与冲突生成低通残差；完整性不足时精确退回 identity，不读取 dataset/tissue/family/path/label 字符串。

SMART（固定 commit `75676546a66d48a7ebfd7c5f3a2758a4c538fcfc`，GPL-3.0）和 SpaBalance（`c3610a638c98c2d525c247ed62eeb41bda430e2d`，AGPL-3.0）均完成源码审计，但固定源码分别涉及 dense pairwise/未注册模块以及 dense adjacency/cross-spot attention 与构造语义不闭合；未修改其方法后伪称官方复现，也未复制其源码。所选 clean-room 核心是合约允许的工程替代，不应写成已经验证的“成熟外部 backbone 等价物”。

## 真实输入与 P0

| 单元 | 原始两模态 shape | 进入统一核心的 shape | K | P0 |
|---|---:|---:|---:|---|
| A1 | 3484×18085；3484×31 | 3484×30；3484×30 | 10 | 两种 clean-room backbone 均训练、checkpoint、fresh reload 通过 |
| P22 | 9196×22914；9196×121068 | 9196×30；9196×50 | 9 | 两种 clean-room backbone 均训练、checkpoint、fresh reload 通过 |

P0 共 4/4 真路径通过；最终形式的可训练参数量为 RNA+protein 118,656、RNA+ATAC 122,516。参数确实更新，optimizer steps 不为 0。

## 绝对 ARI/NMI 主表

数值为 3 个 training seeds × 20 个 matched endpoint seeds 的均值；括号为相对冻结 matched reference 的平均差值。tonsil s2/s3 的 embedding 使用全部 paired observations，评价只在 canonical label 非缺失 subset 上进行。

| 阶段 | 数据集 | total/eval | matched reference | ARI | NMI | endpoint 双胜率 |
|---|---|---:|---|---:|---:|---:|
| 开发 | A1 | 3484/3484 | C00 common head | 0.2257 (+0.0006) | 0.3853 (+0.0068) | 58.3% |
| 开发 | tonsil s1 | 4326/4326 | simple concat | 0.1084 (-0.0676) | 0.2368 (-0.0434) | 0.0% |
| 开发 | P22 | 9196/9196 | N02 common head | 0.4684 (+0.0297) | 0.6205 (+0.0383) | 100.0% |
| 开发 | MISAR E15.5 S1 | 1949/1949 | RNA-only | 0.2706 (+0.0650) | 0.4488 (+0.1375) | 90.0% |
| 冻结确认 | D1 | 3359/3359 | simple concat | 0.2419 (+0.0116) | 0.4043 (+0.0521) | 78.3% |
| 冻结确认 | tonsil s2 | 4519/4518 | simple concat | 0.1428 (-0.0312) | 0.2277 (-0.0387) | 0.0% |
| 冻结确认 | tonsil s3 | 4521/4460 | simple concat | 0.1124 (-0.0699) | 0.1997 (-0.0437) | 0.0% |

共同端点下，RNA+ATAC study-balanced ΔARI/ΔNMI 为 **+0.0474/+0.0879**，联合胜率 95%；RNA+protein 为 **-0.0251/-0.0063**。把 tonsil 三切片合成一个 study effect 后为 -0.0562/-0.0420；lymph-node study effect 为 +0.0061/+0.0294。

这些 common-head 数字不能替代 native full-pipeline context。特别是 P22 的 N02 native 历史高水位约 0.5063/0.6562，仍高于本候选 0.4684/0.6205；A1 的 C00 native 0.2692/0.4087 也未被超过。

## 机制与关键消融

- TCF 将 P22 同一 backbone 的 identity 0.3138/0.4944 提升到 0.4684/0.6205；MISAR 从 0.2035/0.3711 提升到 0.2706/0.4488。
- A1 与 tonsil s1 的正式 TCF 精确退回 identity；冻结确认中 tonsil s2/s3 也退回 identity，D1 则获得小幅收益。这说明 gate 至少没有把一个固定传播强度强加给所有 protein 数据。
- 固定低通在 MISAR 为 0.2916/0.4713，高于正式 TCF；正式 TCF 在 P22 略高于固定低通（0.4684/0.6205 对 0.4642/0.6144）。因此不能声称 TCF 在每个 ATAC 单元都优于普通平滑。
- 离线关键消融中，support-only 在 MISAR 达 0.3137/0.4924、P22 达 0.4707/0.6242，但它在 A1/tonsil s1 回撤更大；integrity 的主要价值是安全拒绝，而不是取得单数据集最高分。
- 预注册的四个 development seed-0 corruption 为 4/4 exact identity。扩展到全部 12 个 development checkpoints 后为 11/12；A1 seed 2 的 gate 为 0.0688，未完全回退。这是实际机制限制，已保留，未补 seed 或改阈值。
- high-pass 10% 消融没有形成一致优势，因此正式主线没有启用 high-pass。

## 最重要的失败与限制

1. **protein 不广泛成立**：D1 改善，但 tonsil 三切片全部下降；这不是一个可包装成 broad unified success 的结果。
2. **未超过 native 历史高水位**：P22/A1 的 common-head 局部收益没有击败其历史 native 完整管线。
3. **TCF 不是所有 ATAC 上的最强滤波器**：MISAR 上固定低通与 support-only 更高，说明当前完整性门偏保守。
4. **corruption 不是 12/12**：A1 seed 2 未精确回退，未来必须把这当作机制修复目标，而不是删掉该 seed。
5. **backbone 外部成熟度未闭合**：官方 SMART/SpaBalance 路径因固定源码的 dense/语义边界未进入真实运行；当前核心是可训练、可复现的 clean-room 起点，不是公平外部 baseline 结论。

## 工程与资源审计

- 完整训练搜索配置：9；所有真实训练审计记录：80；正式冻结训练：21/21。
- 正式训练均为 600 optimizer steps；21/21 strict state load、数值 embedding replay 和正式 W00/W02 分区回放通过。
- fresh-process 全部滤波分区 exact 为 20/21；唯一差异是 P22 seed 1 的非主 W01 在 ≤2.9e-6 CPU/GPU 舍入后跨过 KMeans tie，原 FAIL 与修正前审计均保留。
- 正式训练 GPU 时间合计 224.10 s；全部已登记训练 GPU 时间 922.77 s。
- peak GPU 822.03 MiB，peak RSS 3513.80 MiB；derived root 约 825 MiB。
- 新数据下载 0，dense N×N 0，完整外部 baseline 运行 0，训练标签读取 0，标签进入 loss/gradient/checkpoint selection 0。
- 7 个历史 raw roots 的最终 metadata audit 为 7/7 byte-exact，changed roots 0。
- 定向测试最终为 10/10 通过。

## 导师汇报版（7句）

1. Night-14A 不再修 B10，而是训练了一套两家族共用、参数真实更新的稀疏双视图图模型，并在其后加入拓扑冲突过滤。
2. P22 和 MISAR 的 common-head ARI/NMI 都稳定超过各自强参考，RNA+ATAC study-balanced 增量为 +0.047/+0.088。
3. protein 结果不统一：A1基本持平、D1上升，但 tonsil 三切片均下降，所以不能宣称跨家族统一成功。
4. TCF 的主要机制价值是能在冲突明显时回退 identity；预注册 corruption 4/4 通过，但扩展多 seed 只有 11/12，仍需修。
5. MISAR 上普通低通或 support-only 比正式 TCF 更高，说明当前 gate 不是每个数据集的最优滤波器。
6. 本轮仍未超过 P22 N02 和 A1 C00 的 native 完整管线高水位，也没有做公平外部 SOTA 复现。
7. 因此最合理的下一步是收窄为 RNA+ATAC，修复 corruption fail-safe，并在冻结公式后做强外部 baseline、消融与生物解释。

## 技术附录

- parent：`aec0e07a890f1409ac3e763c0eaff4c7424ccc78` / `night13c-final-20260823`
- branch：`revision/q2-night14a-topology-conflict-sprint-20260823`
- protection tag：`baseline/pre-night14a-topology-conflict-sprint-20260823`
- final tag：`night14a-final-20260823`
- freeze：`NIGHT14A_FREEZE_20260823_04`
- 主 config SHA-256：`79f665781a14a7a8d212c0efef247db18fa4f5e611647221678a595ec04ee266`
- 主 filter-list SHA-256：`d3761db89b5d7cd987e8491b39a3342c1c412e2fc5a500496d554b9d6eebd597`
- 实际 final commit、tag peel、bundle 与 compact index SHA 由 compact 根的 `git_and_delivery_audit.json` 记录。

