# SpaLORA Night-6C 独立审计与 Night-6D 决策

日期：2026-08-17  
审计角色：规划 Worker；仅使用 D 盘交付进行只读核验  
Night-6C 权威终态：`NIGHT6C_BALANCED_AND_OR_ACCURACY_CANDIDATES_LOCKED`  
下一步决策：`PROCEED_TO_NIGHT6D_LOCKED_D1_P22_CONFIRMATION`

## 1. 结论先行

Night-6C 是截至目前最有价值的正向研发结果：预注册组合 `G04_SP10_F10_EUC_UNION/H05_EQUAL3_AFFINITY_SPECTRAL` 在 A1 与 tonsil 的 10 个 dataset-seed 配对 Q 上 10/10 获胜，macro ΔQ 为 `+0.0457485860`，两数据集空间保护门均通过。

但它还不是论文层面的“方法已成功”。主要限制是：

1. A1 平均 ΔQ 只有 `+0.0082704014`；
2. A1 平均 ΔARI 为 `-0.0023900697`，仅 1/5 seeds 的 ARI 上升；A1 收益来自 5/5 seeds 的 NMI 上升；
3. macro 提升主要由 tonsil 的 `+0.0832267707` 拉动；
4. 组合从 9 graph × 12 head 的预注册开发空间中选出，仍存在开发集乐观偏差；
5. 尚未在最重要的 D1 人类淋巴结和 P22 小鼠脑上检验这一新 graph/head 机制。

因此下一步不继续增加候选，也不开始正式 benchmark。应对唯一锁定组合做一次 D1+P22 同时总锁、十 seed 的确认实验，并把 ARI、NMI 分别为正写入 material-gain 硬门。

## 2. 交付与实现完整性独立复核

本地根：

`D:\文档\ChatGPT\博士第一篇科研论文项目\night6c_handoff_20260817\official_compact`

- `handoff/delivery_index.json`：77/77 文件存在、size 与 SHA-256 匹配；索引 SHA `06b2fa0ff9961ce847faea73a973ee6523abb73f75377b2f1d57153a1f2020b9`。
- `external_delivery_index.json`：4/4 匹配。
- `local_post_dispatch_index.json`：3/3 匹配。
- Git commit、远端 branch、final tag 均为 `172cefba7559b34d2894ffa304553c0068ca23b9`；记录显示 final tag 只创建/推送一次，未 force。
- 最终测试 21/21；9/9 graph 与 12/12 head 语义门通过。
- 科学训练 48；48 个 checkpoint file SHA、canonical tensor-state SHA 和 run-manifest SHA 均唯一。
- 48/48 fresh-process round-trip 通过，48/48 H00 reload clusters exact。
- R1 432/432 transforms 均被尝试：395 success，37 个预注册数值失败；R2 48/48 success。
- 37 个数值失败为 H09 的 36 个 mclust/co-association cell 和 H06 的 1 个 invalid-K cell；锁定的 G04/H05 10 个最终 cell 全部完整。
- 两次训练纠正分别来自 GPU 初始化环境和 tensor JSON serialization，实现原因明确且旧 attempt 已保留；不是数值差重跑。
- D1、P22、GSE198353、Night-4B 在 Night-6C 的访问均为零。

## 3. 逐数据集复算

所有下列均由 `per_seed_metrics.csv` 按同 dataset、同 seed 的 fresh `G00/H00` 重新配对计算。

### 3.1 A1 人类淋巴结

| 配置 | mean ARI | mean NMI | mean Q |
|---|---:|---:|---:|
| fresh G00/H00 | 0.2715955221 | 0.3897430123 | 0.3306692672 |
| locked G04/H05 | 0.2692054525 | 0.4086738847 | 0.3389396686 |
| paired delta | -0.0023900697 | +0.0189308724 | +0.0082704014 |

- ARI wins：1/5；NMI wins：5/5；Q wins：5/5。
- 五个 ΔQ 范围：`+0.0035799357` 至 `+0.0162530066`。
- mean Δneighbor `+0.0450505634`，ΔMoran `+0.0617565395`，ΔGeary `-0.0530584561`，Δboundary `-0.0450505634`；空间方向整体改善。

结论：A1 的 Q 增益稳定但较小，且不是 ARI/NMI 双提升。不能把 macro 正 ΔARI 错写成 A1 ARI 提升。

### 3.2 Tonsil section 1

| 配置 | mean ARI | mean NMI | mean Q |
|---|---:|---:|---:|
| fresh G00/H00 | 0.0855772151 | 0.1736463141 | 0.1296117646 |
| locked G04/H05 | 0.1529544047 | 0.2727226658 | 0.2128385353 |
| paired delta | +0.0673771896 | +0.0990763517 | +0.0832267707 |

- ARI wins：4/5；NMI wins：5/5；Q wins：5/5。
- mean Δneighbor `-0.0136303882`，ΔMoran `-0.0033527346`，ΔGeary `+0.0023763837`，空间门通过但存在轻微平均退化。

结论：tonsil 是明确正向来源，但绝对 ARI 仍为中等水平，且该数据集不能单独承担论文主张。

## 4. 机制分解

五 seed、两数据集 summary 显示：

- `G04/H00` graph-only：macro ΔQ `+0.0040503071`；
- `G00/H05` head-only：macro ΔQ `+0.0363770312`；
- `G04/H05` full：macro ΔQ `+0.0457485860`。

但不能简单说“只有 head 有效”：

- A1 上 graph-only ΔQ `-0.0014377170`，head-only ΔQ `-0.0053544111`，full 却为 `+0.0082704014`，对应正 interaction 约 `+0.0150625295`；
- tonsil 上 head-only 已提供主要收益，G04 增量较小。

所以未来主方法必须是完整 `G04/H05` 组合；H05 或 G04 单独都不能替代锁定候选。Night-6D 同时运行 2×2 factorial 只用于机制解释，不能重新选择方法。

## 5. 为什么现在进入 D1+P22

Night-6C 已达到事先锁定的开发门，继续只在 A1/tonsil 增加 head 或 graph 会扩大开发集过拟合而不能回答泛化问题。D1 与 P22 应在同一标签窗口前全部训练和 transform 总锁：

- D1 是与 A1 同研究的人类淋巴结独立 section，具有十类 manual ground truth；它是本轮主要 held-out confirmation。
- P22 是 RNA+ATAC 跨组织/跨模态确认，但此前参加过 Night-3B/Night-5D，不能称 pristine holdout；新 G04/H05 组合从未在 P22 上运行，仍可作为预锁定 cross-dataset confirmation。

两个数据集都固定 seeds 0–9。十 seed 不是 seed search，而是提高 exact paired test 的分辨率，并增加五个未参与 Night-6C 选择的新 seeds。

## 6. Night-6D 设计决策

唯一 primary treatment：`G04_SP10_F10_EUC_UNION/H05_EQUAL3_AFFINITY_SPECTRAL`。  
唯一 reference：同 dataset、同 seed 的 fresh `G00_SP18_F20_CORR_UNION/H00_FUSED_PCA20_MCLUST_EEE`。

训练只含两个 graph：G00、G04；transform 只含两个 head：H00、H05。固定矩阵：

- 2 graphs × 2 datasets × 10 seeds = 40 training units；
- 2 graphs × 2 heads × 2 datasets × 10 seeds = 80 transforms；
- 40 个真实 final checkpoint 与 fresh-process round-trip。

每个数据集的 encoder 训练协议在开标签前由既有权威合同固定：D1 继承 A1 C04/B01 配置；P22 继承 Night-5D P22 B01/C04 配置。候选与 reference 在同一数据集内除 graph/head 外完全相同。这是复用既有 dataset contract，不是根据 Night-6D 标签调参。

## 7. 确认门

对 D1 与 P22 分别以十个 paired ΔQ 做单侧完整 `2^10` sign-flip；两个 primary p-value 做 Holm。固定 100,000 次 paired bootstrap。

某数据集只有同时满足以下条件，才标记 material accuracy confirmed：

- mean ΔARI > 0；
- mean ΔNMI > 0；
- mean ΔQ ≥ +0.010；
- positive-Q wins ≥7/10；
- Holm-adjusted exact p <0.05；
- bootstrap 95% CI 的 ΔQ 下界 >0。

再独立套用既有空间保护门。两个数据集都达到 material gain 且空间门通过，才允许终态 `NIGHT6D_D1_P22_BALANCED_CONFIRMED`。单数据集成功、方向不一致、统计不足或空间权衡均进入 `NIGHT6D_PARTIAL_OR_MIXED_EVIDENCE`，不得包装成普适胜利。

## 8. 操作边界

本次规划审计未连接、查询或唤醒 AutoDL，未修改 Night-6C 或任何历史交付。用户仍需手动以 GPU 模式开机，并把最终 README 提示复制给实验 Codex。
