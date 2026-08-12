# SpaLORA Night‑3B 独立证据审计与决策

日期：2026‑08‑13  
角色：论文规划 Worker（独立读取，未连接 AutoDL，未重跑训练）  
Night‑3B 审计根目录：`C:\Users\李昌赫\Documents\Codex\2026-08-07\https-github-com-abao-519-spalora\work\night3b_handoff_20260810`

## 0. 一句话结论

Night‑3B 的执行与交付完整性为 **带限定通过**；预注册方法级结论经逐 seed 独立复算后仍为 **`MIXED_EVIDENCE`**。

- 120/120 个预注册运行完整，未发现训练期标签泄漏、seed 搜索、失败删除、结果筛选或哈希损坏。
- 7 个消融中没有任何一个达到 `SIMPLIFY_CANDIDATE`；仅 Corr1 correspondence loss 达到跨数据集 `SUPPORTED`。
- learned attention 没有形成跨数据集一致支持；P22 的 learned cross attention 是“稳定但近饱和、且功能上失配”的重要负面边界案例。
- 因此不能写成“FULL_IGE 已证明为最优最终架构”，也不能根据单个数据集事后删组件。
- 下一步不再在现有三数据集上扩展同类模块搜索；先修复证据呈现，再完成独立数据与现代基线的端到端可复现性预检，随后才锁定公平 benchmark。

## 1. 审计范围与边界

本次实际读取并交叉核验了：

1. 迁移包、主交接文档、状态 JSON、Night‑3B 原任务书；
2. `night3b_report.md`、completion、gate、P0‑ARCH、协议偏差、120-run 锁定清单；
3. `per_seed_metrics.csv`、`paired_ablation_deltas.csv`、`summary.csv`、`component_support_matrix.csv`；
4. attention、IGE、gradient influence、QC/domain、per-domain、resource、replay 输出；
5. 24 个 PNG/PDF 图文件及其实际渲染；
6. 源码、配置、专项测试、交付索引、`SHA256SUMS`、Git bundle 与 push/关机记录。

本次没有连接 AutoDL，没有重跑 120 个训练，也没有改变原始交付结果。需要披露的一项本轮过程偏差见第 11 节。

## 2. 迁移包与交付完整性

### 2.1 迁移包

`D:\桌面\jiaojie\SpaLORA_New_Worker_Migration_Pack_2026-08-13.zip`

- 独立计算 SHA‑256：`f41c1087c8ee43a11d7f9a91d0bdfca7f01de41c6b92ed2435140734ac582168`
- 与交接值一致。
- 12/12 个条目均可读取，展开后的字节数与 ZIP 目录记录一致。

### 2.2 Night‑3B 交付

- `delivery_index.json`：55/55 条路径、大小和 SHA‑256 独立匹配。
- 完整归档中的根 `SHA256SUMS`：1186/1186 条独立重算通过。
- 12 组图、24/24 个 PNG/PDF 均存在并进入哈希闭环。
- 大文件独立 SHA‑256：
  - Git bundle：`776ac67c79788fd2993b402b84efb974ae5fca64a213662b3911cfe321cb005b`
  - 完整归档：`f5030232b19ee92888c4c4b710dca1282c757d373c4a080b06fe1f6beacac649`
  - 精简规划包：`ed77df03cc48b4b153b82d9275c95f8a3eacf428a6410478ba5a38ec1e2c4017`
- Git bundle 声明完整历史并通过验证：
  - final commit：`16b04499d41a1e45727eea46a4b6e138e35f599a`
  - final tag：`night3b-final-20260810`
  - Night‑3AF baseline：`384e66587149a687b3eac4a6d1918d8d4972dc06`
  - baseline 是 final 的祖先；对象检查通过。

结论：交付内容可用于继续科研规划；GitHub push 失败不等于科研结果缺失，完整历史已由经验证的 bundle 保留。

## 3. 执行与科研诚信审计

### 3.1 通过项

- P0‑ARCH：3/3 数据集默认路径等价性通过；24/24 probes 通过。
- CPU 的输出、raw losses、IGE 系数、总损失、梯度、一步 Adam 状态/参数为 exact；GPU 数值差在预先规定 envelope 内。
- 120 个唯一 cell 严格覆盖 `3 datasets × 8 variants × seeds 0–4`，运行顺序与预注册顺序一致。
- 120 份 run manifest、0 份 failure；每个运行的 8 个必需产物均存在且哈希闭合。
- 每个 dataset×seed 的八变体初始 state 一致；删除项系数精确为 0；active IGE 系数有限、为正、和为 4；uniform attention forward 精确为 0.5/0.5。
- 训练期只读 immutable cache；训练配置删除 ground-truth 字段；ScientificWindow 记录没有打开 ground-truth CSV，也没有导入禁止的 evaluator。
- 未发现标签指导调参、seed 搜索、失败结果删除或追加变体。

### 3.2 需要限定的证明边界

1. **“6/6 tests passed”仅指 Night‑3B 专项测试。** 最终运行的是 `tests/test_night3b.py` 的 6 项，不可写成“全仓测试全部通过”。首次全仓尝试为 101 passed、7 failed；失败主要来自受保护的历史路径/旧锁，但仍必须保留这一事实。
2. **最终训练轨迹不是 exact replay。** initial state 为 15/15 exact；final-state、embedding、attention 的 all-field exact 为 0/15；clusters 为 14/15 exact。任务书没有把最终 exact replay 设为硬门，因此不推翻 P0‑ARCH，但必须作为复现性风险公开。
3. **评估源代码并非读标签后完全不变。** 最终 `evaluation_config_lock.json` 明确记录 `semantic_label_values_read_before_amendment=true`。第一次评估已读取标签并写出数值后，因 Matplotlib 兼容问题和 replay 角色审计继续修补 evaluator/finalizer。18 个首次数值输出与最终文件逐字节一致，模型、阈值、seed、变体和结论未改变；故未发现标签驱动科学修改，但“评估器全程预先冻结”这一表述不能成立。
4. **标签防火墙的 config lock 未覆盖完整传递依赖链。** 实际导入的若干继承模块未全部进入 `source_sha256`。Git 树表明它们相对基线未改，静态审查未见泄漏，但这仍是证明覆盖缺口。
5. **历史保护证明范围不完全相同。** Night‑3AF 的 709/709 日志证明基线工作树副本完整，不能单凭该日志独立证明原 Night‑3AF 目录当时没有变化。
6. **push 次数和关机顺序只能由现有静态记录佐证。** 日志支持“一次 push 因凭据失败”和“要求最后执行 `/usr/bin/shutdown`”，但本地审计无法穷尽证明不存在其他远端操作。

这些限定不改变 120-run 科学结果与 `MIXED_EVIDENCE`，但必须进入方法/复现性说明。

## 4. 独立复算覆盖

- `per_seed_metrics.csv`：120 行，主键 120/120 唯一、无缺项、所有主要数值有限。
- `paired_ablation_deltas.csv`：133 行，精确等于 `7 × (15 seed rows + 3 dataset summaries + 1 macro summary)`。
- `summary.csv`：456 行；从逐 seed 原表重算 mean、样本 SD、median、min、max，最大误差 `1.11e-16`。
- 全部逐 seed 配对差、数据集汇总、macro 平均与原表一致。
- `component_support_matrix.csv` 的 7 个规则输入、分类和方法级输出均被独立复现。
- `boundary_disagreement = 1 - spatial_neighbor_agreement` 在全表精确成立。

配套最小证据表：`D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night3B_Minimal_Evidence_Table_2026-08-13.csv`

## 5. FULL_IGE 五 seed 基线

| Dataset | ARI mean±SD | NMI mean±SD | Neighbor | Moran’s I | Geary C |
|---|---:|---:|---:|---:|---:|
| A1 | 0.246859±0.009616 | 0.372468±0.006200 | 0.576630 | 0.483487±0.010401 | 0.526283±0.010525 |
| Placenta | 0.604405±0.036103 | 0.659636±0.021608 | 0.477256 | 0.469405±0.023299 | 0.674023±0.013965 |
| P22 | 0.395197±0.038247 | 0.553113±0.012448 | 0.823112 | 0.790867±0.004795 | 0.216013±0.004144 |

五个 seed 是优化随机性的重复，不是五个独立生物样本。bootstrap CI 只能作描述性稳定性证据，不能被写成生物学总体显著性。

## 6. 预注册组件判定

差值均定义为 `FULL_IGE − ABLATION`。ARI/NMI/neighbor/Moran 越大越好，Geary C 越低越好。

| Ablation | Macro ΔARI | ARI+NMI 非劣数据集 | 支持数据集 | 简化空间失败 | 组件判定 |
|---|---:|---:|---|---|---|
| DROP_RNA_RECON | +0.097656 | 1/3 | Placenta | 否 | MIXED |
| DROP_MOD2_RECON | +0.035894 | 0/3 | P22 | 否 | MIXED |
| DROP_CORR1 | +0.045690 | 1/3 | Placenta、P22 | 是 | **SUPPORTED** |
| DROP_CORR2 | −0.016937 | 1/3 | 无 | 否 | MIXED |
| UNIFORM_WITHIN | −0.029494 | 2/3 | P22 | 是 | MIXED |
| UNIFORM_CROSS | +0.021026 | 1/3 | 无 | 否 | MIXED |
| UNIFORM_ALL | −0.040736 | 2/3 | 无 | 是 | MIXED |

独立应用原任务书规则：

- simplification candidate：0/7；
- SUPPORTED loss：Corr1，1 项；
- SUPPORTED attention：0 项；
- `KEEP_FULL` 条件不成立；
- `SIMPLIFY_CANDIDATE` 条件不成立；
- 方法级结论：**`MIXED_EVIDENCE`**。

这不是“每个组件都不可缺少”，也不是“FULL 平均最好”。例如，Placenta 的 `UNIFORM_WITHIN` 和 `UNIFORM_ALL` 在 ARI/NMI 上明显更好，但同时 neighbor 和 Moran 均下降超过预注册空间阈值，所以不能作为跨数据集简化候选。

## 7. 组件与数据集的真实结论

### 7.1 Loss terms

- **Corr1 是证据最明确的保留组件。** Placenta 的 ΔARI/ΔNMI 为 `+0.0441/+0.0440`；P22 为 `+0.1083/+0.0942`，且两者均为 5/5 seed 同方向。
- RNA reconstruction 的强支持主要来自 Placenta：ΔARI `+0.2783`、ΔNMI `+0.2254`，5/5 seed；P22 不支持稳定收益。
- Mod2 reconstruction 的强支持主要来自 P22：ΔARI `+0.1269`、ΔNMI `+0.0760`，5/5 seed；A1/Placenta 不形成跨数据集支持。
- Corr2 没有达到支持标准。删除 Corr2 的平均 ARI 在三数据集均略高，但 NMI/空间条件与跨数据集规则不满足，所以既不能称“有用”，也不能按本轮结果事后删除。

### 7.2 Attention

- learned within、cross、all attention 均为数据集依赖的混合证据。
- Placenta 上等权 within/all 的 ARI/NMI 显著更高，但存在空间连续性损失。
- P22 上 `UNIFORM_CROSS` 在 5/5 seed 的 ARI/NMI 和全部空间方向上均优于 FULL：
  - ΔARI `−0.06133`
  - ΔNMI `−0.03546`
  - Δneighbor `−0.04409`
  - ΔMoran `−0.03472`
  - ΔGeary `+0.03477`
- 因此不能把 learned attention 写成稳定、普适的性能来源。

### 7.3 P22 seed 4 replay 敏感性

P22 FULL_IGE seed 4 是实质数值分叉：

- embedding 最大绝对差：`0.132438`
- attention 最大绝对差：`0.438697`
- 新旧 partition ARI：`0.633287`
- Night‑3B 相对 Night‑3AF 的该 seed ARI：`+0.035489`
- 它使 P22 五-seed FULL mean ARI 增加约 `+0.007098`。

用旧 Night‑3AF seed‑4 值做诊断性反事实替换后：七项组件分类全部不变，仍只有 Corr1 `SUPPORTED`，0 个 simplification candidate，方法级仍为 `MIXED_EVIDENCE`。因此分叉是必须公开的复现性风险，但不是当前决策的翻转点。

## 8. 解释性证据边界

### 8.1 可以支持

- IGE 是 initialization-based、label-free、训练期冻结的 loss-scale calibration；四个 active coefficients 和为 4。
- 初始 weighted-gradient shares 约为等权，训练后重新分配；总体 late entropy 约 `0.900`，没有观察到单项梯度影响坍缩。
- Corr1 的 late gradient share 在 A1/Placenta/P22 分别约为 `0.380/0.387/0.364`，Corr2 最低，和消融的“Corr1 更有证据、Corr2 较弱”方向一致，但不能替代消融。
- attention 在部分通道具有跨 seed 稳定性，并和输入 QC/domain 相关，可作为模型行为审计。

### 8.2 关键负面边界：P22

P22 cross attention 是近二值 spot-level 路由：

- RNA/mod2 mean：`0.8603/0.1397`
- normalized binary entropy：`0.0589`
- `<0.05` 或 `>0.95` 的 spot 比例：`0.9486`

但 `UNIFORM_CROSS` 明确优于 learned cross。最稳妥的结论是：

> P22 学到了可复现但近饱和、功能上失配的 cross-modal routing；attention 的稳定性、极端性和 domain 关联不等于性能贡献、解释忠实度或生物重要性。

### 8.3 不可支持

- attention 等于解释或生物因果重要性；
- 高 RNA attention 证明 RNA 在生物学上主导；
- IGE coefficient 等于某个 objective 的生物学重要性；
- 单个“最敏感 domain”驱动全局 ARI；
- 五 seed 或同一批 spots 的重复检验证明生物学泛化；
- A1 learned attention 稳定提升聚类、只是以边界破碎为代价；
- Placenta 第二模态是“63 个 raw ATAC peaks”。正确表述仍是 `ATAC-derived / TF-associated regulatory features`。

若未来要把 attention 升级为 faithfulness 证据，需在 frozen checkpoints 上做预注册的 matched perturbation，而不是只看 attention map、entropy 或相关性。

## 9. 发现的一项表头语义错误

`night3b_evaluate.py` 对所有指标统一使用 `values > 0` 生成 `*_full_win_count`。这对 ARI/NMI/neighbor/Moran 正确，但：

- Geary C 越低越好；其 `full_win_count` 实际数的是 FULL 的劣势次数。
- boundary disagreement 也越低越好，存在同样问题。

差值、均值、SD、CI 和组件/方法级判定均正确；这两个 win-count 列没有进入预注册决策，所以不影响 `MIXED_EVIDENCE`。后续必须从锁定原表生成更正表，按 `Δ < 0` 计算 lower-is-better 的 FULL wins，或把原列改名为 `positive_delta_count`；不得覆盖原始交付文件。

## 10. 图表审计

24/24 图文件存在、可渲染，但当前整体是审计诊断图，不是 publication-ready：

- forest plot 的 y 轴标签渲染后不可见；
- attention entropy/stability 面板标签重叠；
- P22 heterogeneity 只标 `1–7`，没有变体映射；
- attention distribution 只标 `1–6`，没有通道映射；
- domain 图使用数字索引，没有真实域名；
- heatmap 缺少 seed-level uncertainty；gradient trajectory 缺少 seed ribbon；
- A1 boundary maps 无 ground truth 面板，cluster color 未跨 variant/seed 对齐；
- spatial trade-off 标注拥挤，且没有同时清楚呈现 Geary。

这些问题不改变 CSV 数值，但当前图不能直接进入主文。下一轮只允许从锁定 CSV/NPZ 和固定 ground truth 重绘，不得重新训练或改数值。

## 11. 本轮审计自身的过程偏差

完整性审计代理为独立验证 Git bundle，在原先仅含空 `.git` 的以下目录执行了本地 fetch/checkout：

`C:\Users\李昌赫\Documents\Codex\2026-08-07\https-github-com-abao-519-spalora\work\night3b_handoff_20260810\bundle_verify_repo`

这填充了验证工作树，违反了本轮设定的“只读、不修改文件”边界。发现后已立即停止，没有清理或继续写入。

- 受影响仅为上述验证目录；
- 未修改 bundle、两份 tar.gz、full archive、planner handoff、交付索引或任何科研结果；
- 三份大文件 SHA‑256 在该操作后仍与交付索引一致。

后续必须把该目录视为“2026‑08‑13 审计生成的验证工作树”，不得把它当作 Night‑3B 原始交付证据。为了保留审计链，本次不擅自删除它。

## 12. 论文写作分层

### 主文可写

1. IGE 的定义、label-free 初始化校准和冻结属性；
2. Corr1 是目前跨数据集最清晰的保留组件；
3. RNA/Mod2 reconstruction 的证据分别偏向 Placenta/P22；
4. attention 的贡献明显依赖数据集，存在标签一致性与空间连续性的权衡；
5. P22 是 learned cross attention 的明确负面边界案例；
6. Night‑3B 的方法级结论是 `MIXED_EVIDENCE`，而不是选择性保留有利结果。

### 补充材料

- 全部五-seed 消融、逐 seed 表、bootstrap 描述性 CI；
- per-domain、Geary、boundary disagreement；
- IGE coefficients、gradient influence、attention distribution/stability；
- QC/domain association；
- replay 分叉、协议修补与测试范围；
- 修复后的诊断/空间图。

### 禁止写法

- “FULL_IGE 已证明为最优/最终架构”；
- “所有 loss 都是必要的”；
- “learned attention 普遍优于等权融合”；
- “attention 揭示了生物机制/真实模态重要性”；
- “五个 seed 是五个独立生物重复”；
- “最终训练轨迹 exact reproducible”；
- 隐去 P22 分叉、图表缺陷或评估源修补。

## 13. Q2 readiness 与后续路线

规划估计从 Night‑3AF 审计后的约 45% 上调至 **约 50%**。这是项目管理估计，不是投稿成功概率。

- Gate A（核心候选/可复现性）：基本通过，但保留 replay 与证明覆盖限定；
- Gate B（架构/loss/解释）：消融已完成且负结果可信；没有冻结出可声称普适最优的简化架构，faithfulness 未完成；
- Gate C（现代基线/独立泛化）：尚未完成，是当前最大缺口；
- Gate D（鲁棒性/规模/生物学）：只有部分资源与诊断证据；
- Gate E（代码/论文定稿）：尚未完成。

下一轮采用两级路线：

1. **Night‑4A：证据修复 + 外部数据/现代基线端到端预检。** 修复 lower-is-better win-count 语义和出版图；验证 D1/tonsil/独立候选数据；在目标服务器完整跑通官方 baseline 教程并锁 commit/env/input 兼容性。不得运行正式 benchmark 或根据标签调参。
2. **Night‑4B：预注册公平 benchmark 与独立确认。** 只有 Night‑4A 达到入场门后才制定并执行。SpaLORA 的外部确认只保留预先指定的 `FULL_IGE / UNIFORM_CROSS / UNIFORM_WITHIN` 三个候选，不再扩充同三数据集的新模块或搜索 seed。

Night‑4A 执行任务书：`D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night4A_Evidence_Repair_and_Benchmark_Preflight_Taskbook_2026-08-13.md`

