# SpaLORA Night-5C Laplacian 纠错与候选锁定闭环任务书

日期：2026-08-14  
权威父提交：`2bfb3a0e9d363e90747ec0bd0ec0da92c829bc20`  
权威父标签：`night5b-final-20260813`  
纠错注册表：`SpaLORA_Night5C_Corrective_Registry_2026-08-14.json`  
注册表 SHA-256：`78a5280f3e0f97c897f2d344a93f6a4319b78feabbc0fb4a74dbd6c88d13604e`

## 0. 唯一目标

Night-5C 不是新一轮自由调参，也不是重复 Night-5B。它只做两件事：

1. 按 Night-5B 原注册表的真实语义，独立补跑 B21-B24 的正确 `uniform_all` Laplacian 单元；
2. 将这批纠正结果与 Night-5B 的 84 个未受污染训练单元合并，严格重放原来的 S1/S2 选择规则，形成可审计的完整开发集候选结论。

本轮禁止运行或打开 P22、D1、GSE198353、Night-4B，禁止正式外部 benchmark，禁止新增候选、改阈值、改 seed、看结果后调参或为论文做图。终点只是为下一份独立 P22 任务锁定最多三个候选。

## 1. 已知事实与科学解释

- Night-5B 共执行 108 个训练尝试，0 个运行失败。
- B21-B24 的 24 个单元实际误用了 `shrink_to_uniform_rho025`，而注册表要求 `uniform_all`；它们必须永久标为 invalid，不能用于任何候选选择。
- 其余 84 个训练单元没有受到该实现错误污染，原始结果、manifest、哈希和开发集指标仍然有效，不得重跑、覆盖或丢弃。
- 错误版 B21-B24 的三-seed `delta Q` 均为负，因而没有推动当时的 S2 top-up；但正确的 `uniform_all` 结果未知，仍可能改变完整候选池的排序。因此 Night-5B 的方法级候选锁定被撤回是正确的。
- B10 与 B17 的五-seed 开发集证据可保留为有效探索结果，但在纠正池闭合前，不得称为正式 P22 候选。

## 2. 权威输入与保护门

执行前逐项验证：

1. GitHub 分支 `revision/q2-night5b-secondlook-rescue-20260813` 与标签 `night5b-final-20260813` 均 peel 到权威父提交。
2. 本地 Night-5B 精简交付目录为：
   `/D:/文档/ChatGPT/博士第一篇科研论文项目/night5b_handoff_20260813/official_compact`
3. 下列 SHA-256 完全一致：
   - `night5b_report.md`：`c1cd4c255cb3a6d8af59174f34113e87dc0145ec8f55852cf4abdec435a0f75f`
   - `night5b_planner_handoff_20260813.tar.gz`：`73ee44cc4053395bc799de3ef4bc44f0dd4d0bd50bdc401370665fc777a10ac4`
   - `SpaLORA_night5a_to_night5b_20260813.bundle`：`4e6b8100464e6025903a1f116a4583286e68afa21a76c6dd923abb28a598680f`
   - `delivery_index.json`：`fdd9f3b447a23048345a42e2ab4316911e292c08a6aa919b3a721a15b04a4707`
   - `local_verification.json`：`09b238796834376b6b259e9d3b42a7ddb932e62a11e12d054c0f6d0142a6fff0`
4. 精简包内部 59/59 条目通过。
5. 远端持久盘上的 Night-5B raw runs、Night-5A source runs、Night-3AF cache 均存在；逐文件复核它们的既有 manifest/SHA，不以“目录存在”代替验证。
6. `invalidated_runs.json` 必须精确列出 B21-B24、A1/Placenta、seeds 0-2 共 24 个旧单元；旧目录只能读，不能覆盖。
7. Night-3B 1186/1186 与 Night-4A 76/76 历史保护仍通过。
8. R/mclust、CUDA/GPU、输入缓存、label firewall 或 Git 父提交任一不满足，终止为 `BLOCKED_PREFLIGHT`，不得降级硬跑。

新建隔离 worktree 和普通非 force 分支，建议：

- 分支：`revision/q2-night5c-laplacian-correction-20260814`
- 保护标签：`baseline/pre-night5c-20260814`
- 最终标签：`night5c-final-20260814`

## 3. P0-SEMANTIC：必须验证“运行时语义”，不能只查 JSON

Night-5B 的根因是注册表正确、运行时代码错误。因此以下门必须在任何正式训练前全部通过，并写入 `p0_semantic_contract.json`：

1. 纠错注册表恰有 B21-B24 四个候选，且参数与原 Night-5B 注册表逐字段一致；四个 `source_night5b_config_sha256` 必须匹配原 `candidate_contracts.json`。
2. 为 B00-B24 建立从 registry 到 trainer/model 的 `resolved_runtime_contract`。它至少记录：
   - candidate ID 与 config SHA；
   - 实际 `attention_policy`；
   - 实际 `learned_fraction`；
   - RNA anchor、triplet、DGI、latent reliability、diffusion、Laplacian 的实际启用状态和系数来源；
   - source candidate/source manifest（若复用）。
3. runner 在训练前比较 `resolved_runtime_contract` 与 registry；任何字段不一致时 fail closed。不能只依赖单元测试。
4. 对 B21-B24 逐候选、A1/Placenta、至少 seed 0 做真实 trainer 构造探针，必须证明：
   - `attention_policy == "uniform_all"`；
   - `learned_fraction is None`；
   - forward 中 within/cross 两路融合权重逐 spot 均为 `[0.5, 0.5]`（仅容许浮点误差）；
   - 不存在可学习 attention 参数或其梯度；
   - Laplacian target fraction 分别为 0.05/0.10，step-0 系数 finite、positive 且冻结；
   - RNA-anchor 或 MNN-triplet 与注册表一致。
5. 每个正式 `run_manifest.json` 必须同时保存 declared contract、resolved contract、二者 SHA 和 `semantic_contract_match=true`。缺一即该 run 无效，停止该阶段。
6. 增加负向回归测试：故意把 B21 的 runtime attention 改成 `shrink_to_uniform`，runner 必须在训练前拒绝。
7. 保留原来 sparse、edge dedup/self-loop、gradient scaling、finite forward/loss/Adam、checkpoint resume、label firewall、withheld-path rejection 全套测试。

本轮 P0 测试应在现有 Night-5A/Night-5B 48 项基础上增加上述语义测试；最终报告需列出测试总数和逐门状态，不能只说“通过”。

## 4. 数据、标签与不可变边界

- 训练数据只允许 A1 与 Placenta 的既有不可变 Night-3AF cache。
- 训练、checkpoint 选择、Laplacian 系数、attention、图、triplet、聚类输入生成阶段禁止读取语义标签。
- 所有 24 个 S1 纠错 embedding/checkpoint/manifest/SHA 全部冻结后，才允许第一次打开 A1/Placenta 评价标签。
- S2 run order 在开发集指标上按下述锁定规则生成并落 SHA 后，关闭评价窗口再训练 seeds 3/4；训练阶段仍不得读标签。
- mclust 固定 EEE、PCA 20、random seed 2020；不得逐 epoch 看 ARI/NMI，不得按真实标签选 checkpoint，不得 seed 搜索。
- 代码和路径中对 P22、D1、GSE198353、Night-4B 显式 fail closed。最终运行 `withheld_audit`，四者访问/运行计数必须为 0。

## 5. 运行计划与预算

### S1-CORRECT：24 个必需纠错单元

固定顺序为 dataset → candidate → seed：

- dataset：A1，然后 Placenta；
- candidate：B21、B22、B23、B24；
- seed：0、1、2。

共 `2 × 4 × 3 = 24` 个科学训练单元。输出必须写入新的独立根目录，例如：

`/root/autodl-fs/night5c_corrected_runs_20260814`

禁止覆盖 `/root/autodl-fs/night5b_raw_runs_20260813`。新 manifest 必须用 `supersedes_invalid_run` 指向对应旧 run 的 manifest/SHA，并明确旧 run 仍 invalid。

若发生硬件/进程故障，只能原 tuple 重试，保留失败日志；全轮最多 4 次基础设施重试。不得换 seed、换参数或借重试寻优。

### S1-REPLAY：重建原 S1 top-up 集

24 个纠错单元冻结后：

1. 载入 Night-5B 全部结果；排除且仅排除 24 个旧 invalid B21-B24 单元；
2. 插入 24 个 Night-5C 正确结果；
3. 对所有三-seed eligible candidates 精确重放 Night-5B 的 balanced top3、accuracy top2、latent-family reserve 与既定排序；
4. 得到 `recomputed_s1_topup_candidates.json`，同时写出与旧 S1 决策的逐项差异及原因；
5. 不允许因看到结果改变门槛、排序、family cap 或 top-up 上限。

### S2-CONDITIONAL：只补真正缺失的 seeds 3/4

- Night-5B 已有效完成 B09、B10、B06、B15 的 seeds 3/4；若它们仍在重算后的 S1 top-up 集，直接复用，不重跑。
- 若 B21-B24 中任一进入重算后的 S1 top-up 集，只补该候选 A1/Placenta 的 seeds 3/4，即每候选 4 个单元。
- 若已完成五-seed 的旧候选在重算后被挤出 top-up 集，其数据仍保留为有效探索证据，但不再具有 canonical P22 eligibility。
- 最多四个纠错候选需要 top-up，因此 S2 上限为 16 个科学训练单元。

本轮最多 40 个科学训练单元；计入最多 4 次原位基础设施重试，总训练尝试硬上限为 44。未用预算不得拿来新增候选或扩大搜索。

## 6. 完整候选池与正式 eligibility

最终五-seed 候选池分成两层：

1. canonical eligible：B01-B03 references、B17-B20 预注册 diffusion，以及重算后 S1 top-up 集中拥有完整五-seed 的候选；
2. valid exploratory but noncanonical：结果本身有效、但不属于重算后 S1 top-up 集的额外五-seed 候选。

两层均报告数值，但只有第一层可以进入 future P22 锁定。这样既不浪费 B06/B09/B10/B15 的有效结果，也不利用一次错误候选池带来的额外多重比较机会。

## 7. 数值门与最多三个 future P22 候选

统一以 B00 为历史 reference，并同时报告相对 B01 的差异。Q 定义为 `(ARI + NMI) / 2`。

### balanced frontier

五-seed 同时满足：

- macro delta ARI ≥ +0.01；
- macro delta NMI ≥ +0.01；
- macro delta Q ≥ +0.02；
- 两数据集 worst delta Q ≥ 0；
- 10 个 dataset×seed paired Q 至少 7 个为正；
- 不触发既定空间保护门；
- effective end-to-end runtime ≤ B00 的 2 倍，effective peak GPU ≤ B00 的 1.5 倍。

### accuracy frontier

- macro delta ARI ≥ +0.04；
- macro delta NMI ≥ +0.04；
- macro delta Q ≥ +0.05；
- worst-dataset delta Q ≥ -0.005；
- paired Q 至少 7/10；
- 同一资源门通过。

空间代价必须逐数据集完整报告，但不作为 accuracy frontier 的自动淘汰门。

### 锁定规则

最多三个候选：

1. B01/C04 固定保留为 primary confirmatory candidate；
2. 取 canonical balanced frontier 中按原锁定排序严格优于 B01 的第一名；
3. 取 canonical accuracy frontier 第一名；若与前两项重复则不补名额。

“优于 B01”必须按锁定排序元组严格比较，不能仅仅因为“在 frontier 中且不是 B01”就入选。排序元组保持：worst-dataset delta Q、macro delta Q、paired wins、空间 Pareto、effective runtime、candidate ID。

## 8. diffusion 资源口径修复

Night-5B 的 B17-B20 复用已训练 source embedding，旧 manifest 把 `training_seconds` 与 GPU 记为 0。这只能表示“增量 post-hoc 成本”，不能表示完整方法成本。

Night-5C 不重跑 diffusion，也不修改旧 manifest；另建资源审计表，同时报告：

- `incremental_diffusion_seconds` 与实际 diffusion 内存；
- `source_training_seconds` 与 source peak GPU；
- `effective_end_to_end_seconds = source_training_seconds + incremental_diffusion_seconds`；
- `effective_peak_gpu = max(source_peak_gpu, diffusion_peak_gpu)`。

frontier 资源门只使用 effective end-to-end 口径。必须保留旧 0 值并解释其语义，不得静默改写历史。

## 9. 必须输出的审计材料

至少包含：

- `night5c_report.md`；
- `p0_semantic_contract.json`；
- `corrected_run_order.json`；
- `corrected_training_manifest.json`；
- `supersession_map.json`（24 个旧 invalid ↔ 24 个新 corrected）；
- `recomputed_s1_topup_candidates.json`；
- `candidate_lifecycle_night5c.csv`；
- `per_run_summary_night5c.csv`，含 `validity`、`canonical_eligibility`、source 与 contract SHA；
- `five_seed_summary_night5c.csv`；
- `diffusion_resource_accounting.csv`；
- `selected_for_future_locked_p22.json`；
- `withheld_audit.json`；
- 测试日志、Git/保护哈希、protocol deviations、预算审计与 shutdown 状态。

报告必须明确回答：

1. 24 个新纠错单元是否全部实际使用 `uniform_all`，证据是什么；
2. 原 84 个有效训练单元是否全部原样复用，是否发生重跑/覆盖；
3. 纠正后的 B21-B24 三-seed及可能的五-seed ARI/NMI/Q、空间与资源结果；
4. 重算后的 S1 top-up 集与 Night-5B 旧集合有何差异；
5. B10、B17 以及任何 Laplacian 候选分别处于 canonical 还是 exploratory 层；
6. balanced/accuracy frontiers 及相对 B00、B01 的完整增量；
7. 最多三个 future P22 候选的完整公式、config/source SHA；
8. P22、D1、GSE198353、Night-4B 的访问/运行是否均为 0；
9. 总训练单元、失败、原位重试和总尝试是否符合 40/44 上限；
10. GitHub branch/tag/push、D 盘 compact handoff 与关机命令状态。

## 10. Git、存储与交付

- 普通非 force push；不得改写 Night-5B branch/tag。
- 原始 runs/checkpoints 留在 `/root/autodl-fs` 持久盘；D 盘只回传规划所需 compact handoff、报告、表格、代码、测试、manifest 和增量 Git bundle。
- 不生成或下载 full raw archive，不把重复的历史大文件塞入 D 盘，不向 C 盘写交付。
- 所有 tar/bundle/index 做 SHA-256 与内部条目验证。
- 除非用于判定错误，不制作大批图；本轮没有“图表数量”完成指标。

## 11. 关机操作边界

用户手动负责 AutoDL 开机与最终控制台确认，本任务不开发或调用 AutoDL 开关机 API。实验 Codex 必须：

1. 从任务开始就保留一个可用的远端会话，并设置合理的远端 fail-safe shutdown；
2. 在该会话仍有效时完成 Git push、compact 交付与 SHA；
3. 最后在同一有效会话内执行 `/usr/bin/shutdown`，记录派发时间与 exit status；
4. 派发后不得重新连接；只能如实报告“命令 exit 0”或“无法证明送达”，不能声称控制台已关机。

## 12. 终止状态

只能是：

- `NIGHT5C_CORRECTION_COMPLETE_CANDIDATES_LOCKED_FOR_P22`
- `NIGHT5C_CORRECTION_COMPLETE_C04_ONLY`
- `BLOCKED_PREFLIGHT`
- `CORRECTION_RUN_FAILED`
- `BUDGET_EXHAUSTED`

无论哪一种状态，均不得自行继续 P22、D1、GSE198353、Night-4B 或正式 benchmark。
