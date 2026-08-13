# SpaLORA Night-5D 锁定 P22 确认实验任务书

日期：2026-08-14  
权威父提交：`f8bba65ff3f1a3dacf310dd12f81f8b379315b22`  
权威父标签：`night5c-final-20260814`  
锁定注册表：`SpaLORA_Night5D_Locked_P22_Registry_2026-08-14.json`  
注册表 SHA-256：`59d034fcf371816e24b22477bb757a104c931682e0b60a3d177a0f40e0d94721`

## 0. 唯一目标

在不改变候选、参数、seed、训练协议或统计门的前提下，对 Night-5C 已锁定的三个候选进行 P22 数值确认：

- B01/C04：保守 confirmatory anchor；
- B10：预先指定的主 integrated balanced method；
- B17：预先指定的 accuracy-enhanced diffusion variant。

本轮不再开发方法，不做新候选搜索，不运行 D1、GSE198353、Night-4B 或正式外部 benchmark，不以作图数量作为完成指标。所有训练与无标签变换完成并冻结前，不得读取 P22 语义标签。

## 1. P22 的证据地位必须如实表述

P22 不是整个项目从未使用过的 pristine test set：Night-3B 已在 P22 上做过 FULL_IGE 架构消融，规划过程中也知道 P22 的历史基线与部分 attention 现象。因此最终论文不能称其为“从未查看的独立外部验证”。

Night-5A、Night-5B、Night-5C 对 C04/B10/B17 的定量训练、晋级、排序和锁定只使用 A1 与 Placenta；这三轮对 P22 候选运行和结果访问均为 0。但是候选假设的形成可能继承了 Night-3B 的 P22 架构认识，因此不能把 P22 严格称为 untouched holdout。Night-5D 的合理定位是：

> 在一个未参与 Night-5 候选数值筛选、但存在更早架构分析历史的数据集上，对预先锁定候选进行一次性 cross-dataset confirmation。

本轮一旦打开 P22 标签，P22 永久失去未来调参/选择用途。无论结果好坏，不得依据 P22 返回修改 rho、anchor eta、diffusion alpha、训练轮数、loss 权重、seed 集或候选组合。

## 2. Night-5C 权威输入与保护门

执行前逐项验证：

1. GitHub branch `revision/q2-night5c-laplacian-correction-20260814` 与 tag `night5c-final-20260814` 均 peel 到权威父提交。
2. D 盘 Night-5C 交付根：
   `/D:/文档/ChatGPT/博士第一篇科研论文项目/night5c_handoff_20260814`
3. 下列 SHA-256 一致：
   - `night5c_report.md`：`9daeba69721e37fbbb66628fc54d84b02efbb3f39e3944693844af1a2e2e038b`
   - `p0_semantic_contract.json`：`a15ec5697ae10ca1817bd89003531f9768ee3d907b6eda1c60cc311b8c6067ca`
   - `five_seed_summary_night5c.csv`：`9f634f6c06eaf880823790009a32411d151d474167574f33f5d3748892ba7005`
   - `selected_for_future_locked_p22.json`：`f818ae2309e2faaaa90888853162765f10f0740c3ec80bbd6bfd9a3d727859a4`
   - `delivery_index.json`：`af4867707ea9f51c3f9a71d321963acc27fc7a4c371e3ff62e17c5fd659ea7b7`
   - Night-5C 增量 bundle：`f5a9e3e867e45067df02eb005722486b582608c03515d9a73f150b1d57b306d0`
4. D 盘 delivery index 49/49、Night-3B 1186/1186、Night-4A 76/76 均通过。
5. `selected_for_future_locked_p22.json` 必须精确且仅含 B01、B10、B17，config SHA 与注册表一致。
6. Night-5A/B/C raw 根中不得存在任何既有 P22 候选输出；若存在，停止调查并报告，不能覆盖或忽略。
7. Night-3B P22/FULL_IGE seeds 0-4 的所有 manifest、embedding、clusters、attention、model state 与内部 SHA 必须重新验证，但标签窗口开启前只允许字节完整性读取。
8. Night-3AF 的 P22 deterministic cache、输入 h5ad、坐标、稀疏邻接、HVG 与 observation IDs 必须与原 manifest/SHA 一致；禁止重新预处理。
9. R/mclust、GPU/CUDA、环境锁、Git 父提交或标签防火墙任一失败，终止为 `BLOCKED_PREFLIGHT`。

建立隔离 worktree。建议：

- branch：`revision/q2-night5d-locked-p22-confirmation-20260814`
- protection tag：`baseline/pre-night5d-20260814`
- final tag：`night5d-final-20260814`

普通非 force push，不得改写 Night-5C 及更早历史。

## 3. P0-CONFIRM：公式与运行时语义

在任何科学训练前生成 `p0_confirmatory_contract.json`，至少验证：

### B00/C00 FULL_IGE

- 与 Night-3B `FULL_IGE` 使用完全相同的 trainer、learned attention、Corr2、IGE 与 P22 dataset contract；
- seeds 0-4 只复用并重新验 SHA；seeds 5-9 使用同一实现新增训练；
- 当前代码对 seeds 0-4 的初始状态/forward/loss/active coefficients 做只读 parity probe，不重训、不覆盖历史结果。

### B01/C04 SHRINK25

- `attention=shrink_to_uniform`；
- `learned_fraction_rho=0.25`；
- Corr2 removed；active-set IGE；
- 不含 RNA anchor、triplet、diffusion 或其他候选机制。

### B10 SHRINK25 + RNA anchor

- 与 B01 相同的 shrink25、Corr2 removal 与 active-set IGE；
- 唯一新增机制为 `rna_anchor_eta=1.0`；
- 实际图边权、anchor intersection、coefficient probe 与注册表一致。

### B17 C09 + diffusion0.10

- source C09 实际为 `uniform_all`、Corr2 removed、`rna_anchor_eta=1.0`、active-set IGE；
- 即使模型对象保留未使用的 attention 参数，也必须证明 forward 融合权重精确 `[0.5,0.5]`，attention 参数对输出无影响且梯度为零/None；不得把“参数对象存在”误写成 learned attention 生效；
- B17 必须逐 seed 复用对应 C09 frozen embedding，且 source manifest/config/embedding SHA 完整记录；
- 只允许一次 sparse diffusion：`alpha=0.10`、一步、指定的 immutable normalized P22 spatial adjacency；
- alpha=0 必须与 source embedding bitwise 相等，alpha=0.10 的独立参考实现与正式实现在容差内一致；
- diffusion 必须发生在标签窗口打开前。

所有正式 run/transform manifest 同时保存 declared contract、resolved runtime contract、二者 SHA 与 `semantic_contract_match=true`。任何 mismatch 都 fail closed；不得通过修改注册表迎合代码。

## 4. 为什么固定为 10 seeds

本轮固定 seeds 0-9，全部在评价前锁定。使用 10 seeds 的目的，是为三候选同时确认提供足够的精确配对检验分辨率和更稳定的不确定性区间，而不是 seed 搜索。

- 不得先看 seeds 0-4 再决定是否跑 5-9；
- 不得因中间 loss、embedding 外观或无标签 proxy 改变 seed 集；
- 不得早停整个实验；
- 任一成功 seed 均不得删除或重跑以换取更好值；
- 硬件/进程失败只允许原 tuple 重试并保留失败记录。

## 5. 预注册运行矩阵

固定顺序：

1. B00/C00：只训练 seeds 5-9；seeds 0-4 复用 Night-3B；
2. B01/C04：训练 seeds 0-9；
3. B10：训练 seeds 0-9；
4. C09 source：训练 seeds 0-9；
5. B17：对 C09 seeds 0-9 逐 seed 做固定单步 diffusion，不进行训练。

科学训练单元：`5 + 10 + 10 + 10 = 35`。  
无训练 deterministic transform：10。  
同 tuple 基础设施重试最多 4 次，因此训练尝试硬上限 39；diffusion transform 最多允许 2 次原位基础设施重试。未使用预算不得拿来增加 seed、候选、alpha 或其他实验。

建议新的远端输出根：

`/root/autodl-fs/night5d_p22_runs_20260814`

禁止覆盖 Night-3B 或 Night-5A/B/C 目录。

## 6. 标签防火墙与冻结顺序

训练阶段禁止导入或调用任何 ground-truth loader、ARI/NMI evaluator、标签列、真值路径解析器。允许在 P0 只核验 ground-truth 文件的 SHA/size，不允许解析内容。

打开标签前必须完成并冻结：

1. 35 个训练单元的 embedding、checkpoint、attention、clusters、loss trajectory、coefficient probe、runtime contract、run manifest 与 SHA；
2. 10 个 B17 diffusion embedding、clusters、source link、transform manifest 与 SHA；
3. 5 个历史 B00 与 5 个新增 B00 的统一十-seed baseline manifest；
4. 固定 mclust EEE、PCA 20、random seed 2020 的所有 cluster assignments；
5. `locked_p22_training_and_transform_manifest.json`；
6. scientific-window firewall 审计，证明语义标签访问为 0。

只有上述总锁通过后，才允许一次性读取 P22 标签并完成全部评价。不得按标签选择 checkpoint、重聚类参数、删 seed 或回到训练。

## 7. 评价指标与历史基线复核

逐 seed 统一重算并报告：

- ARI、NMI、`Q=(ARI+NMI)/2`；
- spatial neighbor agreement；
- one-vs-rest Moran mean；
- one-vs-rest Geary mean；
- boundary disagreement；
- end-to-end runtime、peak GPU、RSS；
- candidate/source/config/embedding/cluster manifest SHA。

历史 B00 seeds 0-4 的重算结果必须与锁定 Night-3B 结果在 `1e-12` 绝对容差内一致；否则停止为 `BASELINE_REPLAY_MISMATCH`，保留诊断，不得偷偷改用更有利的一版。

历史 B00 五-seed均值仅作校验参考：ARI `0.3951973766`、NMI `0.5531134690`、Q `0.4741554228`。正式 Night-5D 推断使用完整 seeds 0-9 的 paired baseline。

## 8. 预注册统计规则

### 8.1 三个 primary contrasts

分别比较 B01、B10、B17 与同 seed B00。主终点为 paired `delta Q`。

每个 contrast：

1. 枚举全部 `2^10=1024` 个符号翻转，以 mean delta Q 为统计量、备择方向为 gain；`p = count(T_permuted >= T_observed) / 1024`。因为是完整枚举，不额外加一；
2. 三个原始 p-value 使用 Holm step-down 控制 family-wise alpha=0.05；
3. 固定 seed `20260814`，对 10 个 paired deltas 做 100,000 次 paired bootstrap，给出 mean delta ARI/NMI/Q 的 percentile 95% CI；
4. 报告逐 seed delta、mean、median、sample SD、CI、Holm-adjusted p 与 positive-Q win count。

不得在结果后改用另一种统计检验作为主要结论。可以追加透明的诊断统计，但不能替换预注册结果。

### 8.2 数值确认门

候选满足以下全部条件，才标记 `CONFIRMED_MATERIAL_ACCURACY_GAIN`：

- mean delta ARI > 0；
- mean delta NMI > 0；
- mean delta Q >= +0.01；
- paired Q positive 至少 7/10；
- Holm-adjusted one-sided exact p < 0.05；
- bootstrap 95% CI 的 delta Q 下界 > 0。

若方向为正但未全部过门，标记 `POSITIVE_BUT_INCONCLUSIVE`；若 Q 非正或 ARI/NMI 方向不一致，标记 `NOT_CONFIRMED`。所有实际数值照实报告，不能只给状态。

### 8.3 空间保护与两类确认

沿用既定保护门。相对 B00，若：

`(mean Δneighbor < -0.03 and mean ΔMoran < -0.03)`

或

`(mean ΔGeary > +0.03 and (mean Δneighbor < -0.03 or mean ΔMoran < -0.03))`

则 spatial protection failed。

- 通过数值确认门且空间门通过：`CONFIRMED_BALANCED_GAIN`；
- 通过数值确认门但空间门失败：`CONFIRMED_ACCURACY_WITH_SPATIAL_TRADEOFF`。

空间门失败不能被隐藏，也不应把明确的 accuracy gain 改写成完全失败。

### 8.4 两个 secondary mechanistic contrasts

- B10 − B01：隔离 RNA-anchor 在 shrink25 base 上的增量；
- B17 − C09：隔离单步 diffusion 的增量。

两项使用相同 exact sign-flip、bootstrap 与 Holm（仅在这两个 secondary contrasts 内校正），但不改变三项 primary confirmation，也不用于事后选择新的主模型。

## 9. 解释边界与主方法角色

P22 前已冻结角色：

- B10 是主 integrated balanced method；
- B17 是预指定 accuracy-enhanced optional variant；
- B01 是保守 confirmatory anchor。

不得在看到 P22 后按最高均值重新发明角色。三者均可得到独立、预注册的确认结论；论文可并列报告主方法与可选 refinement，但不能把 P22 当新开发集继续择优。

若结果失败或混合：保留并如实报告。下一步只能转向真正外部数据集验证、基线工程或新的独立研究周期；P22 不再可用于调整当前候选。

## 10. 必须输出

至少包含：

- `night5d_report.md`；
- `p0_confirmatory_contract.json`；
- `locked_p22_run_order.json`；
- `locked_p22_training_and_transform_manifest.json`；
- `baseline_reuse_and_extension_audit.json`；
- `label_firewall.json`；
- `p22_per_seed_metrics.csv`；
- `p22_candidate_summary.csv`；
- `p22_primary_exact_tests.json`；
- `p22_secondary_mechanistic_tests.json`；
- `p22_spatial_protection.json`；
- `p22_resource_accounting.csv`；
- `p22_confirmatory_decision.json`；
- 测试日志、预算/失败/重试审计、Git 审计、delivery index 与 shutdown 状态。

最终报告必须回答：

1. 35 个训练单元与 10 个 diffusion 是否全部按锁定顺序完成；
2. B00 seeds 0-4 是否无重跑复用，seeds 5-9 是否使用同定义实现；
3. 训练/变换完成前是否发生任何 P22 语义标签访问；
4. B01/B10/B17 各自逐 seed和十-seed ARI/NMI/Q、空间、资源结果；
5. 三个 primary contrast 的 exact p、Holm p、CI、wins 与确认状态；
6. B10−B01、B17−C09 是否支持各自新增机制；
7. 是否存在准确率—空间连续性 trade-off；
8. P22 既往 Night-3B 使用历史如何限制“独立验证”措辞；
9. D1、GSE198353、Night-4B、外部 benchmark 是否均未运行；
10. GitHub branch/tag/push、D 盘 compact handoff 与最后关机命令状态。

## 11. Git、D 盘交付与关机

- raw runs/checkpoints 留在 `/root/autodl-fs`；
- D 盘只交付报告、表格、统计 JSON、代码、测试、manifest、必要诊断图和 Night-5C→Night-5D 增量 bundle；
- 不生成/下载 full raw archive，不向 C 盘写交付，不重复打包历史大文件；
- delivery index 独立列 SHA 并在 D 盘逐项复核；
- 图只服务于诊断，不设数量要求。

用户手动负责 AutoDL 开机和控制台关机确认。本任务不开发或调用 AutoDL API。实验开始保留一个可用 SSH 会话并设置 fail-safe；完成 Git push、compact 回传和 SHA 后，在同一有效会话把 `/usr/bin/shutdown` 作为最后一条远端命令。记录 exit status，随后不重连，不虚报控制台状态。

## 12. 终止状态

只能是：

- `P22_CONFIRMATION_SUCCESS`
- `P22_PARTIAL_OR_MIXED_EVIDENCE`
- `P22_NO_LOCKED_CANDIDATE_CONFIRMED`
- `BASELINE_REPLAY_MISMATCH`
- `BLOCKED_PREFLIGHT`
- `BUDGET_EXHAUSTED`

无论何种终态，均不得自行继续 D1、GSE198353、Night-4B、正式外部 benchmark 或新的 P22 调参。
