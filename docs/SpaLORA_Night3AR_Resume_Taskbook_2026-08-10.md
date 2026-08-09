# SpaLORA Night‑3A‑R 续跑任务书

任务名称：Night‑3A Protocol-corrected Resume — Label-free Loss Calibration  
日期：2026‑08‑10  
输入基线：commit `a1d0c04cea279c07d77f253dc173710929c2cb5a`，tag `night3a-final-20260810`  
前序实现 commit：`fff05e4b7e5ad4ac9c39121aa14ee87445eb4d13`  
执行环境：Codex 经本地终端 SSH 到 AutoDL；交付完成后关机。

## 1. 本次续跑的正式判定

Night‑3A 的 P0B 失败是**协议分类错误，不是科学泄漏，也不是 IGE 数值失败**：

- P0A 已通过；
- 15/15 个 IGE 数值探针全部通过；
- 模型没有解析、导入、使用或依据标签值进行任何训练决策；
- 唯一失败是 `verify_lock()` 为验证数据未漂移而以二进制方式重算两个 ground-truth CSV 的 SHA‑256；
- 文件完整性哈希读取不产生标签语义，也不应与标签驱动训练归为同一类；
- 原 Night‑3A 报告及失败记录必须原样保留，作为审计历史；不得回写成 PASS。

本次不是放宽科学标准，而是把防火墙改为正确的数据流规则：**允许完整性字节读取，禁止训练阶段的标签语义解析、传递、评价和决策使用。**

## 2. 本次唯一科学目标

继续原预注册问题：在 corrected sparse pipeline 中检验无标签 Initial Gradient Equalization（IGE）能否替代旧的标签选择 RNA scale 与数据集特异 gamma。

不改变：

- 数据集：A1、Placenta、P22；
- 变体：C0、C1、IGE、ILN；
- seeds：`0,1,2,3,4`；
- 训练 epoch、optimizer、图、预处理、聚类和指标；
- IGE/ILN 数学公式；
- 原随机 60-cell 运行顺序；
- ASR/rescue 冻结；
- 不搜索 scale、seed、tau、clip 或额外公式；
- 不读取 ARI/NMI 后修改候选方法。

允许修改的只有：

1. 标签防火墙对“完整性读取”和“语义读取”的分类与实现；
2. 把原先错误的 scalar-loss contribution collapse 硬门改为 **weighted-gradient influence collapse** 硬门；
3. 为上述两项增加测试、日志和报告字段；
4. 新建独立输出目录，避免覆盖 Night‑3A 失败证据。

## 3. Git、分支与不可覆盖要求

1. 服务器开机后定位仓库，确认 `night3a-final-20260810` 指向 `a1d0c04...`。
2. 只读检查 worktree；若存在未知用户改动且与本任务冲突，停止并报告，不覆盖、不清理。
3. 从 `night3a-final-20260810` 创建：

   `revision/q2-night3ar-20260810`

4. 创建保护标签：

   `baseline/pre-night3ar-20260810`

5. 最终成功或硬停止均创建：

   `night3ar-final-20260810`

6. `outputs/night3a_handoff/`、`night3a-final-20260810` 及既有 Night‑1/2/2B/2C 文件全部只读保护。
7. 新输出根必须为：

   `outputs/night3ar_handoff/`

8. 新配置、脚本和测试使用 `night3ar` 命名；不要覆盖旧 config、report、P0B JSON。

## 4. 协议修订 A：标签防火墙

### 4.1 三类访问

| 类型 | 示例 | P0A/P0B/训练是否允许 |
|---|---|---|
| 完整性字节读取 | `open(path, "rb")` 后计算 SHA‑256，不解析 CSV | 允许，必须记录 |
| 标识符专用审计 | P0A 仅 `usecols=[Barcode]` 检查 spot 集合与顺序 | 仅 P0A 允许，必须记录 `label_values_read=false` |
| 标签语义访问 | 读取 label column、构造 y、计算 ARI/NMI、根据标签选择权重/seed/参数 | 在 60-run manifest 锁定前绝对禁止 |

### 4.2 代码修订要求

在 `night3ar_p0b.py` 和 `night3ar_runner.py` 中：

1. 先加载 config/lock，并执行 `verify_lock()`；此阶段允许哈希 ground-truth 文件。
2. `verify_lock()` 返回结构化的 `integrity_reads`：路径、预期 SHA、实际 SHA、字节数、用途，不返回内容。
3. 只有 `verify_lock()` 完成后才安装 scientific-window audit hook。
4. audit hook 安装后，才允许导入训练模块、准备输入、创建模型和训练。
5. 在 scientific window 中，ground-truth CSV 的任何 `open` 均为失败；完整性哈希已在窗口开始前完成。
6. 禁止导入 `SpaLORA.night1_evaluation`、`scripts.night3a_evaluate`、新 `night3ar_evaluate`，直到 60-run manifest 写入并 fsync。
7. 给 `pandas.read_csv` 增加 ground-truth 路径守卫：P0B/训练进程中若尝试解析这些 CSV，立即报错。该守卫不得妨碍结果 CSV 的写入。
8. 传入 `Night3ATrainer` 的对象不得包含 ground-truth path、label array 或 label column。
9. `prepare_corrected` 返回后断言：训练用 AnnData/中间对象的 `obs` 不含任何列；输出 manifest 仅含 observation ID，不含 label。
10. 胎盘标签位于原 h5ad 的 `obs[cell_type]`：允许底层 h5ad 作为输入文件被读取，但训练数据流在进入预处理/模型前必须删除全部 `obs` columns；不得访问 `cell_type` 属性、复制其数组或用其参与任何逻辑。
11. evaluator 必须是独立进程；只有 `training_complete.json`、`locked_60_run_manifest.json` 和训练防火墙 PASS 后才能读取标签。

### 4.3 必需产物

- `protocol_amendment.json`：说明修订原因、旧失败、允许/禁止边界；
- `integrity_read_manifest.json`：所有哈希字节读取；
- `scientific_window_label_firewall.json`：窗口内打开的 ground-truth 路径、禁止模块、解析守卫触发次数、semantic label read 状态；
- `label_flow_audit.md`：用简洁数据流说明标签未进入训练。

禁止把旧 Night‑3A 的 `night3a_ige_no_go=true` 解释为科学失败。新报告应写：

`previous_night3a_status = ADMINISTRATIVE_HARD_STOP_SCIENTIFIC_GO_NO_GO_NOT_EVALUATED`

## 5. 协议修订 B：用梯度影响而不是 scalar loss 判定塌缩

IGE 的定义是初始梯度等化。Night‑3A 无标签探针已经显示，IGE 会给 RNA 较大的 scalar 系数；这不等于 RNA 在参数更新中垄断，因为原始 RNA 梯度较小。

因此：

- 继续记录每个 raw loss、weighted scalar contribution 和 scalar fraction；
- 但 scalar fraction 仅作描述，不作为 PASS/FAIL 硬门；
- 硬门改为每项损失的 weighted-gradient influence。

对每个检查点和损失项：

```text
g_k(t) = RMS gradient of raw loss L_k with the same P0B definition
q_k(t) = abs(lambda_k) * g_k(t)
gradient_share_k(t) = q_k(t) / sum_j q_j(t)
```

对 C0、C1、IGE、ILN 全部计算同样字段。检查点保持原任务书的初始、20%、50%、final 等记录契约；若成本允许，使用原定 `{0,1,5,10,20,40,80,final}`。

新增测试：

1. IGE 初始化时四个 `q_k` 在浮点容差内相等；
2. IGE 初始化 `gradient_share` 各约 0.25；
3. 计算诊断不改变 parameter、`.grad`、optimizer 或 RNG state；
4. GPU 使用 Night‑2C/P0B 已验证的 envelope，不要求不现实的位级完全一致。

新的塌缩门：后半程所有预定检查点中，若同一损失项持续 `gradient_share < 0.01`，或任一损失项持续 `gradient_share > 0.90`，则机制门失败。单个检查点越界只记录，不自动失败。

raw/weighted scalar loss 出现 NaN/Inf 仍是数值硬失败。

## 6. P0A‑R：修订后重新锁定

新建 `configs/night3ar_ige_feasibility.json`，除以下字段外与旧 Night‑3A config 一致：

- branch/tag/output/source-lock 文件名更新为 Night‑3A‑R；
- label firewall 采用本任务书的新语义；
- collapse gate 字段从 `contribution_fraction_*` 改为 `weighted_gradient_share_*`；
- 记录旧 config SHA、新 config SHA 和本任务书 SHA。

重新运行 P0A，而不是手工宣告旧 P0A 有效。P0A‑R 必须证明：

- 三个数据集的 RNA/modality2 文件 SHA 与 Night‑3A 一致；
- prepared n_obs、spot order、feature order、PCA、四张图、model input SHA 与旧 P0A 一致；
- 原 60-cell run order 的 dataset/variant/seed/ordinal 完全一致；
- 标签只发生允许的 identifier-only P0A 审计；
- 新 source/config lock 在任何性能指标计算前写入并 fsync。

不一致即 P0A‑R 硬停止。

## 7. P0B‑R：快速重跑与旧探针对照

重新执行 3 数据集 × 5 seed 的 15 个 IGE 探针。不要仅修改旧 JSON。

除新防火墙字段外，必须与旧 `night3a_p0b.json` 对照：

- input SHA、初始 state SHA、m_bad；
- raw initial loss；
- RMS gradients；
- IGE weights；
- repeatability、state/RNG preservation、reload envelope。

CPU/哈希字段应精确一致；GPU 浮点字段按已有 P0B envelope 比较。任何超包络差异硬停止。

P0B‑R PASS 条件：

- 15/15 数值单元通过；
- 完整性读取记录完整；
- scientific window 中无 ground-truth CSV open；
- 无标签 CSV parser 调用；
- 无 evaluator import；
- `semantic_label_values_read=false`；
- 新 lock 全部匹配。

满足后授权原定 60 次主实验。

## 8. 60 次主实验

继续以下完全锁定矩阵：

| 代码 | 训练定义 | 论文用途 |
|---|---|---|
| C0 | corrected sparse + 旧 gamma + RNA scale=1、无旧 shape | 开发基线 |
| C1 | corrected sparse + 旧 gamma + 锁定 m_bad、无旧 shape | 阳性机制对照，不能成为最终方法 |
| IGE | corrected sparse + 四 raw losses 的冻结 IGE，无旧 gamma/m_bad/shape | 预注册候选 |
| ILN | corrected sparse + 初始 raw-loss normalization 后等权 | 诊断对照 |

总数：`4 × 3 × 5 = 60`。按旧 preregistered order 执行。

运行要求：

- 支持仅对 SHA 完整且 manifest 匹配的 run 安全 resume；
- partial/mismatched run 不覆盖，立即失败；
- 每次保存 embedding、attention、cluster、model state、loss/gradient trajectory、资源、config 和 artifact hashes；
- 运行完成后先写入并 fsync `locked_60_run_manifest.json`，再启动 evaluator；
- 训练进程不得计算 ARI/NMI 或读取 label values。

## 9. 评估与科学 go/no-go

评估指标与旧任务书保持不变：ARI、NMI、AMI、FMI、homogeneity、V-measure、Hungarian F1/balanced accuracy、spatial neighbor agreement、cluster Moran’s I、silhouette、Davies–Bouldin、时间和内存。

报告每 seed、mean ± SD、median、range、bootstrap CI、配对差和正/负 seed 数；禁止 best-seed 主表。

IGE 相对 C0 的通过条件维持：

1. 满足以下之一：
   - 至少 2/3 数据集的 ARI 与 NMI 配对均值为正，且至少一个数据集 ARI `≥ +0.03`；
   - 胎盘恢复 C1−C0 ARI 增益的至少 60%，且 A1、P22 的 ARI 变化均 `≥ −0.02`。
2. 任一数据集不得同时出现 spatial neighbor agreement 和 Moran’s I 均下降超过 `0.03`。
3. 后半程 weighted-gradient share 不得持续 `<0.01` 或 `>0.90`。
4. attention 不得在后半程全部检查点持续 `<0.05` 或 `>0.95`。
5. 60/60 完成、失败 JSON 为 0、无标签泄漏、无 NaN/Inf、保护文件全部匹配。

科学门失败时，不更改公式、不搜索新 scale/seed，也不自动把 ILN 晋升为最终方法。完整交付后由规划者根据 IGE、ILN 和 C1/C0 结果决定下一方案。

## 10. 结果呈现原则

报告阶段保留完整证据；论文阶段允许以投稿目标为导向进行有重点的组织：

- 主文突出与最终 claim 直接相关、稳定且最有解释力的结果；
- 次要结果、完整逐 seed、扩展指标放 Supplement；
- 已放弃且不构成最终方法的内部探索（例如 ASR rescue）无需成为新论文主线；
- 若某数据集结果中性或轻微下降，可以缩小 claim、解释适用范围，但不能在声称“跨数据集普遍优越”时把该数据集隐去；
- 不允许为本方法单独改变 evaluator、挑 seed、用标签调权重、只报最好结果或删除预注册的相反结果。

目标是最大化真实工作的叙事价值，而不是制造与代码/数据不一致的结论。

## 11. 必需交付物

目录：`outputs/night3ar_handoff/`

- `protocol_amendment.json`
- `integrity_read_manifest.json`
- `label_flow_audit.md`
- `night3ar_p0a.json`
- `night3ar_p0b.json`
- `scientific_window_label_firewall.json`
- `night3ar_gate_status.json`
- `night3ar_completion.json`
- `night3ar_report.md`
- `per_seed_metrics.csv`
- `summary.csv`
- `paired_deltas.csv`
- `loss_trajectories.csv`
- `gradient_influence_trajectories.csv`
- `ige_initial_gradients.csv`
- `ige_weights.csv`
- `attention_summary.csv`
- `resource_usage.csv`
- `locked_60_run_manifest.json`
- 全部图及生成脚本
- `failure_index.json`
- `SHA256SUMS`

报告首页必须明确：

1. P0A‑R/P0B‑R 是否通过；
2. 60/60 是否完成；
3. 这次是否发生任何 semantic label access；
4. IGE scientific go/no-go 为 PASS 还是 FAIL；
5. 三数据集 IGE−C0 与 C1−C0 的五 seed ARI/NMI；
6. 胎盘收益恢复比例；
7. 空间指标 tradeoff；
8. scalar loss 与 gradient influence 的区别；
9. 是否授权规划下一步 architecture ablation。

## 12. Git、归档、推送与关机

1. 测试、报告和结果完成后提交 Git 并创建 `night3ar-final-20260810`。
2. 生成 Git bundle，执行 verify，并在临时目录 clone/checkout 最终 tag。
3. 生成完整 archive；验证内部 manifest、成员路径和 SHA‑256。
4. bundle/archive/report/CSV/SHA256SUMS 下载到本地，双端校验。
5. GitHub 仅尝试 push 一次；无凭据则记录，禁止重复尝试或在日志中暴露 token。
6. 验证 Night‑3A 和 Night‑2C 等全部保护文件未变化。
7. 生成 shutdown checklist/confirmation。
8. `/usr/bin/shutdown` 必须作为最后一条远端命令发出；SSH 断开后不得重新连接。

## 13. Codex 最终回报模板

```text
已严格完成 SpaLORA Night‑3A‑R 协议修订与续跑。
P0A‑R：PASS/FAIL；P0B‑R：PASS/FAIL。
旧 15-cell 探针对照：x/15 在包络内。
完整性字节读取：已记录且仅用于 SHA；semantic label access：0/存在。
主实验：x/60；失败 JSON：x。
IGE scientific go/no-go：PASS/FAIL/NOT_EVALUATED。
IGE−C0 五 seed ARI/NMI：A1 ...；Placenta ...；P22 ...。
胎盘 C1 增益恢复比例：...。
空间门、gradient-share 门、attention 门：...。
测试：x passed，x failed。
保护文件：x/x 一致。
最终 commit：...；tag：night3ar-final-20260810。
GitHub push：成功/失败（仅一次）。
bundle/archive 双端校验：...。
/usr/bin/shutdown 已作为最后一条远端命令发出；之后未重新连接。
本地交付物路径：...。
```
