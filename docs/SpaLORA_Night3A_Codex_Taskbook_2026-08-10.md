# SpaLORA Night‑3A Codex 任务书

任务名称：Label-free Loss Calibration Feasibility  
日期：2026‑08‑10  
输入基线：Night‑2C commit `49764b6d1b9d16510126a879e58f38973b562ce8`，tag `night2c-final-20260809`  
执行环境：Codex 通过本地终端 SSH 到 AutoDL；完成后关机。  
本任务的唯一科学目标：在 corrected sparse pipeline 中，一次性检验预注册的 Initial Gradient Equalization（IGE）能否替代旧的标签选择 scale 和数据集特异 gamma。

---

## 0. 不可违反的规则

1. 不读取 ground-truth 标签做训练、超参数选择、早停、图构建、loss 权重、seed 选择或变体选择。标签只能在 config/代码/哈希锁定后用于最终评估。
2. 不修改、删除或覆盖 Night‑1、Night‑2、Night‑2B、Night‑2C 的报告、结果、标签、bundle、archive 或受保护文件。
3. 不修改 ASR，不恢复 ASR rescue，不搜索 `m_bad`，不试额外 scale，不搜索 seed。
4. 不放宽本任务写死的数值门限。硬门未通过时立即停止后续大实验，生成失败报告和交付物，然后关机。
5. 不把胎盘第二模态写成原始 ATAC counts；统一命名 `ATAC-derived / TF-associated regulatory features`。
6. 不把旧 manuscript/tutorial 数字当作 Night‑3A 目标；不以接近旧数字为调参标准。
7. 所有变体必须共享完全相同的输入、预处理、图、初始化、optimizer、epoch、聚类和评估代码；仅允许任务书定义的 loss calibration 发生差异。
8. 任何额外诊断都必须先写入 config/manifest，不能看完标签指标后临时增加“有利”实验。
9. 最后一个远端动作必须是 `/usr/bin/shutdown`。发出后不得重新连接确认。

## 1. Git 与恢复点

1. 在服务器定位 SpaLORA 仓库，确认当前 Night‑2C commit/tag 可解析。
2. 对工作树、未跟踪文件、当前分支、remote、磁盘空间、GPU、Python/CUDA 环境生成只读审计。
3. 若工作树有用户改动，先保存状态和 SHA，不覆盖、不 stash、不清理；若与本任务冲突，停止并报告。
4. 从 `night2c-final-20260809` 新建分支：

   `revision/q2-night3a-20260810`

5. 新建保护标签（若不存在）：

   `baseline/pre-night3a-20260810`

6. 生成 Night‑2C 受保护文件 SHA‑256 清单，并在结束时逐项验证。
7. 所有实现、测试、config、报告和汇总表都提交 Git；禁止 amend/force-push。

## 2. 输入与数据审计（P0A）

只使用现有三套数据：A1、Placenta、P22。

对每套数据记录：

- 绝对路径、文件大小、SHA‑256；
- `n_obs × n_vars`、dtype、稀疏格式、非零数、NaN/Inf、负数；
- spot/barcode 顺序 SHA；
- feature name 顺序 SHA；
- spatial coordinate 顺序 SHA；
- ground-truth CSV 的 spot 顺序与集合一致性；
- `.raw`、`layers` 和 counts 可用性；
- corrected sparse preprocessing 的 PCA、spatial graph、feature graph 摘要和哈希。

要求：

- 与 Night‑1 corrected pipeline 的输入/预处理契约一致；
- 标签对象在训练模块中不可访问，最好通过独立 evaluation 进程读取；
- 预先生成并锁定 `configs/night3a_ige_feasibility.json`，写入所有数据、seed、epoch、变体、门限和输出路径；
- 对 config、runner、model 文件和数据 manifest 生成 SHA‑256；P0A 通过后不得更改。若必须更改，作废本次运行并重新开始 P0A，不得沿用已看过的标签结果。

P0A 失败：不要运行任何 60 次主实验；生成 `night3a_p0a_failure.json` 和报告，完成保护校验、Git 提交、bundle/archive，然后关机。

## 3. IGE 的精确定义

### 3.1 原始损失

必须从模型中暴露四个**未乘旧 gamma、未乘 m_bad、未应用旧 gene-weight shape**的 scalar loss：

- `L_rna_recon_raw`
- `L_mod2_recon_raw`
- `L_corr1_raw`
- `L_corr2_raw`

保持原有 loss reduction 语义，不为获得更好结果临时改 mean/sum。把 reduction、shape 和公式写入审计报告。

### 3.2 RMS 梯度

在模型正常初始化、第一次 optimizer step 之前，用同一 forward 分别计算每个原始损失对所有实际参与该损失的 `requires_grad` 参数的梯度：

```text
sq_sum_k  = Σ_p Σ_i grad(L_k, p_i)^2
n_elem_k  = Σ_p number_of_elements(p) for non-None gradients
g_k       = sqrt(sq_sum_k / (n_elem_k + eps))
```

实现要求：

- 使用 `torch.autograd.grad(..., allow_unused=True)`；
- 每个 loss 独立计算，不允许梯度累积污染；
- 记录每个参数张量是否有梯度、numel、L2 norm、RMS；
- `eps=1e-12`，写死在 config；
- 不执行 optimizer step；诊断后重新从同一初始 state 开始正式训练；
- 对同一初始 state 重复两次，CPU 必须精确一致，GPU 必须在 Night‑2C 已验证的数值包络思想下稳定。

### 3.3 权重

令 `K=4`：

```text
geo = exp(mean(log(g_k + eps)))
lambda_k_raw = geo / (g_k + eps)
lambda_k = K * lambda_k_raw / sum(lambda_k_raw)
L_total_IGE = Σ_k lambda_k * L_k_raw
```

`lambda_k` 在整次训练中冻结，不进入 optimizer。不得 clip；如果权重超门限，触发硬停止，而不是静默截断。

### 3.4 IGE 硬门（P0B）

三数据集 × 五 seed 全部初始化探针必须满足：

- 所有 `g_k`、`lambda_k` 有限且严格大于 0；
- `1e-3 ≤ lambda_k ≤ 1e3`；
- 权重归一化和为 4，绝对误差 `≤1e-6`；
- 重新装载初始 state 后，第一步正式 forward 与探针前的 forward 满足 Night‑2C 数值包络；
- 训练模块的 import/read trace 中没有 ground-truth 文件；
- IGE 与参考变体初始参数、输入、图、batch/spot 顺序一致。

任一单元失败即 P0B 硬停止，不进入 60 次主实验。

## 4. 四个预注册变体

全部使用 corrected sparse preprocessing、相同模型架构、Adam `lr=1e-4`、`weight_decay=0`、Night‑2C 相同 epoch 和固定 seeds `{0,1,2,3,4}`。

| 代码 | 含义 | 是否允许成为最终方法 |
|---|---|---|
| C0 | corrected sparse + 旧 dataset gamma + RNA global scale=1 + 无旧 shape | 否，开发基线 |
| C1 | corrected sparse + 旧 dataset gamma + 锁定的旧 global scale/m_bad + 无旧 shape | 否，仅阳性机制对照 |
| IGE | corrected sparse + 四个 raw loss 的 IGE；无旧 gamma、无 m_bad、无旧 shape | 是，本次预注册候选 |
| ILN | corrected sparse + `L_k/(L_k_initial+eps)` 后等权；无旧 gamma/m_bad/shape | 否，仅诊断对照 |

ILN 的 `L_k_initial` 必须是正式训练初始 state 上、第一次 optimizer step 前的 raw loss，并在全程冻结。

实验总数：`4 variants × 3 datasets × 5 seeds = 60`。不得增加变体。运行顺序由固定随机排列生成，并在读标签前写入 manifest。

## 5. 必需测试

在主实验前新增并通过：

1. raw loss 与 legacy weighted loss 的代数重构测试；
2. IGE 权重公式单元测试；
3. unused parameter 与零梯度处理测试；
4. 探针不改变 parameter/optimizer/RNG state；
5. IGE 权重冻结测试；
6. label firewall 测试；
7. 四变体仅 loss calibration 不同的 config diff 测试；
8. corrected sparse adjacency 全链路不 densify 的测试；
9. A1/Placenta/P22 spot、feature、graph 顺序测试；
10. CPU 重复 forward/loss/gradient/update 等价；
11. GPU envelope；
12. evaluator 指标与 Night‑2C evaluator 的回归测试；
13. failure JSON、resume、manifest、hash、report schema 测试；
14. 受保护文件不可写测试。

若测试失败，不运行主实验。

## 6. 训练期记录

每次 run 至少记录 epoch `{0,1,5,10,20,40,80,final}`（若总 epoch 不同，保留相同相对检查点）：

- 四个 raw loss；
- 四个最终 contribution；
- contribution fraction；
- IGE/ILN/legacy 权重；
- cross‑omics RNA/modality2 attention；
- 两模态 within‑modality spatial/feature attention；
- gradient norm（至少初始、20%、50%、final）；
- embedding norm、NaN/Inf；
- wall time、GPU allocated/reserved peak、CPU RSS；
- checkpoint SHA。

保存 final embedding、cluster、attention 和完整 run config；不要只保存汇总 CSV。

## 7. 评估与统计

只有 60 次训练全部完成、run manifest 锁定后，独立 evaluation 进程才可读取标签。

每个 seed 计算：

- ARI、NMI、AMI、FMI、homogeneity、V-measure；
- Hungarian macro/weighted F1、balanced accuracy；
- spatial neighbor agreement、cluster Moran’s I；
- embedding silhouette、Davies–Bouldin；
- runtime、GPU peak、CPU peak。

汇总：

- mean、SD、median、min、max、95% bootstrap CI（seed 数只有 5，CI 仅描述，不做夸张显著性声明）；
- IGE−C0、C1−C0、ILN−C0 的逐 seed 配对差；
- 每个数据集的正/负 seed 个数；
- 标签指标与空间指标的 tradeoff 图；
- 初始 IGE 权重与最终 attention/性能的描述性相关，明确标注非因果。

不做多 seed 中“最好 seed”展示。空间图必须用预先指定 seed 0 和一个按 ARI 距离均值最近的代表 seed；代表 seed 的选择规则写入 config，且主表仍报告全部 seed。

## 8. 预注册 go/no-go

### 数值硬门

- 60/60 运行完成；失败 JSON 为 0；
- 无 NaN/Inf/权重退化；
- 受保护文件全匹配；
- 标签防火墙无违规。

### 科学门

IGE 相对 C0 必须满足：

1. 至少 2/3 数据集 ARI/NMI 不劣；并满足下列之一：
   - 至少 2/3 数据集的 ARI、NMI 配对均值均为正，至少一个数据集 ARI `≥+0.03`；
   - 胎盘恢复 C1−C0 ARI 增益的至少 60%，且 A1、P22 的 ARI 均 `≥−0.02`。
2. 任一数据集不得同时出现 spatial neighbor agreement 与 Moran’s I 均下降超过 `0.03`。
3. 后半程任何一个 loss contribution 不得持续 `<1%` 或 `>90%`。
4. cross-omics/within-modality attention 不得持续饱和 `<0.05` 或 `>0.95`。

输出必须同时给出严格判定与完整数值；不得把失败包装成通过。

如果 IGE 失败：

- 不改公式、不试新 tau、不 clip、不搜索 scale；
- 报告失败原因；
- 标记 `night3a_ige_no_go=true`；
- 仍完成交付、Git、归档和关机；
- 下一步由规划者决定是否简化模型或终止 Q2 主线。

## 9. 必需输出

目录建议：`outputs/night3a_handoff/`

- `night3a_report.md`
- `night3a_completion.json`
- `night3a_environment.json`
- `night3a_p0a.json`
- `night3a_p0b.json`
- `night3a_gate_status.json`
- `config_lock.json`
- `data_manifest.csv`
- `ige_initial_gradients.csv`
- `ige_weights.csv`
- `per_seed_metrics.csv`
- `summary.csv`
- `paired_deltas.csv`
- `loss_trajectories.csv`
- `attention_summary.csv`
- `resource_usage.csv`
- `failure_index.json`
- 所有 figure PDF/PNG 与生成脚本
- 完整 run manifests/checkpoint index
- `SHA256SUMS`

主报告必须回答：

1. IGE 是否严格通过 go/no-go？
2. IGE 是否在胎盘复现 C1 的主要收益？比例是多少？
3. A1/P22 是否受损？逐 seed 如何？
4. 空间连续性是否仍下降？是否超过门限？
5. IGE 学到的四个权重分别是多少、跨 seed 是否稳定？
6. loss contribution 与 attention 如何变化？
7. ILN 只作为诊断时表现如何？
8. 是否有任何标签泄漏、数值异常、受保护文件变化或协议偏离？
9. 下一步是否允许进入 architecture ablation？

## 10. 提交、归档和本地下载

1. 所有测试通过、报告完成后提交 Git；创建标签：

   `night3a-final-20260810`

2. 生成可恢复 Git bundle，验证 `git bundle verify` 并在临时目录 clone/checkout 最终 tag。
3. 生成完整 artifacts archive；内部 manifest 和外部 SHA‑256 都要验证。
4. 把 report、CSV、bundle、archive、SHA256SUMS 下载到本地项目目录；本地再次校验 SHA‑256。
5. GitHub 只允许尝试推送一次。若没有凭据，记录失败，不交互式索要用户名/token，不重复尝试。
6. 生成 shutdown checklist，至少验证：

   - 报告/JSON/CSV 存在且 schema 通过；
   - bundle/archive 双端 SHA 一致；
   - 最终 commit/tag 可解析；
   - 受保护文件匹配；
   - 本地下载完成；
   - 没有未记录失败。

## 11. 强制关机

只有全部交付和下载完成后：

1. 写入最终 shutdown confirmation；
2. 确认没有后台实验、传输或校验；
3. 将 `/usr/bin/shutdown` 作为最后一条远端命令发出；
4. SSH 断开后不得重新连接验证；
5. 本地最终回复必须明确说明关机命令是否发出、SSH 返回状态、是否未重连。

---

## 给 Codex 的最终回报模板

```text
已严格执行 SpaLORA Night‑3A 任务书。
P0A：通过/失败；P0B：通过/失败。
主实验：x/60 完成；失败 JSON：x。
IGE go/no-go：PASS/FAIL，具体触发条件：...
IGE 相对 C0 的五 seed ARI/NMI：
- A1: ...
- Placenta: ...
- P22: ...
空间指标门：...
IGE 四损失权重摘要：...
测试：x passed, x failed。
受保护文件：x/x 一致。
最终 commit：...
tag：night3a-final-20260810
GitHub push：成功/失败（仅一次）。
bundle/archive：双端校验结果 ...
/usr/bin/shutdown：已作为最后一条远端命令发出；之后未重新连接。
本地交付物路径：...
```
