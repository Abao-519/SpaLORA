# SpaLORA Night‑3A‑F：确定性 PCA 与主实验任务书

任务名称：Deterministic PCA Freeze and Label-free Loss Calibration Run  
日期：2026‑08‑10  
输入基线：commit `e8ca974d153b37ad4a903e119de75ea7321fafe6`，tag `night3ar-final-20260810`  
执行环境：Codex 通过本地终端 SSH 到 AutoDL；全部交付后关机。

## 1. 本轮决策

Night‑3A‑R 的停止揭示了真实的工程问题，但没有产生任何 IGE 科学失败：

- 原始数据、spot/HVG 顺序、模态 2 特征、空间图与 60-cell run order 均一致；
- 只有 RNA PCA、由其生成的 RNA feature graph、以及包含该图的 model-input SHA 发生变化；
- 旧 `PCA(n_components=n_comps)` 未指定 `random_state` 或 solver，sklearn 在三套 RNA 矩阵上自动采用 randomized SVD；
- RNA feature graph 的 nnz 变化很小：A1 `+0.080%`、Placenta `+0.140%`、P22 `−0.031%`；
- 模型实际 RNA features 仍是固定 scaled HVG matrix，PCA 主要影响 RNA feature graph；
- 标签语义访问为 0，IGE 性能仍未评估。

因此本轮不再要求新结果逐字节匹配旧的随机 PCA。正确标准是：**显式固定 PCA 后，两个独立进程必须生成完全相同的新模型输入与图；四个变体必须只使用同一个锁定缓存。**

## 2. 科学目标与禁止事项

继续原问题：测试 IGE 是否能在无标签条件下替代旧 scale/gamma。

保持不变：

- 数据集 A1、Placenta、P22；
- 变体 C0、C1、IGE、ILN；
- seeds `0–4`；
- epochs、optimizer、HVG、图 K、图 metric、cluster 数、mclust、指标；
- IGE/ILN 公式；
- 原 60-cell dataset/variant/seed/ordinal 顺序；
- 标签防火墙与 weighted-gradient influence gate；
- ASR/rescue 冻结；
- 不搜索 PCA seed、训练 seed、scale、tau、solver 或公式。

唯一新增的预处理参数固定为：

```text
pca_svd_solver = "randomized"
pca_random_state = 0
```

选择 `random_state=0` 是预先固定的常规复现设置，不是搜索结果。不得尝试多个 PCA seed 后选择表现最好者。

## 3. Git 与输出隔离

1. 确认 `night3ar-final-20260810` 指向 `e8ca974...`。
2. 新建分支：

   `revision/q2-night3af-deterministic-pca-20260810`

3. 新建保护标签：

   `baseline/pre-night3af-20260810`

4. 最终 tag：

   `night3af-final-20260810`

5. 新输出目录：

   `outputs/night3af_handoff/`

6. Night‑3A、Night‑3A‑R、Night‑2C 及更早文件全部只读，不覆盖旧失败报告。
7. 新报告明确把 Night‑3A‑R 定义为：

   `DETERMINISTIC_PREPROCESSING_BUG_FOUND_SCIENTIFIC_GO_NO_GO_NOT_EVALUATED`

## 4. 确定性 PCA 实现

### 4.1 不修改冻结 legacy 函数的默认行为

不要直接改变旧 `SpaLORA/preprocess.py::pca()` 的默认语义，以免旧教程/legacy baseline 被静默改写。推荐新增：

```python
def pca_deterministic(adata, use_reps=None, n_comps=10,
                      svd_solver="randomized", random_state=0):
    ...
```

或给 `pca()` 增加可选参数，但必须保证不传参数时旧行为不变；corrected sparse pipeline 显式传入 solver 与 random state。

### 4.2 适用范围

- RNA PCA：三数据集都显式使用 `svd_solver="randomized", random_state=0`。
- 模态 2：保持现有处理不变；P22 使用存档 `X_lsi`，A1/Placenta 当前 PCA 已跨进程精确一致。
- 不改变 HVG、scaling、PCA components、feature graph metric/K 或 sparse normalization。
- 在 config、run manifest 和论文方法记录 sklearn/scanpy/numpy/scipy 版本。

### 4.3 RNG 隔离

测试 PCA 调用前后的 Python、NumPy、Torch CPU/CUDA RNG state。显式 `random_state=0` 的 PCA 不得依赖或改变训练 RNG 流。若 sklearn 内部实现仍改变全局 NumPy RNG，必须在局部保存/恢复 RNG state，不得改变训练初始状态。

## 5. P0D：确定性预处理门

P0D 替代“匹配旧随机 PCA 哈希”的错误门。

### 5.1 两个独立进程生成

在两个全新 Python 进程 A/B 中分别从原始 h5ad 生成 corrected sparse label-free preprocessing。不得在同一进程中连续调用两次冒充跨进程验证。

两个进程必须分别输出每个数据集的：

- observation IDs 与顺序；
- selected gene names 与顺序；
- `features_omics1`、`features_omics2`；
- `weight_vector_omics1`；
- RNA PCA scores；
- 四张 sparse graph 的 coalesced indices、values、shape；
- coordinates；
- PCA explained variance/ratio；
- canonical content SHA‑256。

A/B 以下内容必须精确一致：

- PCA array；
- feature/spatial graphs；
- model-ready dense arrays；
- observation/gene order；
- canonical model-input hash。

任何不一致才触发 P0D 硬停止。

### 5.2 新旧桥接只作诊断

新 deterministic PCA 哈希**预期会与 Night‑3A/Night‑3A‑R 不同**，不得因此停止。

记录：

- 新旧 PCA/graph SHA 不同的原因；
- 新旧 graph nnz；
- 若旧边集合不可恢复，明确写 `edge_overlap_not_available`，不得根据 nnz 伪造 Jaccard；
- 后续用新 C0/C1 与 Night‑2C 对应结果做性能桥接。

### 5.3 发布锁定缓存

进程 A/B 一致后，只发布一份 immutable cache：

`outputs/night3af_handoff/preprocessing_cache/{dataset}/`

建议使用内容确定的独立文件，避免压缩容器时间戳干扰哈希：

- dense arrays：`.npy`；
- graph indices/values：分别 `.npy`；shape 写 JSON；
- IDs/genes：UTF‑8 TSV；
- metadata/config：JSON；
- 每个文件 SHA 和 canonical content SHA 写 manifest。

主实验 runner 禁止重新计算 PCA/feature graph，只能读取并验证该缓存。四个变体和五个 seed 的 input hash 必须相同。

## 6. P0D 必需测试

主实验前全部通过：

1. 两个独立进程 PCA byte-exact；
2. 两个独立进程 RNA feature graph byte-exact；
3. PCA 不受外部 `np.random.seed` 或此前 RNG 消耗影响；
4. PCA 不改变训练 RNG state；
5. corrected pipeline 使用显式 `random_state=0`；
6. legacy pipeline 不传参数时行为未被静默改写；
7. 四变体从缓存读入的 canonical input hash 相同；
8. 缓存损坏、partial cache、manifest 不匹配会拒绝运行；
9. graph 始终 sparse/coalesced，禁止 densify N×N adjacency；
10. 标签防火墙、完整性读取和 evaluator import-order 测试继续通过；
11. weighted-gradient diagnostic state/RNG neutral；
12. Night‑3A/Night‑3A‑R/Night‑2C 保护清单全部匹配。

测试必须 `0 failed` 才能继续。旧 Night‑3A‑R 中因 P0A 预期失败的两项测试应改写为新 P0D 契约，不得保留“expected failure”状态进入主实验。

## 7. P0B‑F：新缓存上的 15 个 IGE 探针

由于 feature graph 已改变，旧 15-cell 数值不能作为新输入的等价目标，必须在锁定 deterministic cache 上重新运行 3×5 探针。

验证：

- 15/15 finite positive gradients/weights；
- weight 范围与和；
- IGE 初始 weighted-gradient influence 约各 0.25；
- CPU state/RNG exact；GPU envelope；
- IGE/C0 初始参数相同；
- 15 个 cell 全部读取同一数据集 cache hash；
- 无标签语义访问、无 evaluator import；
- `m_bad` 仅作为 C1 锁定机制对照，不搜索。

不要要求新 IGE weights 与旧 Night‑3A 完全相同；报告差异即可。只有新探针自身数值失败、缓存不一致或标签泄漏才硬停止。

## 8. 直接执行 60 次主实验

P0D 与 P0B‑F 全部通过后，不再设置额外旧哈希门，立即执行：

`4 variants × 3 datasets × 5 seeds = 60 runs`

| 变体 | 定义 | 用途 |
|---|---|---|
| C0 | deterministic corrected cache + 旧 gamma + RNA scale=1 | 新确定性基线 |
| C1 | 同一 cache + 旧 gamma + 锁定 m_bad | 阳性机制对照 |
| IGE | 同一 cache + 四 raw loss 的冻结 IGE | 预注册候选 |
| ILN | 同一 cache + 初始 raw-loss normalization | 诊断对照 |

所有 run：

- 严格按旧 60-cell ordinal；
- 支持对 manifest/hash 完整 run 的安全 resume；
- partial/mismatched run 不覆盖；
- 保存 embedding、cluster、attention、model、loss trajectory、weighted-gradient trajectory、资源和 hashes；
- 完成 60/60 后先 fsync `locked_60_run_manifest.json`；
- evaluator 只能在 manifest 锁定后独立启动。

## 9. 评估、桥接与 go/no-go

指标与前任务书一致：ARI、NMI、AMI、FMI、homogeneity、V-measure、Hungarian F1/balanced accuracy、neighbor agreement、Moran’s I、silhouette、Davies–Bouldin、时间与内存。

### 9.1 新实验内部判定

IGE 相对新 deterministic C0：

1. 满足以下之一：
   - 至少 2/3 数据集 ARI、NMI 配对均值为正，且至少一套 ARI `≥+0.03`；
   - 胎盘恢复新 C1−C0 ARI 增益至少 60%，且 A1/P22 ARI 变化均 `≥−0.02`。
2. 任一数据集不得同时出现 neighbor agreement 与 Moran’s I 均下降超过 `0.03`。
3. 后半程 weighted-gradient share 不得在全部检查点持续 `<0.01` 或 `>0.90`。
4. attention 不得在全部后半程检查点持续 `<0.05` 或 `>0.95`。
5. 60/60、失败 JSON 0、无 NaN/Inf、无标签泄漏、保护文件匹配。

### 9.2 与 Night‑2C 的桥接

单独输出 deterministic C0/C1 与 Night‑2C 对应 V0/V1 的逐数据集差异：

- ARI/NMI/空间指标；
- 训练输入与 graph 构建差异；
- 明确这些不是完全相同 preprocessing 下的配对检验；
- 不因 deterministic C0/C1 与旧数值不同而自动失败。

这张桥接表用于区分“PCA 固定带来的变化”和“IGE 相对同批基线的真实效果”。

### 9.3 结果组织

- 报告全部五 seed 的 mean±SD、逐 seed、配对差和方向计数；
- 禁止 best-seed 主表；
- 论文可在主文突出最有说服力结果，完整表放 Supplement；
- 如果一套数据轻微下降，缩小 claim 并解释适用范围，不隐藏预注册结果。

## 10. 本轮不再硬停止的情形

以下情况只记录，不得阻断 60 次实验：

- 新 deterministic PCA/graph hash 与旧随机 PCA 不同；
- 新 IGE weights 与旧 Night‑3A 探针不同，但自身有限稳定；
- graph nnz 与旧值小幅变化；
- integrity-only SHA 读取 ground-truth 文件；
- GPU 浮点结果未位级相同但处于已验证 envelope；
- scalar weighted loss 占比不均，但 weighted-gradient influence 未塌缩。

真正硬停止只保留：

- 两个独立 deterministic build 不一致；
- 缓存/输入/四变体不一致；
- 语义标签泄漏；
- 数值 NaN/Inf、state/RNG 污染或 GPU 超包络；
- 保护文件变化；
- 无法安全恢复的 partial/mismatched run。

## 11. 必需输出

目录：`outputs/night3af_handoff/`

- `deterministic_pca_protocol.json`
- `p0d_process_a.json`、`p0d_process_b.json`
- `p0d_cross_process_comparison.json`
- `preprocessing_cache_manifest.json`
- `pca_old_new_diagnosis.json`
- `night3af_p0d.json`
- `night3af_p0b.json`
- `label_flow_audit.md`
- `scientific_window_label_firewall.json`
- `ige_initial_gradients.csv`
- `ige_weights.csv`
- `per_seed_metrics.csv`
- `summary.csv`
- `paired_deltas.csv`
- `night2c_bridge.csv`
- `loss_trajectories.csv`
- `gradient_influence_trajectories.csv`
- `attention_summary.csv`
- `resource_usage.csv`
- `locked_60_run_manifest.json`
- `failure_index.json`
- 图及生成脚本
- `night3af_report.md`
- `night3af_completion.json`
- `SHA256SUMS`

主报告必须首先回答：

1. 两个独立进程是否生成完全相同的 deterministic PCA/graphs/cache？
2. 15/15 新 IGE 探针是否通过？
3. 60/60 是否完成，是否存在任何标签语义访问？
4. IGE scientific go/no-go 是 PASS 还是 FAIL？
5. IGE−C0、C1−C0 的三数据集五 seed ARI/NMI 与空间指标；
6. deterministic C0/C1 相对 Night‑2C 的桥接变化；
7. weighted-gradient influence 是否真正平衡；
8. 是否建议进入 architecture ablation。

## 12. Git、归档与关机

1. 所有测试、结果、报告完成后提交 Git，创建 `night3af-final-20260810`。
2. 生成 Git bundle，verify，并在临时目录 clone/checkout 最终 tag。
3. 生成完整 archive，验证内部清单和成员 SHA。
4. report、CSV、bundle、archive、SHA256SUMS 下载本地并双端校验。
5. GitHub 只尝试 push 一次；无凭据则记录，不重试。
6. 验证 Night‑3A‑R 及全部旧保护文件一致。
7. 生成 shutdown checklist 与 confirmation。
8. `/usr/bin/shutdown` 作为最后一条远端命令；SSH 断开后不得重连。

## 13. Codex 最终回报模板

```text
已严格完成 SpaLORA Night‑3A‑F。
P0D：PASS/FAIL；独立 deterministic build：x/3 数据集 byte-exact。
P0B‑F：x/15 通过。
主实验：x/60；失败 JSON：x。
semantic label access：0/存在。
IGE scientific go/no-go：PASS/FAIL/NOT_EVALUATED。
IGE−C0 五 seed ARI/NMI：A1 ...；Placenta ...；P22 ...。
C1−C0 与 Night‑2C bridge：...。
空间门、gradient-share 门、attention 门：...。
测试：x passed，x failed。
保护文件：...。
最终 commit/tag：... / night3af-final-20260810。
GitHub push：成功/失败（仅一次）。
bundle/archive 双端校验：...。
/usr/bin/shutdown 已作为最后一条远端命令发出；之后未重新连接。
本地交付物路径：...。
```
