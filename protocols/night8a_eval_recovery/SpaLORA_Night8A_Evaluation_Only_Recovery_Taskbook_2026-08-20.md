# SpaLORA Night-8A DEV_WINDOW_1 Evaluation-Only Recovery 任务书

版本：2026-08-20 REV1  
任务性质：只读恢复评价，不训练、不重新 transform、不改写 Night-8A  
目标：在严格证明候选产物未被错误 B00 污染后，用锁定 C00/R02 comparator 重新计算 R1/R2，并锁定最多三个 R3 shortlist ID

## 0. 首条回复与算力

第一条回复必须是：

> 已读取 Night-8A evaluation-only recovery 任务书，开始 P0-RECOVERY-AUTHORITY。本轮 0 training、0 transform、0 external benchmark；原 Night-8A 保持 IMPLEMENTATION_SEMANTICS_INVALID。

服务器建议使用有卡模式开机，因为无卡模式只有 0.5 CPU/2 GB；但本任务不使用 GPU，GPU utilization 为 0 是预期行为。不得调用 AutoDL API。

## 1. 权威输入

Windows compact 根：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8a_handoff_20260820/official_compact`

必须核验：

- compact index：`e74202074a3b4d882670d960bf66c3c1e8667ac3a53baf811ce0658c0ca0d4b6`
- compact root：`e09aec6882a99af7c74d294b6d68a766efc9580e91bb25d0c4ae84424255e265`
- Night-8A report：`22fa3fe04c2548a6a7909f29aa72660f0e157699ceef8904323a83082caa6307`
- Night-8A decision：`bc99e0269857c26986928781101df9662e23a3ab5fa49910d236e47470a8bb6b`
- semantic invalidation：`47bfafbefc0d37e36ad63fa264976fac724fe82a447cd568e03fde215f0aee8c`
- R1 manifest：`15000596a8db3d20d057fd4d3d5d081e275ce7d81d2300edd6683ee89ad17911`
- R2 manifest：`c7be512d335d6e6d45f1d4e566987a23df7ad6d96fc6ff9b696e3b75388537a2`
- failure audit：`f5f6547653db417cd627f2383a3cc94545b0c55579ce8ddfc02852ead3df84ac`
- external dataset lock：`706395103ded06d0c4c4d0d79010b5685adab34e77a1fcc692a72cb6a658e386`

Git：

- base commit：`d09aa00b5e25269e66712dd47d01064b7c9422cf`
- base tag：`night8a-final-20260820`
- branch：`revision/q2-night8a-eval-recovery-20260820`
- protection tag：`baseline/pre-night8a-eval-recovery-20260820`
- final tag：`night8a-eval-recovery-final-20260820`

若 compact 69/69、base/tag、远端或任何 SHA 不符，停止为 `RECOVERY_BLOCKED_AUTHORITY_MISMATCH`。

## 2. 不可改变的原始证据

以下目录只读：

- `/root/autodl-fs/night8a_raw_runs_20260820`
- Night-7A/Night-7B source 与 R02 历史目录
- 原 Night-8A repo outputs、Windows compact 和 final tag
- `/root/autodl-fs/night8a_external_data_20260820`

不得：

- 训练或加载 optimizer 继续训练；
- 运行 `night8a_transform.py` 或任何聚类/affinity 重算；
- 重跑 B03/P22/seed1；
- 修改、覆盖、移动、touch、压缩或删除原始 artifacts；
- 回填原 Night-8A 的空 metrics 文件；
- 打开 MISAR 标签或运行外部 benchmark；
- 创建 R3 work units。

新输出必须进入独立根：

`/root/autodl-fs/night8a_eval_recovery_20260820`

## 3. P0-RECOVERY-AUTHORITY

### 3.1 远端全量 SHA 复核

依据 R1/R2 stage manifests，对全部声明的以下文件重算 SHA：

- 116 个 training artifacts、checkpoint、resolved config、reload audit；
- 所有成功 transform 的 affinity、clusters、manifest；
- 12 个 alias target manifest 与 target artifacts；
- fixed numerical failure 的 training artifacts 与失败日志；
- Night-7B locked source worker inputs、C00 affinity/partition、R02 embeddings/C06 affinity；
- labels snapshot 只核验文件 SHA，不反序列化值。

先生成 `original_artifact_manifest_before.json`。评价完成后再次重算并生成 `original_artifact_manifest_after.json`，要求逐文件 SHA 完全相同。

任何不一致停止 `RECOVERY_BLOCKED_ARTIFACT_MISMATCH`，不得尝试修复原文件。

### 3.2 依赖隔离证明

逐 cell 构建 dependency DAG 并证明：

- B01–B07 的 training reference 不指向 A00/B00 输出；
- B01–B07 的 transform 使用自己的 training embeddings；
- RNA_EPIGENOME 只额外读取锁定 C06/R02 reference；
- B05 RNA_PROTEIN alias 只指向 B04；
- A05 RNA_PROTEIN 是设计上的 no-op，恢复时映射到锁定 C00，而不是错误 A00 transform。

本地先验审计为 109/109 candidate cells 不依赖错误 baseline；远端必须独立得到同样结论。若不是 109/109，停止 `RECOVERY_BLOCKED_CANDIDATE_CONTAMINATION`。

## 4. P0-EVALUATION-VIEW

在不复制大文件的情况下创建只读 `evaluation_view_manifest.json`：

### 4.1 Comparator

- A1、D1、tonsil 的 A00/B00 rows：直接引用 Night-7B source 中 SHA-locked C00 `pseudo_affinity` 与 `pseudo_partition`。
- P22 的 A00/B00 rows：引用已经证明 metric parity `<=4.44e-16` 的锁定 R02/B00 artifacts。
- comparator 的训练输出不参与评价；错误 Night-8A A00/B00 transform 仅进入 invalid-evidence 列表。

### 4.2 新候选

- R1 A01–A07：引用原锁定 candidate transforms；A05 RNA_PROTEIN no-op 映射到 locked C00。
- R2 B01–B07：引用原锁定 candidate transforms；B05 RNA_PROTEIN 映射到原锁定 B04 alias target。
- B03/P22/seed1 保持 `INELIGIBLE_UPSTREAM_FAILURE`，不生成 cluster、不补值。

在打开标签前，evaluation view 必须锁定路径、源 SHA、config、dataset、seed、family、status。预期：

- R1 32 个可评价 rows；
- R2 95 个可评价 rows，1 个固定失败；
- R2 新候选为 83 个可评价 rows，B03 为 11/12 incomplete。

若数量不符，停止。

## 5. 恢复规则冻结

在重新打开开发标签前，把以下规则写入 `recovery_rule_lock.json` 并普通 push：

1. 指标、空间门、权重、seeds、tie-break 沿用原 Night-8A registry；
2. `B00_FAMILY_REFERENCE` 只作 comparator，明确排除在 shortlist、new-candidate Pareto 和 R3 finalist 之外；
3. B03 因不完整不得晋级，但保留 11 个描述性 rows；
4. 不新增配置、不改权重、不放宽阈值；
5. 最多三个 distinct finalist slots：unified、human-lymph frontier、P22 frontier；
6. 本任务只输出 shortlist，绝不启动 R3。

为防止“看结果后改门”，rule-lock commit 必须发生在 label value 第一次反序列化之前。

## 6. 单次授权开发标签窗口

只允许 recovery evaluator 打开 A1、D1、tonsil、P22 的 locked label snapshots 一次。禁止 trainer、transformer、family selector、clustering 或外部数据读取这些值。

### 6.1 Primary evaluator

对 evaluation view 计算：

- ARI、NMI、Q；
- AMI、FMI、homogeneity、completeness、V-measure；
- neighbor agreement、Moran’s I、Geary’s C、boundary disagreement；
- 原任务书的 label-free descriptive metrics；
- paired deltas、wins、Q_HLN、priority macro、spatial protection、complete status。

Comparator parity 硬门：12 个 R2 B00 rows 相对历史 family reference 的 ARI/NMI/Q 最大误差必须 `<=1e-12`。否则停止，不发布候选 metrics。

### 6.2 Independent recompute

用独立脚本重新读取 evaluation view：

- ARI/NMI 不调用 primary evaluator 的函数；至少用独立 contingency-table 实现交叉验证；
- spatial metrics 使用独立向量化路径；
- 对所有成功 rows 核验 primary/independent 最大误差 `<=1e-12`；
- key coverage、K、observation order、cluster SHA 完全一致。

任一不一致终止 `RECOVERY_SEMANTICS_INVALID`。

### 6.3 Shortlist

只在 comparator parity 与独立复算都通过后生成：

- `recovered_r1_per_seed_metrics.csv`
- `recovered_r2_per_seed_metrics.csv`
- `recovered_r2_candidate_summary.csv`
- `recovered_pareto_frontier.csv`
- `recovered_shortlist_ids.json`

Shortlist 必须排除 B00 与 incomplete B03。若没有新候选满足任何 slot，输出空 shortlist，不放宽门。

## 7. 终态

- 通过且有至少一个新 finalist：`NIGHT8A_DEV_WINDOW1_RECOVERED_SHORTLIST_LOCKED`
- 通过但没有 eligible 新 finalist：`NIGHT8A_DEV_WINDOW1_RECOVERED_NO_ELIGIBLE_NEW_CANDIDATE`
- Authority/artifact/contamination 问题：对应 `RECOVERY_BLOCKED_*`
- 评价或独立复算语义问题：`RECOVERY_SEMANTICS_INVALID`

无论终态如何：

- 原 Night-8A 仍是 `IMPLEMENTATION_SEMANTICS_INVALID`；
- 不声称 R3/full-seed/外部确认；
- 不启动训练、transform 或 MISAR benchmark。

## 8. 运行资源与时限

- 科学训练：0；transform：0；GPU：0；external benchmark：0。
- CPU evaluator 最多 3 workers，每 worker 最多 3 threads；缓存各数据集 adjacency，不能每 row 重建。
- 单个评价阶段 wall time 上限 60 分钟；超时停止并报告，不做算法 fallback。
- 若只是 GPU utilization=0，不视为异常；本任务本来就是 CPU evaluation-only。

## 9. Git 与交付

Git 只能普通 push，禁止 force；final tag 只创建一次且在最终报告、测试、delivery index 都进入 final commit 后创建。

D 盘交付根：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8a_eval_recovery_handoff_20260820/official_compact`

至少包含：

- `night8a_eval_recovery_report.md`
- `plain_language_summary.md`
- `recovery_decision.json`
- `p0_recovery_authority.json`
- `candidate_dependency_audit.json`
- `original_artifact_manifest_before.json`
- `original_artifact_manifest_after.json`
- `original_tree_invariance_audit.json`
- `evaluation_view_manifest.json`
- `recovery_rule_lock.json`
- recovered R1/R2 metrics、summary、Pareto、shortlist
- `independent_recompute_audit.json`
- `recovery_label_window_audit.json`
- tests、Git audit、delivery index、planner tar、incremental bundle

不下载 raw runs/checkpoints；compact 目标小于 10 MB。

## 10. 关机

所有远端写入、Git、D 盘复制和 Windows 校验完成后，在保留的同一 SSH 会话中把 `/usr/bin/shutdown` 作为最后一条远端命令。之后不重连、不调用 AutoDL API，只陈述派发与连接状态。

## 11. 给用户的最终汇报

先用通俗语言说明：

1. 116 次训练是否被成功救回；
2. 每个模块在 human lymph、P22、tonsil 上的 pilot 涨跌；
3. 最多三个 shortlist 是谁、为什么；
4. 哪些结果仍只是 3-seed pilot，为什么还不能跑 MISAR。

再报告审计、失败单元、Git、D 盘路径、SHA 与关机状态。
