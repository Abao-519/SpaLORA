# SpaLORA Night‑8B 统一稳健聚类头评价恢复任务书

版本：2026‑08‑20 REV1  
任务性质：pre-label 统一 transform + evaluation recovery  
科学训练：0；adapter 训练：0；固定 recovery transforms：20

## 0. 第一条回复与算力

第一条回复必须原样为：

> 已读取 Night‑8B 统一稳健聚类头评价恢复任务书，开始 P0‑RECOVERY‑AUTHORITY。本轮 0 training、0 adapter、20 个统一 head transforms；原 Night‑8B 保持 INFRASTRUCTURE_BLOCKED，MISAR Y 仍保持未读取。

请由用户在 AutoDL **有卡模式**开机后执行。使用有卡模式是为了获得正常 CPU/内存，不代表本任务应使用 GPU；本轮 GPU utilization=0 是预期现象。不得调用 AutoDL API。

总 wall time 上限 2 小时；单个 transform 上限 10 分钟；评价和独立复算各上限 30 分钟。超时立即停止，不能无限等待。

## 1. 为什么允许恢复

原 Night‑8B 已正确停止为 `INFRASTRUCTURE_BLOCKED`：20/20 训练和 checkpoint reload 均通过，但 `U00/H05/seed6` 的 `SPECTRAL_DISCRETIZE` 只产生少于 12 个非空簇，因此原矩阵只有 19/20 transforms。

MISAR Y 的读取次数仍为 0，所以尚未发生“看结果后选 solver”。本恢复不只修 seed6，也不延续原矩阵：它从相同的不可变 affinity 输入出发，对 U00 和 F00、全部 10 seeds **统一重算 20 个 partitions**。聚类头固定为 Night‑7B 在看结果前已经注册过的 `EIGEN_KMEANS100` 数学实现。

本恢复若成功，回答的是“F00 表示/affinity 在统一稳健 head 下是否优于 U00”，不能声称原来的 H05/discretize endpoint 已完整确认。

## 2. 权威输入

### 2.1 Git

- base commit：`199b2c721b28fb4064be7cfaff696250bd5e8dde`
- base tag：`night8b-final-20260820`
- 新分支：`revision/q2-night8b-uniform-head-recovery-20260820`
- protection tag：`baseline/pre-night8b-uniform-head-recovery-20260820`
- final tag：`night8b-head-recovery-final-20260820`

### 2.2 Windows compact

根目录：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_handoff_20260820/official_compact`

必须独立核验：

- tracked index 38/38，SHA `f15cfaae937b246cb30017a0e8203db04bf6f254cafb8f5fcb9994ae91ff708b`；
- report SHA `980a754561b0ce36dc60af513b6cdbd705575a60929b75561071ff9fce6b82d3`；
- decision SHA `c7eaa285ac413a868bdcad5dbd0b50491454e0410661ec4a07a588e096e848a3`；
- incomplete manifest SHA `eebb21cc24216129dde29415d829f796c22958d3f68b699e696b5778c37d52ce`；
- raw artifact manifest SHA `d45465d1ab2c9b8f7d50aef4bcb8f0d67b869e4071070d7945e73269f5c8b4c9`；
- P0 authority SHA `9371a4b512df65cf3ef3cb16dd9b1cc328b55bca48d99c4557c4240b309699d6`。

机器可读恢复注册表：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_head_recovery_planning_20260820/SpaLORA_Night8B_Uniform_Head_Recovery_Registry_2026-08-20.json`

任务书与 JSON 冲突时停止，不自行决定。

## 3. P0‑RECOVERY‑AUTHORITY 与原证据不变性

原始根只读：

`/root/autodl-fs/night8b_raw_runs_20260820`

新输出根：

`/root/autodl-fs/night8b_head_recovery_20260820`

依据 `raw_artifact_manifest.csv` 重算所有现存原文件 SHA，生成 `original_artifact_manifest_before.json`。恢复交付前再生成 after manifest，要求逐文件一致。不得修改、移动、touch、补写或覆盖原 Night‑8B 文件。

必须定位并独立核验：

- 10/10 base unit、10/10 adapter unit、20/20 reload；
- 10 个 `s04.npz`；
- 10 个 F00 `affinity.npz`；
- 原 9 个成功 U00 affinity 与同 seed `s04.npz` canonical SHA 完全一致；
- observation order SHA 为 `9f0514cee55d307a0ff81d44ffffc2da742dbe2d02b849576b7ef5903743dd1b`；
- prelabel mapping 1949/1949，SHA `322e7bf0f459998c882a0305e8aea129deee29570412982b3697976ac64b8ae5`；
- annotation carrier 的 Y 值尚未反序列化。

任一不符停止 `RECOVERY_BLOCKED_INPUT_INTEGRITY`，不尝试修原文件。

## 4. 恢复输入视图锁

在运行任何 recovery head 前，生成 `recovery_input_view_manifest.json`，固定 20 行：

- `HR_U00/seed0..9`：只引用同 seed `formal/adapter/inputs/seed_{seed}/s04.npz`；
- `HR_F00/seed0..9`：只引用原成功的 `formal/transforms/F00/seed_{seed}/affinity.npz`。

每行锁定 path、file SHA、canonical sparse SHA、shape、nnz、observation SHA、method、seed、K=12。禁止重新 forward、重建 adapter endpoint 或加载 checkpoint 产生新 affinity。

20 行 manifest 与恢复规则必须在 Y 打开前普通 push。

## 5. 唯一允许的统一 head

head ID：`RECOVERY_EIGEN_KMEANS100`。全部 20 cells 必须调用同一个函数、同一个 canonical config SHA；函数不能接收 method 名、seed 结果、标签或指标。

数学语义逐项固定：

1. affinity 转 float64 CSR，按 `0.5*(A+A.T)` 对称化、排序索引、对角清零；
2. 任一 degree `<=0` 直接 `RECOVERY_BLOCKED_HEAD_NUMERICS`；
3. symmetric normalized Laplacian；
4. `eigsh(k=12, which="SM", v0=linspace(1,2,N), tol=1e-10, maxiter=max(10000,N*20))`；
5. 每个 eigenvector 以最大绝对值 pivot 为正进行 sign canonicalization；
6. 逐行 L2 normalize，分母下界 `1e-12`；
7. `KMeans(n_clusters=12,n_init=100,random_state=2020,algorithm="lloyd")`；
8. 必须得到恰好 12 个非空簇，否则硬停止，不改 solver、不 fallback。

该实现来自 Night‑7B 在结果前注册的 `EIGEN_KMEANS100` head family，不得再尝试 `kmeans` spectral、mclust、Leiden、cluster_qr 或其他头。

每个 transform 在同一进程内重复调用 head 两次作 determinism audit，要求 canonical partition SHA 完全一致；只保存一份正式输出。环境固定 `OMP/MKL/OPENBLAS_NUM_THREADS=1`，最多 4 个进程并行。科学 retry=0。

完成后生成 20-row `locked_recovery_partition_manifest.json`。要求 20/20 exact K、20/20 deterministic、20/20 SHA 完整，并在 label window 前普通 push。

## 6. Pre-label head 敏感性

在不读取 Y 的条件下，对原 19 个成功 partition 与同 cell recovery partition 计算 cluster-assignment ARI/NMI，仅衡量 head 改变程度；输出 `prelabel_head_partition_concordance.csv`。

该 concordance 不得影响是否打开标签或终态，只用于解释结果对聚类头是否敏感。

## 7. 单次授权 MISAR 标签窗口

只有在总锁 commit 已普通 push、20/20 recovery partitions 完整后，独立 evaluator 才能读取一次 Y。之后禁止返回 affinity、head、K、mapping、代码或候选修改。

### 7.1 Primary：10-seed uniform-head contrast

对 HR_U00、HR_F00 共 20 行计算：

- ARI、NMI、Q=`(ARI+NMI)/2`；
- AMI、FMI、homogeneity、completeness、V-measure；
- neighbor agreement、Moran’s I、Geary C、boundary disagreement；
- 同 seed `HR_F00-HR_U00` paired deltas；
- 10-seed wins、均值、标准差；
- 对 mean ΔQ 穷举 `2^10` 符号组合的单侧 exact sign-flip test；
- 100,000 次 paired bootstrap，seed=`20260820`，报告 95% CI。

### 7.2 描述性敏感性：原 spectral 9 对完整 seeds

在同一个只读 evaluator 窗口中，可对原 head 的 seeds `0,1,2,3,4,5,7,8,9` 计算 9 对结果，但必须单独放入 `original_spectral_9pair_sensitivity.csv`：

- 不补 seed6；
- 不做终态判定；
- 不替代 10-seed primary；
- 只检查 F00/U00 方向是否与 recovery head 大致一致。

### 7.3 独立复算

独立脚本复算所有 primary metrics、空间指标、统计与 key coverage，最大误差 `<=1e-12`。任一失败终止 `RECOVERY_SEMANTICS_INVALID`。

10 seeds 只衡量算法随机稳定性，不是 10 个独立生物样本。

## 8. 判定规则

科学门沿用原 Night‑8B，不修改：

- mean ΔQ `>=+0.010`；
- mean ΔARI、mean ΔNMI 均 `>=0`；
- Q wins `>=8/10`；
- exact sign-flip p `<0.05`；
- paired bootstrap 95% CI 下界 `>0`；
- neighbor Δ `>=-0.01`、Moran Δ `>=-0.02`、Geary Δ `<=+0.02`、boundary Δ `<=+0.01`。

资源门以“原 base 训练 + F00 adapter + recovery head”的端到端口径计算：F00/U00 runtime `<=1.50x`、peak GPU `<=1.25x`。

终态：

- 科学门与资源门都过：`NIGHT8B_HEAD_RECOVERY_BALANCED_CONFIRMED`；
- 科学门过、资源门不过：`NIGHT8B_HEAD_RECOVERY_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST`；
- mean ΔQ>0 但科学门未全过：`NIGHT8B_HEAD_RECOVERY_PARTIAL_OR_MIXED_EVIDENCE`；
- mean ΔQ<=0 或明确反向：`NIGHT8B_HEAD_RECOVERY_FAMILY_POLICY_NOT_GENERALIZED`。

结论必须写成“统一 EIGEN_KMEANS100 head 下的 family-policy 外部确认”，不能写成原 H05 endpoint 完整确认。原 Night‑8B 仍保持 `INFRASTRUCTURE_BLOCKED`。

## 9. 禁止事项

- 不得训练、reload-forward 生成新 embedding、继续 optimizer 或重建 affinity；
- 不得只重跑 U00 seed6；
- 不得把原 19 个 partitions 混入 10-seed primary；
- 不得试多个 solver/K/标注粒度后选高分；
- 不得 best seed、删失败、fallback 或事后放宽门；
- 不得运行第三方正式 benchmark或声明 SOTA；
- 不得把 raw、affinity、embedding、checkpoint 下载到 Windows；
- 不得 force push、force-with-lease 或移动 final tag。

## 10. 测试、Git 与交付

至少新增并通过：

- original 38/38 compact 与远端 raw manifest SHA；
- original tree before/after 字节不变；
- 20-row recovery input lock；
- original U00 affinity=s04 的 9/9 parity；
- same head/config SHA 20/20；
- head 不接收 method/seed/label/metric；
- exact K 20/20、进程内 determinism 20/20；
- Y 单向窗口、0 training、0 affinity rebuild；
- independent metrics/statistics `<=1e-12`；
- 原 9-pair sensitivity 不影响终态。

Git 全程普通 push。final tag 必须在报告、测试、delivery index 全部进入 final commit 后一次创建。

D 盘 compact：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_head_recovery_handoff_20260820/official_compact`

目标 `<10 MB`；包含协议、注册表、输入视图锁、20-row partitions/metrics、paired summary、prelabel concordance、9-pair sensitivity、统计与独立复算、标签窗口审计、不变性审计、资源审计、测试、Git、delivery index、planner tar 和增量 bundle。不得包含 raw arrays/checkpoints/affinities。

## 11. 关机与通俗输出

全部远端工作、普通 push、D 盘复制和 Windows 验证完成后，在保留的同一 SSH 会话中把 `/usr/bin/shutdown` 作为最后一条远端命令；之后不重连。

最终先用通俗中文回答：

1. 这次只是聚类头故障，还是模型本身失败；
2. 在统一稳健 head 下，F00 是否稳定超过 U00；
3. 这是否支持“RNA+ATAC 使用 R02”的模态家族策略；
4. 结论对聚类头是否敏感。
