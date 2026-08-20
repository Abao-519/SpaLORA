# SpaLORA Night‑8B 标签基数安全评价恢复任务书

版本：2026‑08‑20 REV1  
任务性质：post-lock、evaluation-only、protocol revision  
科学训练：0；transform：0；affinity rebuild：0；GPU：0（预期）

## 0. 第一条回复与本轮目标

第一条回复必须原样为：

> 已读取 Night‑8B 标签基数安全评价恢复任务书，开始 P0‑EVAL‑RECOVERY。本轮 0 training、0 transform、0 affinity rebuild；复用已锁定的 20 个 K=12 partitions，允许 Night‑8B lineage 内第 2 次且最后一次原始 Y 读取，仅用于评价。

本轮只回答一个问题：在已经冻结的统一 `RECOVERY_EIGEN_KMEANS100` 聚类头下，`HR_F00` 是否稳定优于 `HR_U00`。

不得再训练、重聚类、尝试 K、改变标签粒度或改候选。原 Night‑8B 仍保持 `INFRASTRUCTURE_BLOCKED`；上一轮仍保持 `RECOVERY_BLOCKED_INPUT_INTEGRITY`，本轮只增加一个明确降级标注的评价恢复结论。

请由用户在 AutoDL **有卡模式**开机后执行。有卡模式用于正常 CPU/内存配额，本轮 GPU utilization=0 是预期现象。总 wall time 上限 90 分钟；评价与同进程独立复算各上限 30 分钟。超时停止，禁止无限等待。不得调用 AutoDL API。

## 1. 为什么允许这次恢复

上一轮的 20 个分区已经在第一次标签读取前全部锁定并普通 push，且 20/20：

- 使用完全相同的 head/config；
- 得到恰好 12 个非空簇；
- 进程内重复完全一致；
- partitions 与输入 SHA 完整。

第一次授权读取 `Y` 后，评价器发现真实标签类别数不等于注册表写死的 `K=12`，于任何指标前停止。因此没有分数、胜负或候选信息反馈到模型，标签也未用于选择。

这次协议修订纠正两点：

1. `K=12` 是已经冻结的**预测分区簇数**，不是 Figshare `Y` 必须满足的声明。Figshare 官方 README 只把 `Y` 定义为 true labels，没有声明其类别数为 12。
2. ARI 与基于 contingency table 的信息论一致性指标允许两个分区拥有不同类别数。这里 `HR_F00` 与 `HR_U00` 都固定输出 K=12，因此 paired contrast 不会比较不同输出 K 的两个方法。

公开源码旁证仅用于修订语义，不得用于选择结果：SEPAR 的 MISAR 教程先令 `n_cluster=len(unique(Y))`，后续聚类却另行写 `n_cluster=12`，说明参考标签基数与作者选择的输出簇数不是同一概念。

证据降级必须写清楚：由于原始 `Y` 已被第一次评价器打开过，本轮若成功，只能称为 `POST_LOCK_EXTERNAL_EVALUATION_RECOVERY_NOT_PRISTINE_HOLDOUT`。可以评价外部泛化方向，不能声称 pristine holdout 或 SOTA。

## 2. 权威输入

### 2.1 机器可读注册表

Windows：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_cardinality_safe_eval_planning_20260820/SpaLORA_Night8B_Cardinality_Safe_Evaluation_Recovery_Registry_2026-08-20.json`

任务书与 JSON 任一冲突立即停止，不自行决定。

### 2.2 Git

- base commit：`4b4c10c32220c9210a366f8e0b3691b2b46cf4fe`
- base tag：`night8b-head-recovery-final-20260820`
- 新分支：`revision/q2-night8b-cardinality-safe-eval-20260820`
- protection tag：`baseline/pre-night8b-cardinality-safe-eval-20260820`
- final tag：`night8b-cardinality-safe-eval-final-20260820`
- 新 worktree：`/root/autodl-fs/SpaLORA-night8b-cardinality-safe-eval`
- 新输出根：`/root/autodl-fs/night8b_cardinality_safe_eval_20260820`

Git 全程普通 push；禁止 force、force-with-lease、移动旧 tag 或提前创建 final tag。

### 2.3 Windows compact

权威根：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_head_recovery_handoff_20260820/official_compact`

至少独立核验：

- report SHA：`8fb7c8cc44152581a114d2a822503e8e301b2c6961582e4719bb584a174b0757`；
- decision SHA：`54d653c9a33759789a740c3b35d42c4309cdd48fe53d0374987ddf6ef5495252`；
- failure audit SHA：`2e402fe014aa0242b5340379fe91ad66d5412476e9b2cf460f7380a62bb0f739`；
- 20-partition manifest SHA：`8a696ec456b9abe45c9fd65c3b646f2654300f6c51e47d766fb0bfcef0686e6c`；
- tracked delivery index 45/45，SHA：`f660dde18fac9791e4f62cc5b2006cfd411464208b35f0cf252a39f17a247971`；
- input-view manifest SHA：`8e9e042a7de8954425e2b46fc4f878bbc3f7c698f6283a80bb30132d9efcd71e`；
- rule lock SHA：`9b098fff1b80cc3e562c2afa1833255ea050c14601c46e698011a8b0739d6c35`；
- before/after original manifests SHA 均为 `87273b5053588553fe1f6cdc7c51177eb98e0c3dbf84936caf7cdbebd920a6eb`；
- planner tar SHA：`f51ededf76c0a961c82072971226974bf6f65d7a62747ae0d462f8e5de235566`；
- incremental bundle SHA：`9a6395fd684edc2cb2e0ab101c183fb0a9b61232aa3d737f9bf3c2c01faab3b8`。

### 2.4 远端只读根

- 原 Night‑8B：`/root/autodl-fs/night8b_raw_runs_20260820`
- 统一 head recovery：`/root/autodl-fs/night8b_head_recovery_20260820`

两者全程只读。不得修改、移动、touch、覆盖或补写其中任何文件。

## 3. P0‑EVAL‑RECOVERY

在实现评价代码前先完成并保存 `p0_eval_recovery_authority.json`：

1. GitHub base branch/tag 均 peel 到 `4b4c10c...`，工作树 clean；
2. Windows authority hashes 全部匹配；
3. head-recovery tracked index 45/45；
4. 远端 20 个 `clusters.csv`、20 个 transform manifest 与 `locked_recovery_partition_manifest.json` 逐项 SHA 匹配；
5. 20 行仍为 HR_U00 seeds 0–9 与 HR_F00 seeds 0–9，K=12、exact K、deterministic；
6. observation order 1949/1949，SHA `9f0514cee55d307a0ff81d44ffffc2da742dbe2d02b849576b7ef5903743dd1b`；
7. annotation mapping SHA `322e7bf0f459998c882a0305e8aea129deee29570412982b3697976ac64b8ae5`；
8. 第一轮 lineage `Y` read count=1、metrics=0、actual K 未持久化；
9. 原 raw manifest 297/297 与 head-recovery 文件在本轮 before snapshot 中完整；
10. 本轮新增 training/forward/adapter/affinity/head transform 均为 0。

任一不符停止 `EVAL_RECOVERY_BLOCKED_INPUT_INTEGRITY`。

## 4. 在标签前锁死全部评价语义

先新增评价器与测试，并把下列内容写入 `evaluation_contract_lock.json`：

- primary 固定为 20 个现有 K=12 partitions；
- contrast 固定为同 seed `HR_F00-HR_U00`；
- 不要求 predicted K 等于 reference K；
- 不改 K、不重分区、不尝试标注粒度；
- 标签 canonicalization、missing 规则、指标、统计、门槛和终态全部固定；
- 两套独立指标实现必须在同一 evaluator 进程、同一个内存中的 Y 上运行；
- 评价器在打开 Y 前预加载并 SHA 核验全部 partitions、坐标、邻接/空间输入、resource records；打开 Y 后不得再返回方法或代码。

标签 canonicalization 仅允许：

1. `(N,1)` 或 `(1,N)` 压平成 `(N,)`；其他非一维结构停止；
2. bytes 用 strict UTF‑8 解码；
3. 字符串只裁掉首尾空白；
4. 数值保持精确值，不 round、不 bin；
5. 不 merge、不 drop、不 rename 任何类别；
6. IEEE NaN、正负 infinity 或裁剪后空字符串任一存在，停止 `EVAL_RECOVERY_BLOCKED_LABEL_INTEGRITY`，不得自行过滤；
7. `2 <= reference_K < 1949`，长度必须恰好 1949，cell/order mapping 必须完全匹配。

所有代码、测试、协议、评价 contract 必须在第二次原始 Y 读取前 commit 并普通 push。记录 pre-label commit、GitHub branch SHA 与不可变输入 SHA。

## 5. 第二次且最后一次原始 Y 读取

这是 Night‑8B lineage 内明确授权的第 2 次、也是最后一次原始 `Y` 读取。原因是上一评价器在发现 K mismatch 后没有持久化 actual K，导致无法仅用既有交付完成评价。

必须由一个单独 evaluator 进程完成：

1. 打开 Y 前把全部非标签评价输入读入内存并验证；
2. 对原始 carrier 只反序列化一次 Y；
3. 立即计算并以 atomic write + flush/fsync 持久化 `reference_label_contract.json`，至少包含：
   - carrier 文件 SHA；
   - Y dataset shape/dtype；
   - canonical Y SHA；
   - actual reference K；
   - 每个原始类别值及数量；
   - missing-like count；
   - cell 数与 observation-order alignment；
   - lineage read count before=1、this task=1、after=2；
4. K 不等于 12 **不得停止**；只要标签完整且对齐，就继续评价；
5. 不把 raw label vector 写入 Git、Windows compact 或对话；
6. 同一进程用同一个内存 Y 运行 primary 与独立复算；不得由第二个进程再次读取原始 Y；
7. 完成后关闭 carrier，之后禁止任何原始 Y 再读取。

如果 evaluator 在落盘 actual K 前异常退出，不得第三次读取，终止 `EVAL_RECOVERY_SEMANTICS_INVALID`。

## 6. Primary：10-seed uniform-head paired evaluation

对固定 20 行计算：

- ARI、NMI、Q=`(ARI+NMI)/2`；
- AMI、FMI、homogeneity、completeness、V-measure；
- neighbor agreement、Moran’s I、Geary C、boundary disagreement；
- 同 seed `HR_F00-HR_U00` deltas；
- 10-seed wins、均值、标准差；
- 对 mean ΔQ 穷举全部 `2^10` sign assignments 的单侧 exact paired sign-flip test；
- paired bootstrap 100,000 次，seed=`20260820`，报告 ΔARI/ΔNMI/ΔQ 95% CI。

注意：NMI 是未作 chance correction 的指标，但两个方法输出 K 完全相同，因此可作 paired comparison；AMI 必须同时报告为基数敏感性补充。不得把本结果与使用其他 K、其他 carrier 或其他标注粒度的论文绝对分数直接比较。

## 7. 同进程独立复算与描述性敏感性

### 7.1 独立复算

在同一 evaluator 进程中用独立代码路径复算决定终态所需的量：

- ARI：独立 contingency-table 公式；
- NMI 与 Q：独立 contingency-table 公式；
- 四个空间指标；
- paired deltas、wins、exact sign-flip 与 bootstrap；
- 20-row key coverage 和 observation alignment。

所有决定终态的可比数值最大误差 `<=1e-12`；coverage 必须 20/20。AMI、FMI、homogeneity、completeness 与 V-measure 作为 secondary descriptive metrics，要求固定库版本、有限值、范围与 key coverage 审计，但不要求重复实现复杂的 chance-correction 公式。否则终止 `EVAL_RECOVERY_SEMANTICS_INVALID`。

### 7.2 原 spectral 9-pair sensitivity

只在同一内存 Y 尚存时，对原完整 seeds `0,1,2,3,4,5,7,8,9` 做描述性评价，输出 `original_spectral_9pair_sensitivity.csv`：

- 不补 seed6；
- 不参与 terminal decision；
- 不替代 10-seed primary；
- 仅报告方向是否与统一 head 一致。

## 8. 判定规则

不得修改原 Night‑8B 科学门：

- mean ΔQ `>=+0.010`；
- mean ΔARI、mean ΔNMI 均 `>=0`；
- paired Q wins `>=8/10`；
- one-sided exact sign-flip p `<0.05`；
- paired bootstrap ΔQ 95% CI 下界 `>0`；
- neighbor Δ `>=-0.01`；Moran Δ `>=-0.02`；Geary Δ `<=+0.02`；boundary Δ `<=+0.01`。

资源门沿用：F00/U00 端到端 runtime `<=1.50x`、peak GPU `<=1.25x`，使用原训练+adapter+已锁定 recovery head 成本，不把本次轻量评价时间充当训练成本。

终态：

- 科学门、空间门、资源门全过：`NIGHT8B_CARDINALITY_SAFE_FAMILY_POLICY_CONFIRMED`；
- 科学/空间门过、资源门不过：`NIGHT8B_CARDINALITY_SAFE_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST`；
- mean ΔQ>0 但完整门未过：`NIGHT8B_CARDINALITY_SAFE_PARTIAL_OR_MIXED_EVIDENCE`；
- mean ΔQ<=0 或明确反向：`NIGHT8B_CARDINALITY_SAFE_FAMILY_POLICY_NOT_GENERALIZED`。

结论措辞必须是：“在固定 K=12 统一 head、实际 reference-K 标签下的 post-lock 外部评价恢复”。不得写 pristine holdout、原 H05 endpoint 完整确认或 SOTA。

## 9. 禁止事项

- 禁止训练、checkpoint load、forward、adapter、affinity rebuild、head transform；
- 禁止重新生成或改动 20 个 partitions；
- 禁止 K search、用 actual K 重聚类、切换 label carrier 或尝试多种标注粒度；
- 禁止 merge/drop/rename label；
- 禁止第三次读取 raw Y；
- 禁止在读 Y 后回到模型、分区、mapping、代码或门槛；
- 禁止 best seed、删失败、fallback、重试或事后放宽门；
- 禁止把 9-pair sensitivity 写进终态门；
- 禁止第三方 benchmark与 SOTA 声明；
- 禁止将 raw runs、checkpoints、affinities、partitions 或原始 label vector 下载到 Windows；
- 禁止 force push、force-with-lease 或移动 tag；
- 禁止调用 AutoDL API。

## 10. 测试、交付与关机

至少新增并通过：

- authority SHA 与 Git peel；
- 20/20 partition byte/SHA 不变；
- evaluator 不含训练/transform/affinity/head/K 参数入口；
- predicted-K/reference-K mismatch 不再触发错误；
- malformed Y、missing-like、长度或 order mismatch 会在指标前硬停；
- actual K contract 在 metrics 前 atomic 落盘；
- lineage read count 精确为 2；
- 两套指标实现同进程复算 `<=1e-12`；
- primary 20/20、paired 10/10；
- original spectral 9-pair 不影响终态；
- 两个旧根 before/after SHA 不变；
- training/forward/adapter/affinity/head transform 均为 0。

最终交付到：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_cardinality_safe_eval_handoff_20260820/official_compact`

目标 `<5 MB`。包含协议、注册表、authority/contract、reference-label metadata（不含逐 spot label）、20-row metrics、paired summary、statistics、independent recompute、9-pair sensitivity、resource/access/invariance/Git audits、测试日志、report、decision、delivery index、planner tar 与增量 bundle。

final tag 必须在报告、测试、delivery index 全部进入 final commit 后一次创建并普通 push。

全部远端工作、push、D 盘复制与 Windows 校验完成后，在保留的同一 SSH 会话把 `/usr/bin/shutdown` 作为最后一条远端命令；之后不重连。

最终先用通俗中文回答：

1. 实际 reference K 是多少，为什么和预测 K=12 不冲突；
2. F00 相对 U00 的 ΔARI、ΔNMI、ΔQ、wins、p 与 CI；
3. 是否支持“RNA+ATAC 使用 R02”的模态家族策略；
4. 结论对统一 head 与原 spectral head 是否敏感；
5. 为什么本结果只能称 post-lock recovery，不能称 pristine holdout/SOTA。
