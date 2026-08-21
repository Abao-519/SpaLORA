# SpaLORA Night-10A REV1：QCRD 语义补全、恢复与执行任务书

日期：2026-08-21  
性质：Night-10A 零训练失败后的定义级修订；不是事后调分  
运行方式：AutoDL 有卡模式  
科学目标：补全 QCRD 唯一语义后，继续原定指标补算、R1 与条件 R2。

## 1. 为什么允许恢复

原 Night-10A 在正式训练、transform、Stage M 和候选评价全部为 0 时停止。它没有产生科学正负结果，也没有看到任何 QCRD 候选分数。因此 REV1 补充缺失公式和固定权重不属于根据结果调参。

原终态 `IMPLEMENTATION_SEMANTICS_INVALID` 永久保留；REV1 不改写原报告、commit 或 tag。

REV1 的最高新增权威文件是：

`night10a_qcrd_rev1_semantic_contract.json`

它只补充以下缺口：

- global quality 中的空间局部一致性与 graph local residual；
- per-spot local neighbor entropy；
- 15%确定性 mask 规则；
- 所有 loss 的公式和固定权重；
- RNA+protein 与 RNA+ATAC 的固定输出 endpoint。

原 Night-10A 中的数据集、seed、候选 Q00-Q07、三条 frontier、禁止标签训练、禁止 dataset-name routing、禁止 dense N×N、运行预算与交付规则继续有效。

## 2. 权威输入

开始前逐项复算 SHA-256：

1. REV1 semantic contract：
   `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev1_planning_20260821/night10a_qcrd_rev1_semantic_contract.json`
   SHA-256：`ecc91a433a3a4e2dfcf812d40aaa679287b3fcd76d8645f440d2b6015903fe75`
2. 原 Night-10A report：
   `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_handoff_20260821/official_compact/handoff/night10a_report.md`
   SHA-256：`a02f7a32e3f34b89d3d0f7ce9ee97a8eb602c138e1f648296b8a19cc5af02b86`
3. 原 semantic coverage reaudit：
   `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_handoff_20260821/official_compact/handoff/p0_semantic_coverage_reaudit.json`
   SHA-256：`9990265001d75918eca85c640a69f64309a7bf43f93eac7cb531680df4c782dd`
4. 原部分实现：
   `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_handoff_20260821/official_compact/code/night10a_qcrd.py`
   SHA-256：`bbdf94bdff1c9c96e7ae61c6f292d54ab97f386b5f9bcd048aab21b1a2262f57`
5. 原测试：
   `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_handoff_20260821/official_compact/tests/test_night10a_qcrd.py`
   SHA-256：`d0412b8dc65a6e51b098b8ca1de53202ab38d254622477f8b0871b7b404add73`
6. H05 权威实现参考：
   `D:/文档/ChatGPT/博士第一篇科研论文项目/night6c_handoff_20260817/official_compact/SpaLORA/night6c_pipeline.py`
   SHA-256：`b1abd06ab1f5f30d8f3c1d16e89a2ab8c240b61422f8c2b9307adc0619a0e0d1`
7. F00/R02 endpoint 权威实现参考：
   `D:/文档/ChatGPT/博士第一篇科研论文项目/night7b_handoff_20260818/official_compact/scripts/night7b_adapter_stage.py`
   SHA-256：`e8d6d0b6918afe75d9aede5573bfb8d98e3d17ab750df6e43d40c81b8d7ff393`

原 Night-10A 的任务书、registry、指标代码和权威分数表继续从原 compact/protocol 或原 planning 目录读取，其原 SHA 不变。

若上述任一文件 SHA 不匹配，停止；不要使用相似文件替代。

## 3. Git 恢复线

1. 从原 Night-10A final commit `1e576b68938fa194dcdd53ee58915767b7a78325` 新建：
   `revision/q2-night10a-rev1-qcrd-execution-20260821`
2. 创建一次保护标签 `baseline/pre-night10a-rev1-qcrd-20260821`，指向上述父提交。
3. 原 `night10a-final-20260821` 永远不移动。
4. REV1 final tag：`night10a-rev1-final-20260821`，只在最终交付进入 final commit 后创建一次。
5. 普通 push；禁止任何 force。

## 4. P0-REV1：只修语义，不读标签

### 4.1 复用而非重做

- 复用原 `SpaLORA/night10a_qcrd.py`、测试和 P0 证据；不要推倒重写已通过的 teacher stop-gradient、sparse MNN、boundary suppression、checkpoint round-trip。
- 原 P0 已证明环境、五个 raw roots、历史 A1 ARI/NMI parity 和10项测试可用。本轮只需复核文件仍存在、关键 SHA 未变、CUDA 可用；不要再读取 A1 标签重复 parity。
- P0-REV1 label reads 必须为 0。

### 4.2 必须实现的补丁

严格逐字段实现 REV1 contract：

1. modality-specific frozen partitions `p1/p2`；
2. global 五项质量特征及唯一 contrast/weight；
3. modality-specific spatial-neighbor entropy、MNN support、spot logits、boundary risk、base gate；
4. final up-projection 零初始化；
5. 15%逐 spot、逐 epoch确定性 mask；
6. 六个 loss 公式和固定权重：
   - align `1.00`
   - mask `1.00`
   - anchor `0.25`
   - correction `0.05`
   - boundary `0.25`
   - Q07 MNN `0.10`
7. Q00-Q07 每个候选精确开关；
8. RNA+protein 固定 H05 endpoint；P22 固定 F00/R02 E1_ADAPTER_C06_MEAN/H01 endpoint；
9. zero-degree 规则；
10. 全部 frozen quality arrays、partitions、masks、loss config 与 endpoints 的 manifest SHA。

不得新增候选或自行改变 REV1 数值。只要数学结果和 contract 精确一致，允许选择正常的工程实现细节，例如函数拆分、批处理大小和文件组织。

### 4.3 强制测试

除原测试外，新增至少以下测试：

- tiny graph 手工复算五项 global features；
- neighbor entropy、zero-degree、spot logits、gate 手工复算；
- mask 数量、确定性、epoch变化、artifact SHA变化；
- 六个 loss 的独立复算；
- Q00-Q07 semantic truth table；
- 任意遗漏注册 feature/loss 时测试必须失败；
- Q00 对 C00/F00 的 exact alias parity；
- RNA+protein H05 与 P22 E1/H01 endpoint parity；
- 无 dataset-name routing；无 dense N×N；无标签对象进入训练模块。

所有测试通过后生成 `p0_rev1_semantic_contract.json`，逐字段列出 contract path、implementation symbol、test name 和 pass。不得只写概括性 PASS。

P0-REV1 开发阶段允许修正代码和测试，并保留简要更正日志；在正式 config/output 总锁后不得修改科学实现。P0 修正不计 scientific retry。

## 5. 正式输入与 endpoint

### 5.1 RNA+protein：A1、tonsil、D1

- 输入：同 seed 的权威 G04 `private1/private2/fused`、空间图、observation order 和 C00 partition。
- QCRD 输出：`z1c/z2c/zc`。
- 最终聚类：exact `H05_EQUAL3_AFFINITY_SPECTRAL`：三个 corrected views 各自构造 k10 self-tuning sparse affinity，等权平均后固定 spectral。
- Q00：直接 alias 历史 C00 partition，不重算作为 reference。

### 5.2 RNA+ATAC：P22

- 输入：同 seed 的权威 G04 private views、C06 affinity、F00/R02 fused adapter embedding/partition。
- QCRD 输出：`z1c/z2c/zc`，其中 `zf` 使用 F00/R02 fused adapter embedding。
- 最终 affinity：`0.5 * row_sparse(self_tuning_affinity(zc)) + 0.5 * row_sparse(C06)`，sym-zero 后 exact H01 spectral。
- Q00：直接 alias 历史 F00/R02 partition；不能用 C00、C01 或 N02 替代 primary reference。

两类数据允许沿用已经验证的 family-specific preprocessing 与 endpoint；QCRD trainable architecture、超参数、loss 和质量公式必须相同。这回答了“不同测序平台是否允许不同处理”：允许平台家族的输入/endpoint 不同，但核心可学习机制不能按具体数据集写死。

## 6. 正式顺序

### 6.1 代码冻结

P0-REV1 全过后：

1. 生成全部 R1 config，复算 config SHA 唯一性；
2. 将代码、REV1 contract、tests、R1 configs 普通 commit/push；
3. 锁定 code/config commit；从此不得修改 QCRD 数学语义。

### 6.2 R1

- Q00-Q07 × A1/tonsil/D1/P22 × seeds 0/1/2。
- Q00 alias；最多 84 个真实 GPU adapter training units。
- fixed 120 epochs、AdamW、无 early stopping、无 best epoch、无 retry/fallback。
- 先训练和 checkpoint round-trip，再运行固定 endpoint；所有 outputs/partitions/SHA 总锁后才读标签。
- 某个 unit 的非有限或 transform timeout只作废该 unit，保留并继续其他独立 unit；除非发现系统性语义错误，否则不得让单个数值失败作废整轮。

### 6.3 R1 唯一评价窗口，同时完成 Stage M

每个核心数据集最多读取一次标签，完成：

1. R1 Q00-Q07 主指标、扩展标签指标和空间指标；
2. 原计划 Stage M 的历史候选指标补算；
3. 对存在 private views 的候选补 FOSCTTM 与 Recall@K；这些指标不需要标签，可以提前算，但最终一并锁定。

Stage M 缺 artifact 时记录 `MISSING_SOURCE_ARTIFACT`，不得重训补齐。

### 6.4 R1 frontiers 与条件 R2

按原 registry 保留去重后最多三个候选：

- accuracy frontier；
- balanced/worst-family frontier；
- spatial-protected frontier。

普通小幅损失只影响排名。只有系统性语义错误、标签泄漏、错误 source/config 或 artifact 损坏才终止整轮。

若至少一个真实候选在某个 modality-family mean Q 为正且没有明显空间崩塌，进入 R2；否则以 `NIGHT10A_REV1_QCRD_NO_POSITIVE_SIGNAL` 结束。

### 6.5 R2

最多三个候选：

- A1、tonsil seeds 0-4；
- D1、P22 seeds 0-9；
- 最多90个真实训练 units。

先总锁，再进行第二次且最后一次该阶段标签评价。输出完整 per-seed、mean/median/SD/worst seed、paired delta、bootstrap CI、exact sign-flip p、runtime 与 peak GPU。

## 7. 指标与结论边界

主指标：ARI、NMI、Q。  
标签佐证：MI、AMI、FMI、homogeneity、completeness、V-measure。  
无监督几何：Silhouette、DB、CH。  
跨模态：symmetric FOSCTTM、Recall@1/5/10、paired median rank。  
空间：neighbor agreement、Moran's I、Geary's C、boundary disagreement。

- 几何指标不能单独选最终候选。
- A1、tonsil、D1、P22 都是 development evidence，不得称 pristine holdout。
- 本轮不宣称 SOTA；强阳性候选锁定后才进入 Night-10B 端到端整合和后续 benchmark。

## 8. 运行效率

- 使用有卡模式，GPU adapter 必须真实 CUDA。
- CPU transform 先做内存探针，最多4并行。
- 单 transform 30分钟 timeout；干净杀死该子进程、记录 TIMEOUT、继续独立单元，不重跑。
- 整轮12小时 wall-clock。达到上限后停止派发新任务并交付 `WALLCLOCK_BUDGET_REACHED`，禁止无限等待。
- 状态更新仅在 P0-REV1、R1锁定、R1评价、R2锁定、最终交付等边界输出；不要高频轮询或反复打印大日志。

## 9. 合法终态

- `NIGHT10A_REV1_QCRD_BALANCED_CANDIDATE_LOCKED`
- `NIGHT10A_REV1_QCRD_ACCURACY_FRONTIER_LOCKED`
- `NIGHT10A_REV1_QCRD_MULTIPLE_FRONTIERS_LOCKED`
- `NIGHT10A_REV1_QCRD_NO_POSITIVE_SIGNAL`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BLOCKED_MISSING_FROZEN_EVIDENCE`
- `WALLCLOCK_BUDGET_REACHED`

`IMPLEMENTATION_SEMANTICS_INVALID` 只能用于 REV1 contract 与真实实现不一致、标签泄漏或系统性 source 错位，不能用于普通候选失败。

## 10. Git、compact 与关机

D盘目标：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev1_handoff_20260821/official_compact`

只下载：

- 通俗总结、完整报告、decision；
- REV1 contract 与逐字段 semantic coverage；
- QCRD代码、runner、evaluator、测试；
- frozen-quality摘要、config/output manifests；
- metric backfill、per-seed metrics、frontier registry；
- label-read、resource、timeout、independent recomputation audits；
- root-relative delivery index；
- 最小增量 Git bundle。

禁止下载 raw runs、checkpoint、full embedding、affinity 或全历史 bundle。

Windows复算全部 compact SHA。final文件全部进入同一个 commit后创建 final tag一次，普通push并核验。

最后保留同一SSH会话，将 `/usr/bin/shutdown` 作为最后一条远端命令；派发后不重连，只报告命令派发状态。

## 11. 最终回复必须通俗

先回答：

1. QCRD到底学会了什么；
2. A1、D1、P22是否同时提高，tonsil是否保住；
3. 哪个候选是准确率/平衡/空间前沿；
4. 运行代价；
5. 下一步是端到端整合还是放弃QCRD。

随后再给技术表、失败单元、标签读取次数、Git和D盘SHA。不得只返回状态码。
