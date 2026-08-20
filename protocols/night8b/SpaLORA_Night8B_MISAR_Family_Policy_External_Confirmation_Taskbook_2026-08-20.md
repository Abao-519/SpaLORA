# SpaLORA Night‑8B：MISAR 冻结模态家族策略外部确认任务书

版本：2026‑08‑20 REV1  
任务性质：一次性外部确认；两条既有路线正面对比；不搜索新候选  
核心问题：Night‑7B 的 RNA+ATAC 专用 R02 在独立 MISAR E15.5 S1 上是否仍优于通用 C00？

## 0. 第一条回复、算力与总时限

第一条回复必须原样是：

> 已读取 Night‑8B MISAR 冻结模态家族策略外部确认任务书，开始 P0‑AUTHORITY‑AND‑PROVENANCE。本轮固定比较 U00=C00 与 F00=R02，不新增候选、不读取标签、不运行第三方正式 benchmark。

本任务必须由用户在 AutoDL **有卡模式**开机后执行。不得调用 AutoDL API，不得自动开机。训练必须使用 CUDA；聚类/统计阶段允许 CPU，但不能因此改用无卡模式。

硬时限：

- P0 合计不超过 60 分钟；
- 每个训练单元 wall time 不超过 60 分钟；
- 每个 transform 不超过 20 分钟；
- 全任务从 P0 开始到最终交付不超过 6 小时。

达到时限必须保留现场并停止 `TIME_BUDGET_EXHAUSTED`，不能像 Night‑7C replay 那样让单个 CPU transform 无上限运行。

## 1. 权威输入

### 1.1 Git 与恢复结论

- base commit：`b2eb62d8ddb1b80b14f916e4dcbbcb407581c6ca`
- base tag：`night8a-eval-recovery-final-20260820`
- 新分支：`revision/q2-night8b-misar-family-policy-external-20260820`
- protection tag：`baseline/pre-night8b-misar-family-policy-external-20260820`
- final tag：`night8b-final-20260820`

Windows 权威 compact：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8a_eval_recovery_handoff_20260820/official_compact`

必须独立核验：

- compact index：`14b25477e72bb1943620c5fb0056d42913bee201dd2d00d22f8b1070f778e96e`
- compact root：`cb378fdb89c8e56d26aaebb9f7b9dce715ae6971cc314d3152f8cb9759550314`
- report：`22e3c67a2f600a53b50270b8008f2527810b90aee7cb643a7dd88faaf7f539a3`
- decision：`3a92c62fe4be9cfe5909b9e0270504392cec067abd7b59efeb6d642d34f48b77`

原 Night‑8A 仍为 `IMPLEMENTATION_SEMANTICS_INVALID`；恢复轮只救回了评价，且最终 shortlist 为空。不得将 MF‑SPC B01–B07 带入本轮。

### 1.2 机器可读注册表

先读取并逐字段执行：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_planning_20260820/SpaLORA_Night8B_MISAR_Family_Policy_Registry_2026-08-20.json`

该 JSON 与本任务书冲突时立即停止并报告，不自行猜测。

## 2. P0‑AUTHORITY‑AND‑PROVENANCE

### 2.1 纠正来源编号

MISAR E15.5 S1 的权威来源链固定为：

- 原始研究/主编号：`OEP003285`；
- 可验证的 NCBI raw-read 交叉编号：`SRP491963`；
- 官方处理后记录：Zenodo `7480069`，标题 `MISAR-seq`；
- 1949 spots，与原论文 E15.5 S1 数量一致。

原 Night‑8A `external_dataset_lock.json` 中的 `GSE213264` 是已发现的错误交叉引用。GSE213264 属于 spatial‑CITE‑seq RNA+蛋白研究，不是 MISAR RNA+ATAC。新建 `corrected_misar_provenance_lock.json`，必须删除该错误关联并保留纠正说明；不得改写原 Night‑8A 文件。

### 2.2 训练数据复核

只使用 `/root/autodl-fs/night8a_external_data_20260820` 中已下载的四个官方文件，核验 bytes、MD5、SHA‑256 与注册表完全一致。训练数据为官方 Zenodo 原始处理矩阵、坐标和 barcode 文件，不使用任何 label carrier 中的表达矩阵。

要求确认：

- 1949 个选中 tissue spots；
- RNA 32,285 features、ATAC 141,420 peaks；
- RNA/ATAC 同 spot 配对；
- 坐标唯一且与 barcode 确定性对应；
- 所有训练副本为零 `obs` 列或等价的无标签数组。

### 2.3 标注载体与 K

下载到远端持久盘，不下载到 Windows：

- Figshare article `21623148` v5；
- `MISAR_seq_mouse_E15_brain_data.zip`；
- bytes `21670493`；MD5 `76fb78c20d218baf8c964d214a5899b2`；
- URL `https://ndownloader.figshare.com/files/42520831`。

只允许在 pre-lock 阶段：

- 核验 ZIP、MD5、内部文件名；
- 通过 `h5py` 低层接口读取 HDF5 key、shape、dtype；
- 读取 `cell` 与 `pos` 用于建立确定性 observation mapping；
- **禁止读取、切片、转数组、统计或输出 `Y` 的任何值**。

主标注载体固定为 `MISAR_seq_mouse_E15_brain_ATAC_data.h5::Y`，K 在看值前依据公开 benchmark 锁为 12。用 cell ID 优先、坐标为独立复核，生成 `prelabel_observation_mapping.csv` 与 SHA。若无法得到 1949/1949 一一映射，停止 `BLOCKED_PROVENANCE_OR_ANNOTATION`。

在最终评价窗口打开 Y 后，如果实际 unique K 不是 12、存在无法解释的缺失值、或 mapping 不再成立，停止同一状态；禁止重聚类、换粗粒度 7 类标注或试多个 label mapping。

## 3. P0‑DEPLOYABILITY‑AND‑CUDA

### 3.1 两条路线必须可在新数据上从零构造

只允许两条已冻结方法：

1. `U00_UNIVERSAL_C00 = C00_G04_H05_CONFIRMED`；
2. `F00_FAMILY_R02 = R02__E1_ADAPTER_C06_MEAN__H01`，即 RECON+MNN、equal fusion、E1 adapter/C06 mean、锁定 H01。

必须从 Night‑7B 注册表解析精确算法和超参数，并生成 canonical config SHA。不得用数据集名、组织名或标签决定参数；唯一允许的路由信息是输入模态 schema=`RNA_EPIGENOME`。

F00 必须对 MISAR 自己的六视图重新训练 adapter。严禁把 P22 的 embedding、spatial support、checkpoint、MNN pairs 或 cluster 当成 MISAR 输入。P22 历史文件只可用于代码语义 parity，不可进入训练 DAG。

预处理沿用 Night‑7B 已冻结的 RNA_EPIGENOME 路径；不得为 MISAR 另选 HVG 数量、LSI 维数、邻居数、epoch、loss weight 或聚类头。若旧代码不能对新路径运行，只能做“路径/数据适配”，且要通过与历史 P22 的只读 parity 测试；不得改变数学语义。

### 3.2 CUDA 与性能探针

在正式训练前记录：GPU 型号、CUDA、PyTorch、CPU/内存、驱动和环境 lock。必须：

- `torch.cuda.is_available()==True`；
- 真实 CUDA tensor 运算通过；
- 一个最小模型 forward/backward 确认参数、输入和 loss 均在 CUDA；
- 每单元 manifest 写入 device、wall time、peak GPU MiB、CPU time、checkpoint SHA。

若训练进程自称 CUDA，但连续 2 分钟 GPU memory/利用率均为 0，停止正式矩阵，先做一次全局基础设施诊断；不能悄悄退回 CPU。

### 3.3 P0 smoke

仅用固定 seed 0 的小步数临时 smoke 验证数据流、checkpoint reload 和两条 transform 可以结束。smoke 不评价、不保存为正式结果、不读取 Y。完成后删除的只能是明确标记的临时 smoke 目录，不得删科研失败。

P0 全过后先普通 push protocol/config/测试 commit，再开始正式训练。

## 4. 正式训练与总锁

固定 seeds：`0,1,2,3,4,5,6,7,8,9`。

每 seed：

1. 训练一次可复用的 C00 六视图/base state；
2. 从该 seed 自己的六视图训练一次 R02 adapter；
3. 立即保存 final checkpoint、optimizer/config/RNG state 与完整 SHA；
4. reload 必须逐元素复现 embedding；
5. 生成 U00、F00 两个 K=12 partition 与 affinity；
6. 锁定 observation order、embedding、affinity、clusters、checkpoint、config 和 manifest SHA。

预算：10 base + 10 adapter = 20 个科学训练单元；20 个正式 transforms。科学 retry=`0`。label 打开前只允许最多 4 次“影响全局且有明确根因”的基础设施/实现纠正；一旦修正，所有受影响单元必须一致补跑，不能只补负结果。

全部 20 个 partition、20 个 checkpoint round-trip 和总 manifest 完成并普通 push 后，生成：

`locked_misar_training_and_prediction_manifest.json`

在这之前 evaluator 不得拥有 Y 路径或文件句柄。

## 5. 单次授权标签窗口

只允许独立 evaluator 在总锁 commit/push 之后读取一次 `Y`。读取后禁止返回训练、adapter、affinity、聚类、mapping 或候选修改。

对 20 rows 计算：

- primary：ARI、NMI、Q=`(ARI+NMI)/2`；
- secondary：AMI、FMI、homogeneity、completeness、V‑measure；
- spatial：neighbor agreement、Moran’s I、Geary C、boundary disagreement；
- paired contrast：同 seed `F00-U00`；
- runtime/peak GPU：端到端口径，同时另表 adapter 增量成本。

统计固定为：

- ΔQ 单侧 exact paired sign-flip permutation：对 10 个 paired deltas 穷举全部 `2^10` 个符号组合，以 mean ΔQ 为统计量；
- 100,000 次 paired bootstrap，seed=`20260820`；
- 报告 95% CI、10 seed wins 和全部逐 seed 值。

必须明确：10 seeds 衡量算法随机稳定性，不等于 10 个独立生物样本。

独立脚本复算全部指标与 paired contrast，最大误差 `<=1e-12`；否则 `IMPLEMENTATION_SEMANTICS_INVALID`。

## 6. 预注册判定

### 6.1 科学门

F00 相对 U00 必须同时满足：

- mean ΔQ `>= +0.010`；
- mean ΔARI `>= 0` 且 mean ΔNMI `>= 0`；
- Q wins `>=8/10`；
- 单侧 exact sign-flip p `<0.05`；
- paired bootstrap 95% CI 下界 `>0`；
- neighbor Δ `>=-0.01`、Moran Δ `>=-0.02`、Geary Δ `<=+0.02`、boundary Δ `<=+0.01`。

### 6.2 资源门

- F00 端到端 runtime/U00 `<=1.50x`；
- F00 peak GPU/U00 `<=1.25x`。

### 6.3 终态

- 科学门和资源门都过：`NIGHT8B_MISAR_FAMILY_POLICY_BALANCED_CONFIRMED`；
- 科学门过、资源门不过：`NIGHT8B_MISAR_FAMILY_POLICY_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST`；
- mean ΔQ>0 但科学门未全过：`NIGHT8B_MISAR_PARTIAL_OR_MIXED_EVIDENCE`；
- mean ΔQ<=0 或出现明确反向：`NIGHT8B_MISAR_FAMILY_POLICY_NOT_GENERALIZED`。

不得放宽阈值，不得把最高 seed 当结果，不得换 7 类标注补救。

## 7. 公开成绩只作语境，不作本轮选择

在且仅在确认同一 1949 spots、同一 12-region annotation 后，可以在报告中列出：SEPAR 论文报告 ARI 0.644、其文中 SpatialGlue ARI 0.371。它们是文献中的单点语境，不是本轮重跑结果，也不能据此宣称 SOTA。

Night‑8B 不运行 SMART、SEPAR、SpaLLM、SpatialGlue 或其他第三方正式 benchmark。无论本轮绝对 ARI 多高，都只写“外部确认结果”；正式 SOTA 比较需后续固定 method 后另开任务。

## 8. 测试、Git 与交付

至少新增并通过：

- GSE213264 provenance 负向回归测试；
- prelabel HDF5 审计绝不访问 Y 的测试；
- 1949 observation mapping 测试；
- U00/F00 config SHA 和 dataset-identity-blind selector 测试；
- P22 artifact 不进入 MISAR DAG 的测试；
- CUDA device、checkpoint round-trip、K=12、label-window 单向门测试；
- timeout、无 retry、独立 metrics 复算测试。

Git 只普通 push；禁止 force/force-with-lease；final tag 只创建一次，且必须在报告、测试、delivery index 全部进入最终 commit 后创建。

D 盘 compact 根：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night8b_handoff_20260820/official_compact`

compact 目标 `<10 MB`，包含协议、注册表、报告、通俗总结、provenance lock、mapping SHA、20-row metrics、paired summary、统计/独立复算、资源审计、测试、Git 审计、delivery index、planner tar 和增量 bundle。不得下载 raw matrices、raw runs 或 checkpoints 到 Windows。

## 9. 关机

所有远端工作、普通 push、D 盘复制与 Windows 独立校验完成后，在保留的同一 SSH 会话中把 `/usr/bin/shutdown` 作为最后一条远端命令。派发后不重连、不调用 AutoDL API，只如实陈述派发和连接状态。

## 10. 给用户的最终表述

最终输出先用通俗语言回答三件事：

1. R02 在新 MISAR 上到底有没有比通用 C00 更好；
2. 提升是否同时体现在 ARI/NMI、空间结构和多数 seeds；
3. 这意味着“按模态家族选择融合策略”可以保留，还是 P22 只是专科特例。

随后再列科学表格、偏差、commit/tag、D 盘路径和关机派发状态。
