# SpaLORA Night-7C Stage-W 立即停止 W00 与语义分诊修订

日期：2026-08-19  
性质：用户在标签仍关闭、W00 仍无终态 manifest 时作出的资源终止决定。本文件覆盖此前“等待 W00 连续运行满 16 小时”的条款；其余 60 分钟单元上限、候选熔断、总资源上限、标签防火墙、Git、交付和关机要求继续有效。

## 1. 立即停止决定

不再等待当前 `W00_FILTER75/u000/A1 seed0/H01` 自然返回。

实验 Codex 收到本修订后必须：

1. 先只读捕获 driver、guard、resource-bounded coordinator、manifest、label-access、进程树、PID start-time、PGID/SID、CPU/RSS、日志和部分目录状态。
2. 若 W00 在本消息到达前已经写出完整且可验证的自然 success/failure manifest，保留自然终态，不得改写为资源截尾。
3. 若仍无完整 manifest，先精确暂停/终止会自动接管的 coordinator，再精确停用 guard，确认二者不会与手工停止发生竞态。
4. 对核验后的 W00 driver 先 `SIGSTOP`，再次确认 PID start-time、PGID 与命令行；保存现场后 `TERM`，60 秒后仅在同一进程仍存活时才 `KILL`。
5. 禁止模糊 `pkill python`，禁止影响 SSH、训练 checkpoint、Stage T、其他用户进程或持久盘成果。
6. 无 manifest 的 W00 登记为 `RESOURCE_CENSORED_USER_STOP_AFTER_EXTREME_LONGTAIL`；不是 success、不是自然 ARPACK failure、不是 scientific retry。
7. partial 文件、资源日志和进程证据移入新的 immutable incident 目录并逐文件记录 SHA；不得删除、覆盖或复用。

## 2. W00 候选结论

若 W00/u000 被资源截尾：

- `W00_FILTER75` 整个候选标为 `INELIGIBLE_RESOURCE_CENSORED`；
- 其余 7 个尚未开始的 transform 标为 `SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER`；
- 不运行、不补值、不读取标签、不计算不完整均值；
- 48/48 已完成 training/checkpoint 保留，W00 checkpoint 可用于未来纯工程 solver 诊断，但不能冒充本轮聚类结果。

## 3. 在继续前先回答“48 个是否存在系统性错误”

终止 W00 后，不得立即盲目启动 40 个正式谱聚类。先完成一次 label-free semantic triage：

### 3.1 训练与输入核验

- 48/48 training manifest、checkpoint、embedding、observation order、K、candidate/unit 主键和声明 SHA 全部复核；
- embedding 必须 shape 正确、finite、无 NaN/Inf、无全常数维度异常；
- 确认训练确实使用锁定 CUDA 路径、固定 epoch、无 retry/fallback；
- 确认不存在把错误 checkpoint、错误 dataset/unit 或旧 invalid 结果送入 transform 的映射错误。

### 3.2 affinity 的廉价结构诊断

只从锁定 embedding/checkpoint 机械构造 pre-clustering affinity，不调用正式 H01 谱聚类、不读取标签。对 48 cells 输出：

- shape、dtype、nnz、density、finite/nonnegative、symmetry error；
- degree min/median/max、zero-degree 数；
- connected-components 数及最大/最小 component size；
- isolated/near-isolated spots；
- affinity file/canonical SHA；
- 与同 dataset/seed 的已知快速 C00/Stage-T affinity 的上述结构差异；
- candidate×dataset 的异常汇总。

不得用 ARI/NMI/Q 或标签判断异常。不得用另一个 eigensolver 的聚类结果替代正式结果。

### 3.3 代码语义核验

- 核对 `transform_cell -> affinity build -> SpectralClustering` 的唯一调用链、sklearn/SciPy 版本、`K`、`eigen_solver/eigen_tol/maxiter`、`assign_labels`、random state；
- 确认 W00 没有意外稠密化、错误矩阵、维度放大、重复 block、错误线程/进程环境或输出锁竞争；
- 将 W00 与同一 A1 seed 的 W01–W05 预聚类 affinity 结构并列表比较。

## 4. 分诊后的机械决策

### A. 只有 W00/hard-filter family 明显异常，W01–W05 输入与 affinity 合约全部通过

- 锁定 triage 报告和全部 SHA；
- 启动 W01–W05 共 40 个 transforms；
- 4 workers，完全相同的正式 H01/ARPACK；
- 每单元 60 分钟硬上限；候选首个资源截尾即熔断；总继续阶段 12 小时；
- scientific retry=0、fallback=0；完整 manifest 禁止重跑。

### B. 多个 W01–W05 也出现合约错误、错误映射、非有限值、零度或明显系统性实现问题

- 不启动正式 40-cell 队列；
- 终态为 `IMPLEMENTATION_SEMANTICS_INVALID` 或准确的 preflight block；
- 交付分诊证据和最小修复建议，等待新的预注册任务，不得边修边跑。

### C. W01–W05 合约合法但多个 affinity 具有相似病态结构

- 不用当前无界 H01 继续烧资源；
- 终态为 `BLOCKED_NUMERICAL_ENDPOINT_SCALABILITY`；
- 保留 checkpoint/affinity，下一轮独立预注册 bounded ARPACK tolerance、LOBPCG/AMG、cluster_qr、Leiden 或 component-aware endpoint；不得在本轮偷换 solver。

## 5. 后续科研与交付

- Stage T 已锁结果不因 W00 截尾作废；若 Stage-W 无完整候选，仍可按原规则评价完整 Stage-T router。
- 全部运行/截尾/跳过与 triage manifest 锁定前，label access 必须为 0。
- 只评价完整 eligible candidates；不完整 W00 不进入科学均值。
- 报告须通俗说明：这是训练后谱聚类的数值长尾，不是 48 个 GPU 训练失败；但分诊会检查是否存在系统性输入/映射错误。
- Git 只普通 push；旧 tag 不移动；raw/partial 留远端，D 盘仅 compact。
- 全部审计、报告、Git、D 盘校验完成后，`/usr/bin/shutdown` 作为最后一条远端命令。

## 6. 权限与自主执行

用户明确授权实验 Codex在现有可用权限内完成上述精确进程控制、只读分诊、有界运行、Git/交付和最终关机，不需再次请求逐条确认。若应用或系统本身强制审批，Codex不能自行提高安全权限；应仅报告唯一真实阻塞，不得反复请求或绕过平台安全机制。

