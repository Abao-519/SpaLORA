# SpaLORA Night-7C Stage-W 资源截止与有界继续修订

日期：2026-08-19  
性质：标签窗口开启前的基础设施与资源治理修订；不修改训练结果、科学输入、模型、loss、affinity、聚类算法或评价指标。

## 1. 现场与判断

- Stage T 已锁；Stage W 的 48/48 training/checkpoint 已完成。
- 当前 `W00_FILTER75/u000/A1 seed0/H01` 在固定 `SpectralClustering -> ARPACK/eigsh` 内持续单核计算，仍未写出终态 manifest。
- A1 规模约 3,484 observations；Stage T 的 240 个同类谱聚类平均约 19.5 秒。当前十余小时长尾不能用数据规模正常解释。
- 固定后端的 `eigen_tol=0` 对应机器精度，`eigsh` 默认最大迭代为 `n*10`。进程持续占用 CPU 说明它不是睡眠死锁，但并不代表继续无限计费具有工程合理性。

因此，将此现象定义为 **数值求解长尾与资源不可接受**。它可能最终收敛，也可能到迭代上限失败；本修订不猜测结果，而是设置统一、事前锁定的资源边界。

## 2. 当前 W00 的一次性截止线

当前 PID 必须按 `/proc/<pid>/stat` 的 start-time 精确识别。其连续 wall time 达到 **16 小时**时为硬截止：

- 预计北京时间约 `2026-08-20 01:50`，但服务器 monotonic/process start-time 是唯一权威；
- 若部署本修订时已经达到 16 小时，完成一次只读状态捕获后立即执行截止，不再追加宽限；
- 若截止前自然写出完整 success/failure manifest，沿用既有边界守卫：立即暂停旧 driver、验证并保留该自然终态；
- 若截止时仍无完整 manifest，先 `SIGSTOP` 精确 PID/PGID，记录进程树、CPU/RSS、日志和部分文件 SHA，再精确 `TERM`；60 秒后仅在 PID start-time 仍相同时才 `KILL`；
- 禁止模糊 `pkill python`，禁止影响守卫、SSH、协调器或无关进程。

无 manifest 的 W00 必须登记为：

`RESOURCE_CENSORED_WALLTIME_16H`

它不是 success，不是 ARPACK 自然 numerical failure，也不是 scientific retry。不得补值、推断 partition、读取标签评价或从中间续算。

## 3. W00 候选的处理

原加权 MNN promotion gate 要求候选 8/8 cells 完整。

- 若 W00 在 16 小时内自然终态且有效：保留 u000，并继续其余注册单元。
- 若 W00/u000 被资源截尾：`W00_FILTER75` 整个候选标为 `INELIGIBLE_RESOURCE_CENSORED`；其尚未开始的另外 7 个 transforms 机械跳过，因为完整性门已不可能满足。不得为了观察分数继续消耗资源。

跳过必须写 manifest，原因只能是由本修订预先定义的 candidate-level resource circuit breaker；不得伪称运行失败。

## 4. 后续单元的统一资源门

对截止后尚未开始的每个 transform：

- scientific function、H01、ARPACK、输入、affinity、K、random state、容差、maxiter 与 schema 均保持原样；
- 最多 4 个独立 worker，每个 transform 使用独立子进程组和独立预先不存在的输出目录；
- **每个 transform wall-time 上限 60 分钟**；
- 到时仍无完整终态 manifest，按 `RESOURCE_CENSORED_WALLTIME_60M` 保存现场并终止精确子进程；
- scientific retry=0，fallback=0；资源截尾单元不得重跑；
- 任一候选出现首个资源截尾，其未开始单元全部按 `SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER` 终止调度；
- 候选必须 8/8 自然成功/注册允许的完整终态且满足原 promotion contract，才有资格进入标签后比较；不得以少量 seed 均值代替；
- 资源有界继续阶段总 wall-time 上限 12 小时。达到总上限时停止所有未完成子进程、锁定资源截尾证据并进入有界收尾。

四进程只是并行不同 transform，不得替换为 GPU eigensolver、LOBPCG、AMG、Leiden、较松 tolerance 或其他新科学 endpoint。那些替代方案只能在下一轮作为新方法预注册测试。

## 5. 协调器切换

当前已部署的自动协调器在 W00 自然终态后会启动无单元超时的 47-cell 队列。为避免它抢先启动：

1. 在不触碰 W00 driver 和边界守卫的前提下，精确停止旧协调器；
2. 保留旧协调器代码、日志与 SHA，不覆盖；
3. 新建 resource-bounded coordinator，先通过静态调用链核对和 synthetic process-timeout 测试；
4. 新协调器同时处理“W00 截止前自然终态”和“W00 到 16 小时被截尾”两条路径；
5. 已存在的完整 manifest 永不重跑，partial 永不复用。

若部署时旧协调器已启动后续单元，立即停止新增提交，保留已完整结果，精确隔离 partial，并按本修订重新机械计算 remaining set；不得覆盖。

## 6. 标签与结论

- 所有已运行、自然失败、资源截尾和规则跳过单元全部锁定前，标签访问保持 0。
- 标签打开后只评价完整 eligible candidates；资源截尾/跳过候选只报告计算可行性，不计算不完整 seed 的科学均值。
- Stage T 结果不因 Stage W 资源截尾而作废；最终仍可对完整 Stage T router 按原规则评价。
- 若没有完整 eligible weighted-MNN candidate，如实报告 `NO_ELIGIBLE_WEIGHTED_MNN_CANDIDATE_UNDER_RESOURCE_BOUND`，保留 Night-6D/7A 的 C00 与 Night-7B 的 P22 frontier，不把运行问题包装成性能结论。

## 7. 交付与关机

- 报告必须列出：两次历史 W00 中断、当前连续运行时间、16h/60m/12h 资源门、每个资源截尾与跳过单元、四进程资源轨迹。
- Git 只普通 push；旧 tag 不移动；raw/partial 留远端，D 盘只交 compact。
- 全部审计、Git、compact 与本地校验完成后，才把 `/usr/bin/shutdown` 作为最后一条远端命令派发。

