# Night-7C Stage-W：保留 W00 长尾并在完成边界切换并行的权威修订

日期：2026-08-19  
性质：标签前、结果前的纯调度修订；不修改任何科学算法、输入、参数或输出语义。

## 1. 用户明确决定

当前正在运行的 `W00_FILTER75 / u000 / A1 seed0 / H01` 不得中断，让其使用现有固定 solver 自然返回成功或固定数值失败。不得因为耗时而改变 solver、容差、迭代上限、affinity、K、random state 或输出 schema。

本文件只覆盖此前 `SpaLORA_Night7C_StageW_LongTail_Parallel_Recovery_Amendment_2026-08-19.md` 第 3 节中“无 manifest 时立即终止并重跑 W00”的条款。此前文件的 exact-parity、四进程、标签防火墙、Git、交付和关机要求继续有效。

## 2. 必须使用事件守卫，而不是聊天高频轮询

旧串行 driver 在 W00 返回后会立即进入下一个注册单元。为既保留 W00 又避免与后续并行队列冲突，实验 Codex 必须在 W00 返回前部署一个不读取标签、不进行科学计算的服务器端边界守卫：

1. 精确记录 driver PID、PID start-time、PGID、完整 command line、当前 W00 输出目录及唯一终态 manifest/marker 的确切路径与写入方式。
2. 守卫只观察 W00 的自然 success/failure 终态文件。优先使用文件 close/move 事件；不可用时可用低频文件存在性检查。不得解析标签或评价指标。
3. 终态文件一旦出现，守卫必须先核对 PID start-time 未发生复用，然后立即向精确 driver PID/独立 PGID发送 `SIGSTOP`，阻止旧串行循环继续推进。
4. 守卫、SSH、Codex 或无关进程不得与 driver 位于同一被暂停的进程组。禁止模糊 `pkill python`。
5. 守卫将时间、PID、信号及命中的终态路径写入独立基础设施日志。平时不向聊天持续输出。

如果无法唯一确定终态文件或安全暂停边界，不得猜测；保持当前 W00 不动并报告阻塞。

## 3. W00 返回后的处理

1. driver 被 `SIGSTOP` 后，先只读验证 W00 manifest、全部声明 artifact、SHA、cluster count 和终态语义。
2. 若 W00 完整有效，将它计入正式 Stage-W transform，禁止重跑。
3. 只有完成第 2 项后，才精确终止旧 serial driver。先 `TERM`，60 秒后仅在同一 PID/start-time 仍存活时才 `KILL`。
4. 如果旧 driver 在守卫响应的极短窗口已建立下一单元的 partial 目录，将其原样移入新的 `invalid_attempts/stagew_post_w00_boundary_spill_20260819/` 并记录 SHA；不得计入 transform、不得覆盖或复用。这属于调度 spill，不是 scientific retry。
5. 若 W00 终态文件不完整，保持 driver 暂停，保全现场并报告；不得把不完整结果当作完成，也不得静默重跑。

## 4. 剩余 47 个单元

W00 完整有效后，剩余集合必须由注册表机械计算为“全部 48 个锁定单元减去有效 W00”；预期为 47 个，若不是 47 立即停止说明差异。

剩余单元采用此前已验证 scientific artifacts exact 的固定四进程后端：

- 最多 4 个独立 worker processes；
- 每个 worker 内仍调用完全相同的 H01 / `SpectralClustering` / ARPACK / `assign_labels=discretize`；
- 输入、affinity、K、random state、容差、迭代上限和 schema 全部不变；
- 每个单元写独立且预先不存在的目录；完整 manifest 单元禁止重跑；
- scientific retry 仍为 0；自然数值失败原样保留并继续；
- 提交按 registry 顺序，完成可乱序，最终聚合按 registry 顺序；
- 总 RSS 和线程设置仍服从此前并行修订，不得扩到 12 workers，也不得换 GPU solver。

## 5. 低 Token 监控约束

- W00 等待期间由服务器端事件守卫负责边界动作，不依赖大模型高频检查。
- Codex 主动状态检查最多每 60 分钟一次；状态未改变时不重复长篇解释，不反复读取大日志。
- 只在 W00 终态、守卫暂停成功、核验异常、四进程启动或需要用户处理的真实阻塞发生时输出简短更新。
- 这项低频约束不能降低边界保护：文件事件守卫应以秒级在服务器本地运行，但它不消耗对话 Token。

## 6. 其余硬门

- Stage T 与已完成的 48/48 training/checkpoint 不修改、不重跑。
- 全部 Stage-W transform 与总锁完成前，标签访问必须为 0。
- 不在本地 Windows 分担正式 transform。
- Git 只普通 push；旧 tag 不移动；D 盘只保留 compact。
- 全部科研、审计、Git 与本地交付完成后，才把 `/usr/bin/shutdown` 作为最后一条远端命令派发。

