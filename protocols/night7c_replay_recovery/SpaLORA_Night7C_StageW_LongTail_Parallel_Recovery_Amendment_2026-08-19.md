# SpaLORA Night-7C Stage-W 长尾并行恢复修订

日期：2026-08-19  
性质：标签前、科学结果前的基础设施调度修订；不修改模型或算法

## 1. 当前事实

实验 Codex 的只读现场审计确认：

- Night-7C recovery 已完成 Stage T；Stage W 的 48 个训练/checkpoint 已完成并验证。
- Stage W transform 尚为 `0/48`，标签窗口尚未打开。
- 当前仅-transform driver 的第一个单元是 `W00_FILTER75 / u000 / A1 seed 0 / H01`。
- 当前进程持续单核约 100%，整机 12 核归一化约 8–9%，GPU 0%，RSS 约 575 MiB。
- 栈位于固定 `sklearn SpectralClustering(assign_labels="discretize")` 的 SciPy/ARPACK 特征分解；不是磁盘等待或死锁。
- 该次调用已经超过 8 小时仍无 manifest；此前同一单元还有一次因实例关闭而中断、约 3 小时 44 分的无效尝试。

`SpectralClustering` 当前实现没有显式 `eigen_solver/eigen_tol`，在 sklearn 1.1.1 下默认走 ARPACK 且 `eigen_tol=0.0`。SciPy `eigsh(tol=0)` 追求机器精度，默认 `maxiter=n*10`；数值条件差的稀疏图可能出现极长单核尾部。

## 2. 为什么允许修订调度门

原 Night-7C P2 已在四个无标签历史单元上比较 serial 与 4-worker：输入、canonical affinity、partition、cluster count 和 pre-label artifacts 均 exact；4-worker 无新增数值失败。它未被采用的唯一原因是短任务墙钟只达到 `1.17498×`，低于人为预设的 `1.5×` 工程门。

该基准的单元约几十秒，启动/加载开销占比很高；现在正式 workload 暴露了超过 8 小时的 ARPACK 长尾。因此旧速度门对正式 workload 的外推已经失效。调度速度不是科学候选或评价结果，且目前 0/48、0 标签；在看到任何 ARI/NMI/Q 之前修订该工程门，不构成按结果调参。

本修订只允许把多个互相独立的 transform 同时调度。每个 transform 内部仍调用完全相同的 affinity、`SpectralClustering`、ARPACK、`assign_labels="discretize"`、random state、K、输入、容差和输出 schema。

## 3. 对当前长尾进程的授权

用户把本修订提示词手动发给实验 Codex后，视为授权以下一次性基础设施纠正：

1. 先只读记录当前 driver/child PID、PGID、完整 command line、start/CPU time、RSS、线程、GPU、Stage-W manifest 数量、标签访问计数和输出目录。
2. 若当前单元已经生成完整、SHA 可核验的 success/failure manifest，保留它，不重跑。
3. 若仍无 manifest，向**精确核验后的当前 driver 进程组**发送 graceful TERM；不得用模糊 `pkill python`，不得影响 SSH、其他任务或已完成训练。
4. 等待 60 秒；只在同一已核验进程组仍存活时才 KILL，并记录信号、时间和进程树。
5. 将无 manifest 的部分目录连同日志和文件 SHA 移入新的 `invalid_attempts/stagew_serial_arpack_longtail_20260819/`；不得删除、覆盖或复用。
6. 该中断记为一次 infrastructure scheduling correction，不记为 scientific retry；只有原 recovery registry 的 implementation-correction 预算尚有至少 1 次时才允许执行。

## 4. 新的正式调度后端

后端固定为原 P2 已验证过的：

- `4` 个独立 worker processes；
- 每 worker：`OMP_NUM_THREADS=8`、`MKL_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=8`；
- 每个 worker 同一时刻只运行一个 transform；
- 总 RSS `<48 GiB`；不得 oversubscribe 到超过 4 个 worker；
- 每个 unit 使用独立、预先不存在的 attempt/output 目录；
- task submission 按 registry 顺序，完成允许乱序；最终 manifest/评价严格按 registry order 排序；
- 任何已经有完整有效 manifest 的 unit 不重跑；任何被中断且无 manifest 的 unit 从头运行一次；
- scientific retry 仍为 0，solver failure 原样保留并继续下一注册单元。

旧 `>=1.5×` 墙钟接受条件被本修订替换为：

1. 启动正式 worker 前，重读原 P2 parity report，确认四个 reference units 的 serial/parallel scientific artifacts exact 且无失败增量；
2. 新调度代码不得改变 per-unit transform 函数或参数；做静态调用链/hash 对比；
3. 4-worker 启动后记录实际 CPU/RSS/GPU 和每单元墙钟，不再因短任务速度比低于 1.5 而退回 serial；
4. 任一输入、算法参数、affinity 或 output schema 发生变化则 `IMPLEMENTATION_SEMANTICS_INVALID`。

## 5. GPU 为什么暂不启用

GPU eigensolver 会更换数值后端和浮点归约顺序。原任务书要求 60 个历史 C00/R02 单元 canonical partition SHA 全部 exact 才能正式采用；该门尚未完成。当前临时切换 CuPy/cuGraph/PyTorch LOBPCG 会改变科学路径，因此本修订不授权安装、替换或使用 GPU spectral backend。

GPU 加速可在 Night-7C 完成后独立做 60/60 parity 工程任务，不应拿正在运行的 Stage W 冒险。

## 6. 标签、结果与交付

- Stage T 和 48 个已锁训练/checkpoint 不得修改或重跑。
- 4-worker 完成全部 remaining transforms、全部 manifest/affinity/partition SHA 总锁之前，标签访问必须保持 0。
- 标签窗口后禁止返回 transform 或调度修改。
- 报告必须披露：旧短任务 P2 为 1.17498×、为何修订、两个 W00/u000 中断尝试、并行资源轨迹、每单元耗时分布和最终失败数。
- Git 只普通 push；旧 tag 不动；raw/partial artifacts 留远端，D 盘仍只交 compact。
- 全部交付完成后才派发 `/usr/bin/shutdown`，且作为最后一条远端命令。

## 7. 通俗结论

当前不是“12 个核都能共同加速一个 transform”，而是“一个 transform 只能吃一个核，但可以同时跑四个 transform”。这不会让已经卡住的 W00/u000 本身突然变成 12 倍快，却能让它运行期间另外三个单元同时推进，避免 11 个核心长期闲置。对当前 48 个相互独立任务，这是最快且不更换科学算法的恢复方式。
