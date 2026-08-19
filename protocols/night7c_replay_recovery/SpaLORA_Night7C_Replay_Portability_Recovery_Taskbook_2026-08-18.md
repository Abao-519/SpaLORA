# SpaLORA Night-7C replay portability recovery 任务书

日期：2026-08-18  
父提交：`e34567db5ace4f0fcdd2526cfb94a84fc9148020`  
父标签：`night7c-final-20260818`

## 0. 用通俗语言说明

上一轮没有跑正式实验，而是在第六个无标签复放单元停下。原因不是标签泄漏或 checkpoint 损坏，而是当前 RTX 4080 SUPER 把 Night-7B 在 RTX 4080 上记录的一个 float32 首轮 loss 复算出了约 `0.028%` 的相对差异，旧协议却要求绝对误差小于 `1e-7`。

本恢复轮先用真正与历史训练相同的操作顺序重放全部 30 个单元。路由最终固定使用 Night-7B 当时真实记录并哈希锁定的首行 MNN，而不是选择当前显卡上更有利的重算值。恢复门通过后，才继续原 Night-7C 尚未开始的 routing、weighted-MNN 和运行加速实验。

## 1. 不可修改的边界

1. 原 `night7c-final-20260818`、commit、报告和 invalid 终态永久保留，不得 force、移动 tag 或修改历史交付。
2. 本轮只修订 `m_initial` authority/replay 语义。原 registry 的 T00–T09、W00–W05、阈值、seed、预算、顺序、评价门和保护门一个字符都不能改。
3. 用户手动派发；不得调用 AutoDL API，不得自动启动其他 Codex。服务器使用有卡模式。
4. 不确定立即停止。不得猜 checkpoint、补造 initial state、放宽新门、按 dataset/标签/指标选择数值。
5. 总锁前禁止打开原始 h5ad 或读取 obs/标签。恢复诊断只能使用 Night-7B 已登记的 stripped arrays、checkpoint、indices、loss curves 和 manifests。
6. invalid 远端目录中的 `p1_features` 一律不复用；新根固定为 `/root/autodl-fs/night7c_replay_recovery_20260818`。

## 2. 权威文件与先读顺序

先核验本轮 planning delivery index，再完整读取：

1. `SpaLORA_Night7C_Replay_Portability_Independent_Audit_2026-08-18.md`
2. `SpaLORA_Night7C_Replay_Portability_Recovery_Registry_2026-08-18.json`
3. 本任务书
4. 原 Night-7C planning index、source audit、registry、taskbook
5. Night-7C invalid compact index/report/P1 contract
6. Night-7B compact index、report、locked R1/R2 manifests、source index、训练源码及 tests

Recovery registry 是本轮 revision 的机器可读权威；原 Night-7C registry 继续是科学候选权威。两者若出现 revision scope 以外的冲突，立即 `BLOCKED_PREFLIGHT`。

## 3. P0-RECOVERY-AUTHORITY

正式动作前必须：

- D 盘 Night-7C compact `38/38` 重新核验，index SHA 和 compact root 精确匹配。
- Night-7B compact `67/67`、root SHA、30 个 R02 unit、60 个 six-view archives、R02 checkpoints、fixed indices 和 loss curves 重新核验。
- 核验 Night-7C branch/tag peel 到 `e34567...`，新 branch 从该 commit 创建；旧 tag 不动。
- 证明 Night-7C 正式 training/transform/label access 均为 0。
- 建立路径级 deny guard；新增测试证明 replay 脚本不能读取原始 h5ad、obs、labels、dataset identity 或 metrics。
- 记录 GPU/driver/CUDA/cuDNN/cuBLAS/PyTorch、TF32、`CUBLAS_WORKSPACE_CONFIG`、deterministic 与 `warn_only` 状态。

任一失败：`BLOCKED_PREFLIGHT`，0 replay、0 正式单元。

## 4. P1A-HISTORICAL-ORDER REPLAY

### 4.1 必须拆成真正独立的进程

每个 unit 的 initial replay 进程在执行首轮 forward 之前：

- 不得加载 final checkpoint 到 CUDA；
- 不得执行 final model forward；
- 不得执行 endpoint、affinity 或 clustering；
- 不得先运行同 shape 的其他模型 forward。

必须按 Night-7B `night7b_train.py::train` 的历史顺序逐步实现：seed → summaries/reliability → model init/to CUDA → targets → relation/MNN/mask tensors → optimizer/scheduler → pre-forward RNG → first training forward → MNN。不得用“数学上应该等价”的重排代替。

checkpoint 只可在 CPU 上读取已保存的 `initial_rng` 和 config/contract；若 `torch.load` 本身可能影响 CUDA，使用 `map_location=cpu` 且在任何 CUDA context 初始化前完成，记录顺序。

### 4.2 30/30 × 两次重复

每个单元运行两次独立历史顺序 replay，记录 registry 要求的全部环境、SHA、state/RNG、历史值、重算值和 router 稳定性字段。两次必须使用同一固定代码和环境，不得单元级调参或重试。

另外对 A1、tonsil、D1、P22 各第一个 unit 运行一次旧 Night-7C 顺序 probe，仅用于判断“先 final forward”是否改变结果；它不是权威值，不能用于 routing。

### 4.3 权威值与通过条件

最终 `m_initial` 必须直接来自已锁定的 Night-7B `loss_curve.csv` 首行 `MNN`。生成一个 30 行 authority manifest，至少包括 unit、历史 loss path/SHA、training manifest path/SHA、固定 index SHA、历史值与 authority row SHA。

严格执行 recovery registry 全部门：

- 当前硬件两次 replay 误差 `<=1e-7`；
- 相对历史绝对误差 `<=1e-3` 且相对误差 `<=1%`；
- T04 hard decision 0 翻转；T02/T03 alpha 最大绝对差 `<=0.01`；
- initial state SHA 两次一致；公式/indices 0 mismatch；全部有限。

全部通过后状态写为 `PASS_BY_LOCKED_HISTORICAL_FEATURE_AUTHORITY`。任一失败为 `BLOCKED_REPLAY_PORTABILITY_RECOVERY`，不得训练、transform 或开标签。

## 5. P1B-FINAL STATE 与特征冻结

P1A 通过后，在另外的 fresh process 中完成原 Night-7C 要求的 30/30 final checkpoint、embedding、gate、endpoint affinity 与 canonical partition exact parity。不得把 final-forward 进程和 initial replay 合并。

从 final checkpoint 和锁定 arrays 派生 `c_i/rank_c_i/q_i/s_i`；从 historical authority manifest 读取 `m_initial`。冻结 30 个 feature files 的 SHA。全部 router formula、边界和 label-deny 测试按原任务书执行。

## 6. 条件继续原 Night-7C

只有 P0/P1A/P1B 全通过后，才按原任务书原顺序继续：

1. P2 runtime profiling 与 exact-output acceleration；
2. Stage T：240 个无训练 transforms；
3. Stage W：48 个固定 pilot training + 48 transforms；
4. 两条路径全部总锁 SHA；
5. 只开一次标签窗口；
6. 完全按原 registry selection order、门槛和终态判定。

原预算不因 invalid Night-7C 消耗而减少，因为旧轮正式 scientific training/transform 均为 0。scientific retry 仍为 0；implementation correction 总上限仍为 8，并且不能用于修补某个不理想结果。

## 7. GPU/CPU 与时间

- 模型训练必须实测 CUDA tensors/parameters/loss 和正 peak GPU memory。
- CPU 谱分解、聚类与 sparse transform 可导致 GPU=0；必须输出 phase-aware 1 秒采样日志。
- 先跑原 CPU backend；并行/GPU backend 只有历史 reference exact parity 后才可采用。
- 长日志写文件，聊天中每阶段用通俗中文报告：做了什么、通过多少、下一步是什么。

## 8. Git 与交付

- Branch：`revision/q2-night7c-replay-portability-recovery-20260818`
- Protection tag：`baseline/pre-night7c-replay-portability-recovery-20260818`
- Final tag：`night7c-replay-recovery-final-20260818`
- 全程普通 push；禁止 force/force-with-lease；final tag 只在 tracked 报告/索引提交完毕后创建一次。
- raw/checkpoint/affinity 留远端持久盘；D 盘 compact `<25 MB`，只交报告、manifest、表、测试、源码、增量 bundle 和索引。
- 交付必须明确区分旧 invalid run、replay diagnostic 与新的正式科学结果。

## 9. 关机

完成所有下载、hash、GitHub peel、D 盘校验后，保留同一 SSH 控制会话，把 `/usr/bin/shutdown` 作为最后一条远端命令。派发后不重连、不调用 AutoDL API，只报告客户端派发状态，不虚报控制台电源状态。
