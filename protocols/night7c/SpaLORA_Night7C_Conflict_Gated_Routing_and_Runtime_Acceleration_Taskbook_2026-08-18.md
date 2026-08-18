# SpaLORA Night-7C：冲突门控、置信度加权 MNN 与运行加速任务书

日期：2026-08-18  
性质：当前四数据集开发面板上的预注册 R&D；不是外部 benchmark，也不是 SOTA 确认  
父提交：`32d6ed947b313423805ee0f80c9dada06bb6a28d`  
父标签：`night7b-final-20260818`

## 0. 用通俗语言说明这轮要做什么

Night-7B 已经找到一个很强的 P22 专家 R02：P22 的 ARI/NMI/Q 分别比 C00 高约 `0.0833/0.0601/0.0717`，但把它无条件用在人类数据上会轻微掉分。R02 训练开始前的无标签 MNN 冲突恰好把 P22 与三个人类数据清楚分开。

Night-7C 要回答两个问题：

1. 能否只在跨模态冲突足够强、匹配质量足够可靠的 spot 上启用 R02，从而保留 C00 在 A1、D1、tonsil 上的成绩，同时保留 P22 的收益？
2. 能否借鉴 MaxFuse 的低质量匹配过滤和 ARISE 的共享邻边思想，让 MNN loss 本身更稳健，而不是把所有 reciprocal-MNN pair 等权处理？

本轮还必须解释用户看到的 GPU 长时间为 0 的现象，并测试不改变任何数值结果的 CPU transform 并行调度。正式训练仍必须使用 CUDA；谱分解、Leiden/mclust/KMeans 等 CPU 后处理期间 GPU 为 0 是允许的。

### 0.1 跨数据集训练语义（必须按此理解）

当前开发面板是 **4 个数据实例、3 个数据家族**：A1 与 D1 属于同一人类淋巴结技术/数据家族，另有 human tonsil 与 P22 mouse brain。这里的“统一候选”不表示把四个数据实例联合训练成一个模型，也不表示四者共用同一个 checkpoint 或同一组已学习参数。

- 每个 dataset × seed 都必须从确定性初始化开始**独立训练**，产生自己的 checkpoint；不得跨数据集共享 learned model state、embedding 或梯度。
- 跨数据集共同锁定的是 architecture、候选公式、预算、selection order、评价门和一套不看身份的 label-free router 规则。
- 不要求一个固定数值的 modality/loss 权重在所有平台上强行最优。允许同一个已注册公式根据该单元自身的无标签冲突、匹配置信度和 shared-neighbor 证据产生不同的 per-unit/per-spot 权重。
- 禁止的是依据 dataset/tissue/organism/platform/modality/file name 手工查表选参数，或看标签/ARI/NMI/Q 后为某数据集单独改权重、epoch、seed、阈值。
- SpatialGlue 官方论文和固定 commit `7c976d811d27ace51ce47ae0ad94a068a7d222fa` 的源码确实按技术类型设置不同 `weight_factors` 与 epochs；这证明 platform-specific preset 是既有可接受实践，但 Night-7C 选择检验更强的“同一无标签自适应规则”命题，而不是检验“一个冻结模型跨平台直接推理”。

因此，若某个安全 router 在低冲突人类数据上主要回退 C00、在高冲突 P22 上主要启用 R02，并通过 registry 的预注册保护门，这属于本轮预期的成功形态，不得因其对不同数据实例产生不同权重而判为“不统一”。

## 1. 最高优先级规则

1. 用户会手动把本任务交给实验 Codex；不得调用 AutoDL API，不得自动启动其他 Codex。
2. 用户启动服务器时使用**有卡模式**。GPU 训练和 CPU 图后处理都在同一有卡实例完成。
3. 不确定就停止并报告；不得猜路径、补造 checkpoint、绕过 SHA、硬着头皮继续。
4. 绝不使用标签、ARI、NMI、Q、数据集名、组织名、物种名、文件名或模态名做路由、训练、阈值、epoch、seed 或候选选择。
5. 不允许 seed search、best epoch、删除失败、科学重试、数值 fallback、事后改阈值或按数据集手写分支。
6. Night-6A 的 invalid 科学输出不得使用；Night-7B H01/H02 已逐行相同，本轮只允许 H01，禁止浪费计算重复 H02。
7. 原始 `.h5ad` 在总锁前一律不得用 `anndata.read_h5ad` 打开；也不得低层读取 `obs` 的值。优先只读远端已有 stripped cache、locked arrays、affinity、checkpoint 与 manifests。
8. 所有正式 routing 输出和 weighted-MNN pilot 输出必须先完成并锁 SHA，然后只开一次标签窗口。标签窗口后禁止返回训练、transform、阈值或候选修改。
9. 所有失败原样保留。scientific retry=`0`；仅允许在正式首单元前发生的全局实现纠正，或 registry 允许的最多 8 次 fail-closed 实现纠正，且必须逐次记录、重新从未评价状态开始，不能针对某一不理想结果修补。
10. 速度永远不能覆盖精确性；只有输出 SHA/数组逐字节或按规定容差完全一致时，才允许采用并行或 GPU 后处理。

## 2. 权威文件与先读顺序

先读取并核验同目录的 planning delivery index，再按以下顺序完整读取：

1. `SpaLORA_Night7B_Independent_Audit_and_Night7C_Decision_2026-08-18.md`
2. `SpaLORA_Night7C_Source_Code_Transfer_Audit_2026-08-18.md`
3. `SpaLORA_Night7C_Conflict_Gated_Affinity_Registry_2026-08-18.json`
4. 本任务书
5. Night-7B `official_compact/compact_delivery_index.json`
6. Night-7B report、P0 contract、locked H/R1/R2 manifests、R2 full metrics、source snapshot 与 tests

JSON registry 是候选、公式、顺序、预算、门槛和终态的机器可读权威。本任务书解释执行流程；如二者有歧义，立即 `BLOCKED_PREFLIGHT`，不得自行选择。

## 3. Git、工作区与存储

1. 从远端 GitHub 的 `night7b-final-20260818` 建立独立 worktree，验证 peel 到父提交。
2. 工作分支：`revision/q2-night7c-conflict-gated-mnn-rnd-20260818`
3. 保护标签：`baseline/pre-night7c-conflict-gated-mnn-rnd-20260818`
4. 最终标签：`night7c-final-20260818`
5. 全程普通 push，禁止 force/force-with-lease，禁止移动任何已有标签。
6. **最终标签只创建一次**：必须等所有应纳入 Git 的脚本、测试、协议、报告和 delivery index 已提交后，再创建 annotated tag 并普通 push。若 final tag 名已存在或 push 冲突，停止，不得移动它。
7. raw runs、checkpoint、大 affinity 留在 `/root/autodl-fs`；D 盘只放规划需要的小型报告、CSV/JSON、测试日志、源码、增量 bundle 和索引。目标 compact `<25 MB`，不得把 raw runs 或 model state 塞进 D 盘。

## 4. P0-AUTHORITY：未通过不得开始实验

### 4.1 本地与 Git 权威

- planning index 全部 SHA/size 通过。
- Night-7B compact 67/67 重新核验通过，root SHA 必须为 `eb52cc09e58722fb0a0f7c083d4099b74e526afbf71f1af750244194f3136eb0`。
- 父 commit/tag/远端 branch 一致。
- 找到 Night-7B 124 个正式训练单元、checkpoint、affinity、六视图/locked arrays 和 manifests；逐项核验登记 SHA，禁止用“文件名看起来一样”代替哈希。
- 核验 124/124 training entries 的 GPU model 为 RTX 4080 且 peak GPU memory 均大于 0；若任何一项不符，停止并报告。
- R02 H01/H02 历史指标逐行完全相同；本轮只登记 H01。

### 4.2 标签防火墙

- 在代码入口建立路径级 deny guard：总锁前禁止打开任何原始 `.h5ad`，禁止触碰登记的 label/annotation/ground-truth 路径。
- 对允许使用的 stripped cache/locked arrays 建白名单和 SHA 清单。
- 增加负向测试：模拟 `anndata.read_h5ad(original)`、低层读取原始 `obs` 值、评价函数提前调用、从文件名/数据集名传入 router；都必须 fail closed。
- 预锁阶段的任何日志、manifest、exception 不得含标签值。

### 4.3 环境与预算

- 记录 Python、PyTorch、CUDA、GPU、CPU 核数、RAM、R/mclust、SciPy、igraph、sklearn 版本。
- 正式 CUDA smoke 必须确认模型参数、六视图和 loss tensor 位于 CUDA；不能只看 `torch.cuda.is_available()`。
- 锁定预算：routing 新 transforms `240`；weighted-MNN pilot 训练 `48`、transforms `48`；scientific retry `0`；总训练尝试最多 `56`。

P0 全通过后，用通俗语言回复用户：

`已完成 Night-7C 权威文件、checkpoint、GPU 训练证据和标签防火墙校验，开始无标签路由与运行加速预审。`

## 5. P1-SEMANTICS：先证明实现的是注册公式

### 5.1 无标签特征

严格实现 registry 中：

- unit-level `m_initial`
- spot-level `c_i`、`rank_c_i`、`q_i`、`s_i`

其中 `q_i/c_i/s_i` 必须从锁定的 R02 checkpoint、固定 reciprocal-MNN indices、locked arrays 与确定性邻接派生，并在任何新训练前冻结 SHA。weighted-MNN pilot 的权重固定使用这份预先冻结特征，不能随 epoch 更新。

必须完成：

1. 30/30 R02 单元 fresh reload 后 endpoint、affinity、partition 与 Night-7B exact parity。
2. 30/30 `m_initial` 重算与第一条 loss curve 的误差不超过 `1e-7`。
3. 对 tied ranks、`n=1`、zero-degree、无 second alternative、非有限距离、空 shared-neighbor 等边界建 fail-closed 单元测试。
4. 对 T02–T09 每个公式做手算小矩阵测试；权重非负、逐 spot 和为 1、结果 CSR canonical、对称、零对角。
5. 对 W00–W05 做权重与 mean-one normalization 测试；所有权重为零必须失败，不得 fallback 到等权。
6. 静态扫描 router/trainer 的参数和调用链，证明不存在 dataset/tissue/organism/modality/file-name/label/metric 分支。

任一语义门不通过：终态 `IMPLEMENTATION_SEMANTICS_INVALID`，0 正式单元。

## 6. P2-RUNTIME：解释并缩短 CPU 长尾

### 6.1 分阶段采样

实现 1 秒采样器并输出 phase-aware CSV：

- `phase`: load/cache、CUDA forward/backward、checkpoint reload、affinity build、spectral eigensolve、Leiden、mclust、KMeans、metric-only
- timestamp、PID、CPU%、RSS、线程数
- GPU utilization、GPU memory、power、SM clock（NVML/nvidia-smi 可用项）

正式训练期间如果模型/loss 实际不在 CUDA，立即停止。CPU transform 期间 GPU=0 只记录，不判失败。

### 6.2 完全等价的并行调度

先在不读标签的四个 P22 单元上比较：

- serial reference
- 4 个 worker process；每 worker 固定 `OMP_NUM_THREADS=8`、`MKL_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=8`；总 RSS `<48 GiB`

接受条件同时满足：

1. 输入、canonical affinity、partition、cluster count、全部 pre-label artifacts SHA 与 serial 一致；若某文件包含无意义时间戳，必须先把时间元数据从科学 artifact 中剥离并在正式运行前锁定 schema，不能在比较时忽略科学字段。
2. 四单元墙钟速度至少 `1.5x`。
3. 无数值失败增量。

若不满足，正式阶段使用 serial，并在报告中说明瓶颈。CuPy/cuGraph 仅可作可选诊断；不得为了它阻塞安装，且只有在 30 个 C00 和 30 个 R02 历史单元 canonical partition SHA 全部 exact 时才可正式使用。

## 7. Stage T：冲突门控 affinity（0 新训练）

### 7.1 输入与候选

- 只复用 immutable specialists S0/C00、S2/R02；T09 另复用 S8/R08。
- T00/T01 是历史 reference，不运行新 transform。
- 按 registry 固定顺序运行 T02–T09：8 candidates × 30 units = `240` transforms。
- partition 只用 Night-7B H01 exact spectral-discretize endpoint、锁定 K 和原 seed。
- 不允许 H02、额外 endpoint、额外阈值或可视化后人工挑选。

### 7.2 总锁

在读取任何标签前输出并锁定：

- `routing_feature_manifest.csv`
- `routing_weights_manifest.csv`
- `routing_transform_manifest.json`
- 每 unit 的 affinity/partition/artifact SHA
- 失败清单
- formula/config SHA
- runtime phase log

完整主键必须为候选 × 数据集 × seed；失败保留为空指标行，不能重试或 fallback。

## 8. Stage W：置信度加权 MNN pilot

### 8.1 固定训练

- 精确复用 Night-7B R02 架构、初始化、optimizer、scheduler、float32、160 epochs、equal fusion、`RECON + 0.2*weighted_MNN`。
- 仅改变 registry 明列 W00–W05 的固定 per-spot loss weights。
- 6 candidates × 4 datasets × pilot seeds `[0,1]` = `48` 正式训练。
- 每次必须从同 seed 的确定性初始化开始；不能 warm-start R02，也不能 best-epoch。
- 每个成功 checkpoint 做 fresh-process round-trip；endpoint 只运行 E1 + H01，共 `48` transforms。
- 正式科学失败 0 retry；失败必须保留。

### 8.2 总锁

标签窗口前输出并锁定：

- 48 行 training manifest 与 48 行 transform manifest
- checkpoint/config/input/output SHA
- loss curve、固定权重摘要、GPU/CPU phase log
- checkpoint round-trip 结果
- 全部 partitions 与失败记录

Stage T 与 Stage W 两边都锁完才允许开标签。

## 9. 唯一标签窗口与评价

### 9.1 打开条件

先提交并普通 push：registry、实现、测试、两个 locked manifests、SHA 总锁和 label-window authorization。确认 GitHub branch 指向该锁提交后，只通过一个审计函数一次性打开登记标签。不得回到训练或聚类。

### 9.2 指标

- 每 unit：ARI、NMI、`Q=(ARI+NMI)/2`、neighbor agreement、Moran's I、Geary C、boundary disagreement。
- 与同 seed C00 比较；weighted pilot 还与同 seed R02 比较。
- 逐 seed 表、数据集均值、wins、priority-weighted mean、balanced macro、最差数据集和完整失败计数都要给出。
- 独立第二实现从冻结表复算，数值误差容限 `1e-12`。

### 9.3 路由门

逐项执行 registry 的 `routing_success_gate_vs_C00`、`strong_unified_gate_vs_C00` 和空间保护门：

- neighbor mean Δ ≥ `-0.01`
- Moran mean Δ ≥ `-0.02`
- Geary mean Δ ≤ `+0.02`
- boundary disagreement mean Δ ≤ `+0.01`

若多个路由通过，排序固定为：

1. strong unified gate 通过优先；
2. priority-weighted ΔQ 降序；
3. 最差数据集 ΔQ 降序；
4. total Q wins 降序；
5. 更低 wall time；
6. registry order。

只锁第一名，不得按某一张图或某个 seed 改选。

### 9.4 weighted-MNN pilot 晋级

严格按 registry 的 OR material gate、空间门和固定排序，最多晋级 2 个。8-cell pilot 只能标记“ready for full-seed confirmation”，不能称确认成功。不得因为 pilot 不理想补跑别的 seed。

## 10. 终态

只允许 registry 中一个终态：

- `NIGHT7C_STRONG_UNIFIED_CONFLICT_ROUTER_LOCKED`
- `NIGHT7C_SAFE_P22_CONFLICT_ROUTER_LOCKED`
- `NIGHT7C_WEIGHTED_MNN_CANDIDATES_READY_FOR_FULL_SEEDS`
- `NIGHT7C_SAFE_ROUTER_AND_WEIGHTED_MNN_CANDIDATES_READY`
- `NIGHT7C_NO_SAFE_ROUTER_OR_WEIGHTED_MNN_CANDIDATE`
- `BLOCKED_PREFLIGHT`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BUDGET_EXHAUSTED`

任何已知失败都必须写入最终报告，不能把 safe P22 router 写成统一架构胜利，也不能把当前四数据集开发结果写成 SOTA 或外部泛化。

## 11. 测试与交付

至少覆盖：authority/hash、label deny guard、router formula、weighted loss、canonical sparse output、zero-degree、tied rank、checkpoint round-trip、exact parity、budget、single label window、selection order、spatial gate、Git/tag immutability、delivery index。

运行 Night-7C 定向测试和全仓库测试。旧 raw 缺失造成的历史测试失败可如实分类，但 Night-7C 新增或触碰功能的失败为硬停止条件。

D 盘交付根：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night7c_handoff_20260818/official_compact`

至少包含：

- 通俗版结论 + 完整报告
- protocol/registry 与 source-transfer audit
- P0 authority/firewall/runtime contract
- routing/weighted-MNN manifests、逐 seed 指标、summary、gate audit
- phase-aware resource log 与 serial/parallel parity/speed report
- tests 与 Git audit
- 源码快照或增量 bundle
- internal/external/post-dispatch delivery indexes 与 SHA

不要下载 raw runs、checkpoint 或大型 affinity。报告中给出它们的远端持久盘路径和 SHA index 即可。

## 12. 用户可读输出与 token 节制

每完成一个阶段，只输出 4–8 行通俗状态：做了什么、是否通过、累计训练/transform、当前 GPU/CPU 情况、下一步是什么。不要把大 JSON、逐 seed 全表或重复日志贴进聊天；详细内容写入文件并给路径。

最终回复先用通俗语言回答：

1. 是否保住 C00 的人类结果并保留 P22 增益；
2. 哪个候选通过哪个门；
3. weighted-MNN 是否有候选需要全 seed；
4. GPU 低利用的定量原因和加速是否成功；
5. 训练/transform/失败/retry 数；
6. 科学限定；
7. commit/branch/tag；
8. D 盘关键路径和 SHA。

## 13. Git final 与关机

1. 完成交付、Windows/D 盘校验和所有应跟踪文件的最终 commit。
2. 再创建且只创建一次 `night7c-final-20260818` annotated tag；普通 push branch/tag，并核验 GitHub peel。
3. 保留同一 SSH 控制会话。所有下载、push、索引和本地校验结束后，把 `/usr/bin/shutdown` 作为最后一条远端命令。
4. shutdown 派发后严禁重连、严禁调用 AutoDL API。只能报告“命令已派发”和客户端状态，不得虚报控制台已关机。
