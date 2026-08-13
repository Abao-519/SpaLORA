# SpaLORA Night-6A：独立外部开发与结构性救援任务书

日期：2026-08-14  
任务性质：GPU 研发实验；外部开发、候选冻结；**不是** P22 再调参  
上游权威提交：`fccffcff8fb591467b4f7a390897217ffb2617ff`  
计划分支：`revision/q2-night6a-structural-rescue-20260814`  
保护标签：`baseline/pre-night6a-structural-rescue-20260814`  
最终标签：`night6a-final-20260814`

## 0. 唯一目标

以公平、label-free、同一全局规则提升 A1 人类淋巴结和一个独立人类扁桃体数据集的 ARI/NMI，同时保护空间结构与跨组织泛化。候选一旦冻结，下一轮才允许读取 D1 和 P22。

本轮禁止：

- 读取、列目录、解析、复制或运行 D1/P22 的任何数据、标签、缓存、embedding 或结果；
- 根据 P22、D1、真实标签、单个 seed 或人工观察修改候选；
- 用真实标签 early stop、保存最佳 checkpoint、选 epoch、选 seed 或选超参数；
- 为不同数据集设置不同学习率、epoch、优化器、损失权重或候选定义；
- 启动 Night-4B、正式 benchmark、投稿制图或论文结果包装；
- 覆盖 Night-3B 至 Night-5D 的历史结果；
- force push 或移动已经发布的标签。

## 1. 本地权威输入

从 D 盘直接读取，不要求用户重复上传：

1. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Post_Night5D_Research_Decision_2026-08-14.md`  
   SHA-256 `c8bc3b0ef260529c8815ca923a4b1d1fa429109e23317f2066e52104058ef445`
2. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night6A_Source_Code_and_Dataset_Audit_2026-08-14.md`  
   SHA-256 `653249e46ab727e8c4877933cd17f22c9c675520d9f2c6cf63b3683292c6e41d`
3. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night6A_Candidate_Registry_2026-08-14.json`  
   SHA-256 `869ceb59c25b5abd02f2242a4379acf91589e647e6e71a0ce38e694625bbe62f`
4. Night-5D evaluation recovery 交付目录：  
   `D:\文档\ChatGPT\博士第一篇科研论文项目\night5d_eval_recovery_handoff_20260814`

启动后先逐个核验 SHA；不匹配即 `BLOCKED_INPUT_INTEGRITY`，不得训练。

## 2. 终态枚举

最终状态只能是以下之一：

- `NIGHT6A_CANDIDATES_LOCKED_FOR_D1_P22`
- `NO_STRUCTURAL_RESCUE_CANDIDATE`
- `BLOCKED_INPUT_INTEGRITY`
- `BLOCKED_EXTERNAL_DATA_PREFLIGHT`
- `IMPLEMENTATION_SEMANTICS_INVALID`
- `BUDGET_EXHAUSTED`

任何终态都必须完整交付已产生的证据；只要曾连接远端，最后仍必须执行第 14 节关机流程。

## 3. P0-PROTECT：代码、历史和 Git

1. 从 GitHub 获取权威历史，核验上游提交存在且可达；不得用未审计工作目录当基线。
2. 在 `/root/autodl-fs/SpaLORA-night6a` 创建独立 worktree/clone，并从上游提交建立计划分支。
3. 创建保护标签并普通 push；若同名标签已存在且指向不同提交，立即停止，不得移动它。
4. 核验 Night-5D recovery 本地索引 16/16；报告原始 Night-5D 25 文件不变性清单 SHA `a29853735d502baadd0e44be092bfa47e9b9d86299759d54422a6d1503f36560`。
5. 将三份本地规划文件原样复制到仓库 `protocols/night6a/`，再次核验 SHA。
6. 记录 Python、R、PyTorch、CUDA、GPU、mclust、关键包和 Git 版本。
7. 对包含 `P22`、`D1`、`night5d` 结果根的路径设置访问审计；本轮只允许读取 recovery 的交付索引和协议，不允许打开其中 P22 指标内容来辅助设计。

P0 完成后，在日志中明确写：`P0-PROTECT PASS; BEGIN P0-DATA`。

## 4. P0-DATA：独立人类扁桃体硬门

目标数据：SpaMosaic/SpaMode 公开的人类扁桃体 Stereo-CITE-seq section 1，配对 RNA/ADT、空间坐标、四区域 annotation。优先使用官方文档链接的 Zenodo 记录 `https://zenodo.org/records/12654113`。

### 4.1 定位和下载

1. 先只读搜索 `/root/autodl-fs` 是否已有官方文件；不得重复下载。
2. 若缺失，只下载到 `/root/autodl-fs/datasets/human_tonsil_official/`；不得把原始矩阵塞入 D 盘。
3. 保存来源 URL、record/version、下载时间、文件名、字节数、SHA-256 和解压映射。
4. 不得从第三方二次打包、教程内隐藏缓存或作者未指向的网盘补齐缺失文件。

### 4.2 模态和坐标门

- RNA、ADT、坐标必须来自同一 section；
- barcodes 一一对应，重复为 0；
- 矩阵形状、稀疏度、非有限值、全零行列、坐标范围全部审计；
- 只使用与 A1 相同的全局预处理规则；模态固有的 RNA/ADT 变换可不同，但规则必须由模态类型决定，不得由数据集名字决定；
- 生成处理缓存和 manifest，缓存内容不得包含 annotation 值。

### 4.3 标签防火墙门

P0 只允许：定位 annotation 文件、哈希整个字节文件、读取表头，以及仅按列选择读取 barcode 列以验证对齐。禁止读取、统计、打印或传入 domain/label 值。已知 cluster 数 4 来自公开数据说明，固定用于所有候选。

若 paired modalities、坐标、官方来源或 barcode 对齐任一不成立，终止为 `BLOCKED_EXTERNAL_DATA_PREFLIGHT`；不得临时换数据集继续。

### 4.4 MISAR 仅做元数据登记

可以登记 MISAR E15.5 的官方入口和未来 provenance 审计清单，但本轮不得下载大文件、训练或用它选候选。它不是 Night-6A 预算的一部分。

## 5. P0-SEMANTIC：必须先证明实现正确

逐项实现注册表中的图剪枝、PCGrad、MinNorm、Barlow 和 neighbor InfoNCE。正式训练前至少通过以下测试：

1. 16/16 配置解析，配置 SHA 全部唯一；N00 与 Night-5 B01/C04 精确对应。
2. 手工小图验证 hard/soft prune 的每一条边权；soft epsilon=0.10/0.25/0.50 不得互相映射。
3. hard prune 孤点 rescue 只使用原空间边，距离同值按 barcode 决定；soft prune 不应产生新边。
4. 图对称、self-loop 和归一化与注册表完全一致；保存可复核矩阵。
5. 三个解析梯度例验证 PCGrad：正交不变、同向不变、反向发生确定性投影；重复运行逐 bit 一致。
6. MinNorm 在解析二/三向量例上与闭式或穷举最优值一致，权重非负且和正确。
7. Barlow 在相同标准化表示上对角损失接近 0，维度置换和破坏配对会按预期改变损失。
8. neighbor InfoNCE 的正样本集合与手工小图完全一致；chunked 与 exact 结果在 `1e-10` 内一致。
9. 新损失初始梯度校准只读取训练输入和模型梯度；标签字段进入函数必须触发负向测试失败。
10. 至少一次真实 A1 小步构造探针证明 N02、N05、N07 的图 SHA、梯度轨迹或 loss 项与 N00 确实不同。
11. 所有 lower-is-better 指标测试明确覆盖 Geary C 与 boundary disagreement 的方向。
12. 标签防火墙、固定 endpoint、seed 列表、候选不可变和 P22/D1 路径拦截均有正负测试。

若发现语义映射错误，必须先修复并重跑全部受影响测试。不得让错误实现进入正式训练；无法在尝试预算内完成即 `IMPLEMENTATION_SEMANTICS_INVALID`，不能把错误候选判为失败。

## 6. 固定训练、聚类和指标规则

- seeds 固定为 0–4；不得搜索或删除数值分叉 seed。
- 训练 epoch、optimizer、学习率、batch、特征数、图 k、聚类器和 cluster 数沿用 N00；所有候选和数据集使用同一规则。
- 固定最终 checkpoint；保存中间 checkpoint 仅供数值诊断，禁止根据其指标选择。
- 每个训练单元保存 config SHA、data/cache SHA、code commit、seed、最终 state SHA、embedding SHA、cluster SHA、runtime、peak GPU、loss 轨迹、非有限计数。
- 聚类和所有 label-free 输出完成并总锁后，才可读取本轮允许的标签。
- 主准确性：`Q=(ARI+NMI)/2`。
- 空间指标：neighbor agreement、Moran's I、Geary C、boundary disagreement。必须分别报告，不合成可掩盖退化的单一分数。
- 空间保护门沿用 Night-5 修正后的方向：相对同数据同 seed N00，(a) neighbor 与 Moran 同时下降超过 0.03；或 (b) Geary 上升超过 0.03 且 neighbor/Moran 至少一项下降超过 0.03，即失败。boundary 作为独立诊断，不参与既有门，避免再次后设阈值。

## 7. 预算

计划科学训练最多 96：

- R1：16 配置 × A1 × seeds 0–1 = 32；
- R2：N00 + 6 晋级配置，各补 A1 seed 2，并运行 tonsil seeds 0–2 = 28；
- R3：N00 + 3 晋级配置，各补 A1 seeds 3–4、tonsil seeds 3–4、placenta seeds 0–4 = 36。

硬上限：

- 有效科学训练：96；
- 兼容性/实现纠正重试：最多 12；
- 总训练尝试：108；
- diffusion transforms：0；
- P22/D1/GSE198353/MISAR 训练：0。

只有明确的基础设施中断或已证明的实现错误才可重试，并必须保留旧尝试。数值不好、loss 分叉或 seed 异质性不得重试。若系统性错误需要超过 12 次纠正，不淘汰候选，终止为 `BUDGET_EXHAUSTED` 或 `IMPLEMENTATION_SEMANTICS_INVALID`。

## 8. R1：A1 双 seed 机制筛选

1. 先完成全部 32 个训练单元，再锁定 manifest；任何单个结果都不得提前触发修改。
2. 锁定后读取 A1 标签一次，按既有评估代码计算全部指标。
3. 以 N00 同 seed 为参照，计算每个候选 mean ΔARI、ΔNMI、ΔQ 和空间 delta。
4. 候选若两 seed 平均 ΔQ > 0 且不触发空间门，可进入排序池。
5. 先按注册表执行 family rescue：graph、optimization、alignment 各保留表现最好的一个有效正向候选；再按 mean ΔQ 填满最多 6 个名额。
6. combination 的唯一晋级受注册表可解释性约束；accuracy-spatial frontier 必须另表保留。
7. 不得因单 seed 失败直接淘汰；若某候选两 seed 方向相反，标记 heterogeneous，而不是删除原始证据。

若有效正向候选不足 6 个，只晋级实际满足条件者，不用负向候选凑数；N00 始终作为 reference 进入后续轮。

## 9. R2：A1 + 独立扁桃体三 seed

1. 对 N00 与最多 6 个晋级候选补齐 A1 seed 2，并运行 tonsil seeds 0–2。
2. 先完成并锁定全部 R2 embedding/cluster/manifest，才读取 tonsil domain 值和再次计算 A1 指标。
3. 定义：

```text
Q_A1      = mean_seed((ARI + NMI) / 2)
Q_tonsil  = mean_seed((ARI + NMI) / 2)
Q_core    = 0.60 * Q_A1 + 0.40 * Q_tonsil
```

4. R3 晋级门：
   - 两个数据集均不触发空间保护门；
   - ΔQ_core ≥ +0.010；
   - `min(ΔQ_A1, ΔQ_tonsil) ≥ -0.005`；
   - 6 个 dataset-seed paired Q 中至少 4 个胜 N00。
5. 满足门者按 ΔQ_core 排序，最多 3 个。分差 ≤0.005 时优先选择来自不同机制族且资源更低的候选。

不得在看到 R2 后修改 epsilon、温度、warmup、loss weight、候选组合或数据预处理。

## 10. R3：五 seed 冻结与胎盘保护

1. N00 与最多 3 个 R3 候选补齐 A1/tonsil seeds 3–4，并在胎盘运行 seeds 0–4。
2. 胎盘只作补充泛化和空间保护门，不进入核心加权分数。
3. 全部训练、embedding、cluster 与 manifest 总锁后，才打开允许的标签并评价。
4. 正式候选锁定门：
   - `ΔQ_core ≥ +0.020`；
   - `ΔQ_A1 ≥ +0.015`；
   - `min(ΔQ_A1, ΔQ_tonsil) ≥ 0`；
   - A1+tonsil 共 10 个 paired Q 至少 7 个胜 N00；
   - A1、tonsil、placenta 均不触发空间保护门；
   - placenta `ΔQ ≥ -0.010`；
   - 无标签泄漏、语义错误、失败运行删除或 seed 搜索。
5. 竞争力标志（不替代锁定门）：A1 五 seed mean ARI ≥ 0.316 且 NMI ≥ 0.406。
6. 报告 exact paired sign-flip 和 bootstrap 区间，但明确标注为 development inference；因为候选在这些开发集上筛选过，不得称 confirmatory p-value。
7. 同时报告 runtime ratio、peak-GPU ratio、参数量和图边数变化；资源不藏入准确性结论。

若至少一个候选过门，终态 `NIGHT6A_CANDIDATES_LOCKED_FOR_D1_P22`；否则 `NO_STRUCTURAL_RESCUE_CANDIDATE`。不得为了产生候选而放宽门槛。

## 11. 必须回答的科学问题

报告必须明确回答：

1. 真实剪弱不受 RNA 支持的空间边，是否比 Night-5 的“只增益交集边”更有效？
2. hard prune 是否因图破碎而失败，soft prune 是否存在剂量趋势？
3. IGE 之后是否仍存在持续梯度冲突？PCGrad/MinNorm 是否把冲突下降转化为准确性收益？
4. Barlow 或 neighbor alignment 的收益是否跨 A1 与 tonsil，而非只在一个 seed/数据集出现？
5. 组合收益是否超过各单模块，还是只是复杂度堆叠？
6. accuracy 提升是否以 Moran/neighbor/Geary 或 boundary 的退化为代价？
7. 结果与近期公开方法的报告区间相比处于何处？必须同时披露流程不可比性和标签选 checkpoint 风险。

## 12. Git 与可复现性

- 每个语义模块独立 commit；禁止改写历史。
- 测试、注册表、完整运行 manifest、汇总和报告进入 Git；raw runs/checkpoints 不进入 Git。
- 最终交付索引提交完成后，才创建 final tag；final tag 只普通 push 一次，绝不移动。
- branch 与 tag push 后分别查询远端 peel commit 并记录。
- GitHub 凭据失败只尝试一次，不做盲目重试；但当前 SSH 已由用户配置，应优先使用 SSH remote。

## 13. D 盘紧凑交付

交付根：

`D:\文档\ChatGPT\博士第一篇科研论文项目\night6a_handoff_20260814\official_compact`

必须包含：

- `night6a_report.md`
- `p0_data_and_label_firewall_audit.json`
- `p0_semantic_contract.json`
- `candidate_registry_resolved.json`
- `all_attempts_manifest.csv`
- `per_seed_metrics.csv`
- `round1_decision.json`, `round2_decision.json`, `round3_decision.json`
- `graph_and_gradient_diagnostics/` 的紧凑表格
- `tests_and_invariance_audit.json`
- `delivery_index.json`
- 增量 Git bundle
- compact planner tar.gz
- shutdown 状态记录

raw matrices、raw runs、全 checkpoint 和大图不得下载到 D 盘；保留在 `/root/autodl-fs`，以路径、大小和 SHA manifest 保护。D 盘官方紧凑交付目标小于 100 MB；若超过，先剔除可由代码重建的中间物，不得删除报告、逐 seed 指标、配置、代码、测试、manifest 和最终候选证据。

交付完成后，在 Windows 本地独立复算每个 delivery-index SHA，并报告 `N/N`。

## 14. 强制关机流程

不调用 AutoDL API。只要本轮曾连接远端，无论成功、阻塞、预算耗尽还是实现无效：

1. 在仍然打开的远端控制会话内完成全部同步和 Git 操作；
2. 将 `/usr/bin/shutdown` 作为最后一条远端命令直接执行；
3. 记录 SSH exit status/断连现象；
4. 此后绝不重连、不得为了“确认关机”再次执行远端命令；
5. 只能声称“关机命令成功派发”或如实声称失败，不能冒充控制台已关机。

若关机派发失败，在最终回复第一行用醒目标记提醒用户立即手动关机。

## 15. 最终回复格式

第一行给终态。随后依次给：

1. P0 数据/标签防火墙与语义测试；
2. R1/R2/R3 完成数、失败数、重试数和预算；
3. 每个最终候选的 A1、tonsil、placenta ΔARI/ΔNMI/ΔQ、空间门、paired wins、资源；
4. 是否达到 A1 竞争力标志；
5. 明确说明 D1/P22/MISAR/GSE198353/Night-4B 访问与运行次数；
6. commit、branch、protection/final tag、远端 peel；
7. D 盘文件绝对路径、SHA、索引 N/N；
8. 关机命令状态，且不夸大控制台状态。
