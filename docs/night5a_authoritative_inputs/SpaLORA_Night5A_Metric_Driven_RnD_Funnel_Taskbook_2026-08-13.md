# SpaLORA Night-5A：指标驱动方法研发漏斗任务书

日期：2026-08-13  
状态：规划 Worker 授权执行  
父提交：`4a22cfb4afe331e1ca2edcc0b86a01fa3892a452`  
父标签：`night4a-preflight-final-20260813`  
建议分支：`revision/q2-night5a-metric-rnd-20260813`  
建议 baseline tag：`baseline/pre-night5a-20260813`  
候选注册表：`SpaLORA_Night5A_Candidate_Registry_2026-08-13.json`

## 0. 本轮唯一目标

本轮只做 A1 与 Placenta 上的受控方法研发漏斗，最多从 16 个新/复现候选中筛出两个候选，完成 R0、R1、R2、R3 后停止。

本轮不得：

- 运行任何新候选的 P22；
- 打开或评价 D1 lymph node 的 ARI/NMI；
- 运行 Night-4B 正式 benchmark；
- 根据 P22、D1、GSE198353 结果修改候选；
- 修改 Night-3B/Night-4A 原结果、commit、tag 或结论；
- 生成“论文已成功”“达到 SOTA”或方法优于现代 baseline 的结论；
- 自动扩展候选、超参数值、seed 或运行预算。

终点必须是以下之一：

- `R3_CANDIDATES_READY_FOR_LOCKED_P22`：最多两个候选满足全部门槛；
- `NO_DEV_CANDIDATE`：没有候选满足 R3；
- `BLOCKED_PREFLIGHT`：输入、Git、环境、标签防火墙、源码/数据哈希或实现门无法满足；
- `BUDGET_EXHAUSTED`：到达硬预算但仍未完成预注册漏斗。

## 1. 权威状态与协议修订

用户的新指令正式开启新的 metric-driven R&D cycle。

Night-3B 的全部原始结果、commit、tag 和 `MIXED_EVIDENCE` 结论保持不可变。Night-4A 的 `BLOCKED_PREFLIGHT` 也保持不可变。

本任务书显式修订旧的“不得再用 A1/Placenta/P22 搜索架构”边界：

- A1 与 Placenta 永久降级为 development datasets；
- P22 仅作为下一轮 locked robustness gate；
- D1 lymph node 作为一次性 within-study final accuracy confirmation；
- 三个旧数据集以后均不得包装为独立泛化证据。

若执行 Worker 认为这条修订仍与任何旧任务书冲突，应以本节为当前用户授权，但不得反向改写旧文档。

## 2. 必须读取并核对的本地输入

执行前完整读取：

1. `D:\文档\ChatGPT\博士第一篇科研论文项目\night4a_handoff_20260813\night4a_report.md`
2. `D:\文档\ChatGPT\博士第一篇科研论文项目\night4a_handoff_20260813\delivery_index.json`
3. `D:\文档\ChatGPT\博士第一篇科研论文项目\night4a_handoff_20260813\SpaLORA_night4a_20260813.bundle`
4. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night3B_Independent_Evidence_Audit_and_Decision_2026-08-13.md`
5. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Metric_Driven_RnD_Rationale_2026-08-13.md`
6. `D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night5A_Candidate_Registry_2026-08-13.json`
7. 本任务书。

必须复核用户已给出的 Night-4A 关键 SHA：

- report：`bf33cede8105e65662263bca3dc28a9c5f531999332f1610a9f63da4121da958`
- planner handoff：`7ffc09b1b7e87340a45bc7abe38036b517fca0d9b4cf05b646c42389fd7c8cac`
- full archive：`f18e429b17bef564f4bab73e41f8cb22bf768543b5758faf2ff22ce2eda01dcc`
- Git bundle：`94cc16f2a626af3e57812618a3452727ac24b5cf2f08403603e8b6a8d9e65fa3`
- delivery index：`7f636e87c1a7df77b0fe1538500b67d4167a946003ec9f547714a6f8e7835581`
- shutdown confirmation：`f5bc6b0725c24fdd659df64785d3aa858e654801b0eaa163f6cde761df4dfcf8`

候选注册表与本任务书的 SHA 由规划 Worker在交付提示中给出；执行 Worker必须在任何远端写操作前复算。

## 3. P0-GIT：先把 Git 真正打通

用户已在 AutoDL 创建并注册：

- 私钥：`/root/autodl-fs/.ssh/id_ed25519_spalora`
- 公钥 fingerprint：`SHA256:tvie+fmn2BEgpSEY+ckTTRqBUYbbLn/mF79UDyVrwkE`
- GitHub 账号：`Abao-519`
- SSH 认证输出：`Hi Abao-519! You've successfully authenticated, but GitHub does not provide shell access.`

### 3.1 持久化 SSH 主机验证

当前 `accept-new` 可能把 `known_hosts` 写在临时系统盘 `/root/.ssh`。必须建立：

```text
/root/autodl-fs/.ssh/known_hosts
```

要求：

- 权限：目录 700、私钥 600、public key/known_hosts 644；
- 只接受与 GitHub 官方公布 fingerprint 匹配的 host key；
- GitHub ECDSA fingerprint 应为 `SHA256:p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM`；
- 或 Ed25519 fingerprint `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`；
- 不允许长期使用 `StrictHostKeyChecking=no`。

仓库级固定：

```bash
git config core.sshCommand "ssh -i /root/autodl-fs/.ssh/id_ed25519_spalora -o IdentitiesOnly=yes -o UserKnownHostsFile=/root/autodl-fs/.ssh/known_hosts -o StrictHostKeyChecking=yes"
git remote set-url origin git@github.com:Abao-519/SpaLORA.git
```

### 3.2 从权威 Night-4A commit 开始

不得从本地 `night4a_local_repo` 的 untracked 状态直接继续。远端必须验证：

- `night4a-preflight-final-20260813` peel 为 `4a22cfb4afe331e1ca2edcc0b86a01fa3892a452`；
- worktree clean；
- Night-3B、Night-3AF、Night-2C 保护清单通过；
- 以该 commit 新建隔离 Night-5A worktree/branch。

### 3.3 push 规则

先执行只读：

- `git ls-remote origin`；
- 枚举当前分支历史中的最大 blob；
- 若发现 GitHub 100 MiB 单文件限制风险，先报告，不得自动重写受保护历史。

若无风险，允许正常、非 force push：

1. Night-4A final branch；
2. `night4a-preflight-final-20260813` 与必要 baseline tag；
3. Night-5A 实现 commit；
4. Night-5A final branch/tag。

禁止：

- `--force`、`--force-with-lease`；
- 重写 Night-1 至 Night-4A 历史；
- 把 token、私钥或 SSH 配置内容提交进仓库；
- 因一次网络失败反复盲目 push。

如果 GitHub 因大 blob 或远端非 fast-forward 拒绝，保留完整错误，继续本轮科研但交付增量 bundle；不得伪称 push 成功。

P0-GIT 输出：`outputs/night5a_handoff/git_preflight.json`。

## 4. P0-PROTECT：历史保护与新分支

新 worktree 建议：

```text
/root/autodl-fs/SpaLORA-night5a
```

要求：

- 父 commit 精确为 `4a22cf...`；
- 建立 `baseline/pre-night5a-20260813` 指向父 commit；
- Night-3B 1186/1186、Night-4A 76/76（以实际权威清单为准）及历史保护全部核对；
- 不使用 `bundle_verify_repo` 作为原始证据；
- 所有新输出只写 Night-5A worktree 与 `/root/autodl-fs/night5a_*`；
- 不修改既有 outputs 下的任何文件。

## 5. 标签防火墙与数据角色

### 5.1 DEV-A：A1

- 允许候选训练完成且 manifest 锁定后读取标签计算指标；
- 训练 payload 删除全部 `obs` 列；
- K=10；
- 只用于研发筛选。

### 5.2 DEV-B：Placenta

- 标签位于 `obs[cell_type]`，进入训练前必须删除全部 `obs`；
- K=10；
- 只用于研发筛选。

### 5.3 WITHHELD

- P22：本轮不得运行新候选、不得读取新候选评价；
- D1：不得运行候选或打开标签结果；
- GSE198353：本轮不得用于候选选择；
- tonsil/GSE213264：不进入本轮。

### 5.4 允许查看的历史信息

可以读取 Night-3AF/Night-3B 已锁定的历史指标，用于建立 reference 与 historical envelope。不得读取任何本轮 withheld 新结果。

## 6. 实现原则

### 6.1 不复制许可证不兼容源码

- SMART、SpatialCOC 为 GPL-3.0：不得复制源码；
- ARISE、SpaMFG、CoMo 未找到明确 LICENSE：不得复制源码；
- COSMOS 为 MIT：若实际复制任何实现，必须保留 LICENSE/notice 并在 provenance manifest 逐函数注明；
- 优先依据论文和数学定义 clean-room 实现；所有来源写入 `method_provenance.json`。

### 6.2 旧路径必须保持 exact

`C00_FULL_IGE`、`C01_DROP_CORR2`、`C02_UNIFORM_ALL` 必须通过 CPU forward/loss/gradient/one-step optimizer parity 与 Night-3B 对应语义一致。

### 6.3 候选必须来自注册表

只允许 `SpaLORA_Night5A_Candidate_Registry_2026-08-13.json` 中 C00–C16。

不得：

- 看结果后增加 C17；
- 调整 rho、eta、lambda、margin、temperature、k、beta；
- 为 A1/Placenta 使用不同候选超参数；
- 用标签决定 early stopping 或图参数。

## 7. 候选数学契约

### 7.1 等权简化母体 C03

- within 与 cross 都使用精确 `[0.5, 0.5]`；
- Corr1 保留；
- Corr2 系数精确为 0；
- 其余三项使用 active-set IGE，总和精确为 4；
- 参数对象不必删除，以保持可审计初始化；但 Corr2 必须无目标贡献。

### 7.2 收缩融合 C04/C05

对 learned attention `a`：

```text
a_final = (1-rho) * [0.5, 0.5] + rho * a
```

- C04 rho=0.25；
- C05 rho=0.50；
- within 与 cross 使用同一 rho；
- 不得加额外熵正则。

### 7.3 label-free local reliability C06/C07

在模型训练前、标签封锁内，对每个模态 PCA 表示建立 k=20 邻域。对每个 spot，分别计算该模态表示由本模态邻居与另一模态邻居预测时的误差/affinity ratio，转换成两模态互补权重 `w1+w2=1`。

最终：

```text
w_final = (1-rho) * [0.5, 0.5] + rho * w_reliability
```

- C06 rho=0.25；
- C07 rho=0.50；
- 权重必须冻结、detach、逐 spot 保存；
- k、PCA、距离、epsilon、clip 全局一致；
- 必须测试交换模态时权重互换、常量输入时回到 0.5、无 NaN/Inf；
- 不得使用聚类、标签或 domain QC 调权。

### 7.4 RNA-anchor residual consensus graph C08/C09

```text
A_common = binary_intersection(A_spatial, A_rna_feature)
A_shared_raw = A_spatial + eta * A_common
A_shared = symmetric_normalize(A_shared_raw + I)
```

- C08 eta=0.5；
- C09 eta=1.0；
- 两模态 spatial encoder 使用同一 `A_shared`；
- 原 modality feature graph 保持不变；
- 不删除原空间边，因此不得因 intersection 造成孤点；
- 保存 edge count、degree、connected components、common-edge fraction。

### 7.5 MNN-triplet C10/C11

- 在训练前的模态 PCA 空间分别建立 top-3 mutual nearest pairs；
- negative 从预注册最远 40% 集合中按固定 preprocessing RNG 选取；
- pair/triplet manifest 在训练前锁定，对所有 model seeds 相同；
- 在 fused embedding 上使用 margin=0.5 的 triplet loss；
- C10 weight=0.1，C11 weight=0.3；
- 不得使用标签、伪标签或 K 构造 pair；
- 保存 triplet 数、spot coverage、重复率与每模态来源。

### 7.6 Neighbor-aware contrastive C12

- positives 为锁定空间图中的相邻 spot；
- negatives 为固定 RNG 下的 degree-stratified non-edges；
- temperature=0.2，weight=0.1；
- 不使用预测 cluster、已知 K 或标签；
- 必须避免构造完整 N×N GPU 相似矩阵，使用边/负采样实现。

### 7.7 Hybrid IGE C13/C14

对 active losses：

```text
raw_k = legacy_k ** (1-beta) * (gradient_k + eps) ** (-beta)
c_k = 4 * raw_k / sum(raw_active)
```

- C13 beta=0.25；
- C14 beta=0.50；
- Corr2 精确为 0；
- 权重只在 step 0 计算并冻结；
- 记录 legacy prior、raw gradient、最终 coefficient；
- 不得根据 dataset name 改 beta。

### 7.8 Residual nonlinear encoder C15

独立 clean-room 稀疏实现：

```text
h1 = LayerNorm(GELU(A @ X @ W1))
h2 = LayerNorm(h1 + 0.5 * GELU(A @ h1 @ W2))
```

- 不依赖 PyG；
- decoder 保留简单线性版本；
- 参数量不得超过 C00 的 2 倍；
- 无 dataset-specific depth/dropout；
- 检查 oversmoothing：记录初末 embedding pairwise variance 与 neighbor/non-neighbor cosine gap。

### 7.9 DGI C16

- fused embedding 与其全局 summary 的 node-global discrimination；
- corruption 为训练前锁定的 spot permutation；
- weight=0.1；
- 不得从 COSMOS/PyG 复制实现；
- 保持 O(Nd)，不构造 N×N；
- 保存 positive/negative logits 与 loss trajectory。

## 8. P0-ARCH/ENGINEERING 门

先实现完整 registry、runner、evaluator 与测试，不运行正式候选。

必须通过：

- C00/C01/C02 parity；
- 17 个配置可序列化、可复现并有唯一 SHA；
- synthetic/subsample forward/backward 全 finite；
- 所有应激活模块有非零梯度；
- Corr2-off objective contribution 精确为 0；
- reliability 交换/常量测试；
- anchor graph 不删空间边、对称归一化正确；
- triplet/contrast pair manifest 在 model seed 间不变；
- 无标签模块 import、open、parser access；
- checkpoint/resume 不改变 run identity；
- 参数量、峰值显存与估算 runtime；
- 至少 20 个专项测试，0 fail。

任何候选工程门失败：保留失败并淘汰；不得换 seed 或悄悄简化定义。

## 9. 评价协议

主研发分数：

```text
Q = (ARI + NMI) / 2
```

- A1 与 Placenta 等权；
- ARI、NMI 必须分别报告；
- reference 为同阶段 C00；
- Night-3B 八变体的每数据集最高 Q 形成 `HISTORICAL_ENVELOPE`，只作难度标尺；
- 主 evaluator 固定 `L2 normalize -> PCA20 -> mclust EEE, seed 2020`；
- 不允许根据结果切换 Leiden/mclust、covariance model、PCA 维数或行归一化。

空间保护门失败条件：

1. neighbor 与 Moran 同时相对 C00 下降超过 0.03；或
2. Geary 上升超过 0.03，且 neighbor/Moran 至少一项下降超过 0.03。

空间指标是保护门，不得替代准确性。

## 10. R1：seed 0 双开发集筛选

矩阵：17 configs（含 C00）× 2 datasets × seed 0 = 34 runs。

所有数据使用正式 full epochs；不得用短 epoch 排名。

最多保留 6 个非 reference 候选：

- 两数据都完整、finite；
- 相对 C00 dev macro `delta Q >= +0.005`；
- 任一数据 `delta Q >= -0.02`；
- 不触发空间保护门；
- 每个机制 family 最多保留一个，但 C03 simplified base 可单独占位；
- family 内先按 dev macro Q，再按 worst-dataset delta Q，再按 runtime 决定；
- 不得人为挑选更“新颖”的较差配置。

输出：`r1_decision.json`，包含 34 个 run，无论成功失败。

## 11. R2：三 seed 稳定性门

对 R1 最多 6 个候选增加 seeds 1、2；C00 同阶段 reference 必须可复用精确 Night-3B 或重跑并说明。新增上限 24 candidate runs，reference 不计入候选硬预算但必须完整。

最多晋级 3 个：

- 3/3 seeds 全完成；
- dev macro `delta Q >= +0.015`；
- A1、Placenta 任一 `delta Q >= -0.01`；
- 四个 dataset×metric mean cells 至少 3 个非负；
- 六个 dataset×seed paired Q 至少 4 个胜 C00；
- 不触发空间保护门。

输出：`r2_decision.json`。

## 12. R3：五 seed 决策门

对最多 3 个候选增加 seeds 3、4；新增上限 12 candidate runs。

最多选择两个候选：

- 5/5 seeds 全成功；
- dev macro `delta ARI >= +0.01`；
- dev macro `delta NMI >= +0.01`；
- dev macro `delta Q >= +0.02`；
- 两开发集均 `delta Q >= -0.005`；
- 10 个 dataset×seed paired Q 至少 7 个为正；
- 不触发空间保护门；
- runtime 不超过 C00 2 倍；
- peak GPU allocated 不超过 C00 1.5 倍。

超资源但数值明显更高的候选只可标记 `PARETO_CANDIDATE`，不得隐去成本；它可占两个名额中的一个，但需同时保留一个资源合格候选。

若超过两个候选通过，以以下字典序选二：

1. 最大化 worst-dataset delta Q；
2. 最大化 dev macro delta Q；
3. 最大化 paired Q wins；
4. 最小 runtime；
5. 最小 candidate ID。

输出：`r3_decision.json` 与 `selected_for_p22_lock.json`。本轮到此停止，不运行 P22。

## 13. 预算硬门

候选新增 run 上限：

- R1：32 非 reference candidate runs；
- R2：24；
- R3：12；
- 合计：68 candidate runs。

加上 reference 重跑最多 10 个 dataset×seed runs，总训练单元不得超过 78。

以下不计入训练单元但必须记录：synthetic probe、CPU parity、失败的启动前测试。

不得自动用未消耗预算去增加超参数或 P22。

## 14. evaluator 加固但不消耗 withheld 结果

本轮可以实现并测试通用 strict-ID evaluator：

- observation ID 必须一一对应、无重复、无缺失；
- 明确 metric direction；
- D1 接口必须要求 3359 spots 精确对齐；
- 用 synthetic labels 测试 D1 evaluator，不读取真实 D1 结果；
- GSE198353 label-free evaluator 只做接口与结构测试，不尝试用 silhouette/DBI 选择候选。

真实 D1/GSE198353 不进入 Night-5A 决策。

## 15. 统计与失败规则

- 五 seeds 是优化重复，不是五个生物学重复；
- 开发集经过候选筛选，CI 只能描述，不能宣称确认性显著；
- 报告 mean、SD、median、每 seed paired delta、全部失败；
- 不筛 seed、不删除数值分叉；
- 数值失败直接算稳定性失败；
- 只允许外部抢占后从同一 checkpoint、同一 config、同一 seed 恢复；
- OOM 若需统一降 batch，必须对该阶段所有候选/reference 从头一致应用，并形成协议修订；
- 只允许一次明确的依赖兼容性重试；不得改科学公式救活候选；
- run order 在训练前生成、固定 RNG、哈希锁定。

## 16. 交付与存储控制

### 16.1 本地只下载精简交接

默认只下载到：

```text
D:\文档\ChatGPT\博士第一篇科研论文项目\night5a_handoff_20260813
```

精简包必须包含：

- report；
- config/registry/locks；
- per-run summary 与 paired deltas；
- R1/R2/R3 decisions；
- 代码、测试、provenance；
- Git incremental bundle；
- delivery index 与 shutdown confirmation。

不得默认下载：

- 每 run model state；
- 重复 embedding；
- full raw archive；
- 完整历史 bundle 的又一份副本。

raw runs 留在 `/root/autodl-fs`，由 manifest 和 SHA 保护。只有规划 Worker明确要求或服务器存储即将销毁时，才生成/下载完整 raw archive。

### 16.2 增量 Git bundle

本地已经保存完整 Night-4A bundle。本轮优先生成只覆盖：

```text
night4a-preflight-final-20260813..night5a-final-20260813
```

的增量 bundle，并验证可与 Night-4A bundle 合并恢复。不要再下载 1+ GB 完整历史 bundle。

## 17. Git 提交点

至少两个提交：

1. 实现、registry、tests、locked manifest；
2. 完整 R0-R3 结果、决策、报告、交付索引。

建议 final tag：`night5a-final-20260813`。

如果 GitHub SSH push 成功，记录远端 branch/tag object ID；如果失败，记录唯一错误和增量 bundle SHA。

## 18. 最终报告必须回答

1. GitHub SSH 是否在重启可持久配置下通过？
2. Night-4A branch/tag 是否成功无 force push？
3. 17 个配置中每个在 R0/R1 的命运是什么？
4. 哪些机制带来可重复增益，哪些失败？
5. A1 与 Placenta 是否同时提高？
6. 是否存在空间连续性或资源代价？
7. 最多两个被选候选的完整公式、参数、代码 SHA 是什么？
8. 最终状态是 `R3_CANDIDATES_READY_FOR_LOCKED_P22`、`NO_DEV_CANDIDATE`、`BLOCKED_PREFLIGHT` 还是 `BUDGET_EXHAUSTED`？
9. 明确写出：P22/D1/Night-4B 未运行。
10. 下一轮只可由规划 Worker另行签发，不得自行继续。

## 19. 关机契约

完成所有本地精简交付下载与双端 SHA 校验后：

- 确认无训练/压缩/push 进程；
- 生成并下载 `shutdown_confirmation.json`；
- 最后一条远端命令严格为 `/usr/bin/shutdown`；
- 之后不得重连。

如果执行中因硬门提前停止，也必须完成可获得的精简日志交付后按同一关机契约处理。
