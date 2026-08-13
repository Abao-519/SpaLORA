# SpaLORA Night-5B 二次复核、空间救援与组合研发任务书

日期：2026-08-13  
权威父提交：`f9aeed223d38a897e190ac55ca741af1246071d2`  
权威父标签：`night5a-final-20260813`  
候选注册表：`SpaLORA_Night5B_Candidate_Registry_2026-08-13.json`  
注册表 SHA-256：`86f45f10949dce3a5de2b8233a786fb394608f2f78461975c0538eeca22c221a`

## 0. 唯一目标

本轮继续只使用 A1 与 Placenta 两个开发数据集，完成三类工作：

1. 给 Night-5A 因 family cap 而没有获得多 seed 机会的候选一次完整二次复核；
2. 组合 C04 的稳健融合与 C09/C10 的高准确率机制；
3. 尝试用轻量、无标签的空间救援保持 C09/C10 的准确率优势，同时减小连续性代价。

本轮不得运行 P22、D1、GSE198353、Night-4B 或任何现代基线正式 benchmark。终点只能是开发集候选锁定，P22 必须留给下一份独立任务书。

## 1. Night-5A 已知事实，不得重写

- Night-5A 状态为 `R3_CANDIDATES_READY_FOR_LOCKED_P22`；
- C04 是预注册平衡门的唯一入选候选，且在本轮始终保留为 primary locked candidate；
- C09 与 C10 的开发集准确率提升更大，但触发 Night-5A 空间保护门；
- 这不等于 C09/C10 无效。它们必须作为 accuracy-frontier reference 完整保留；
- P22、D1、GSE198353、Night-4B 尚未用于候选选择；
- 74 个 Night-5A 训练单元、所有失败尝试、测试和原始结果不得修改、删除或覆盖。

## 2. 权威输入与保护

执行前必须逐项验证：

- 父提交与父标签均 peel 到 `f9aeed223d38a897e190ac55ca741af1246071d2`；
- Night-5A report SHA 为 `f7a1ae426ec6e019f49639e67d58f775e902cdbc7718cad866a40678c2a2282b`；
- Night-5A compact handoff SHA 为 `7fd03857e38919603702835cb4eb7bc7af249d4533469b5c795dfc28f7c30673`；
- Night-5A 增量 bundle SHA 为 `f5d43ad20e5ee0eefb4ee6435051e10839b40a4be6d1ca721d92c6f2482c4ef4`；
- compact handoff 79/79 内部条目通过；
- Night-3B 1186/1186、Night-4A 76/76 历史保护仍通过；
- GitHub 远端 Night-5A branch/tag 与本地 object ID 一致；
- 新建隔离 worktree，不在 Night-5A 工作树上原地修改。

建议分支：`revision/q2-night5b-secondlook-rescue-20260813`  
建议保护标签：`baseline/pre-night5b-20260813`  
建议最终标签：`night5b-final-20260813`

任何输入、哈希、历史保护、GPU、R/mclust 或标签防火墙失败，立即进入 `BLOCKED_PREFLIGHT`，不得降级协议硬跑。

## 3. 标签防火墙

- 训练、warmup、可靠性权重、图构建、triplet、DGI、Laplacian、diffusion 和 checkpoint 生成阶段不得读取任何语义标签；
- 所有候选、所有 seed 的 embedding/checkpoint/manifest 完整冻结并写入 SHA 后，才允许开启评价窗口；
- mclust 固定为 `EEE`、PCA 20、random seed 2020；
- 不允许读取每 epoch ARI/NMI，不允许用真实标签挑 checkpoint，不允许 seed 搜索；
- 不允许看到 P22 后返回修改候选；本轮代码中 P22 路径必须显式拒绝。

## 4. 为什么需要本轮

Night-5A 的 one-seed family cap 有节省预算的作用，但 C06、C07、C08、C14 都曾通过 R1 numeric/spatial gate，只因 family/candidate cap 没有进入完整多 seed。C16 在 seed 0 上 accuracy 为正但空间门失败。为了避免把一次 seed 或 family cap 当作科学否定，本轮不再做 seed-0 早停，而是在打开标签前先完成三 seed。

C09/C10 在 Placenta 上的五 seed mean 分别达到：

- C09：ARI `0.722441`，NMI `0.759290`；
- C10：ARI `0.748868`，NMI `0.781323`。

它们的空间下降是真实代价，但数值信号足够强，必须进行组合和救援，而不是永久丢弃。

## 5. 候选集合

注册表精确包含 B00-B24，共 25 个唯一配置。不得新增、删减或改参数。

### 5.1 锁定复用 reference

- B00：Night-5A C00 FULL_IGE；
- B01：Night-5A C04 SHRINK25，primary locked candidate；
- B02：Night-5A C09 RNA_ANCHOR10；
- B03：Night-5A C10 MNN_TRIPLET01。

上述候选必须复用已验证的五 seed 结果，不得为了获得更好值重跑。

### 5.2 二次复核

- B04/B05：C06/C07 frozen input-space reliability；
- B06：C08 RNA anchor eta=0.5；
- B07：C14 hybrid IGE beta=0.5；
- B08：C16 DGI weight=0.1。

seed 0 必须复用 Night-5A。先补 seeds 1、2；若进入五 seed 阶段，再补 seeds 3、4。

### 5.3 机制组合

- B09/B10：C04 shrink25 + RNA anchor eta 0.5/1.0；
- B11/B12：C04 shrink25 + MNN triplet weight 0.05/0.10；
- B13：C04 shrink25 + DGI 0.10；
- B16：C04 shrink25 + anchor 0.5 + MNN 0.05。

### 5.4 warmup latent reliability

B14/B15 先按 C04 训练 100 epochs。第 100 epoch 后：

1. 从当次 seed 的 `latent1`、`latent2` 计算 PCA-free、k=20 的 within/cross local-predictability error；
2. 将两模态 score 归一化为每 spot 两通道权重；
3. 分别按 rho=0.25/0.50 向 `[0.5,0.5]` 收缩；
4. 权重冻结，继续训练至 epoch 200；
5. 不读取标签，不跨 seed 共用 learned latent weights；
6. 保存每 seed 权重 SHA、均值、极值、entropy、extreme fraction 和 modality-swap test。

这借鉴 COSMOS 源码中“先学习 latent、再一次性计算 WNN 权重”的真正实现逻辑，但必须 clean-room 编写，不复制第三方函数。

### 5.5 单步空间 diffusion

B17-B20 复用 C09/C10 已锁定 embedding，不重新训练：

```text
z_out = L2Normalize((1-alpha) * z + alpha * A_spatial_hat * z)
```

- `A_spatial_hat` 必须来自不可变 Night-3AF cache，含自环并对称归一化；
- alpha 只能为注册表中的 0.10 或 0.25；
- 只能一步；
- diffusion 在标签评价前完成并冻结 SHA；
- 不得对 cluster labels 做多数投票或依据真值平滑。

### 5.6 训练期 Laplacian 救援

B21-B24 使用：

```text
z = L2Normalize(embedding)
L_sp = mean_(i,j in undirected spatial edges) ||z_i - z_j||_2^2
g_sp = RMS(parameter gradients of L_sp at step 0)
g_active = mean RMS gradient of the active Night-5A raw losses at step 0
c_sp = target_fraction * g_active / (g_sp + 1e-12)
L_total = L_night5a + c_sp * L_sp
```

- target fraction 只能为 0.05 或 0.10；
- `c_sp` 在 step 0 计算一次后冻结；
- 边去重且不计 self-loop；
- 保存原始/加权 gradient RMS、系数、loss contribution 轨迹；
- 不允许依据空间指标动态调权。

## 6. P0 工程门

至少覆盖：

- B00-B03 复用结果的 manifest、embedding、metric 和 source/config SHA；
- B04-B08 对 Night-5A 实现逐项 parity；
- B09-B24 25/25 config SHA 唯一；
- combined mechanism 的 forward/loss/gradient/one-step Adam finite 且非零；
- latent reliability modality-swap、row-sum、freeze、checkpoint resume parity；
- diffusion alpha=0 与 source embedding bitwise parity；
- Laplacian empty-edge、duplicate-edge、self-loop、gradient scaling 测试；
- CPU reference probe 和 GPU probe；
- label-firewall negative tests；
- P22/D1 路径拒绝测试；
- 所有稀疏图继续保持 sparse，不得意外 densify。

首次工程失败允许修复实现 bug，但必须保存失败日志、原因与修复差异。不得借工程失败修改科学参数。

## 7. 运行次序与预算

### Stage S1：完整三 seed 冻结

在任何标签评价前，先完成：

- B04-B08 的缺失 seeds 1、2；
- B09-B16、B21-B24 的 seeds 0、1、2；
- B17-B20 对 C09/C10 已有 seeds 0-4 的无训练 diffusion；
- B00-B03 只复用，不重跑。

新增科学训练单元精确为 92 units。若发生硬件/进程失败，最多允许 4 次同一 tuple 原位重试；原失败必须保留，且不可换 seed 或换参数。因此本阶段最多 96 次训练尝试。

全部训练 manifest 与 embedding SHA 冻结后，统一评价。不得边看 seed 0 标签边决定是否继续。

### Stage S2：五 seed top-up

最多 6 个尚未拥有五 seed 的候选进入 top-up，只补 seeds 3、4，新增上限 24 units。候选来源取以下集合的并集后按锁定排序截断：

1. balanced frontier 前 3；
2. accuracy frontier 前 2；
3. 若 B14/B15 中任一前三 seed `delta Q > 0` 且未在前两类，保留 latent-reliability family 最优 1 个。

锁定排序：worst-dataset delta Q、macro delta Q、paired wins、spatial Pareto domination、runtime、candidate ID。

本轮新增科学训练单元硬上限 116 units；包含故障原位重试时最多 120 次训练尝试。不得扩大。

## 8. 双前沿决策，避免误杀高准确率候选

统一以 B00 为历史 reference，同时报告相对 B01 的增量。

### 8.1 balanced frontier

五 seed 要求：

- macro delta ARI >= +0.01；
- macro delta NMI >= +0.01；
- macro delta Q >= +0.02；
- 两数据 worst delta Q >= 0；
- 10 个 dataset×seed paired Q 至少 7 个为正；
- 不触发 Night-5A 空间保护门；
- runtime <= B00 2 倍，peak GPU <= B00 1.5 倍。

### 8.2 accuracy frontier

五 seed 要求：

- macro delta ARI >= +0.04；
- macro delta NMI >= +0.04；
- macro delta Q >= +0.05；
- worst-dataset delta Q >= -0.005；
- paired Q 至少 7/10；
- 资源门通过。

空间指标不作为 accuracy-frontier 的自动淘汰门，但必须逐数据集报告 neighbor、Moran、Geary、边界碎片度和 Pareto 状态，不能把 trade-off 隐藏成“全面提升”。

### 8.3 下一轮 P22 候选锁

最多锁定 3 个：

1. B01 C04 必须保留为 primary confirmatory candidate；
2. 若存在比 B01 更优的 balanced candidate，取锁定排序第一；
3. 取 accuracy frontier 第一；若与前两项重复则不补名额。

本轮只写 `selected_for_future_locked_p22.json`，绝对不得运行 P22。

## 9. 源码审阅边界

必须在 `method_provenance.json` 记录以下事实：

- SMART 官方代码实际使用共享 embedding 上的 reconstruction + triplet，Laplacian 默认为 0；
- COSMOS 官方代码在 epoch 100 左右用 learned latent 一次性计算 spot-wise WNN 权重并冻结；
- COSMOS 默认 accelerated spatial regularization 源码中 `c2` 使用了与 `c1` 相同的 index，不能照搬；
- ARISE 官方训练代码逐 epoch 读取 `true_labels` 并按 best ARI 保存 embedding，不能作为我们的 checkpoint 策略；
- SpaMFG 官方代码高度 dataset-specific/hard-coded，不能直接移植为通用模块；
- SMART/SpatialCOC 为 GPL-3.0，ARISE/SpaMFG 未见明确可复用许可证，禁止复制源码；
- COSMOS 为 MIT 但包含第三方 notice，本轮仍优先 clean-room 实现。

## 10. Git、存储与交付

- 普通非 force push；不得改写 Night-5A branch/tag；
- raw runs、checkpoints 留在 `/root/autodl-fs`；
- 本地只回传 D 盘 compact handoff、报告、表格、代码、测试、必要图和 Night-5A→Night-5B 增量 bundle；
- 默认不得生成或下载 full raw archive；
- 不得向 C 盘交付；
- 所有 bundle/tar/manifest 做内部 SHA 验证；
- 交付图只作诊断，不把图数量当完成指标。

## 11. 开关机与计费硬契约

本轮不得依赖“最后再新建一次 SSH 执行 shutdown”。

执行前：

1. 用户手动有卡开机，或在已经完成 API 实机验收后使用 GPU power-on API；
2. 用 D 盘 `autodl_control/AutoDL-ProInstance.ps1 -Action Status` 确认实例状态；
3. 在已建立的 SSH session 内设置最长 6 小时的远端 fail-safe shutdown，并把确认记录写入日志；若平台不支持定时参数则记录并继续，但不得伪报；
4. P0 与科学运行开始后，不允许自动取消 fail-safe，除非任务确需延期且已经先完成一次交付快照。

结束时：

1. 先完成 Git push、compact 回传、SHA 和 delivery index；
2. 最后一次远端命令只需 `sync`/状态落盘，不再把关机押在新 SSH handshake 上；
3. 从 Windows 侧调用 AutoDL Pro `power_off` API；
4. 轮询官方 status，只有观察到明确 stopped/off 类状态并连续确认三次，才写 `POWER_OFF_CONFIRMED`；
5. 若 API 返回 Success 但状态未确认，立即在本地给用户发出醒目告警，不得沉默结束；
6. 不得调用 release/delete/image reset 等破坏性接口。

## 12. 必答问题

最终报告必须回答：

1. 25 个配置是否均有唯一 SHA，P0 是否全部通过？
2. Night-5A 哪些结果被复用，哪些是新增训练，是否有任何重跑或覆盖？
3. family-cap second-look 中是否出现被 Night-5A 低估的候选？
4. C09/C10 的空间代价能否通过 diffusion、Laplacian 或 C04 组合救回？
5. latent reliability 是否优于 input-space frozen reliability？
6. balanced frontier 与 accuracy frontier 分别有哪些候选？
7. 相对 B00 与 B01 的 ARI/NMI/Q、paired wins、空间和资源结果是什么？
8. 最多三个 future locked P22 候选及其完整公式、config/source SHA 是什么？
9. 明确写出：P22、D1、GSE198353、Night-4B 未运行；
10. GitHub commit/tag/push、compact handoff、增量 bundle 和最终 API power-off 状态是什么？

## 13. 终止状态

只能是：

- `NIGHT5B_CANDIDATES_LOCKED_FOR_FUTURE_P22`；
- `NO_ADDITIONAL_CANDIDATE_C04_REMAINS_PRIMARY`；
- `BLOCKED_PREFLIGHT`；
- `BUDGET_EXHAUSTED`。

无论哪种状态，都不得自行继续 P22 或正式 benchmark。
