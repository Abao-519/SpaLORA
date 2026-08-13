# SpaLORA Night-5D 前置门勘误 1：历史 B00 权重文件并非官方交付项

日期：2026-08-14  
状态：`AUTHORITATIVE_REVISION`  
原任务书：`SpaLORA_Night5D_Locked_P22_Confirmation_Taskbook_2026-08-14.md`  
原任务书 SHA-256：`69b328755216519fe816391c891003b2618a64079c3011d665f8ef57f99d69a8`  
锁定注册表 SHA-256：`59d034fcf371816e24b22477bb757a104c931682e0b60a3d177a0f40e0d94721`  
历史证据契约：`SpaLORA_Night5D_Historical_B00_Evidence_Contract_REV1_2026-08-14.json`  
证据契约 SHA-256：`1b3fd2821e3111d9ddacf19ba34fbc18182b9aaa17a654031b0e21a93ecd31a5`
独立原始归档审计：`SpaLORA_Night5D_B00_Official_Archive_Audit_2026-08-14.json`  
独立审计 SHA-256：`ed89c20e0bc909f7ea2ee479ef66c46a8fedba73399014a8be3d5fcb76c2c286`

本勘误与原任务书共同构成 Night-5D 的完整权威协议。若二者冲突，仅在本文件明确列出的历史 B00 证据范围内，以本勘误为准；候选、公式、seeds 0-9、35 个训练单元、10 个 transforms、统计门、标签防火墙、预算和终止规则全部不变。

## 1. 触发原因

Night-5D 第一次 preflight 正确停止为 `BLOCKED_PREFLIGHT`：原任务书第 2.7 条错误要求 Night-3B P22/FULL_IGE seeds 0-4 存在并重新验证 model state/checkpoint 文件。

复核原始 Night-3B 任务书、runner 和官方完整归档后确认：

- Night-3B 的正式输出清单要求 `checkpoint_index.csv` 和 `run_manifest.json`，没有要求保存模型权重文件；
- runner 明确将每个 checkpoint 的 `state_file_saved` 写为 `False`；
- manifest 保存 initial/final state SHA，但 `artifact_sha256` 合理地不包含不存在的 model state；
- P22/FULL_IGE 五个 seed 各有 8 个受 manifest SHA 保护的结果 artifact；官方原始完整归档独立复核为 40/40 哈希通过；
- 每个 checkpoint index 均为 10 行、最终 step=1600、全部 `state_file_saved=False`，最后一行 state SHA 与 manifest final state SHA 一致；
- Night-3AF `model_final.pt` 的状态不同，不能替代 Night-3B FULL_IGE。

因此这不是历史文件丢失，而是我方 Night-5D 任务书错误扩大了 Night-3B 的原始交付契约。实验 Codex 的停止行为完全正确；0 训练、0 transform、0 标签访问，不构成科学失败或预算消耗。

## 2. 对原任务书第 2.7 条的替换

原第 2.7 条整体废止，替换为：

> Night-3B P22/FULL_IGE seeds 0-4 必须按 REV1 历史证据契约验证：逐 seed 验证 run manifest、8 个正式 artifact 及其 SHA、checkpoint index 的 10 个 state-hash 记录、initial/final state SHA、last-row/final-state 一致性和官方交付覆盖。历史模型权重文件从未被官方保存，因此其缺失是 expected，不是 blocker。严禁用 Night-3AF 权重替代、伪造或从 hash“恢复”权重。

只有 artifact/manifest/checkpoint-index SHA 或结构不满足 REV1 契约时，才停止为 `BLOCKED_PREFLIGHT`。

## 3. 对 P0-CONFIRM 中 B00 parity 的澄清

P0 对历史 B00 seeds 0-4 应执行：

- 不重训、不覆盖；
- 从锁定代码、seed 与 immutable cache 重建 initial model，比较 initial-state SHA；
- 重算 initial no-label forward/loss/gradient/coefficient probe，与历史 `coefficient_probe.json` 核对；
- 验证历史 embedding、attention、clusters、observation IDs、loss/gradient trajectories、checkpoint index 与 manifest SHA；
- 验证 checkpoint index 最后一行 state SHA 等于 manifest final-state SHA。

由于最终权重文件不存在，明确禁止且不要求：

- 加载历史 final state；
- 从 final state 重新 forward；
- 进行历史权重扰动、checkpoint faithfulness 或 weight-level parity；
- 声称历史最终权重可重载。

这项限制必须写进最终报告，但不妨碍使用 SHA-verified embedding/clusters 作为数值基线。

## 4. 对总锁与评价阶段的澄清

- Night-5D 新产生的 35 个训练单元仍必须按新协议保存 model state/checkpoint 与完整 SHA；该要求不变。
- 历史 B00 seeds 0-4 在统一 baseline manifest 中记录：
  - `historical_model_state_file_available=false`；
  - `absence_expected_by_original_contract=true`；
  - `state_hash_only=true`；
  - `numerical_artifacts_verified=true`；
  - `weight_level_analysis_forbidden=true`。
- 总锁后，从历史 SHA-verified clusters/embeddings 重新评价 ARI/NMI/Q 与空间指标；与 Night-3B 锁定 metrics 在 `1e-12` 内一致。
- `BASELINE_REPLAY_MISMATCH` 只由 artifact/manifest/index 不一致、initial parity 失败或锁后 metrics 不一致触发；不能由 expected 的 model-state 文件缺失触发。

## 5. 对第一次停止记录的保留

重新执行时必须保留一份 `preflight_attempt1_blocked_authority_overreach.json`，如实记录：

- 第一次终态 `BLOCKED_PREFLIGHT`；
- 原因是原 Night-5D 任务书超出 Night-3B 官方输出契约；
- 科学训练 0/35、diffusion 0/10、P22 语义标签访问 0；
- 未建立 worktree/branch，未修改历史；
- 本勘误和 REV1 证据契约解除的是错误的权重文件要求，而不是降低 artifact/metric 完整性标准。

不得删除或隐藏第一次停止。

## 6. 恢复授权

在下列文件及 SHA 均通过后，可以重新从 preflight 开始 Night-5D：

1. 原 Night-5D 任务书；
2. 原锁定注册表；
3. 本勘误；
4. REV1 历史 B00 证据契约；
5. Night-5C 权威交付。

如果 REV1 契约全部通过，历史 B00 seeds 0-4 被授权用于数值基线复用，Night-5D 可以继续其余 P0-CONFIRM 与预注册运行矩阵。任何候选、参数、seed、统计或标签规则均不得改变。
