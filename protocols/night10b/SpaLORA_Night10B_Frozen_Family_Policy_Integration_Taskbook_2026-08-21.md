# SpaLORA Night-10B 冻结模态家族策略整合与复现封口任务书

日期：2026-08-21  
执行类型：代码整合、零标签重放、两条固定工程 smoke  
不是：候选搜索、科学重新评价、外部 benchmark、QCRD 恢复

## 0. 执行 Codex 的首条确认

开始任何远端操作前，先完整读取同目录的决策、JSON 合约和启动提示词，并回复：

> 已完整读取 Night-10B 决策、冻结 family-policy 合约和任务书；本轮不恢复 QCRD、不读取任何标签、不新增数据或 benchmark。正式目标是 30/30 Q00 零标签精确重放和 2/2 固定 fresh engineering smoke。

若任一规划文件缺失、JSON 不能解析、SHA 与 planning index 不一致，立即停止并报告，不得连接或唤醒 AutoDL。

## 1. 本轮唯一目标

将已冻结的两条权威路线整理为单一、可测试、内容寻址的软件入口：

```text
RNA + PROTEIN -> C00_G04_H05_CONFIRMED
RNA + ATAC    -> F00_R02_FULL
```

路由只能依据显式 assay pair。数据集 ID 只允许由 data steward 用于定位已经注册并哈希的输入，不能进入 policy resolver、trainer、endpoint 或 selector。

本轮成功不产生任何新的 ARI/NMI/Q 结论。历史科学结果保持不变；Night-10A QCRD 继续是负结果。

## 2. 不可变范围

允许使用的现有数据单元：

- A1 seeds 0–4；
- tonsil seeds 0–4；
- D1 seeds 0–9；
- P22 seeds 0–9。

正式 replay 总数固定为 30。只允许以下两个 fresh smoke：

- `S00_RNA_PROTEIN_A1_SEED0`；
- `S01_RNA_EPIGENOME_P22_SEED0`。

全轮固定：

- 新科学候选：0；
- 新科学训练单元：0；
- 标签读取：0；
- MISAR Y：0；
- E18.5：0；
- 第三方 benchmark：0；
- QCRD 训练、修复或评价：0。

## 3. P0-LOCAL：Windows 权威定位，不开服务器

在用户开机前完成：

1. 读取 `night10b_frozen_family_policy_integration_contract.json`；
2. 逐个定位 `authority_files`；
3. 独立复算每个文件 SHA-256；
4. 重新核验 Night-10A REV2 compact 为 182/182；
5. 读取 30-row Q00 audit，独立核对单元集合正好为 A1 5、tonsil 5、D1 10、P22 10；
6. 检查 parent commit/tag、目标 branch/tag/raw root 不与历史对象冲突；
7. 输出本地只读 `P0_LOCAL_AUTHORITY_PASS` 或 fail-closed 结果。

任何文件不匹配时停止。不得用同名近似文件替代，不得从报告手工回填 SHA。

## 4. P0-REMOTE：父提交、保护边界与独立工作区

只有 P0-LOCAL 通过、用户明确以有卡模式开机后才能连接 AutoDL。

远端要求：

1. 找到包含 parent commit `8b83fe38f3fbd4d3cb24fef6512596be2d49408b` 的权威仓库；
2. 验证 `night10a-rev2-final-20260821^{}` peel 精确等于 parent；
3. 验证工作树和历史 raw roots 未被改变；
4. 从 parent 新建独立 clone/worktree `/root/autodl-fs/SpaLORA-night10b`；
5. 创建新 branch `revision/q2-night10b-frozen-family-policy-integration-20260821`；
6. 仅创建一次 protection tag `baseline/pre-night10b-frozen-family-policy-integration-20260821`，指向 parent；
7. 保护 tag 若已存在但不指向 parent，立即停止；若正确存在则只记录，不移动；
8. 新 raw root 必须为 `/root/autodl-fs/night10b_family_policy_integration_20260821`；不得覆盖或移动 Night-6C/6D/7B/8B/9A/9B/10A raw。

所有 push 使用普通 push，禁止 `--force`、`--force-with-lease` 和移动已有 tag。

## 5. P0-SOURCE：冻结 recipe 的源代码闭合

在写新入口前，执行 Codex 必须亲自读取当前 parent repo 中与下列语义相关的源代码、测试和冻结配置：

- Night-6C/6D 的 G04 training、H05 transform、graph/head 定义；
- Night-7B 的 R02 adapter、C06 endpoint、H01 partition；
- Night-8A/8B 的 identity-blind family policy 与 full F00 训练路径；
- Night-9A 对 full F00 resource semantics 的审计；
- Night-10A 的 Q00 references、input manifests 和 30-row lock。

生成 `outputs/night10b_handoff/authority_audit.json`，至少列出：

- 每个被采用函数/类/配置的 repo-relative path 与 SHA；
- 每条 recipe 的唯一语义来源；
- 是否存在多个冲突实现；
- 30 个 reference 的 raw absolute path、size、SHA 是否仍可访问；
- 标签路径是否从训练/重放进程完全隔离。

如果不能唯一重建 recipe，停止 `BLOCKED_AUTHORITY_OR_ARTIFACT`。不得把“看起来相近”的函数当权威实现。

## 6. P1：统一代码入口

在不删除历史脚本的前提下，新增生产化入口。文件名可以适配当前仓库，但至少应形成以下职责：

```text
SpaLORA/family_policy.py            # 纯路由与冻结 recipe 解析
SpaLORA/family_recipes.py           # 两条内容寻址 recipe
scripts/night10b/run_family_policy.py
configs/family_policy/*.json        # 冻结配置
tests/night10b/*                     # 语义、身份盲、重放和 CLI 测试
```

### 6.1 路由 API

建议最小接口：

```python
resolve_family_policy(primary_assay: str, auxiliary_assay: str) -> ResolvedPolicy
```

硬要求：

- 只接受 `RNA/PROTEIN` 与 `RNA/ATAC`；
- 大小写/空格规范化规则必须固定并测试；
- 不接受 dataset、tissue、species、stage、shape、path 或 score；
- 不从 feature dimension 推断 family；
- 未知、倒置、缺失、多值输入 fail closed；
- 返回内容包括 recipe ID、config SHA、backbone、adapter、endpoint 和 preprocessing contract ID；
- 相同 assay pair 重复解析必须 byte-exact；
- resolver 不 import evaluator、annotation reader 或历史 metric table。

### 6.2 两条 recipe

`RNA_PROTEIN` 必须保持：

- `G04_SP10_F10_EUC_UNION`；
- `H05_EQUAL3_AFFINITY_SPECTRAL`；
- 现有 family preprocessing；
- 现有固定随机种子、训练终点与 checkpoint 语义。

`RNA_EPIGENOME` 必须保持：

- 完整 G00 与完整 G04 双骨干；
- R02 `RECON+MNN`；
- `equal` fusion；
- fixed 160 adapter epochs；
- `E1_ADAPTER_C06_MEAN__H01`；
- 稀疏 ATAC/LSI 路径；
- 不使用 Night-9A E00–E08 替代候选。

### 6.3 数据表示和稀疏边界

新入口要显式区分：

- RNA counts；
- normalized/log RNA；
- model features；
- auxiliary modality model features；
- ordered observation IDs；
- selected feature names 与 weight vector。

禁止含义不清的 `raw_feat`。图和 MNN 路径必须保持 sparse；增加静态和运行时守卫，拒绝构造 dense N×N array/tensor。

## 7. P1-TEST：训练前 fail-closed 测试

至少覆盖：

1. 两个合法 assay pair 精确映射；
2. dataset/tissue/species/stage/shape/path 不在 resolver signature；
3. source AST 中没有 dataset-name 分支、标签读取或 metric 选择；
4. 用完全不同的 opaque data-steward IDs，路由结果仍只由 assay pair 决定；
5. unknown/reversed/missing pair 全部 fail closed；
6. 两条 resolved config canonical SHA 固定；
7. C00 只含 G04/H05；
8. F00 必须含 G00+G04+R02/C06/H01，不能落到 E00–E08；
9. 标签模块不能被 trainer/replay worker import；
10. dense N×N guard；
11. CLI 一条配置只生成一个明确 output row；
12. manifest 原子写、失败不伪装 success。

所有测试通过并普通 push 后，才允许正式 30-row replay。

## 8. P2：30/30 Q00 零标签正式重放

权威行来自 Night-10A REV2：

`outputs/night10a_rev2_handoff/r2_lock/r2_total_lock_independent_audit.json::q00_references`

### 8.1 重放顺序

严格固定：

1. A1 seeds 0–4；
2. tonsil seeds 0–4；
3. D1 seeds 0–9；
4. P22 seeds 0–9。

不得并发改变记录顺序。可以在资源审计允许时并行执行独立 transform，但最终 manifest 必须恢复上述 canonical row order。

### 8.2 每行必须核对

- unit ID、dataset data-steward ID、seed；
- data steward 提供的 assay pair；
- resolved family 与 recipe；
- ordered observation SHA；
- registered input SHA；
- input manifest SHA；
- source reference embedding SHA；
- source canonical partition SHA；
- implementation SHA、resolved config SHA；
- replay embedding/endpoint/partition；
- fresh-process reload；
- 唯一 canonical row SHA。

如果 reference 在该历史流程中本来就是不可重新训练的权威 no-op alias，可以做**哈希验证后的 no-op replay**，但必须明确记录 alias source、文件 SHA、loader、shape、finite 和 fresh-process reload；不能只复制报告里的 SHA 字符串。

### 8.3 通过门

- 30/30 行存在；
- 30 个 row SHA 唯一；
- 30/30 source artifacts 实际重哈希通过；
- 30/30 assay pair→family→recipe 正确；
- 30/30 ordered rows 正确；
- 30/30 embedding canonical SHA 精确，或合法 no-op alias 精确；
- 30/30 canonical partition SHA 精确；
- 标签对象反序列化 0。

任何一行缺失即整体不通过。不得删掉失败行、换 seed、换 solver、改 K、只报告 29/29 或把 partition agreement 近似 1 当作 SHA exact。

### 8.4 正式 correction

正式 replay 开始后只允许最多两个**全局**实现修正周期。每次必须：

1. 在读取任何标签或历史分数之前明确定位同一个全局 root cause；
2. 保留旧 attempt；
3. 使此前所有正式 replay 行全部作废；
4. 从第 1 行重新执行完整 30 行；
5. 不得只重跑失败 dataset/seed。

普通数值失败按原样锁定并报告，不允许 seed/solver/K/threshold 补救。

## 9. P3：两个固定 fresh engineering smoke

只有 30/30 replay 通过后执行。

### 9.1 S00 RNA_PROTEIN

- data：A1 seed 0；
- route：显式 `RNA+PROTEIN`；
- full recipe：G04 backbone + H05 endpoint；
- 从权威 label-free 输入开始；
- 保存原子 checkpoint、resolved config、input/graph SHA、六视图或对应冻结输出、endpoint、resource row；
- 新进程严格加载 checkpoint，重复 forward，数值容差 `atol=1e-6, rtol=1e-5`，canonical partition exact。

### 9.2 S01 RNA_EPIGENOME

- data：P22 seed 0；
- route：显式 `RNA+ATAC`；
- full recipe：完整 G00 + 完整 G04 + R02 fixed adapter + C06/H01；
- 不得使用 topology-transfer 或历史 P22 embedding 代替 fresh smoke 的训练输出；
- 保存两个 backbone checkpoints、adapter checkpoint、resolved config、全部 input/graph SHA、endpoint 与资源；
- 每个 checkpoint 和最终 endpoint 均做 fresh-process round-trip。

### 9.3 禁止解释

不得读取标签，不得计算 ARI/NMI/Q/AMI/FMI/空间标签指标，不得比较哪次 smoke “更好”。Fresh smoke 只回答：统一软件链能否按冻结 recipe 跑完并重载。

## 10. P4：独立审计与终态

由与执行循环分离的审计脚本重新计算：

- authority file hashes；
- resolver signature/AST/runtime identity blindness；
- resolved config canonical SHA；
- 30-row set、顺序、唯一性和所有绑定 SHA；
- 30/30 replay embedding/partition；
- 2/2 smoke checkpoint/endpoint round-trip；
- labels/MISAR/E18.5/benchmark read count；
- correction、failure、runtime、GPU/CPU/RSS；
- 历史 raw before/after invariance。

成功终态只能是：

`NIGHT10B_FROZEN_FAMILY_POLICY_INTEGRATION_LOCKED`

如果 30-row replay 通过，但 fresh smoke 因明确基础设施原因在任何科学输出前阻塞，可使用：

`NIGHT10B_REPLAY_LOCKED_FRESH_SMOKE_INFRASTRUCTURE_BLOCKED`

它不等同于整合完成。权威缺失、实现语义不一致、资源超限或标签防火墙破坏必须使用合约规定的失败终态。

## 11. Git 与交付

### 11.1 Git

- 每个阶段小 commit；
- 所有 push 普通 push；
- 不改历史 branch/tag/report/raw；
- final tracked repo index 在 final commit 中；
- final tag 只在 final commit 后创建一次；
- 远端 branch 与 final tag peel 必须等于 final commit；
- 生成从 parent `8b83fe...` 开始的增量 bundle，并验证其 prerequisite 与 tag peel。

### 11.2 Windows compact

下载范围只包含：

- 源代码、测试、冻结配置；
- JSON/CSV/MD/log 审计；
- Git 增量 bundle 和验证记录。

禁止下载 raw data、checkpoint、embedding、affinity、`.pt/.pth/.ckpt/.npy/.npz/.h5/.h5ad/.arrow` 等大工件。

compact 必须含 root-relative path、size、SHA-256 的索引；Windows 独立复算全部文件。报告实际 `N/N`，任何 missing/size/SHA mismatch 均为失败。

## 12. 关机纪律

只有在以下全部完成后，才能派发 `/usr/bin/shutdown`：

1. final commit/tag/remote peel；
2. bundle 验证；
3. compact 下载；
4. Windows 独立 SHA 复算；
5. 最终用户摘要已整理。

`/usr/bin/shutdown` 必须是最后一条远端命令。SSH 随后关闭即可；不得重连确认。只报告“关机命令已派发”，不得声称控制面板已经关机。

## 13. 最终报告必须回答

1. 两条 frozen recipe 是否被唯一重建；
2. router 是否真正 dataset-identity-blind；
3. 30/30 Q00 replay 是否精确通过；
4. 两个 fresh smoke 是否完整通过；
5. 是否发生任何 correction、普通数值失败或资源超限；
6. labels、MISAR Y、E18.5、第三方 benchmark 的访问数；
7. final commit/tag/bundle/compact 的 SHA 与验证数；
8. 为什么本轮不能被包装成新的科学性能提升。
