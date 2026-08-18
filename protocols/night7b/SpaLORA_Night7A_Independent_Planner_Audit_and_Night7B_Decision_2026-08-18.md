# SpaLORA Night-7A 独立规划审计与 Night-7B 决策

日期：2026-08-18  
审计性质：Windows 本地只读复核；未连接 AutoDL、未运行训练、未修改 Night-7A 结果  
Night-7A 权威终态：`KEEP_CONFIRMED_G04_H05_FOR_EXTERNAL_VALIDATION`  
权威提交：`8b67e4bc09196f6c44b20e7afcfd0c3f0345e88b`

## 1. 给非技术读者的阶段结论

项目已经完成三个重要台阶：

1. **旧基线已经被稳定超过。** Night-6D 的 `G04+H05` 在 D1 人类淋巴结和 P22 小鼠脑都比同轮旧基线高，且不是单个 seed 偶然；这个阳性结论仍然有效。
2. **主要有效部件已经找到。** 真正稳定带来提升的是 H05 聚类头；G04 图本身不是所有数据都更好。
3. **下一处突破口已经定位。** Night-7A 的 C06 在 P22 上比当前确认结构又提高约 `0.045 Q`，D1 也略升，但 A1/tonsil 各退约 `0.003`。这不是“方案失败”，而是说明应让模型按数据和 spot 自动决定更信任哪张图。

因此，项目仍处在**以 ARI/NMI/Q 提升为核心的代码研发阶段**，尚未进入最终投稿制图或正式 SOTA 宣称阶段。下一轮 Night-7B 使用有卡模式，先筛聚类头，再训练轻量自适应关系融合模块，目标是同时保留 C06 的 P22 收益和 C00 的淋巴/tonsil 稳定性。

## 2. 独立完整性复核

本地交付根：`D:\文档\ChatGPT\博士第一篇科研论文项目\night7a_handoff_20260818\official_compact`

复核结果：

- internal delivery index：`97/97` 通过；
- external delivery index：`4/4` 通过；
- local post-dispatch index：`3/3` 通过；
- `night7a_report.md` SHA-256：`274cf68d82634ab935b894a101e439904f0ce3f4e0d16f6d146b2d7ee95e4de3`；
- `night7a_decision.json` SHA-256：`43a24db0e129c2c9914992c8dda84be35d55c51957697efba609d7c89e16dd3e`；
- Git final commit：`8b67e4bc09196f6c44b20e7afcfd0c3f0345e88b`；
- annotated final tag object：`a746367...`，peeled commit 与 final commit 一致；
- branch/tag 证据显示普通 push、无 force、final tag 只创建一次。

Night-7A 报告的 60/60 views、120/120 历史预测、H05 30×2 partition parity、360/360 transforms、32/32 tests 与独立复算记录彼此一致。6 个 D1 C04/C05 数值失败被保留且未 fallback，不影响 C00/C06 的比较。

## 3. 关键数值复核

### 3.1 当前确认结构 C00 的绝对水平

| 数据集 | mean ARI | mean NMI | mean Q | 相对 G00/H00 的 ΔQ |
|---|---:|---:|---:|---:|
| A1 | 0.269205 | 0.408674 | 0.338940 | +0.008270 |
| tonsil | 0.152954 | 0.272723 | 0.212839 | +0.083227 |
| D1 | 0.241201 | 0.377656 | 0.309428 | +0.030332 |
| P22 | 0.420274 | 0.590306 | 0.505290 | +0.031418 |

### 3.2 C06 相对 C00 的变化

| 数据集 | C06 mean Q | C06 − C00 ΔQ | 解释 |
|---|---:|---:|---|
| A1 | 0.335738 | -0.003202 | 小幅回退 |
| tonsil | 0.209513 | -0.003325 | 小幅回退 |
| D1 | 0.314880 | +0.005452 | 小幅提升 |
| P22 | 0.550345 | +0.045055 | 明显提升 |

四数据集等权 macro Q 下，C06 比 C00 约高 `+0.010995`；按本轮用户优先级权重 A1/D1/P22/tonsil=`0.25/0.25/0.35/0.15`，C06 比 C00 约高 `+0.015833`。它之所以没有在 Night-7A 替代 C00，是因为预注册复杂度门要求双图方案补偿两次 encoder 成本，并且不能损害 A1/tonsil；不是因为 P22 提升不存在。

## 4. 当前成功概率的更新

这里只给条件判断，不伪造精确概率：

- **把方法做到稳定高于原 SpaLORA/SpatialGlue 基线：高概率，且已经在 D1/P22 实现。**
- **再做出一个在当前四套切片多数或全部提升的统一版本：中等偏高。** C06 已提供明显可利用的 P22 信号，下一步不是盲搜。
- **直接达到或超过真正同协议的 2025–2026 SOTA：目前未知。** 还没有在相同输入、相同 K、相同 fixed-final evaluator 下跑现代方法，不能把不同论文表格直接横比。
- **形成可投稿 Q2 论文：可行但尚未锁定。** 需要一个可复现的统一赢家、至少一至两套新增独立数据，以及公平现代基线。若只有当前绝对分数而没有同协议比较，成功率判断会失真。

## 5. Night-7B 决策

Night-7B 定位为：`GPU_SCORE_FIRST_ADAPTIVE_RELATIONAL_FUSION_RND`。

它包含两个有顺序的漏斗：

1. **H 阶段（便宜）：** 在 30 个现有 dataset-seed 单元上完整测试固定的 row-weight、WNN reliability 和多种聚类头；先找出“不重新训练也能提分”的空间。
2. **A 阶段（GPU）：** 在相同输入上训练轻量 residual adapter，分别测试 relational KL、CLIP、MNN triplet、masked reconstruction、pseudo-cluster semantic alignment、MoE gate 和 DCCA；先用 seeds 0–1 全候选筛选，再把自动晋级的固定候选扩到剩余 seeds。

所有当前数据都明确标为 development。真实标签可以用于每个阶段结束后的统一评价和自动晋级，但不能进入 loss、early stopping、checkpoint 选择、失败重试或 dataset-specific 分支。每次成功训练必须保存 final checkpoint 并做 round-trip parity，避免再次出现 Night-5A/Night-6B 的“有结果但无模型权重”阻塞。

## 6. Night-7B 的成功门

Night-7B 的新候选若要锁定，至少必须同时满足：

- priority-weighted mean ΔQ vs C00 `>= +0.018`；
- human-lymph aggregate（A1 与 D1 等权）ΔQ `> 0`；
- P22 ΔQ `>= +0.030`；
- A1、D1、P22 各自 mean ΔQ 不为负；tonsil mean ΔQ `>= -0.005`；
- 四数据集 paired Q wins 至少 `22/30`，且每个数据集达到固定最低 wins；
- 空间保护门通过；
- 不能靠 best epoch、seed 搜索、标签训练或隐藏 per-dataset recipe。

若没有候选通过，不把最好的失败者包装成成功；保留 C00 为确认结构，同时把 C06 作为 P22 accuracy frontier，为下一轮更强模型继续提供方向。

## 7. 资源与沟通决定

1. Night-7B 必须在 AutoDL **有卡模式**启动；0.5 CPU/2 GB 无卡模式不再用于此类矩阵任务。
2. 规划 Worker 不自动启动实例、不自动唤起实验 Codex；用户开机后复制 bootstrap prompt。
3. 实验 Codex 的每次最终回复必须先用通俗语言说明：做了什么、分数是否提高、为什么成功/失败、下一步需要什么，再给技术证据。
4. Windows D 盘只收 compact；raw runs、checkpoints、外部源码和 cache 留在 `/root/autodl-fs`，避免 C 盘和 D 盘无意义膨胀。

