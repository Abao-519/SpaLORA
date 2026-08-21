# SpaLORA Night-10A REV2 后科研决策与 Night-10B 定位

日期：2026-08-21  
角色：规划 Worker 2  
决策：`PROCEED_TO_NIGHT10B_FROZEN_FAMILY_POLICY_INTEGRATION`

## 1. 结论先行

Night-10A REV2 可以正式封口为：**工程修复成功，QCRD 科学候选失败**。

下一步不应继续围绕 QCRD 调参，也不应立刻增加数据、第三方 benchmark 或新的统一融合模块。当前证据真正支持的是两条已经冻结的模态家族路线：

- `RNA_PROTEIN`：`C00 = G04_SP10_F10_EUC_UNION + H05_EQUAL3_AFFINITY_SPECTRAL`；
- `RNA_EPIGENOME`：`F00 = R02__E1_ADAPTER_C06_MEAN__H01`，保留完整 G00+G04 双骨干和固定 `RECON+MNN/equal/160` adapter。

Night-10B 的任务不是创造新的科学候选，而是把这两条路线整理成一个**只根据显式 assay pair 路由、完全不读取数据集名字或标签、可以从单一 CLI/配置入口运行**的冻结 family policy，并对 Night-10A 已锁定的 30 个 Q00 reference 做零标签精确重放。

## 2. 已经站住的科学证据

### 2.1 RNA+protein 路线

相对同 dataset、同 seed 的 fresh `G00/H00`，`G04/H05` 的 mean ΔQ：

| 数据集 | mean ΔQ | 证据角色 |
|---|---:|---|
| A1 | +0.008270 | development |
| tonsil | +0.083227 | independent development section |
| D1 | +0.030332 | locked confirmation，10/10 Q wins |

Night-6D 中 D1 的 mean ΔARI、ΔNMI、ΔQ 均为正，exact paired test 与 bootstrap 均通过。机制分解同时表明，不能把收益只归因于 G04：H05 是主要贡献者，A1 还存在 graph/head interaction。因此正式 recipe 必须保留完整 `G04/H05` 身份。

### 2.2 RNA+epigenome 路线

`F00/R02` 在 P22 上是高准确率路线；在 MISAR E15.5 S1 的锁定恢复评价中，相对统一参考 U00：

- mean ΔARI `+0.013742`；
- mean ΔNMI `+0.020982`；
- mean ΔQ `+0.017362`；
- Q wins `10/10`。

但端到端 runtime 约为 U00 的 `2.261×`，所以只能称为准确率优先路线，不能称为高效率路线。Night-9A 已经否决了已注册的低成本替代方案；Night-10B 不得偷偷换成近似压缩版本。

### 2.3 统一新模块已经得到的负证据

- Night-9B：MF-RACF 没有跨 A1/P22 候选；N02 只在 P22 提高，在 A1 下降并触发保护门。
- Night-10A REV2：QCRD 最终 macro ΔQ `-0.0024105`，protein family `-0.0002966`，ATAC family `-0.0087522`，只有 D1 独立为正。

这两轮都说明，继续在已经反复使用的四个数据集上寻找一个统一 residual/gating/hierarchy 模块，容易形成事后研发和标签消费，却没有可靠的跨家族收益。

## 3. 为什么 Night-10B 是整合封口，而不是新实验

原论文和重构蓝图要求最终代码具备：统一配置、完整训练/评价入口、固定种子、环境与 provenance、每行主表可复现。当前证据虽然丰富，但实现仍分散在 Night-6C、Night-7B、Night-8B 等历史脚本中，尚缺一个正式、稳定、身份盲的主入口。

Night-10B 解决的是这个缺口：

1. 明确论文当前可辩护的方法身份是**模态家族策略**，不是跨所有模态共享同一 trainable module；
2. 把两条权威 recipe 固定进内容寻址配置；
3. 通过显式 `primary_assay`/`auxiliary_assay` 选择 recipe；
4. 路由器不得接收或检查 dataset、tissue、species、stage、文件名或标签；
5. 对已锁定 30 个 family-reference embedding/partition 做精确重放；
6. 只做两个固定的无标签 fresh smoke，证明从原始 label-free 输入到输出的完整链路仍可运行。

本轮不得计算 ARI、NMI、Q、空间标签指标或任何候选排名。既有科学结果只作为不可变历史证据引用，不进入 Night-10B 的选择过程。

## 4. 冻结方法身份

### 4.1 路由输入

唯一合法路由依据是数据管理员在模型运行前提供的显式 assay pair：

- `RNA + PROTEIN -> RNA_PROTEIN`；
- `RNA + ATAC -> RNA_EPIGENOME`。

不得从输入维度、数据集名称、组织名称、物种、spot 数、annotation、历史分数或文件路径推断 family。未知、倒置、缺失或含糊的 assay pair 必须 fail closed。

### 4.2 RNA_PROTEIN

- graph：`G04_SP10_F10_EUC_UNION`；
- endpoint：`H05_EQUAL3_AFFINITY_SPECTRAL`；
- reference identity：`C00_G04_H05_CONFIRMED`；
- family preprocessing：沿用 A1/tonsil/D1 已冻结 RNA+protein 语义；
- 禁止以 dataset-specific trainable weights、标签或数据集名字改变配置。

### 4.3 RNA_EPIGENOME

- backbones：同 seed 的完整 `G00_SP18_F20_CORR_UNION` 与 `G04_SP10_F10_EUC_UNION`；
- adapter：`R02`，loss=`RECON+MNN`，fusion=`equal`，fixed epochs=`160`；
- endpoint：`E1_ADAPTER_C06_MEAN__H01`；
- reference identity：`F00_R02_FULL`；
- family preprocessing：稀疏 ATAC/LSI 路径，不允许 dense N×N；
- 禁止使用 Night-9A 未通过的 topology-transfer 近似替换完整 F00。

## 5. Night-10B 验证对象

Night-10A REV2 在 R2 总锁中已经记录 30 个 Q00 reference：

| 数据集 | family | seeds | 单元数 |
|---|---|---|---:|
| A1 | RNA_PROTEIN | 0–4 | 5 |
| tonsil | RNA_PROTEIN | 0–4 | 5 |
| D1 | RNA_PROTEIN | 0–9 | 10 |
| P22 | RNA_EPIGENOME | 0–9 | 10 |
| 合计 |  |  | 30 |

每个单元已有 ordered observation SHA、registered input SHA、input manifest SHA、reference embedding SHA 与 canonical partition SHA。Night-10B 必须以这些 30 行为唯一正式 replay 总体；不得少跑、替换 seed、增加数据集或仅抽样声称完成。

## 6. Fresh smoke 的边界

只授权两个预先固定的工程 smoke：

- A1 seed 0：完整 `RNA_PROTEIN/C00` 路线；
- P22 seed 0：完整 `RNA_EPIGENOME/F00` 路线。

它们仅用于验证训练、checkpoint round-trip、endpoint、manifest 和资源记录可以通过统一入口完成。不得打开标签，不得计算科学指标，不得与历史分数比较，不得因为结果形状、聚类外观或无标签诊断“不好看”而重跑。

## 7. 成功与停止规则

允许的成功终态：

`NIGHT10B_FROZEN_FAMILY_POLICY_INTEGRATION_LOCKED`

必须同时满足：

1. 权威输入和父 commit/tag 完整通过；
2. 路由器纯函数测试与 AST/运行时身份盲审计通过；
3. 30/30 Q00 reference 的内容、family、recipe、embedding 和 canonical partition 重放闭合；
4. 两个 fresh smoke 均完成，并通过 checkpoint fresh-process round-trip；
5. 标签读取为 0，MISAR Y/E18.5/第三方 benchmark 访问为 0；
6. 无 dataset-name routing、无 dense N×N、无科学 retry/fallback；
7. Git、compact、Windows SHA 复算和关机派发闭合。

若权威源语义或 raw artifact 不能支持精确重放，停止为 `BLOCKED_AUTHORITY_OR_ARTIFACT`；若统一实现与冻结 recipe 不一致，停止为 `IMPLEMENTATION_SEMANTICS_INVALID`。不得通过重新训练历史 baseline、放宽 SHA/partition 门或新开标签来补洞。

## 8. 对论文叙事的约束

Night-10B 成功只允许说明：

> 已冻结的 RNA+protein 与 RNA+epigenome 路线已被整理为一个显式模态感知、数据集身份盲、可审计的统一软件入口。

它不允许自动升级为：

- 一个共享 trainable module 在所有模态上普适优于 baseline；
- SOTA；
- 高效率方法；
- 新外部数据已确认；
- QCRD 被整合后转为正向结果。

工作名继续使用 SpaLORA。是否改名应在最终方法和论文主张锁定后决定，而不是由失败模块名称驱动。

## 9. 下一里程碑

Night-10B 完成后，优先进行一次论文证据账本与复现入口审计，再决定正式 benchmark 的最小公平集合。不得在 Night-10B 执行过程中顺手运行第三方方法或扩展数据范围。
