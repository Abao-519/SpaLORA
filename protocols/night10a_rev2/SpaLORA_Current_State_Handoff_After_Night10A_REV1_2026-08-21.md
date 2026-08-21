# SpaLORA 当前状态交接：Night-10A REV1 之后

日期：2026-08-21  
用途：新规划 Worker 的第一份当前状态文件。

## 当前一句话状态

项目已经得到一个跨 D1/P22 确认的强基线 `G04+H05`，并得到一个适合 RNA+ATAC 的高准确率但较慢的 `F00/R02` 路线；正在探索的 QCRD 尚未产生合法分数，最新失败是正式训练前未覆盖 P22 的 128/64 混合维度接口。

## 已经站住的阶段性结果

相对同 seed 的 G00/H00，`C00=G04+H05` 的 mean ΔQ：

- A1：`+0.00827`；
- tonsil：`+0.08323`；
- D1：`+0.03033`；
- P22：`+0.03142`。

Night-6D 对 D1/P22 的锁定确认：

- D1 mean ΔQ `+0.030332`，10/10 Q wins，Holm p `0.001953`；
- P22 mean ΔQ `+0.031418`，8/10 Q wins，Holm p `0.008789`；
- 机制诊断表明收益主要来自 H05，G04 本身不能宣称跨数据集普适优势。

RNA+ATAC family 的 `F00/R02` 在 MISAR 外部恢复评价中相对 U00：mean ΔARI `+0.013742`、ΔNMI `+0.020982`、ΔQ `+0.017362`，10/10 wins；但 runtime 为 `2.261×`，所以是准确率路线，不是高效率路线。

Night-9B 的层次融合 N02 在 P22 有 `+0.030665` ΔQ，但 A1 为 `-0.005923`，未通过跨平台保护门。COSMOS common-head 相对 F00 ΔQ `-0.014569`，没有超过当前 P22 路线。

## 还没有站住的内容

- 不能宣称 SOTA：尚缺完整同协议主流 benchmark 和更多外部数据确认。
- QCRD/MF-RACF 等新模块尚未形成跨 A1、D1、P22 且保住 tonsil 的统一候选。
- QCRD 两次终止均是实现/权威语义问题，不是有效负分结果；标签未打开，因此仍可在新定义下做一次干净恢复。
- MISAR E18.5 数据契约仍存在 provenance/feature/annotation 不匹配，不是当前主线。

## 当前科研优先级

1. 先用 REV2 把 QCRD 的真实混合维度接口和全量组合测试做正确；
2. 在不重跑无关计算的前提下，评价 QCRD 是否真正带来协同提分；
3. 若 QCRD 无明确正向信号，及时停止该分支，回到 `G04+H05` 与 `F00/R02` 的端到端家族策略整合；
4. 数据集搜集和第三方 benchmark 暂居次位，不能继续挤占核心模型研发资源。

## 下一步必须读取的文件

- `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev2_planning_20260821/Night10A_REV1_Failure_Root_Cause_and_REV2_Safety_Audit_2026-08-21.md`
- `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev2_planning_20260821/night10a_qcrd_rev2_dimension_and_reuse_contract.json`
- `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev2_planning_20260821/SpaLORA_Night10A_REV2_Execution_Taskbook_2026-08-21.md`
- `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev1_handoff_20260821/official_compact/handoff/night10a_report.md`
- `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev1_handoff_20260821/official_compact/handoff/p0_rev1_retrospective_false_pass_audit.json`

新规划 Worker 不得仅根据本摘要直接改任务书；必须独立读取上述证据和源代码，复核维度修复是否数学和工程上闭合。

