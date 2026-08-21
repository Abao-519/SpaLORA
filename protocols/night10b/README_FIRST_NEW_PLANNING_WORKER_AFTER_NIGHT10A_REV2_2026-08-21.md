# 新规划 Worker 启动文件：Night-10A REV2 之后

你将接棒 SpaLORA 论文项目的规划工作。不要直接根据本文件安排实验，必须亲自读取和复核证据。

## 当前终态

- Night-10A REV2 工程修复通过：真实 P0 210/210，P22 64→128 冻结矩形 Procrustes 闭合。
- QCRD 科学候选失败：最终 macro ΔQ `-0.0024105`，protein 与 ATAC family 均为负，frontier entry=false。
- D1 的独立正向结果可以保留，但不能据此声称 QCRD 成功。
- 当前站得住的路线是 RNA+protein 的 `C00/G04+H05` 与 RNA+epigenome 的 full `F00/R02`。

## 先读与先核验

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev2_delivery_20260821/official_compact/compact_delivery_index.json`  
   SHA-256：`d2b1a1c6baf469aede5b7042b2133b6d065b6ffe0f30919682a952b08decc804`；应为 182/182。
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev2_delivery_20260821/official_compact/repo/outputs/night10a_rev2_handoff/SpaLORA_Night10A_REV2_Final_Report_2026-08-21.md`  
   SHA-256：`4cb99b6f89ee1354feb44ef3adc646d728df78877ccb5547f7bf7f0402b0826e`。
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev2_delivery_20260821/official_compact/repo/outputs/night10a_rev2_handoff/r2_lock/r2_total_lock_independent_audit.json`  
   SHA-256：`b85ca47915589471afee0732bc336fca31071f1bfa114494f538dcd0618adedc`；应含 30 个 Q00 references。
4. 本目录的 Post-Night10A 决策、Night-10B JSON 合约、Taskbook、执行 Codex 提示词与 planning index。

还必须读取 Night-6D、Night-7B、Night-8B、Night-9A、Night-9B 的权威 report/decision/source，确认两条 frozen recipe 的来历和限制。不得自动派发另一个规划代理，不得开服务器，不得改历史交付。

## Night-10B 的定位

Night-10B 是零标签代码整合与复现封口：

- 路由只看显式 assay pair；
- `RNA+PROTEIN -> C00_G04_H05_CONFIRMED`；
- `RNA+ATAC -> F00_R02_FULL`；
- 30/30 Night-10A Q00 reference 精确重放；
- 两个固定 fresh engineering smoke；
- 不产生新科学分数，不扩数据或 benchmark。

若 Night-10B 成功，只能说明统一软件入口和冻结路线的复现链闭合；不能声称新的统一模型、SOTA 或效率优势。

## 规划 Worker 的审核职责

执行 Codex 回报后，必须在 Windows 独立完成：

1. compact 根索引 `N/N` 与内部 repo index `N/N`；
2. 全部 size/SHA 与禁带大文件类型；
3. 30-row 单元集合、顺序、唯一 row SHA 和 family/recipe 绑定；
4. router AST 与运行时 dataset-identity-blind 审计；
5. 2/2 smoke checkpoint fresh-process round-trip；
6. label/MISAR/E18.5/benchmark 计数为 0；
7. final commit/tag/bundle prerequisite 与远端 peel 封口证据；
8. 关机只确认命令派发，不额外声称控制台状态。

任何文件不匹配、authority 语义不唯一或 30-row 未闭合时立即停止，不得“硬着头皮”接纳交付。
