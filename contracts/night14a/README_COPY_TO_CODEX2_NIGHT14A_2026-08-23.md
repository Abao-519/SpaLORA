# 复制给 Codex 2：Night-14A

你现在接手 SpaLORA 项目的 Night-14A。请直接执行，不等待用户逐步批准。

先完整读取并遵守：

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14a_topology_conflict_sprint_planning_20260823/Night13C_Worker2_Independent_Audit_and_Night14A_Decision_2026-08-23.md`
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14a_topology_conflict_sprint_planning_20260823/SpaLORA_Night14A_Topology_Conflict_Sprint_Taskbook_2026-08-23.md`
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14a_topology_conflict_sprint_planning_20260823/night14a_topology_conflict_sprint_contract.json`
4. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14a_topology_conflict_sprint_planning_20260823/planning_delivery_index.json`

Night-13C 权威 compact 是：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night13c_delivery_20260823/official_compact`

其 `compact_delivery_index.json` 应为 59/59，SHA-256：

`bfdd7996fe742f19e1f9cd9782e85e5fb082d40409507c449cac47587cae189a`

先独立定位并复算；若不匹配立即停止报告。若匹配，不要在之后反复做整套文件审计。

这轮的核心判断是：Night-13C 否定的是被 simple PCA 强锚定、只有 80–160 步训练的浅层原型，不是否定成熟统一模型。停止继续调 B10/E11/E12。选择一个真实可训练的成熟统一 backbone（优先审计 SMART 与 SpaBalance，最多真实比较两条），在其上开发工作代号 TCF 的“跨模态拓扑冲突滤波器”。TCF 要用两个模态对局部空间边的支持和冲突，控制 identity/低通，以及必要时的高通通道；不能按数据集名称切模型。

你拥有充分研发自由：可以修改具体公式、loss、normalization、latent dimension、graph k、训练轮数和数值超参数；可以读取公开标签做开发评价与 HPO；可以进行合理工程修复，不受一次 correction cycle 限制；失败尝试保留到 ledger 即可。若仍声称训练无监督，标签不能进入 loss、gradient 或单次训练 checkpoint selection。不同数据集允许透明的数值超参数，真正禁止的是 dataset-name-driven backbone/flow switching。

先完成短小的评价口径修正：candidate 与 reference 用完全相同的 endpoint seeds；确定性 embedding 不复制成虚假的多 model-seed；common-head 与 native full-pipeline 分开。然后用 A1、tonsil s1、P22、MISAR 做开发；架构稳定后用 D1、tonsil s2/s3 做冻结后内部确认。不要要求所有数据集全部上涨，按 study-balanced ARI/NMI、matched strong reference、endpoint 稳定性、最坏回撤和资源共同排序。若最后只在 P22+MISAR 上形成稳健主线，可以诚实登记 `ATAC_FOCUSED_SIGNAL`，不必强行否定整个候选。

把主要精力放在真实模型性能、收敛、关键消融和机制诊断上，不要花大量 token 重复无关审计，也暂时不要复现一整张外部 benchmark 表。官方源码只用于 backbone 选择、语义学习和必要对照，固定 commit、许可证与 attribution。

执行完成后，以“我现在需要知道的三件事”开头，先给通俗结论和明确分类，再给绝对 ARI/NMI 主表、开发/确认结果、最重要失败、导师汇报版，最后放 Git/hash/compact 技术附录。完成新 branch、普通 push、annotated tag、compact 和 Windows 独立复算后，再把 `/usr/bin/shutdown` 作为最后一条远端命令派发。

