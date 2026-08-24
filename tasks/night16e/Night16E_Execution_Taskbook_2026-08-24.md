# Night-16E 执行任务书

## 目标

在一个连续任务中完成两件相互依赖的事：

1. 以最小官方下载闭合 MISAR 多发育阶段的 processed 数据与 reference protocol；
2. 在 Night-15F 直接能量上实现 TSRE（三态关系能量），同时给出 family-frozen 方法板和 per-lane 分数前沿板。

AutoDL 当前有卡在线。本轮结束后保持开机，**不要派发 shutdown**。

## 0. 权威起点

完整读取并独立核对：

- Night-16D official compact 的 report、decision、absolute table、candidate ledger、correction audit、source/tests；
- Night-15F 的 method semantics、decision、matched ablation 和实现；
- Night-15G、Night-16B、Night-16C 的最终强起点 partition/embedding/registry 及 hash；
- Worker2 的近期论文数据家族与协议审计。

不要只按提示词重写。实际强起点、同 head teacher、历史开发高分必须分开；每个比较必须是同一输入与同一 evaluator。

## 1. 数据阶段

1. 先检查 AutoDL 持久盘已有 Zenodo/INSTINCT/PRESENT/SMART 资产和剩余空间。
2. 若无缓存，下载并验证 Zenodo 14789361 的 268.2 MB zip；安全列包后只读解压到新 external root，原文件不可修改。
3. 对 E11.0、E13.5、E15.5、E18.5 的每个物理单元登记：来源 URL/DOI、archive 与文件 SHA、shape、spot IDs、RNA/ATAC 层、坐标字段、reference 字段、K、mask、是否人工/人工核验/算法派生。
4. 真实端到端 P0：真实 preprocessing → 稀疏图 → TSRE 数据结构 → no-op partition → checkpoint/artifact reload → evaluator。先做 E18.5，再并行处理其余可用阶段。
5. 若小包缺必要对象，保留失败证据并按方法合约中的官方来源优先级继续；不要凭名称猜矩阵或 annotation。

## 2. 方法实现

实现统一 TSRE core。优先复用 Night-15F 经验证的稀疏图、dynamic unary、stay cost、alpha-expansion 和能量审计代码；新增代码集中在三态关系权重、boundary-exclusion unary 与 private-modality unary。不要复制 GPL/AGPL 第三方源码。

最低语义测试：

- 三态权重有限、非负、可解释且边顺序置换不改变结果；
- support 只增加合法非负平滑；boundary 不通过负 Potts 破坏 submodularity；
- conflict/private 的模态选择只由当前数值可靠度产生；
- 全拒绝时 exact self-return；全 support、全 boundary、全 conflict 的小图方向正确；
- 每个接受 move 降低当前 frozen energy；
- cluster label permutation 不改变评价；
- evaluation mask 与 full partition 的簇大小分别正确；
- 两个家族各一条真实 P0 和 fresh-process byte-exact partition replay。

允许正常工程修正，不设武断的“一次修正额度”。每次修正记录原因、受影响输出和重算范围；不因 smoke 分数不好偷偷换成另一套公式。

## 3. 有界搜索与冻结

先用少量结构候选验证 TSRE full 是否能在同起点超过 Night-15F。若完全无表示/能量增益，停止同义网格并如实封口；可继续完成数据资产闭合，不用靠扩大数据掩盖方法失败。

若出现信号：

- 蛋白家族在 A1 + tonsil s1 上选择一个 study-balanced family profile；冻结后一次性跑 D1 + tonsil s2/s3；
- 染色质家族在 P22 K9 上选一个 family profile；冻结后一次性跑 MISAR E15.5 与新闭合阶段；P22 K18 仅 sensitivity；
- 主方法板不得为 transfer lane 重新选 profile；
- 分数前沿板可逐 lane 扩展 label-assisted HPO，目标是刷新 ARI/NMI Pareto 前沿，但与方法板分表；
- 先单 seed/确定性开发；只有出现有意义信号才做 3 个模型种子或多 endpoint 稳定性，不为无信号候选消耗 GPU。

无需所有数据集都提高才继续。保留每条 lane；科学结论按 study block 与家族汇总，分数前沿按 lane 汇总。

## 4. 资源与效率

- 重训练/大矩阵 preprocessing 留在 AutoDL；纯 evaluator、表格、hash、compact 检验优先在 Windows 本地完成；
- 不运行完整第三方基线；只核对论文/官方源码公开 context；
- 不生成 dense N×N；使用 CSR edge list；
- 避免重复读取数百 GB fragments；优先复用冻结、已哈希的 feature/embedding/graph；
- 保持合理并行，先做内存探针；不要为交付审计反复重跑科学计算。

## 5. 必交付

- `night16e_report.md` 与通俗总结；
- dataset/protocol registry；
- P0/replay audit；
- family-frozen 方法主表；
- per-lane score-frontier 表（绝对 ARI/NMI/AMI/FMI、permutation-invariant spatial metrics、簇大小、运行时间与资源）；
- matched ablation 表；
- candidate ledger（含失败/作废行）；
- label/HPO flow、source collision、resource、raw immutability、correction audit；
- source、targeted tests、root-relative size/SHA index、incremental bundle。

从 Night-16D final commit 新建 `revision/q2-night16e-tristate-relation-energy-and-stage-transfer-20260824`，最终 annotated tag `night16e-final-20260824`。普通 push；GitHub key 若仍缺失，记录并保留可恢复 bundle，不 force push。

最终回复先给“我现在需要知道的三件事”，再给绝对结果、方法贡献、数据新增、导师汇报版，最后才放 Git/hash。Night-16E 完成后保持 AutoDL 在线，等待 Worker2 下一条任务。

