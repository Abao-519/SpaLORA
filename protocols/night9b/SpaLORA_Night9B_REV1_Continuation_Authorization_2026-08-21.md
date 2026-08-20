# SpaLORA Night-9B REV1：P0 接受与后续阶段继续授权

日期：2026-08-21  
适用父任务：`SpaLORA_Night9B_SOTA_Gap_and_RNA_Anchored_Cooperative_Fusion_Taskbook_2026-08-20.md`  
适用注册表：`night9b_racf_locked_20260820_v1`

## 1. P0 正式接受

规划方已在 Windows D 盘独立复核 P0 compact：16/16 文件的 size 与 SHA-256 全部匹配。

接受以下权威状态：

- P0 commit：`397638728debfe2c9aef0b6425776b471bfe4977`
- 远端分支：`revision/q2-night9b-rna-anchor-cooperative-fusion-rnd-20260820`
- 保护标签继续 peel 到父提交：`aa933b8fc11fc05470287a21f80facd03d0acfb9`
- P0 合约 SHA：`f693215e148bc8c879914f489b3923b48cd04e9953281be83e3c77531d57ecf8`
- Night-9A 资源语义审计 SHA：`517137eec884123b2afcf89f199dda6aa777521dcfffe32686ebf97203fe0ede`
- P0 compact index SHA：`29a7ad8a7485203781204e1c8e081e51be09eb2c4cde837bf309c3cf1457c8a4`
- 正式训练、benchmark、embedding、affinity、partition 与 A1/P22/MISAR 标签读取均为 0。
- `next_phase_authorized=true`、`next_phase_started=false`、P0 errors=0。

P0 合约中的 `git.head` 是创建 P0 commit 之前的父提交，而 `p0_git_receipt.json` 记录提交后的本地/远端 `397638...` 一致；这是时间顺序差异，不是 lineage mismatch。后续不得回写 P0 合约。

## 2. 对“4 次基础设施修正预算”的权威澄清

注册表字段：

`infrastructure_corrections_before_science_max = 4`

它在本轮的权威含义限定为：**P0 权威/环境审计阶段允许记录的基础设施与验证元数据修正上限**。P0 已使用并关闭这项预算，四项记录全部接受：文件名、更准确的 COSMOS checkout 路径、正确 Python 环境、手抄日志 SHA 字符串修正。

以下行为不计入该 P0 字段，也不计为 scientific retry：

- P1 新模块首次实现后的编译、单元测试、真实构造测试和负向回归测试；
- 在任何正式 R1/P2 科学单元启动前，对 P1 实现缺陷进行修复并重新运行同一语义测试；
- 仅用于验证 checkpoint round-trip、梯度到达、图构造和无标签 smoke 的开发运行。

但 P1 不是无限调试：

- 最多允许 8 个具有不同根因、完整留痕的 P1 implementation correction cycles；
- 最多允许 2 个 P1 environment-only correction cycles；
- correction 不能改变注册的 7 个候选、参数、seed、shortlist gate 或 benchmark endpoint；
- 一旦第一个正式 COSMOS/R1 scientific training unit 启动，结构与 evaluator SHA 全锁，后续 scientific retry 仍严格为 0，fallback 仍严格禁止；
- 若在上述 P1 预算内不能通过全部真实语义测试，终态必须为 `IMPLEMENTATION_SEMANTICS_INVALID` 或 `INFRASTRUCTURE_BLOCKED`。

这项澄清只修复任务管理语义，不放宽标签防火墙、科学重试或候选选择规则。

## 3. 后续阶段连续执行授权

从 P0 commit 继续执行，不重做 P0。授权顺序为：

1. P1 外部源码语义锁、MF-RACF 实现、真实语义测试与无标签 smoke；
2. P1 全通过后，按原任务书执行 P2 COSMOS 公平校准和 R1；
3. 全部训练/endpoint 输出总锁并普通 push 后，才打开一次 A1/P22 标签窗口；
4. shortlist 非空才执行 R2 seeds 3、4；为空则按原合法终态停止；
5. 完成独立复算、测试、Git、compact 交付与最终关机。

不需要在 P1 通过后再次等待用户复制“继续”。每个阶段只做低频、通俗的里程碑更新；只有硬门失败、权限阻塞或输入权威不确定时才停下请求用户。

## 4. 保持不变的边界

- MISAR Y 永久禁止第三次读取。
- A1/P22 总锁前标签读取为 0；禁止 `anndata.read_h5ad` 打开含标签原文件。
- ARISE 只做源码语义审计与结构溯源，禁止运行其真实标签逐 epoch ARI 选最佳 embedding 路径。
- COSMOS 只能使用指定 commit；native fixed 与 common-head 输出必须共享同 seed 的一次训练。
- 7 个 MF-RACF 候选、42 个 R1 units、最多 16 个 R2 units、5 个 COSMOS training units 均不变。
- 允许 balanced、lymph specialist、brain specialist；禁止 dataset/seed/label-specific 新权重。
- raw/checkpoint/embedding/affinity 留在远端持久盘，D 盘 compact 目标仍小于 10 MB。
- 禁止 AutoDL API。最终最后一条远端命令仍必须为 `/usr/bin/shutdown`；派发后不得重连。

