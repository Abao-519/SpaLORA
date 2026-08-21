# 复制到另一条 Codex 对话

```text
现在执行 SpaLORA Night-10A REV1。原 Night-10A 的零训练终态必须永久保留；本轮是定义级恢复，不得改写原 commit/tag/report。

请先完整读取并复算以下两个最高权威文件：

1. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev1_planning_20260821/night10a_qcrd_rev1_semantic_contract.json
   SHA-256: ecc91a433a3a4e2dfcf812d40aaa679287b3fcd76d8645f440d2b6015903fe75

2. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev1_planning_20260821/SpaLORA_Night10A_REV1_QCRD_Recovery_and_Execution_Taskbook_2026-08-21.md
   SHA-256: 08b387da1323dac1ab6d1ccb8640da52d71aca83f0f1800f5ce8dcff05637233

同时读取任务书列出的原 Night-10A compact、原 QCRD 部分实现/测试，以及 Night-6C H05、Night-7B F00/R02 endpoint 权威实现。不要请求用户重新上传；全部文件可由共享 D 盘直接读取。

本轮从原 Night-10A final commit 1e576b68938fa194dcdd53ee58915767b7a78325 新建 revision/q2-night10a-rev1-qcrd-execution-20260821。普通 push，无 force，不移动原 tag。

先完成 P0-REV1：严格实现新增 semantic contract，补全 global/per-spot quality、15%确定性 mask、全部固定 loss 权重、zero-degree 规则和两类固定 endpoint，并完成逐字段 semantic coverage 与负向测试。P0-REV1 不得读取任何标签，不要重复原 A1 parity 标签读取。

如果 P0-REV1 通过，先冻结并普通 push 代码/config，然后自主继续 R1、R1总锁后的唯一评价窗口和 Stage M 指标补算；按三条 frontier 条件进入 R2。无需用户逐项审核。

必须使用 AutoDL 有卡模式。单 transform 30分钟 timeout、整轮12小时 wall-clock；禁止再次单核无限等待。独立 unit 的普通数值失败应保留并继续，不能让一个失败作废整轮；只有系统性语义错误、标签泄漏、错误 source/config 或 artifact 损坏才硬停。

本轮不访问 MISAR Y、E18.5 或其他外部数据，不运行第三方 benchmark。D盘只交付 compact，不下载 raw/checkpoint/full embedding/affinity。最终最后一条远端命令为 /usr/bin/shutdown，派发后不重连。

请先用通俗中文回复：已读取 REV1 权威文件、REV1 补全了哪些定义、是否可以开始 P0-REV1。最终也必须先用通俗语言说明 A1/D1/P22 是否协同提高、tonsil 是否保住、创新模块是否值得进入端到端整合，再给技术审计。
```
