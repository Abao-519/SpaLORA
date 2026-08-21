# 复制到另一条 Codex 对话的第一条消息

```text
现在开始执行 SpaLORA Night-10A。AutoDL 已由用户以“有卡模式”开机后再开始连接；如果当前实例未开机，只报告未开机，不要结束或改写任务。

请先完整读取以下共享 D 盘文件，逐项复算 SHA-256，并把任务书与注册表加入你的工作计划：

1. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_local_planning_20260821/SpaLORA_Night10A_QCRD_Score_RnD_Taskbook_2026-08-21.md
   SHA-256: 5551270041576969e1a777a6dcf8f1b8113d2fdefe7b7322738a4d57c8b4bd0d
2. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_local_planning_20260821/night10a_qcrd_candidate_registry.json
   SHA-256: b310e232282840772b69ad9d1c1d1de6e04c92e8b7b013a7e5247c628dfd959d
3. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_local_planning_20260821/metric_expansion_reference.py
   SHA-256: 33397cada3701108fbfb15d58fafd6f1f9e20417ac7bc14cce1e8aee993cd159
4. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_local_planning_20260821/test_metric_expansion_reference.py
   SHA-256: 4100b23ab30d74db02cb2e540112b9548080ae1c17817011c202a409b48c6ab0
5. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_local_planning_20260821/Night10A_Local_Evidence_and_Innovation_Audit_2026-08-21.md
   SHA-256: ea920aeee3593a7903b6189119dc60b7327cee48f9d8b00e0c4283420049b34c
6. D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_local_planning_20260821/authoritative_scoreboard_20260821.csv
   SHA-256: 951a9e0449e6e083834cd15268bb4c316b011726c80bd122dd9a7516626a0d68

本轮最重要的目标不是找新数据、跑更多第三方方法或做论文图，而是：

- 只读复用 AutoDL 持久盘已有 clusters/views/checkpoints，补齐当前候选的扩展指标；不得为补指标重训旧模型；
- 实现并运行 QCRD（质量校准的跨模态残差去噪），重点解决“P22 上涨但人类淋巴下降”的冲突；
- 保留 accuracy、balanced、spatial 三条前沿，不再因为一个数据集轻微下降就清空全部 shortlist；
- 禁止 dataset-name 路由、标签参与训练/选参、seed search、dense N×N 运算、科学重试和 fallback；
- 本轮不访问 MISAR Y、E18.5 或任何新外部数据，不运行新的 COSMOS/SMART/PRESENT/ARISE benchmark；
- 使用 GPU 模式；CPU transform 最多 4 并行，单任务 30 分钟 timeout，整轮 12 小时 wall-clock 上限，禁止再次单核卡一整天；
- D 盘只交付 compact，不下载 raw/checkpoint/embedding/affinity；Git 普通 push，无 force，final tag 一次创建；最后一条远端命令为 /usr/bin/shutdown，派发后不重连。

请先只做 P0 权威性、资源和真实语义检查。P0 完成后用通俗中文告诉用户：检查了什么、是否可以开始正式实验、预计训练单元和最长运行时间；随后如果 P0 通过，按任务书自主继续 Stage M、R1 和有条件的 R2，不需要用户逐项审批。

最终输出也必须先写通俗总结，让用户能直接向导师说明“这一步做了什么、哪些数据涨了、哪些没涨、创新点和代价是什么”，再给技术审计、Git、D 盘路径和 SHA。
```
