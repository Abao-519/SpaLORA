# Night-7C 实验 Codex 第一条消息

使用方法：先在 AutoDL 控制台用**有卡模式**开机，再把下面代码块中的全部内容复制给负责实验的 Codex。不要上传文件；所有权威文件都在共享 D 盘路径中。

```text
请执行 SpaLORA Night-7C。你必须先完整读取并逐 SHA 核验以下权威文件，然后严格按任务书执行；不要自行扩展候选、阈值、seed、预算或评价规则：

1. D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night7C_Planning_Delivery_Index_2026-08-18.json
2. D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night7B_Independent_Audit_and_Night7C_Decision_2026-08-18.md
3. D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night7C_Source_Code_Transfer_Audit_2026-08-18.md
4. D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night7C_Conflict_Gated_Affinity_Registry_2026-08-18.json
5. D:\文档\ChatGPT\博士第一篇科研论文项目\SpaLORA_Night7C_Conflict_Gated_Routing_and_Runtime_Acceleration_Taskbook_2026-08-18.md
6. D:\文档\ChatGPT\博士第一篇科研论文项目\night7b_handoff_20260818\official_compact\compact_delivery_index.json

本轮核心目标是：用完全无标签的跨模态冲突、匹配置信度和共享邻边支持，在 C00 与 Night-7B 的 P22 专家 R02 之间进行预注册门控；同时运行固定的六个置信度加权 MNN pilot，并在不改变结果的前提下缩短 CPU transform 长尾。

跨数据集语义必须按任务书 0.1 节理解：A1/D1、tonsil、P22 不做联合训练，也不共用 checkpoint；每个 dataset × seed 独立训练。所谓统一，是 architecture、候选、保护门和一套 identity-blind label-free router 公式统一。允许同一公式从每个单元自己的无标签统计产生不同权重；禁止按 dataset/tissue/platform/modality/file name 手工查表路由。不要把“统一规则”误解为“一个固定数值权重或一个 frozen model 强行适配所有平台”。

重要执行要求：

- 当前服务器应由用户以有卡模式开机；正式训练必须实际在 CUDA 上，CPU 谱分解/聚类期间 GPU=0 允许但必须分阶段记录。
- 不得调用 AutoDL API，不得启动其他 Codex，也不得要求用户逐操作审批。请在当前已授予权限内自主完成整轮；若权限或证据不足，停止并一次性说明缺什么。
- 总锁前禁止以任何方式打开原始 h5ad 或读取 obs/标签值；只可用登记的 stripped cache、locked arrays、checkpoint、affinity 与 manifests。
- 不得标签路由、数据集名路由、seed search、best epoch、科学重试、fallback、删除失败、事后改阈值或重复 H02。
- 先做 P0/P1/P2；通过后 Stage T 为 240 个无训练 transforms，Stage W 为 48 次固定 pilot 训练与 48 transforms；两边全部锁 SHA 后只开一次标签窗口。
- 详细日志和大表写入文件；聊天中每阶段只用通俗中文报告关键进度，避免重复贴长日志，不能以节省 token 为由减少验证。
- Git 只能普通 push。最终 tag 必须等所有 tracked 交付完成后创建一次，禁止 force 或移动 tag。
- raw/checkpoint 留在 /root/autodl-fs；D 盘 compact 目标小于 25 MB，不下载大文件。
- 所有交付和校验完成后，保留同一 SSH 会话，把 /usr/bin/shutdown 作为最后一条远端命令；此后不重连。

如果任一权威 SHA、checkpoint、语义或标签防火墙不能成立，立即按任务书终止，不要硬着头皮蛮干。完成六个文件的读取与 SHA 校验后，请先准确回复：

“已完成 Night-7C 权威文件校验，开始 P0-AUTHORITY 与资源/防火墙审计。”
```
