# SpaLORA Night-14A：成熟 Backbone 与拓扑冲突滤波性能冲刺任务书

日期：2026-08-23  
任务性质：方法研发与性能开发；允许试错；不要求本轮直接形成论文终稿。  
工作代号：TCF（Topology-Conflict Filter，拓扑冲突滤波器；不是最终论文名）。

## 1. 唯一核心问题

在不按数据集名称切换模型的前提下，能否把跨模态局部一致/冲突转化为边级或节点级图频率响应，使同一成熟空间多组学 backbone 在利用 P22/MISAR 空间连续性的同时，减少 A1/tonsil 的过平滑与负迁移？

## 2. 执行风格

这是“目标与预算型授权”，不是逐行操作清单。执行 Codex 有权：

- 审计论文、Methods、Supplement 和官方源码后改变具体实现细节；
- 在 SMART、SpaBalance 或可追溯 clean-room backbone 中选择一条主骨架；
- 调整 latent dimension、graph k、学习率、训练轮数、loss weights 和 normalization；
- 使用公开标签做开发榜与超参数选择，只要完整登记；
- 修复工程问题、环境问题和 API 兼容问题，不受“一次 correction cycle”限制；
- 在证据显示当前 TCF 公式不合理时重构它，但必须保留失败记录并说明为什么改。

执行 Codex 不必为了审计而重复无意义的审计；把时间优先用于真实模型、真实性能和关键消融。

## 3. 必须先读的权威材料

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night13c_delivery_20260823/official_compact/compact_delivery_index.json`
2. `.../outputs/night13c_handoff/night13c_report.md`
3. `.../outputs/night13c_handoff/night13c_decision.json`
4. `.../outputs/night13c_handoff/b10_endpoint_robustness_summary.csv`
5. `.../source/SpaLORA/night13c_core.py`
6. `.../source/scripts/night13c/night13c_stage_a.py`
7. `.../source/scripts/night13c/night13c_stage_b.py`
8. 本目录的 `Night13C_Worker2_Independent_Audit_and_Night14A_Decision_2026-08-23.md`
9. 本目录的 `night14a_topology_conflict_sprint_contract.json`
10. 项目治理与阅读清单：
   - `research_governance_20260821/PROJECT_MEMORY_PLUGIN_RESEARCH_WORKFLOW_2026-08-21.md`
   - `research_governance_20260821/Spatial_MultiOmics_Reading_List_2025_2026.md`

先独立复算 Night-13C compact 的 59/59 和 index SHA。若不匹配，停止并报告；若匹配，后续不再重复整套清单审计。

## 4. Phase A：纠正评价语义（短、一次完成）

1. 确定性 embedding 只登记一份，不能把未消费的 model seed 复制成独立重复。
2. Candidate 与 reference embedding 使用同一批 KMeans endpoint seeds（建议 0–19 或 0–29）、同一 `n_init` 和同一 mask。
3. 同时保留两条语义：
   - `COMMON_HEAD_ROBUSTNESS`：完全相同聚类端点下比较表示；
   - `NATIVE_FULL_PIPELINE_CONTEXT`：登记历史完整管线高水位，不与 common head 混为一个胜负。
4. 从同一冻结 checkpoint 离线评价 residual grid，不为每个 residual 重新训练。
5. 按研究单位汇总：lymph-node、tonsil、P22、MISAR；tonsil 三切片不能当三项独立研究票。

这一步只修口径，不应吞掉整轮时间。

## 5. Phase B：成熟统一 backbone 选择

只审计和真实运行最多两条候选骨架，优先 SMART 与 SpaBalance。目的不是复现整张外部 benchmark 表，而是为我们自己的方法选一个可信起点。

对每条骨架至少完成：

- 固定官方 commit、许可证、论文版本、核心数据流和 loss 语义；
- A1 与 P22 的真实 preprocessing → training/forward → checkpoint reload → fused embedding → clustering endpoint；
- 报告原始和处理后 tensor shape；
- 训练 loss 轨迹、early stopping/convergence、实际参数更新和 peak resource；
- 在完全相同 common endpoint 下给出 A1/P22 绝对 ARI/NMI；若成本允许加入 tonsil s1 与 MISAR。

选择依据为真实性能、代码可维护性、8 GB GPU 可行性和 TCF 插入点清晰度。若官方代码含 dataset-specific 配置，这不自动判失败；记录并区分“数值配置”与“架构分支”。

## 6. Phase C：TCF 方法开发

### 6.1 必须保持的本质

TCF 必须是一套共享结构，至少显式拥有以下两类通道：

- identity/保真通道：不传播，保护局部边界和原表示；
- low-pass/聚合通道：只沿被两个模态共同支持的空间边传播。

若证据支持，可加入 high-pass/diversification 通道。通道或 residual 权重必须由下列真实内容的一部分产生：RNA 局部关系、第二模态局部关系、跨模态边一致、跨模态边冲突、空间距离、局部重构误差或扰动稳定性。不得由数据集名称直接产生。

### 6.2 可自由探索的实现

执行 Codex 可在下列形式中选择、组合或改写：

- 解析式 agreement/conflict edge weights；
- 小型可训练 edge/node gate，直接由 backbone 的无标签目标端到端训练；
- identity/low/high 三通道 softmax mixer；
- 以 backbone 为锚的 safe residual；
- 多空间尺度但保持稀疏的 filter bank；
- modality dropout、edge perturbation 或 reconstruction 作为辅助信号；
- 稳健 prototype/Leiden endpoint 仅作为 robustness facility，不作为主创新。

不要再用同一 embedding 的平均 cosine 作为唯一 edge regression target。若训练 gate，必须有能够区分“传播有益/有害”的更独立信号，或让 gate 从端到端任务中学习并做 collapse 诊断。

### 6.3 开发预算

- 先跑冻结 embedding 上的低成本解析式/filter-bank 扫描，再进入完整训练。
- 建议总计 30–60 个低成本 filter/config 单元、6–12 个完整训练配置；这是预算指南，不是硬上限，执行 Codex可根据饱和情况调配。
- Top 配置再做 3 个真实 training seeds；普通探索不必每个都多 seed。
- 若早期发现明显更好且文献碰撞更低的邻近机制，可在同一任务内替换具体 TCF 公式，但核心问题仍是拓扑冲突与安全传播。

## 7. 数据与调参策略

### 开发板

- A1 lymph node
- tonsil slice 1
- P22 mouse brain
- MISAR E15.5

公开 annotation 可以用于开发评价、候选排序和逐数据集数值超参数选择。所有 HPO 尝试、失败和选择依据必须写入 ledger。若模型训练本身不读标签，论文中仍可称“训练无监督、benchmark 开发有标签调参”；不得把两者混写为盲测。

### 确认板

- D1 lymph node
- tonsil slice 2
- tonsil slice 3

当 architecture 和默认搜索空间稳定后再运行确认板。若确认数据此前已多次查看，称为“冻结后内部确认”，不要声称 pristine blind test。

### 无标签扩展（可选）

P5S1/S2/S3 用于大规模真实 RNA+ATAC 路径、资源扩展、扰动稳定性与跨重复一致性；只有主候选已经出现时才投入，不让它阻塞性能开发。

## 8. 评价与晋级

主表必须包含绝对 ARI、NMI、AMI、FMI、Moran、Geary、endpoint seeds 分布、胜出数、运行时间、峰值显存和峰值内存。

候选排序采用以下原则，而不是“七个数据集全部双升”的硬门：

1. 研究单位平衡后的 ARI/NMI 平均变化与平均排名；
2. 相对 matched strong reference 的 common-head 差值；
3. endpoint seeds 稳定性和 paired win rate；
4. 最坏数据集的回撤；
5. 机制指标：可靠边/冲突边分布、边界保护、corruption 时是否退回 identity；
6. 资源成本。

本轮可以形成以下不同结果，不强迫只有一个全胜门：

- `BROAD_SIGNAL`：两模态家族总体均有收益，至少 3/4 开发研究单位方向良好；
- `ATAC_FOCUSED_SIGNAL`：P22 与 MISAR 对强 matched reference 稳健提升，protein 不出现灾难回撤；后续论文范围收窄为 RNA+ATAC；
- `LOCAL_SIGNAL`：只有部分数据/endpoint 有可复现收益；
- `BACKBONE_ONLY_SIGNAL`：成熟 backbone 提升，但 TCF 没有新增价值；
- `SCIENTIFIC_NEGATIVE`：实现正确但主机制没有有效增益；
- `IMPLEMENTATION_FAILURE`：真实路径、训练或评价语义未成立。

不要因单个指标小幅下降就停止，也不要因单个数据集高分就宣布论文成立。

## 9. 最小完整性边界

仅保留真正必要的边界：

- 不伪造、删除或隐藏失败 run/seed；
- 不修改历史 raw；新缓存放到新根；
- 不按数据集名称切换 backbone、公式或整条 flow；
- 若标签进入 loss、gradient 或 within-run checkpoint selection，必须明确把相应路线重分类为监督/半监督，不得仍称无监督；
- 不把 endpoint 不同的数值当公平胜负；
- 不 force push，不覆盖历史 tag；
- 第三方源码遵守许可证并保留 attribution；
- 禁止意外构造无法承受的 dense `N×N`，除非先估算内存且只用于小样本诊断。

工程修复、依赖兼容、合理重构、重新启动失败的工程单元均允许；记录即可，不设置人为的一次修复上限。

## 10. 交付

交付目录建议：`outputs/night14a_handoff/`，至少包括：

- `night14a_report.md`
- `night14a_plain_summary.md`
- `night14a_decision.json`
- `source_and_collision_audit.md`
- `backbone_selection_board.csv`
- `development_leaderboard.csv`
- `absolute_metrics.csv`
- `endpoint_robustness.csv`
- `hpo_and_failure_ledger.csv`
- `mechanism_diagnostics.csv`
- `resource_audit.json`
- 关键源码、配置、测试与 fresh-process 证据

最终报告先写“我现在需要知道的三件事”和结果分类，再写技术附录。Git 使用新 branch、普通 push、新 annotated tag；compact 用根相对 size+SHA-256 index，并在 Windows 独立复算。

## 11. 导师汇报需要回答的六句话

1. 固定空间图为什么会在 P22/MISAR 有益、在 lymph/tonsil 有害？
2. TCF 比普通 graph attention、dynamic graph 和 B10 多解决了什么？
3. 选用了哪个成熟 backbone，为什么？
4. 最重要的绝对 ARI/NMI 和强参考差值是什么？
5. 哪些数据集失败，失败幅度与机制解释是什么？
6. 现在证据属于局部信号、ATAC 主线、跨家族信号，还是仍需停止？

