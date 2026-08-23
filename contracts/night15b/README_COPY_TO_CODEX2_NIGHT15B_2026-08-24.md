# 复制给 Codex 2：Night-15B

请接续 SpaLORA 项目，执行 Night-15B“稳定性锚定原型与分数冲刺”。本轮以自身绝对分数和可形成统一故事的新模块为主，不复现外部整张 benchmark。

先完整读取并复核：

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_delivery_20260823/official_compact/outputs/night15a_handoff/night15a_report.md`
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_delivery_20260823/official_compact/outputs/night15a_handoff/night15a_decision.json`
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_delivery_20260823/official_compact/outputs/night15a_handoff/main_results_table.csv`
4. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_delivery_20260823/official_compact/outputs/night15a_handoff/all_run_ledger.csv`
5. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_delivery_20260823/official_compact/SpaLORA/night15a_mcdf.py`
6. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15b_stability_anchored_prototype_score_sprint_planning_20260824/Night15A_Worker2_Independent_Audit_and_Night15B_Decision_2026-08-24.md`
7. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15b_stability_anchored_prototype_score_sprint_planning_20260824/SpaLORA_Night15B_Stability_Anchored_Prototype_Score_Sprint_Taskbook_2026-08-24.md`
8. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15b_stability_anchored_prototype_score_sprint_planning_20260824/night15b_stability_anchored_prototype_score_contract.json`

Night-15A 应闭合为：compact 64/64，index SHA-256 `ebd87d5bc4bcb18c52e8e6a48bde519e4b897ba758751265837f33c6f012299f`，final commit `ba8f775739ab63475c6a5318e1f9ccce3e089ae2`，tag `night15a-final-20260823`。若不一致，停止并报告。

Worker 2 的独立结论是：MCDF 已科学失败，不再继续修补。Night-14B/Night-15A 的有效资产是强 molecular representation、P22/MISAR 高分 chain、P22 K=18 强 ATAC latent，以及 label-free partition medoid。当前主要缺口是 learned representation 的 loss 与最终 cluster separability 错位。

本轮分两条互相衔接的工作流。

第一条是绝对分数冲刺。复用所有已有强 embedding，对 P22 K=9/K=18、MISAR K=7/K=12、A1、D1、tonsil s1/s2/s3，以及标签协议能闭合时的 GSE213264，做表示混合、PCA/whitening、坐标 basis/weight、稀疏滤波、KMeans/GMM/可用的其他 head、spatial refinement、partition medoid/consensus 的 coarse-to-fine HPO。公开标签可以用于 known K、开发 HPO、候选排序、BEST_RUN 选择和评价；所有运行行保留。允许每个数据集不同数值参数，不要求所有 seed 或所有数据集同时赢。优先目标：P22 K=9 ARI 过 0.60；P22 K=18 过 0.65；MISAR K=7 过 0.55；MISAR K=12 先过 0.50，再向 SEPAR 的 0.644 背景线靠近。其他 protein 数据尽量取得多数提升。

第二条是一个小型统一 trainable core。工作名 SAPR，即“稳定性锚定的原型残差”：从多个高分、结构不同的 label-free partitions 中对齐 cluster ID，找出稳定内部点和不稳定空间边界；用内部点初始化 prototypes，只让一个小型残差网络重点修正边界，同时用 trust region 保护稳定内部。可候选使用 confidence-weighted prototype sharpening、balanced/Sinkhorn assignment、cross-view consistency、boundary graph consistency 和 residual norm constraint。不要机械全上，选最简有效组合。RNA+ATAC 与 RNA+protein 允许不同 modality adapter 和平台级参数，但 residual/prototype 主体相同，代码不能按数据集名称切换整套模型。

SAPR 是工作假设，不是死命令。请先读 PRAGA、Proust、SpaMV、soFusion、SEPAR、3d-OT 的论文与真实源码，避免把已有的动态图、普通 prototype contrastive、shared/private MoE 或模态专用 decoder 重新命名成创新。若你发现更有希望且仍为统一核心的结构，可以直接替换 SAPR，并留下理由。重点是：新模块必须在同一 strongest representation/head 上产生独立分数增益。

实验漏斗要灵活高效：先 1 个 training seed、大胆筛 10–20 个有针对性的候选；只把 top 2–3 扩展到另外两个 training seeds。没有信号时不要先跑 30 个模型。工程错误可修复并重跑，不设一次 correction 上限；科学公式改变时登记 revision。只对每个模态家族做 1 个真实完整 P0，只对 finalist 做 checkpoint fresh-process 回放。最小贡献对照只有三行：retained teacher/strong embedding、SAPR full、SAPR residual disabled。

请把计算严格拆开以减少 AutoDL 时间。AutoDL 只承担 raw fragments/peaks 的重型预处理、raw-feature 神经训练、GPU screen 和 finalist checkpoint。重型预处理完成后导出 `local_compute_kit`，每个数据集包含 ordered IDs、coordinates、labels、mask、K、reduced modality views、历史强 embeddings、sparse graph CSR、candidate partitions 和配置注册表；压缩目标不超过 2 GiB，不包含 raw fragments 或 dense N×N。提供 Windows runner，使 AutoDL 关闭后可在本地完成 embedding-level head HPO、medoid/consensus、metrics、表图、ledger、compact/hash 和 CPU endpoint 回放。本机为 Ryzen 7 6800H、15.19 GiB RAM、无 CUDA、D 盘约 94.77 GiB 可用，因此 runner 要支持 memmap、按数据集串行和缓存复用。

外部工作只做“论文报告分数+协议可比性”背景板，不跑完整外部方法。对每个数字登记 K、annotation、mask、observation 和来源；不同协议只作 context。绝大多数时间投入自身分数与新核心。

你拥有实现、公式、实验顺序、参数范围和停止点的自由度。不要因形式审计拖慢研发，也不要把失败 seed 删除。BEST 是合法开发结果，同时报告 median/mean 以便判断稳定性。标签不直接作为模型输入或 prototype target；否则就变成监督分类，不能再与无监督空间域 SOTA 并列。

结果分类：仅 head 提分为 `HEAD_ONLY_SCORE_GAIN`；新残差只在部分数据有效为 `CLUSTER_AWARE_RESIDUAL_LOCAL_SIGNAL`；同一核心跨 RNA+ATAC 与 RNA+protein 多数据产生独立增益为 `CROSS_DATASET_METHOD_SIGNAL`；都没超过旧高位为 `NO_ADDED_SCORE_SIGNAL`；真实路径或资源问题另归实现/设施失败。

报告先写“我现在需要知道的三件事”，再给绝对指标主表和 5–8 句导师汇报版。主表含数据集、K、历史参考、BEST、绝对 ARI/NMI、Δ、AMI/FMI、Moran/Geary、training/endpoint seed、BEST/median/mean、wall/GPU/RAM。另列本地 compute kit、AutoDL GPU 时间、迁移到本地的 CPU 时间、三行最小贡献对照和完整失败 ledger。技术哈希放附录。

最终普通 push branch/tag，生成 root-relative compact index，在 Windows 独立复算。GPU 资产与 local compute kit 已下载并验证后，最后一条远端命令为 `/usr/bin/shutdown`；派发后不重连。
