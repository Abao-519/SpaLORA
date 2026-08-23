# 复制给 Codex 2：Night-15A

请接续 SpaLORA 项目并执行 Night-15A。先完整读取：

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14b_delivery_20260823/official_compact/outputs/night14b_handoff/night14b_report.md`
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14b_delivery_20260823/official_compact/outputs/night14b_handoff/night14b_decision.json`
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14b_delivery_20260823/official_compact/outputs/night14b_handoff/all_run_ledger.csv`
4. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_multimodal_contribution_and_score_stability_planning_20260823/Night14B_Worker2_Independent_Audit_and_Night15A_Decision_2026-08-23.md`
5. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_multimodal_contribution_and_score_stability_planning_20260823/SpaLORA_Night15A_Multimodal_Contribution_and_Score_Stability_Taskbook_2026-08-23.md`
6. `D:/文档/ChatGPT/博士第一篇科研论文项目/night15a_multimodal_contribution_and_score_stability_planning_20260823/night15a_multimodal_contribution_contract.json`

Night-14B 的 45/45 compact、index SHA `dd985418f48e556164d38e034aabbbc699e65074df4e7a9661d01c00cf35f003` 和 final commit `026a15bf8b44a816ffe8f0699bd63c4a83b83f9a` 应先复核。

本轮的第一要务不是继续包装 TSPR，也不是复现外部整张 benchmark，而是回答：P22 与 MISAR 的高分究竟来自空间坐标、分子多模态，还是聚类 head。请先在相同 K/mask/seed/head 下完成 coordinate-only、RNA-only、ATAC-only、fused、无坐标、无滤波、无 refinement 与 Night-14B best chain 对照。公开专家标签可以用于固定 K、开发 HPO、候选排序和评价；请标明 known-K unsupervised clustering，标签不要进入模型输入或无监督 loss。

P22 主协议先用现有 K=9 groundtruth；同时从 3d-OT 官方 Zenodo/教程闭合 K=18 的 exact h5ad、annotation、mask 与 IDs，不能把 K=9 标签拆成 K=18。MISAR 同时保留 K=7 与 K=12，优先核对 SEPAR exact artifact。再核查 GSE213264 Spatial-CITE-seq 与现有 tonsil 数据是否重复；A1、D1、tonsil s1/s2/s3 都保留为统一模型旁路验证，不要求本轮全部提升。SEPAR 的 DLPFC、Stereo-seq、osmFISH、MERFISH、CRC 数据只在低成本直接兼容时作为辅助，不要挤占主模型研发。

独立复核显示 Night-14B 缺少 coordinate-only control，而且 P22/MISAR 的坐标增强贡献很大。若 fused 不能超过 coordinate-only 和最强单模态，就如实登记 `GEOMETRY_ONLY_SIGNAL`，但继续 raw-feature backbone 研发。若多模态贡献成立，研发 MCDF（受模态贡献约束的多尺度扩散融合）：geometry、RNA-guided、ATAC-guided、joint bilateral 稀疏专家，多尺度内容门，modality-dropout/cross-view/masked reconstruction，外加防止坐标支配的约束。这个结构是工作假设，不是死命令；你可依据源码与真实数值改成更有希望、可独立消融的统一 RNA+ATAC 模块。

重点投入 raw-feature RNA/ATAC，尤其追 MISAR K=12。RNA 可试 HVG/PCA 或轻量 masked encoder；ATAC 可试 TF-IDF/LSI、可追溯 gene activity 或稀疏 peak encoder；允许平台级数值调参。单 seed 达标就登记阶段性胜利，不要求每个 seed 都赢；但所有运行行都保留。把最有希望的 3–5 个候选冻结后，再用未参与 HPO 的 backbone seeds 3–7 诊断稳定性。BEST、median、mean、min 分开报告。

你拥有实现与实验顺序的自由度。工程错误可以修复并重跑受影响单元，不设置机械的一次 correction 限额；若改变科学公式或数据协议，登记 revision boundary 即可。不要按数据集名称切换整套模型；允许公开平台/模态统计量驱动参数和平台级超参数。不要把 consensus、bilateral、PCA 或 GMM 单独包装成原创贡献，新模块必须在 strongest preprocessing/head 上做独立消融。

目标：P22 K=9 保住 BEST ARI≥0.55，并争取 mean/median 超过 0.5063/0.6562；MISAR K=7 保住 BEST≥0.50、优先把 median ARI 提到≥0.45；MISAR K=12 先过 0.50 再冲 0.644 背景线；P22 K=18 先形成 exact-protocol 基线。

报告必须先写“我现在需要知道的三件事”，再给主表和导师汇报版。主表包含数据集、K、坐标/单模态/融合对照、绝对 ARI/NMI、AMI/FMI、Moran/Geary、BEST/median/mean/min、seed、wall/GPU/RAM。保留失败 ledger、checkpoint/reload、commit/tag、root-relative compact index 和 Windows 独立 SHA 复算。最后一条远端命令为 `/usr/bin/shutdown`。
