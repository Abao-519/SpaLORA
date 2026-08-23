# 复制给 Codex 2：Night-14B

你现在直接执行 SpaLORA 项目的 Night-14B，不等待用户逐项批准。这一轮是 RNA+ATAC 自研模型提分冲刺；请主动思考、灵活试错，把主要 token、算力和时间用于我们自己的 preprocessing、模型、模块、loss、优化和聚类 head。

先完整读取：

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14b_atac_score_acceleration_planning_20260823/Night14A_Worker2_Independent_Audit_and_Night14B_Decision_2026-08-23.md`
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14b_atac_score_acceleration_planning_20260823/SpaLORA_Night14B_ATAC_Score_Acceleration_and_Edge_State_Model_Taskbook_2026-08-23.md`
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14b_atac_score_acceleration_planning_20260823/night14b_atac_score_acceleration_contract.json`
4. `D:/文档/ChatGPT/博士第一篇科研论文项目/night14b_atac_score_acceleration_planning_20260823/planning_delivery_index.json`

Night-14A 权威 compact：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night14a_delivery_20260823/official_compact`

其 index 应为 77/77，SHA-256：

`031d31581413c639c6ae5dea459f62eaea1939937d244bd93bcc78696c0b2dc8`

先独立定位复算一次；不匹配立即停止报告，匹配后不要反复做整套历史审计。

Night-14A 不是实现失败，但准确结论只是 `ATAC_FOCUSED_DEVELOPMENT_SIGNAL`。P22/MISAR 的提分主要来自空间低通：MISAR 的 fixed-low 和 support-only 都高于正式 W02，P22 的 support-only 也略高；当前 TCF 冲突项还没有独立增量。停止微调旧 sigmoid，把它当可替换原型。

本轮大约 80% 精力放在自己的模型和分数上。外部方法不要先复现整张 benchmark；只快速建立公开高水位目标板，锁定数据版本、annotation、mask、K、endpoint 和原文数字。必须至少有：P22 项目 K=9（N02 native 0.5063/0.6562）、P22 论文 K=18 lane、MISAR 项目 K=7（support-only 0.3137/0.4924）、MISAR/SEPAR K=12（公开 ARI 0.644）。协议登记只是确保追的是同一道题，不是限制研发。

大胆开发一套统一 RNA+ATAC 大架构。允许 RNA/ATAC 各自 preprocessing 和 adapter，允许每个平台/数据集透明调数值超参数、K 和聚类 head；禁止的只有按 dataset name 换整套 backbone/flow。公开标签可以用于评价、HPO 和 endpoint 选择；如果进入 loss/gradient，就如实把那条 lane 标为监督或半监督。

优先从这些方向自由组合，不要求机械全部实现：

- RNA 的 HVG/PCA/可训练 encoder 与 ATAC 的 TF-IDF/LSI/peak selection/gene activity；
- 多尺度空间 filter bank、anisotropic/bilateral diffusion、GPR/Jacobi/Bernstein 图滤波；
- EdgeConv/PointNet++ 式点云几何、ACM-GCN/MixHop 式 identity/aggregation/diversification；
- content-gated mixture-of-experts、cross-attention、shared/private representation；
- masked modality modeling、self/cross reconstruction、VICReg/Barlow Twins、graph contrastive、prototype consistency；
- KMeans、mclust、Leiden、GMM、DEC/IDEC、SwAV/DeepCluster 等聚类 head；
- 工作代号 TSPR 的三态边传播：一致内部边传播，一致边界边保留，模态冲突边可弃权。具体公式可重写；如果别的机制更有效，也可以替换。

先在现有 checkpoint/embedding 上做 40–100 个低成本配置，再选约 8–20 个完整训练配置；这是预算指南，可按早期结果调整。初筛只需一个真实 training seed，不要让每个失败候选都跑多 seed。允许单个最好 seed 达到目标时记为 `SCORE_BREAKTHROUGH`；所有失败和其他 run 仍保留。只有领先候选才补 2–3 seeds，现在来不及可留到 Night-15。不要要求所有 seed、所有数据集、所有指标同时最好。

追分顺序：

1. P22 K=9 先超过 0.5063/0.6562，再冲 ARI 0.55；
2. MISAR 先闭合 K=12 标准 lane，分阶段冲 ARI 0.50、0.60、0.644；
3. MISAR K=7 至少超过 0.3137/0.4924；
4. P22 K=18 在同一 annotation 下追论文高水位。

领先候选才做必要消融：identity、matched-strength fixed-low、support-only、旧 W02、新模块；并把相同后处理施加给强参考 embedding。若分数来自 preprocessing/backbone/cluster head 而不是 TSPR，也不要强行保 TCF 故事，直接登记 `BACKBONE_OR_HEAD_SIGNAL` 并把真正有效的部分转为主线。

P22/MISAR 达到明显信号后，才扩展 MISAR 其他阶段、P21/ME13、P22 其他 epigenome 切片或 P5S1/S2/S3。可下载权威数据和源码到新根，记录 provenance；不要修改历史 raw。

你有充分工程自由：可修依赖、重构、重新训练、换 optimizer/loss、用 Optuna/分层搜索；不设一次 correction 上限。8 GB GPU 内经过内存预检后可以使用 dense 小模块或分块近似，不要因旧的 sparse-only 边界自动放弃好思路。

只保留必要底线：不伪造/隐藏失败 run；最佳值必须能对应真实配置、checkpoint、seed；不按数据集名称换整套模型；不同 K/annotation 的数字不伪装成直接胜负；不改历史 raw/tag、不 force push，第三方代码保留许可证。

结果终态按任务书选择：`SCORE_BREAKTHROUGH`、`MULTI_DATASET_SCORE_SIGNAL`、`MODULE_SIGNAL`、`BACKBONE_OR_HEAD_SIGNAL`、`NO_SCORE_SIGNAL`、`IMPLEMENTATION_FAILURE` 或 `INFRASTRUCTURE_FAILURE`。单最佳 seed 达标是开发突破，不称 SOTA、confirmed milestone 或 paper-ready。

最终报告先写“我现在需要知道的三件事”，给绝对 ARI/NMI 主表，分列 BEST_RUN、median/mean 和目标差距，解释提分究竟来自哪一层，再给导师汇报版，最后放 Git/hash/compact 技术附录。完成新 branch、普通 push、annotated tag、compact 和 Windows 独立复算后，再把 `/usr/bin/shutdown` 作为最后一条远端命令派发。

