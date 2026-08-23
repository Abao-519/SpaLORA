# Night-15A 独立复核与 Night-15B 决策

日期：2026-08-24  
负责人：Worker 2

## 我现在需要知道的三件事

1. Night-15A 要回答的是：Night-14B 的高分究竟来自新可训练融合、分子表示、坐标，还是聚类 head。独立复核确认答案不是“新融合成功”，而是“强表示与聚类 head 仍有价值，新 MCDF 没有新增价值”。
2. 我已在 Windows 对 official compact、64 个 payload 的 size/SHA、bundle、commit/tag、主表、运行 ledger 和真实源码逐项复核。结果与正式报告一致；MCDF（受模态贡献约束的扩散融合）确实训练了，但它优化重构与跨模态一致性，没有直接优化最终空间域的可分性。
3. 下一步不再修补 MCDF。保留 P22/MISAR 的高分链、P22 K=18 的强 ATAC 表示和无标签 partition medoid（多个分区中选最居中的稳定分区），转向“强表示之上的小型聚类感知残差”；同时把 CPU 聚类搜索和审计迁到本机，AutoDL 只做不可替代的 GPU 工作。

## 明确分类

- Night-15A 主终态：`NO_ADDED_METHOD_SIGNAL`，属于 `SCIENTIFIC_NEGATIVE`。
- 次级可复用结果：`STABLE_HEAD_SIGNAL`，属于工程/局部信号，不是论文主方法。
- Night-14B 的历史开发峰值仍真实存在：P22 K=9 ARI 约 0.5691，MISAR K=7 ARI 约 0.5143；它们不能归因于 MCDF。

## 独立文件复核

- compact index SHA-256：`ebd87d5bc4bcb18c52e8e6a48bde519e4b897ba758751265837f33c6f012299f`，与交付声明一致。
- payload：64/64；missing、size mismatch、SHA mismatch、extras 均为 0。
- bundle SHA-256：`2fd51dd2bc586d2978e9b957f369e836969abc0dae82ebb7266c518bfc7e8e06`，与交付声明一致。
- final commit：`ba8f775739ab63475c6a5318e1f9ccce3e089ae2`；final tag：`night15a-final-20260823`；tag peel 与报告一致。
- formal unseen training 30/30、formal endpoint 45/45、active checkpoint replay 62/62；一条 superseded 早期失败被保留。

## 真正的科学原因

Night-15A 源码中的 MCDF 使用同一图自编码器和 geometry、RNA、ATAC、joint 四类专家，训练目标主要是自重构、跨模态重构、方差/协方差和 gate 约束。它没有让最终聚类边界或 prototype（每个候选空间域的表示中心）进入训练目标。因此模型可以学到“能重建输入的表示”，却不一定学到“更容易被 KMeans/GMM 分开的表示”。

这与结果完全吻合：

| 数据与协议 | 最强单模态+坐标 | 融合均值 | 融合优势 | 冻结新模型结论 |
|---|---:|---:|---:|---|
| P22 K=9 | ARI 约 0.482 | 约 0.496 | 约 +0.014 | R21 MCDF unseen best 0.4939，未保住 0.5691 峰值 |
| MISAR K=7 | ARI 约 0.419 | 约 0.404 | 负 | 最优冻结配置主要依赖单模态 latent |
| MISAR K=12 | ARI 约 0.279 | 约 0.274 | 负 | unseen best 0.3737，仍远低于 0.50 第一目标 |

所以问题不再是“门控权重调得不够好”，而是训练目标与最终聚类目标错位。继续增加扩散专家、重构 loss 或门控复杂度，成功概率低且故事会与 PRAGA、Proust、SpaMV、soFusion 等现有路线重叠。

## 文献与源码后形成的方向约束

- PRAGA 已经使用动态自适应图和原型对比学习，不能把“再加一个 prototype loss”直接当成原创核心：https://ojs.aaai.org/index.php/AAAI/article/view/32010
- Proust 已经用图自编码器与对比自监督做空间多组学域识别：https://genome.cshlp.org/content/35/7/1621
- SpaMV 已经覆盖 shared/private 表示、Mixture-of-Experts 和跨模态重构：https://www.nature.com/articles/s41467-026-74718-1
- soFusion 已经用统一 GCN、模态内/模态间学习和模态专用 decoder 覆盖 RNA、ATAC、protein：https://github.com/sunxue-yy/soFusion

因此 Night-15B 不把“图、对比、原型、混合专家”任何一个常见零件单独声称为创新。工作假设是：**不替换强表示，只识别多个高分聚类之间稳定的内部点与不稳定的边界点，并让一个受限残差只修正不稳定边界。**这是一项较小、可实现、可验证的增量创新；是否最终足够投稿，取决于它能否在多个真实数据集产生独立分数增益和可解释边界变化。

## Night-15B 工作假设：SAPR

SAPR 是暂定工作名，展开为 `Stability-Anchored Prototype Residual`，中文为“稳定性锚定的原型残差”。造这个词只是为了指代要实现的真实模块，不预先宣称它是最终论文名或已经原创。

它对应四个真实对象：

1. 用同一强表示运行多种合理 head，得到多个候选分区；
2. 对齐 cluster ID 后，找出多分区一致的高置信内部点和不一致的边界点；
3. 用内部点初始化 prototype，并训练同一小型残差网络；
4. 对内部点设置 trust region（不允许大幅漂移），主要让跨模态证据修正边界点。

它与 PRAGA 的预期区别是：PRAGA 从动态图和 prototype 联合学习表示；SAPR 从已经验证较强的固定表示出发，以分区稳定性定义锚点，只学习有界的边界残差，目的首先是保住峰值、降低坍塌，再争取额外增益。这个差异必须在源码和更广文献中继续核对，当前不把“未撞车”写成定论。

## 速度决策：本地与 AutoDL 分工

本机实测为 AMD Ryzen 7 6800H（8 核/16 线程）、15.19 GiB RAM、AMD 集显、无 NVIDIA/CUDA，D 盘约 94.77 GiB 可用。

### 留在 AutoDL

- 数亿 fragment 的原始预处理；
- raw RNA/ATAC/ADT 的神经网络 forward/backward；
- GPU candidate screen 和 finalist 训练；
- 最终少量 checkpoint 导出。

### 移到本地

- compact/index/SHA/Git bundle 复核；
- 指标复算、报表、图和运行 ledger 汇总；
- reduced embedding 上的 PCA、KMeans、GMM、坐标权重、稀疏 refinement、partition medoid/consensus 搜索；
- 文献/源码审查和配置生成；
- final candidates 的 CPU endpoint 回放。

Night-15B 必须导出 `local_compute_kit`：每个数据集的有序 IDs、坐标、公开标签与 mask、K、压缩后的多视图 embedding、稀疏图 CSR、历史强表示和配置注册表。目标压缩体积不超过 2 GiB，不包含数亿 raw fragments。这样 AutoDL 关机后，Codex 2 仍可在 Windows 完成大部分 head 搜索和审核。

## 执行决策

1. 保留 Night-14B/Night-15A 全部有效表示和高分 head，不从头重建大模型。
2. 先做大范围、低成本的 head/表示混合搜索，明确各数据集可达到的真实上限；公开标签允许用于开发 HPO 和候选排序，所有运行保留。
3. 用最稳的多个高分分区产生无标签 teacher，再训练 SAPR；ground-truth 标签不直接成为模型输入或 prototype target，否则就变成监督分类，不能与无监督空间域方法比较。
4. 同一 SAPR 核心服务 RNA+ATAC 与 RNA+protein；允许模态类型适配器、平台级数值超参数和每个数据集不同的 HPO 最优值，但代码不按数据集名称切换模型结构。
5. 先用一个 training seed 大胆筛选；只把最有希望的 2–3 个候选扩展到额外 seeds。单次峰值是合法的开发胜利，同时保留 median/mean 以判断后续写作风险。
6. 外部方法本轮只建立“论文声称的分数与协议背景板”，不运行整套复现；主要资源投入自身模型和分数。
7. 最小消融只保留三行：强 teacher head、SAPR full、SAPR residual disabled。它不是形式负担，而是防止再次把无效模块写成提分来源。

## 导师汇报版

Night-15A 独立复核确认，MCDF 没有带来新的方法增益，但 Night-14B 的 P22 与 MISAR 高分链仍是可用资产。失败的根因不是代码没训练，而是重构式训练目标与最终聚类可分性不一致。近期方法已经大量覆盖动态图、对比学习、共享/私有表示和混合专家，因此继续堆同类模块既不易提分，也不易形成新故事。下一步将保留强表示，只对多种聚类结果不稳定的空间边界学习小型残差。该设计以统一模型服务 RNA+ATAC 和 RNA+protein，允许不同平台数值调参，但不按数据集名称切换整套流程。工程上将把轻量聚类搜索和审计迁到本机，AutoDL 只承担 GPU 训练和重型预处理。Night-15B 首先追求 P22、MISAR 和 protein 数据中的绝对分数突破，再根据跨数据集独立增益判断能否发展为论文核心。
