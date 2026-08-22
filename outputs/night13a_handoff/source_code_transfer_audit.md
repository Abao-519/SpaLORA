# Night-13A 官方源码迁移审计

本审计读取了实际模型、训练、预处理与教程代码，不以 README 代替实现。外部源码只保存在独立 source root；没有复制进 SpaLORA 仓库或 compact。

| 方法 | 官方仓库与冻结 commit | 许可证 | 实际模型 / loss / 默认配置 | 数据特例、endpoint 与人工编辑 | Night-13A transfer decision |
|---|---|---|---|---|---|
| SpatialGlue | `JinmiaoChenLab/SpatialGlue` / `7c976d811d27ace51ce47aee0ad94a068a7d222fa` | AGPL-3.0 | 双模态空间图/特征图 GCN 与双注意；两项重构 MSE 加两项跨模态 correspondence MSE；构造器默认 600 epochs、权重 `[1,5,1,1]` | `datatype` 会切换 epochs/weights；预处理把两个空间邻接和两个特征邻接转为 dense N×N；教程常用 mclust | `BLOCKED_UNSUPPORTED`：dense N×N 与 datatype routing 均违反本轮硬边界；在 wrapper 内替换图实现或配置路由会改变正式语义，因此未伪称复现 |
| SMART | `Xubin-s-Lab/SMART-main` / `75676546a66d48a7ebfd7c5f3a2758a4c538fcfc` | GPL-3.0 | 每模态 GraphSAGE encoder、共享投影、每模态 decoder；重构 MSE + triplet loss，可选 Laplacian；API 默认 500 epochs | 官方 A1/P22 教程分别改变 epochs、K 和 farthest ratio；P22 教程含手工 cluster-to-anatomy 映射；triplet builder 先构造完整 `pairwise_distances(X)` N×N | `BLOCKED_UNSUPPORTED`：官方 triplet 路径构造 dense N×N；改用稀疏近邻会改变 triplet 语义。手工映射也被明确跳过 |
| ARISE | `XiangxiangWang-code/ARISE` / `fefdd849494c0d08e755052a7a31b20169945e40` | 未检测到 LICENSE | RNA 锚定 similarity/spatial/common-edge GCN；RNA、第二模态、融合重构与空间正则 | `process.py` 先生成完整 cosine similarity 与 dense adjacency；`train.py` 接收 true labels，每 10 epochs 计算 ARI/NMI 并按最高 ARI 保留结果；另有数据集脚本 | `BLOCKED_UNSUPPORTED`：dense N×N、训练期标签读取和结果驱动 epoch selection 三项均触碰硬边界；无明确源码许可证，不能复制 |
| simple standardized concatenation | SpaLORA clean-room `night13a_runner.py` | 项目代码 | RNA/ADT/ATAC 合法 adapter 后逐视图标准化、直接拼接、seed-0 PCA；统一 common KMeans endpoint | 只由显式 adapter 类型和输入 shape 决定，不读取 dataset/tissue 名称；无人工 cluster 编辑 | `EXECUTE` |
| SpaMV | `ericcombiolab/SpaMV` / `d7105ef70e9276350e8a12bddfbd3d396d1c33d2` | 未检测到 LICENSE | Pyro shared/private latent，cross reconstruction，shared/private 分离与 HSIC | 输入矩阵在构造期 `.toarray()`；本轮只做迁移审计 | `AUDIT_ONLY_FOR_NIGHT13B`；无许可证，不复制 |
| SpaMode | `bridge1924/SpaMode` / `d8d8e2b70c6ad47ef12aa1a5d9a65cf4fb226c00` | 未检测到 LICENSE | invariant/variant VAE 表示、PoE、noisy top-k MoE、重构/KL/双向/对抗/MoE losses | 教程级数据配置与图预处理；本轮不执行 | `AUDIT_ONLY_FOR_NIGHT13B`；无许可证，不复制 |
| SpaBalance | `nudt-bioinfo/SpaBalance` / `c3610a638c98c2d525c247ed62eeb41bda430e2d` | AGPL-3.0 | 双流表示、cross reconstruction、Barlow Twins、contrastive proxy 与梯度相似度动态平衡 | 继承 SpatialGlue/GraphST 结构，含生成式 contrastive labels（不是公开组织标签） | `AUDIT_ONLY_FOR_NIGHT13B`；不复制 AGPL 源码 |
| CANDIES | 论文给出 conditional diffusion + contrastive integration；本次在论文与已登记 reading list 中未闭合可复算官方源码仓库 | `MISSING_EVIDENCE` | 论文描述 dual-GAE、条件 DiT denoising、双图 GCN 与 attention | 未获得可冻结源码 commit，不能审计默认配置与数据分支 | `AUDIT_ONLY_BLOCKED_NO_SOURCE_COMMIT` |

## 复用资产边界

- C00/G04+H05、F00/R02、N02 与 Night-9B COSMOS 只有在现有 checkpoint、embedding、partition 和 manifest SHA 全部闭合时才记为 `LOCKED_REUSE`。
- 同名重构不算复用；缺 checkpoint 或 partition 的历史结果只进入背景说明，不进入可复算主表。
- common endpoint 是主比较；native endpoint 只作背景。本轮不执行任何看分数后的合并、拆分、重命名或人工 anatomy 映射。

## 结论

源码审计本身闭合，但三个计划执行的外部强基线都在正式代码路径触发不可通过 wrapper 消除的硬冲突：SpatialGlue 和 SMART 需要 dense N×N，ARISE 还把公开标签和 ARI 用进训练期选择。按 fail-closed 规则，这些 lane 必须保留为 blocked/unsupported，而不能改写模型、图或 loss 后再称为官方复现。因此 Night-13A 成功门中的“至少两个强外部基线跨两家族通过”在当前冻结边界下不可满足；后续结果最多可分类为 `NIGHT13A_PARTIAL_BENCHMARK_BOARD`。
