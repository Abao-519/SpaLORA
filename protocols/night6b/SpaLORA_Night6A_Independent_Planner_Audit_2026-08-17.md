# SpaLORA Night-6A 独立规划审计

日期：2026-08-17  
审计性质：只读、独立复核；不重新连接 AutoDL，不修改 Night-6A 结果，不把诊断结果升级为科研结论

## 1. 权威结论

Night-6A 的唯一权威终态是：

`IMPLEMENTATION_SEMANTICS_INVALID`

这一判定正确，原因也足够明确：正式训练锁定前，`scripts/p0_data_numeric.py` 对原始 tonsil RNA/ADT 文件调用了 `anndata.read_h5ad`，从而把 `obs` 中的值反序列化进进程内存。即使这些值没有被索引、打印、导出、传入训练、用于候选选择或 checkpoint 选择，也已经违反了 Night-6A 自己预注册的“锁定前不得读取 annotation 值”的严格防火墙。

因此：

- 45/45 训练、270/270 raw artifact SHA、8/8 语义测试均可证明执行和保存完整；
- 它们不能把本轮恢复成一个有效的科学实验；
- `NO_STRUCTURAL_RESCUE_CANDIDATE` 只能作为失败尝试的诊断描述，不能作为 Night-6A 的正式科学结论；
- 不能把 Night-6A checkpoint、embedding 或候选胜负直接带入后续确认性实验。

这不是“模型偷偷用了标签”。它是实验隔离协议的实现语义失败。两者必须同时说清楚。

## 2. 本地交付完整性复核

复核根目录：

`D:\文档\ChatGPT\博士第一篇科研论文项目\night6a_handoff_20260814\official_compact`

独立按 `handoff/delivery_index.json` 所在目录解析相对路径，逐文件重算字节数和 SHA-256：

- 申报文件：16
- 实际检查：16
- 缺失：0
- 字节数不符：0
- SHA-256 不符：0

关键文件：

| 文件 | SHA-256 |
|---|---|
| `handoff/night6a_report.md` | `8cba120b670d2778353d7e314b7f2c9a7c3c215ef690d5ecfbc247700dceb26e` |
| `handoff/p0_data_and_label_firewall_audit.json` | `425c980122906e714efde13f7586e5d44b23fddad335379f737cef336e0ad515` |
| `handoff/delivery_index.json` | `27ae79af08fc9a0d9b030b3ea27cc752c78beec80f31f75422b3c199b79f752b` |
| `night6a_planner_handoff_20260814.tar.gz` | `e0743e9e22a11f90bec3310175eadafaf0305aed9c755543bc296cb5cc4b8531` |
| `night6a_incremental_20260814.bundle` | `d7e28c8631ec92921e7486d06632f02d9bf4a5d225c11fbe26c1f2759a20a41d` |
| `shutdown_dispatch_status.json` | `0110aa40ffee265b1e7efbe27537d4f4c393cb4573953513efdae0ffea01da07` |

Git bundle 已在 D 盘只读恢复出提交链；`night6a-final-20260814` 指向 `7f204a56690768f22bd06e0dac1b5785c97c4c70`。

## 3. 交付内部字段冲突

下列较早生成的字段不能继续被当作权威事实：

- `p0_data_numeric_audit.json: semantic_label_values_read=false`
- `p0_semantic_contract.json: label_values_read=false`
- 缓存 metadata 中继承的 `semantic_label_values_read=false`

它们与随后完成的权威防火墙审计冲突。正确事实是：

1. 原始 `obs` 值被 `anndata.read_h5ad` 反序列化进内存；
2. 值没有被显式索引、查看、打印、导出或用于训练/选择；
3. 后续正式训练使用的是零 `obs` 列的 label-free copies/cache；
4. 但预锁定防火墙仍已不可逆失败。

后续脚本必须使用三态或更精确字段，例如：

- `deserialized_into_memory`
- `explicitly_indexed_or_observed`
- `used_for_training_or_selection`

不得再用一个含义不清的 `label_values_read` 布尔值掩盖差异。

## 4. 第二项高优先级设计问题：tonsil ontology/K 未被正确资格审计

Night-6A 任务书把 tonsil 描述成“四区域 annotation”，runner 又硬编码：

`n_clusters = 4`

但外部权威资料并不支持把 4 直接当成唯一正确的 `final_annot` 聚类数：

- SpaMosaic 对同名 `s1_adata_rna.h5ad` / `s1_adata_adt.h5ad` 的官方 section-1 教程使用 `n_cluster=6`：  
  https://spamosaic.readthedocs.io/en/latest/tutorials/integration/vertical/Tonsil_vertical.html
- SpaMICS 论文说明：公开手工图可概括为 4 个大区，但原研究还给出了更细的 7-domain scheme；其 section-1 benchmark 使用更细方案：  
  https://www.sciencedirect.com/science/article/pii/S1566253525005019
- SpaMICS 源码对 `Human_tonsil_1` 读取 `obs['final_annot']` 并编码；默认流程由实际唯一标签数决定 K，而另一个 `Human_tonsil` 分支又显式写 7：  
  https://github.com/SZU-CGC/SpaMICS

Night-6A 本地结构审计只记录了 `final_annot / lab / lab_lynn / src` 四个列名，没有在一个独立 data-steward 阶段记录：

- 每列的非空唯一值数；
- category 名称与计数；
- RNA/ADT 每个 barcode 的 annotation 是否完全一致；
- `final_annot` 与 4/6/7 三种公开口径的对应关系；
- 最终 benchmark 应采用哪一套 ontology 及其理由。

因此，“K=4”不能继续沿用。当前 D 盘没有原始 tonsil 文件，不能在本地补做真实唯一值检查；在服务器下一次开机前，实际 K 仍是未知量。任何人都不得凭印象宣布它一定是 6 或 7。

这可能解释 Night-6A tonsil N00 的异常低指标（五 seed 平均约 ARI 0.0815、NMI 0.1612、Q 0.1214），但目前只能称为强疑点，不能称为已证实原因。

## 5. Night-6A 诊断结果能告诉我们的内容

在不把它升级为正式证据的前提下，失败尝试仍提供以下工程信息：

- 原样重跑 PCGrad、MinNorm、Barlow、neighbor InfoNCE 和原注册表图剪枝没有明显成功信号；
- 15 个结构候选中，没有一个在 A1 双 seed 上同时达到正向平均 ΔQ 和原空间保护门；
- 表现最接近的 N10 平均 ΔQ 仍为 `-0.000879` 且空间门失败；
- 空间门通过者中 N05 最接近，平均 ΔQ `-0.002516`，且 seed 方向异质；
- neighbor InfoNCE 平均 ΔQ `-0.005132`，运行时间约参考的 51.65 倍；
- hard/soft pruning 往往损害空间连续性；
- N00 初始训练贡献中 RNA reconstruction 约占总 loss 的绝大多数，但 Night-5/6 的动态平衡与梯度协调尝试没有稳定改善，因此继续堆损失函数不是最高优先级。

这些结果支持“换研发层级”，而不是支持“这些思想永远无效”。

## 6. 当前真正的数值瓶颈

有效 Night-5 B01/C04 在 A1 五 seed 的均值为：

- ARI `0.262235`
- NMI `0.383919`
- Q `0.323077`

这仍低于近年同方向方法在 A1 上公开呈现的水平。跨论文数值不能在未统一数据、标签、K、预处理、seed 和聚类器时作严格横向结论，但足以说明 A1 仍有实际提升空间。

代码审计发现两个直接瓶颈：

1. 当前 A1/tonsil 同时把空间 kNN 设为 18；该图不仅进入 GNN，还参与 ASR/Moran 预处理。SpaBalance 对 10x 默认空间邻居数为 3，SpaMosaic tonsil 教程为 10，SpaMICS 默认 spatial k 为 10 并用 feature graph 交集细化空间图。18 很可能过宽，但必须以全局、跨 A1+tonsil 的固定候选验证，不能按数据集单独调参。
2. 当前训练器实际上产出 `emb_latent_omics1`、`emb_latent_omics2` 和 fused embedding，但 runner 只保存 fused `SpaLORA`，随后固定做 PCA20 + mclust EEE。模态特异表示、跨 view 邻接一致性和多 seed 共识全部被丢弃。SpaMICS、SMODEL 等源码提示，最后的 affinity/ensemble/cluster head 本身可能带来显著收益。

## 7. 规划结论

不批准原样重跑 Night-6A。

下一轮应是一个全新的、语义清洁的 Night-6B，依次解决：

1. 独立建立 tonsil ontology/K contract；
2. 用低层 HDF5 构建零 annotation 的训练副本，并通过进程级路径守卫；
3. 固定 C04/B01 encoder，系统比较空间/特征图尺度和 feature-spatial intersection；
4. 保存全部三类 latent view；
5. 在同一批冻结 embedding 上比较 mclust、稀疏 affinity、view consensus 和轻量空间后处理；
6. 保留 balanced 与 accuracy frontier 两条路线；
7. D1、P22 继续完全封存，直到候选在 A1+tonsil 五 seed 上冻结。

这一路线比继续增加 loss 模块更接近当前实际瓶颈，也能在很低的单次训练成本下广泛试错。

