# Night-16B 独立复核与 Night-16C 决策

日期：2026-08-24

## 我现在需要知道的三件事

1. Night-16B 的分数资产是真实、可复算的：Windows 端独立复核为 49/49，D1 达到 0.365174/0.444577，P22 K=9 达到 0.595552/0.717931。
2. 但这还不是新模型证据。D1 的高分依赖既有强初始化与通用聚类修复；从普通 D1 start 直接运行完整 decoder，ARI 反而由 0.254658 降到 0.213976。P22 的小幅提升含空间边界修复贡献，但不足以证明统一 decoder 在多 study 独立生效。
3. Night-16C 不再继续堆 per-dataset HPO。下一轮转向一个可独立消融的新机制：用两种组学在每条空间边上的变化，显式区分“域内支持、共同边界、模态冲突”，再用可信核心约束 prototype 更新；同时扩充同家族真实切片，冻结家族级配置后检验转移。

## 1. 文件与交付复核

- Compact payload：49/49。
- Missing / size mismatch / SHA mismatch / extras：0 / 0 / 0 / 0。
- Compact index SHA-256：`420a8db2b12b3e029fe6bdc0e55b696db0eab688011d5d21ae49c155ee937a6c`。
- Bundle SHA-256：`ac3c4a2ff6404b30b08decc1dc21b7f4e56ed377f085b8eab35e7120d82a92a4`。
- Final commit：`80e584ebfc82544f1e37f7eed81942d55878a31e`。
- Annotated tag：`night16b-final-20260824`。
- 普通 push 因 GitHub SSH 私钥缺失失败；bundle 可恢复，未 force push。

关键文件独立 SHA-256：

- `night16b_report.md`：`250b359d1a9e1d4032879af4bb3fb73e26ad16cd9c4064e7426f6861650b9a54`
- `night16b_decision.json`：`cce9c6d6b1530a1f3c46ef4cd7d897939b8136fe69c2ced2c294024af1f24dc2`
- `minimal_contribution_table.csv`：`0009f49b999db1ebdc1f5a15fbb1ed017f5483a34d8029bb06f11fb62115ab1a`
- `method_semantics_and_selection_contract.md`：`d29c1782f94c52176c2125cf89651ac1085d56f9e8162daf6fcda0593666e99b`

## 2. Night-16B 的科学含义

### 2.1 可保留的分数前沿

| 协议 | ARI | NMI | 说明 |
|---|---:|---:|---|
| A1 K10 | 0.276003 | 0.421740 | 未刷新 |
| D1 K10 | 0.365174 | 0.444577 | 明显刷新；最小簇 55 |
| tonsil s1 K4 | 0.236536 | 0.317118 | 未刷新 |
| tonsil s2 K4 | 0.258264 | 0.314324 | 未刷新 |
| tonsil s3 K4 | 0.350644 | 0.309771 | 未刷新 |
| P22 K9 | 0.595552 | 0.717931 | 小幅刷新 |
| MISAR E15.5 K7 | 0.541424 | 0.666798 | 未刷新 |
| P22 K18 sensitivity | 0.745939 | 0.766121 | 不与 K9 合并计票 |
| MISAR E15.5 K12 sensitivity | 0.454117 | 0.607191 | 不与 K7 合并计票 |

这些值可以作为公开 benchmark 的开发前沿，但不能把不同 annotation、K、mask 或 endpoint 的数字直接当成公平胜负。

### 2.2 为什么不能把高分直接归因于 unified decoder

D1 最小贡献对照为：

| D1 路径 | ARI | NMI |
|---|---:|---:|
| start partition | 0.254658 | 0.389044 |
| full common path | 0.213976 | 0.361442 |
| reliability disabled | 0.213207 | 0.360420 |
| self-return disabled | 0.213207 | 0.360420 |
| repair disabled | 0.254381 | 0.388974 |

0.365174 的 D1 headline 来自 Night-16A 已经达到 0.338775 的 authority start，再叠加结构修复。其 boundary refinement 的 `beta=0`，实际更新主要是受最小簇约束的 prototype 重新分配，并非跨模态空间边界证据。P22 的最佳配置含非零空间项，因此 P22 有局部边界修复信号；但它的增量很小，尚未跨 study 独立复现。

因此 Night-16B 的正确分类仍是：

- 主结果：`PAPER_COMPATIBLE_TUNED_SCOREBOARD + SCORE_FRONTIER_ADVANCE`
- 方法贡献：`HEAD_ONLY_OR_INITIALIZATION_SIGNAL`
- 不能宣称：统一 decoder、SOTA、confirmed milestone 或 paper-ready evidence

## 3. 数据家族与 annotation 协议的新问题

公开论文和官方教程显示，当前若只按数据集名称/现有 K 直接延续，会产生协议混淆：

- SMART 论文以 A1 的 7 类 H&E annotation 为主，并在 5–10 类范围做敏感性；当前项目 A1/D1 primary 是 K10。
- SMART-MS 对三张 tonsil 切片联合使用 6 类；当前项目逐切片 primary 是 K4。
- SMART 的 MISAR 教程用 E18.5、K10；当前项目主要是 E15.5、K7，另有 K12 sensitivity。
- SMART 的 P22 H3K27me3 教程使用 K12；COSMOS 的 P22 主要背景是 K9；3d-OT 资产另有 18-state assignment。

这些都不是“谁对谁错”，而是不同 annotation/粒度/任务。Night-16C 必须为每个协议登记 annotation 来源、物理样本、评价 mask、K 和用途；不同协议不得混表冒充同一任务。

## 4. Night-16C 的核心机制

工作名：`CMBF-TPR`。

- `CMBF` 是 Cross-Modal Boundary Field，即“跨模态边界场”。造这个词是因为现有项目只有普通邻接权重，没有一个显式对象同时表示每条边的域内支持、共同边界和模态冲突。
- `TPR` 是 Trusted Prototype Refinement，即“可信原型约束修复”。它对应稳定区域核心、簇原型和不确定边界点；区别于 Night-16B 的普通 prototype 重分配在于，移动必须由跨模态边界证据与多起点稳定性共同授权。

在稀疏空间边 `(i,j)` 上，对每个模态先计算经过局部秩/稳健尺度校准的变化量，再形成三种状态：

1. 两模态都低变化：域内支持边，可传播与平滑；
2. 两模态都高变化：共同边界边，应切断或强烈衰减传播；
3. 一高一低：模态冲突边，不让较差模态强迫另一模态传播。

随后把三状态场用于：

- 可微的各向异性稀疏图传播；
- boundary-aware 对比目标：支持边为高置信正对，边界边为负对，冲突边不强制共享；
- 可信 prototype 核心锚定；
- 只移动低信任且靠近候选边界的点；
- 以边界证据提出 split/merge，而不是仅以簇大小修复。

方向性邻域均值/梯度可以借鉴 BANKSY 与 stLVG，RNA 锚定图应与 ARISE 区分，prototype 机制应与 PRAGA 区分。若最终增益实际来自这些已有组件，必须如实归因并重新收紧 novelty，不得把组合包装成完全原创。

## 5. 家族冻结决定

Night-16C 只允许两套 headline family config：

1. `RNA_PROTEIN`：A1、D1、tonsil s1/s2/s3，以及审计通过后加入的 SPOTS/P10/Stereo-CITE 等 RNA+protein 单元。
2. `RNA_CHROMATIN`：P22、MISAR，以及审计通过后加入的其他 MISAR stage、P22 ATAC/CUT&Tag、P5/GSE205055 等 RNA+chromatin 单元。

两家族调用同一模型、同一计算图和同一参数 schema，只允许家族级数值配置不同。K、是否有 morphology、是否有 batch ID 属于输入/评价协议，不是另换模型。正式 family config 一经在 discovery units 上选择，held-out family units 不得按当前 lane 标签再次调参。

## 6. 数据扩展决定

先建立 12–20 个公开物理单元的资产表，再按以下条件选择本轮 4–8 个新单元进入 P0/评价：

- paired spot/cell IDs 可闭合；
- 组学、坐标、annotation 来源明确；
- 有官方 processed artifact 或可追溯 clean-room preprocessing；
- 能补足已有家族的独立切片/重复；
- 下载体量与 inode 风险可控。

优先审计：SMART Zenodo 中的 MISAR 8 sections、SPOTS 2 sections、P22 其他 epigenome sections；GSE308623 P5/P10；GSE205055；GSE263333；Stereo-CITE；GSE213264。没有可信 annotation 的数据仍可用于无标签空间、表征、跨模态保真与生物标志验证，但不得制造 ARI。

## 7. 成功边界

Night-16C 不是以“所有数据集都涨”为硬门。优先顺序为：

1. 新模块相对同一强 start 的独立增益；
2. 同一家族固定配置在至少两个独立物理单元上转移；
3. 现有分数前沿继续推进；
4. 数据集扩展与 annotation 协议闭合。

允许终态：`FAMILY_FROZEN_METHOD_SIGNAL`、`BOUNDARY_FIELD_LOCAL_SIGNAL`、`START_GENERATOR_SIGNAL`、`SCORE_FRONTIER_ADVANCE`、`DATASET_EXPANSION_READY`、`NO_ADDED_METHOD_SIGNAL`，以及实现/设施失败。不得因为只有一条 lane 的 tuned best 上升就自动把新机制判为论文贡献。
