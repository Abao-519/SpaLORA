# 复制给 Codex 2：Night-16C

请接续当前 SpaLORA 项目，执行 Night-16C“家族冻结的跨模态边界场与数据扩展”。这轮不要继续把主要精力花在逐数据集大网格调参上。Night-16B 的高分可以保留，但它没有证明 unified decoder 独立有效；本轮要同时推进真正的新机制、家族级参数转移和更多真实数据单元。

先完整读取并独立复核：

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/compact_delivery_index.json`
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/windows_independent_verification.json`
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/night16b_report.md`
4. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/night16b_decision.json`
5. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/all_candidate_hpo_ledger.csv`
6. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/per_lane_parameter_table.csv`
7. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/family_default_table.csv`
8. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/minimal_contribution_table.csv`
9. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/frontier_dependency_graph.json`
10. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/label_flow_audit.json`
11. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16b_delivery_20260824/official_compact/outputs/night16b_handoff/failure_and_correction_ledger.csv`
12. `D:/文档/ChatGPT/博士第一篇科研论文项目/research_governance_20260821/PROJECT_MEMORY_PLUGIN_RESEARCH_WORKFLOW_2026-08-21.md`
13. `D:/文档/ChatGPT/博士第一篇科研论文项目/research_governance_20260821/Spatial_MultiOmics_Reading_List_2025_2026.md`
14. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16c_family_frozen_boundary_field_and_dataset_expansion_planning_20260824/Night16B_Worker2_Independent_Audit_and_Night16C_Decision_2026-08-24.md`
15. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16c_family_frozen_boundary_field_and_dataset_expansion_planning_20260824/SpaLORA_Night16C_Family_Frozen_CrossModal_Boundary_and_Dataset_Expansion_Taskbook_2026-08-24.md`
16. `D:/文档/ChatGPT/博士第一篇科研论文项目/night16c_family_frozen_boundary_field_and_dataset_expansion_planning_20260824/night16c_family_frozen_crossmodal_boundary_contract.json`

父级应闭合为：compact 49/49，index SHA-256 `420a8db2b12b3e029fe6bdc0e55b696db0eab688011d5d21ae49c155ee937a6c`，bundle SHA-256 `ac3c4a2ff6404b30b08decc1dc21b7f4e56ed377f085b8eab35e7120d82a92a4`，final commit `80e584ebfc82544f1e37f7eed81942d55878a31e`，tag `night16b-final-20260824`。若文件/hash/commit 不一致，立即停下。仅 GitHub SSH 私钥缺失不阻塞研究，可继续并最终生成 bundle。

先确认 Night-16B 的贡献边界：D1 0.365174/0.444577 与 P22 K9 0.595552/0.717931 是可信 development frontier，但 D1 高分来自 Night-16A 强 authority start 加通用 repair；从普通 D1 start 跑完整 decoder，ARI 由 0.254658 降至 0.213976。不要把 Night-16B 继续包装成统一 decoder 成功。

本轮工作名为 `CMBF-TPR`。CMBF 是“跨模态边界场”：在每条稀疏空间边上，根据 RNA 与另一模态的局部稳健变化，显式区分域内支持、共同边界、模态冲突。TPR 是“可信 prototype 修复”：只有多起点稳定性、各模态 prototype margin 和边界场共同判为低信任的边界点才允许移动，高信任核心保持锚定。这个名字只是工作名，不预先锁定论文命名。

你有充分的研究与实现自由。可以并行尝试确定性 robust-rank 边界场、小型单调可训练 edge calibrator、boundary-aware contrastive objective、方向性邻域特征和多起点 trust；根据 Stage-1 诊断机械缩小到最有希望的路线，不必把每个想法都跑完。真正的原则只有两个：最终 candidate 必须有一条共同计算图；增益必须相对同一强 start 做最小贡献对照。

先做数据家族与 annotation 协议审计。headline 只允许两套 family config：

- `RNA_PROTEIN`：A1、D1、tonsil s1/s2/s3，加审计通过的 SPOTS、GSE308623 P10、Stereo-CITE 等；
- `RNA_CHROMATIN`：P22、MISAR，加其他 MISAR stage、P22 ATAC/CUT&Tag、GSE308623 P5、GSE205055、GSE263333 等。

两家族使用同一模型、同一 API 和同一参数 schema，只允许家族级数值配置不同；不能为 A1、D1、P22 等分别换整条 pipeline。morphology/batch ID 是有 presence mask 的可选输入。K 跟随已登记 annotation protocol，不算换模型。

必须首先闭合四组协议冲突：A1/D1 当前 K10 与 SMART A1 K7；tonsil 当前逐切片 K4 与 SMART-MS 联合 K6；P22 K9/K12/K18；MISAR 不同 stage 的 K7/K10/K12。建立 `annotation_protocol_registry.csv`，记录物理样本、annotation 文件/hash/来源、mask、K、用途。不同 label 粒度不得混表或冒充同一协议。

同时建立 12–20 个公开物理单元的 `dataset_family_registry.csv`。优先查已有磁盘与官方 processed assets，再考虑下载。优先审计 SMART Zenodo/reproduce branch 中的 MISAR 8 sections、SPOTS 2 sections、P22 其他 epigenome sections，以及 GSE308623 P5/P10、GSE205055、GSE263333、Stereo-CITE、GSE213264。每个候选登记 accession/DOI、平台、模态、分辨率、N/features、paired IDs、annotation provenance/K、下载体量、family、独立 study 身份。本轮选择 4–8 个最有价值且资源可承受的新单元进入 P0/评价；没有可信 annotation 的数据可以用空间、表征、跨模态保真和生物标志指标，但不能制造 ARI。

下载前必须闭合 exact asset、官方来源、大小、spot-ID 合约、annotation provenance、可用空间与 inode。优先 h5ad/单 archive，避免大量小文件；不要重复下载现有资产。

在 development units 上先做边界误差诊断：label boundary precision/recall/F1、错误到真实边界的图距离、split/merge 分解、每模态 edge change、跨模态 agreement/conflict、start-bank stability、prototype margin。labels 只在 producer outputs 已保存/hash 后由独立 evaluator 读取，用于公开 development diagnostics 和 family-level HPO，不能进入 features、graph、loss/gradient、单次 move acceptance 或 checkpoint selection。若误差主要不在边界，或 edge field 与真实边界无关系，立即转向 stronger start/representation objective，别在无效 post-processing 上跑大网格。

CMBF 必须为每条 sparse edge 输出 support/boundary/conflict 与 conductance；support 边传播，consensus boundary 边切断/衰减，conflict 边不让较差模态污染另一模态并保留 self-return。若采用可训练 calibrator，可使用 masked-view consistency、augmentation stability、mutual-neighbor support 等无标签目标。TPR 必须锚定高信任核心，只移动低信任边界点；split/merge 需要边界和多模态证据，不能只是最小簇修复。

如果 post-processing 的诊断上限不足，优先实现真正改善表示的 boundary-aware contrastive objective：support edge 作为高置信正对，consensus boundary edge 作为负对，conflict edge 不强迫 shared alignment。RNA+protein 与 RNA+chromatin 仍使用同一 encoder/fusion/core，仅输入 adapter 处理维度。

读并审计 BANKSY、stLVG、ARISE、PRAGA、SpatialCOC、SpaMV 的论文 Methods 和官方源码真实入口，固定 commit/许可证，输出 `source_code_collision_and_transfer_audit.md`。BANKSY 的邻域均值/方位梯度、stLVG 的方向权重、ARISE 的 RNA 锚定图交集、PRAGA 的 prototype aggregation 都可以借鉴，但必须准确归因。若真正增益来自已有组件，就按实际来源写；如果 CMBF-TPR 高度撞车，改变机制而不是只换名字。

每个 family 至少一条现有真路径、每个新增平台至少一条新真路径先做 P0：真实输入→preprocessing→sparse graph→forward/loss→optimizer step（若可训练）→checkpoint strict reload→CMBF→TPR→endpoint→fresh-process replay。列全 tensor shapes，先每家族 1 seed smoke，再扩展。工程 bug 可以修并登记，不设僵硬 correction 次数；formal freeze 后若改科学公式，完整重跑受影响单元。所有失败结果保留。

搜索按效率漏斗执行：先 CPU/单 seed 做诊断和每家族约 50–150 个低成本 config screen，每家族只晋级少数候选；真正可训练模块才用 GPU；复用 reduced views、sparse graphs、start banks 和 local compute kit，禁止 dense N×N。不要再生成每个 lane 一万条近似同义参数。

公开 annotation 可以用于 discovery units 的 `label-assisted family-level benchmark HPO`。候选先 materialize/hash，再评价。每家族只冻结一组 headline config，机械排序为：ARI/NMI 双升的 discovery study 数最多→最差 study ΔARI 最大→study-balanced mean ΔARI→mean ΔNMI→更低复杂度。可以保留 per-dataset tuned oracle，但必须标成 development ceiling，不能当 family-frozen headline。

若 2026-08-21 长期治理文件中“不得根据标签事后选配置”的旧句与本轮冲突，以用户在 Night-16A/16B 后作出的最新明确决定和本 Night-16C 合约为准：公开 discovery annotations 可以用于透明、可复算的跨运行 benchmark HPO；但权限只到 family-level config 冻结为止，不授权标签进入单次 producer，也不授权对 held-out/new unit 再调参。

建议 RNA_PROTEIN 用 A1 + tonsil s1 选 family config，冻结后运行 D1、tonsil s2/s3 及新增 SPOTS/P10；RNA_CHROMATIN 用 P22 K9 + MISAR E15.5 primary 选 config，冻结后运行新增 MISAR/P22/GSE205055/P5。现有 lane 都参与过历史开发，不能称 pristine blind confirmation；只有没有进入 family HPO 的新增物理单元可称 frozen transfer。

每条 primary lane 至少做：strong start only、CMBF-TPR full、boundary disabled、conflict disabled、trust gate disabled、generic repair only，以及 directional features only（若使用）。如果 full 不超过同一 strong start，就不能把分数归因于新模块。

现有 frontier 不是上限：A1 .276003/.421740、D1 .365174/.444577、tonsil s1 .236536/.317118、s2 .258264/.314324、s3 .350644/.309771、P22 K9 .595552/.717931、MISAR K7 .541424/.666798。继续朝 same-protocol 公开高位推进，但优先形成家族冻结的新模块证据。

主结果至少报告绝对 ARI/NMI、AMI/FMI/Homogeneity/V-measure、Moran/Geary、几何与跨模态保真、best/median/mean/min、胜出 seed 数、最小簇、wall/GPU/RSS。无可信 labels 的单元只报无标签和生物学指标。论文主表未来可以选平台/物种/组织/annotation 可信度有代表性的子集，但项目 ledger 与补充材料必须保留所有实际运行的 eligible 单元和排除理由，不能按分数秘密删除失败数据。

终态按事实选择：`FAMILY_FROZEN_METHOD_SIGNAL`、`BOUNDARY_FIELD_LOCAL_SIGNAL`、`START_GENERATOR_SIGNAL`、`SCORE_FRONTIER_ADVANCE`、`DATASET_EXPANSION_READY`、`NO_ADDED_METHOD_SIGNAL` 或实现/设施失败。不要因为一条 per-dataset best 上升就自动判定方法成功。

必须交付任务书列出的 registry、annotation audit、source collision audit、P0、family search/frozen config、per-dataset oracle、boundary diagnostics、绝对主表、最小贡献对照、完整失败 ledger、label/resource audit、Methods/novelty 草案和 reviewer risk register。报告先写“我现在需要知道的三件事”，所有缩写第一次出现即解释，主结果在技术 hash 之前，并给 5–8 句导师汇报版。

从 `80e584ebfc82544f1e37f7eed81942d55878a31e` 新建 `revision/q2-night16c-family-frozen-crossmodal-boundary-field-20260824`；普通 push，final tag `night16c-final-20260824`，不 force push。若 SSH key 仍缺失，保留事实并生成可恢复 bundle。生成 root-relative compact index，Windows 独立复算。所有交付验证完成后，最后一条远端命令严格为 `/usr/bin/shutdown`；派发后不重连。
