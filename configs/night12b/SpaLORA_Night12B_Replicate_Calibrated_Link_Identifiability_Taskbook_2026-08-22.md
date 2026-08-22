# SpaLORA Night-12B：跨重复校准链接证据可识别性任务书

日期：2026-08-22  
状态：用户已授权，可执行  
唯一数学权威：同目录 `night12b_replicate_calibrated_link_identifiability_contract.json`

## 我现在需要知道的三件事

1. 这一步只想回答：两个发现重复里可信的“同基因跨模态连接”，能否预测第三个独立重复里这条连接是否真的有用。
2. 实际工作位于 feature-level 数据、固定空间回归和统计评价层；不训练新神经网络、不运行融合模型、不聚类、不读取标签。
3. 两个模态家族都通过严格门，最多得到 `LOCAL SIGNAL`，再授权下一轮设计小型统一门控层；任一家族失败就是 `SCIENTIFIC_NEGATIVE`，该新方向停止。

## 1. 为什么做这一轮

Night-12A 只证明六个真实重复能进入统一工程接口。Night-12B 第一次检验新方法故事的核心前提：权威 gene link 的证据能否跨重复复现。这里的 link 指同一 canonical gene 在 RNA 与 ADT 或 ATAC gene score 之间的结构对应；不是聚类标签，也不是按数据集名称挑模型。

为了防止问题再次扩大，本轮只有一个检验对象：`replicate-calibrated linked utility`。该术语表示“先在发现重复中用正确链接相对于全部错误链接的优势和空间块稳定性评分，再看它能否预测未参与评分的第三个重复”。它与 STORM 等重复级空间统计的区别是：本轮关注的是将来是否允许跨模态信息传递的 linked utility，不声称首次使用生物重复。

## 2. 权威输入与启动定位

执行前必须在 Windows 亲自定位并完整读取：

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/research_governance_20260821/PROJECT_MEMORY_PLUGIN_RESEARCH_WORKFLOW_2026-08-21.md`
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/post_night12a_p0_ident_readiness_review_20260822/Night12A_Independent_Review_and_Night12B_P0_IDENT_Proposal_2026-08-22.md`
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/post_night12a_p0_ident_readiness_review_20260822/decision.json`
4. Night-12A `official_compact` 的 report、decision、P0-IDENT input contract、ADT mapping、ATAC link contract、shape audit、source audit、firewall、resource audit、manifest 和源代码。
5. 本目录的冻结数学合约与本任务书。

先独立复算：Night-12A compact `35/35`、index SHA-256 `27a102c731f91078d519d08da2cf01067f4037dd8b9075bf41f69a075f69d933`、bundle SHA-256 `ce0b8f60a050579693fb63ff4764fba3c367640c1dd80c3181e0e4b96e80fbf8`。任何文件、数量、大小、SHA、commit、tag 或 raw provenance 不一致，立即停止，禁止“按大概意思继续”。

旧 `decision.json` 中 `night12b_authorized=false` 是授权前的历史事实；本任务书和冻结合约记录了用户之后的明确授权，不是文件冲突。

## 3. P0-PROTECT：先保护历史结果

1. 从 Night-12A final commit `ed7c93d979a85eb8b907058ea463d93ba65f6956` 建立独立 workspace `/root/autodl-fs/SpaLORA-night12b`。
2. 验证 `night12a-final-20260822` peel 精确指向该 commit。
3. 在修改前创建并普通 push 保护 tag `baseline/pre-night12b-replicate-calibrated-link-identifiability-20260822`。
4. 新分支固定为 `revision/q2-night12b-replicate-calibrated-link-identifiability-20260822`。
5. Night-12A raw root 只读；所有派生矩阵、中间量和报告写入新的 Night-12B derived root。
6. 开始前和结束后分别记录所有历史 raw roots 的 root-relative path、size、mtime_ns 和 SHA（如原审计合同对大文件使用既定元数据范围，则沿用并解释）；任何历史 raw 变化都 fail-closed。

禁止 force push、覆盖历史 tag、在旧 checkout 上直接开发或把派生数据写回历史 raw。

## 4. P0-INPUT：从已登记的 83-gene 上限冻结 bridge panel

`bridge panel` 是为两家族使用同一 gene identity 而建立的固定公平接口，不是新模型名。

必须在读取任何数值矩阵之前，仅用标识符完成：

1. P10S1、P10S2、P10S3 中 `status=unique` 的 canonical gene 交集；
2. 与六个 RNA matrices 的 feature IDs 交集；
3. 与 Ensembl release 79 GRCm38 中只有一个合法 interval 的 genes 交集；
4. 精确 Unicode 排序，UTF-8 每行一个 gene，末尾换行并哈希。

独立规划阶段观察到 P10 三重复交集是 83，排序文本 SHA-256 为 `ac1b801fceba359bbd7e9454c6811400443440575e4022376c9f64940e612a43`。这 83 个是机械交集的上限，不是假定六个 RNA 与 Ensembl79 后仍必为 83：先要求 P10-only 的 83 数量和 hash 完全一致，再按上述结构规则得到最终 `m<=83`。若 RNA/Ensembl 结构交集删除 gene，逐个登记标识符和结构原因，不得补入替代 gene；只有 `m<3` 才因无法形成穷尽 decoy 而 blocked。任何步骤都发生在读取数值矩阵之前。

禁止使用 Night-12A 的 P5 lexical-first 256；它们只用于工程 smoke。禁止按表达、方差、marker、初步 utility 或 held-out 结果选 feature。常数或零特征保留并标注 non-identifiable，不能静默删除。

## 5. P0-REAL：六个真实路径先闭合

对 P5S1/S2/S3 与 P10S1/S2/S3 全部完成：

1. 从完整 RNA library 计算冻结的 `log1p(10000*count/library)`，再选最终冻结的 `m` 个 genes。
2. P5 只用 Night-12A 已下载官方 fragments 和已审计的 Ensembl79 clean-room 规则重新生成 `m`-gene score；保持“非 ArchR 数值等价”声明。
3. P10 先在每个 spot 的全部 deposited ADT targets 上做冻结 CLR，再选最终 `m` 个 unique targets。
4. P5 仅做切片内部 exact barcode join，报告 RNA rows、ATAC barcodes、交集、两侧未匹配数量；不得为了外观一致下采样。
5. 生成 5 个坐标块，记录 spot-to-block 文件和 SHA。
6. 对每个真实 unit 运行一条 baseline/linked OLS 路径，写出 artifact 并在 fresh process 严格 reload；只报告 shape、ID、finite、hash、round-trip 和资源，不打印/比较科学 utility。

必须 6/6 通过才能冻结正式运行。不同切片之间不做 spot 配准、最近邻映射、图像对齐或逐点比较。

## 6. 实现与最小测试

优先复用本项目 Night-12A 的已审计解析器，新增代码保持模块小、公式直译、无 dataset-name routing。P5 与 P10 可有输入预处理 adapter，但从 83-gene矩阵开始，fold、OLS、null、bootstrap、score 和 gate 必须完全共用一份代码与参数。

冻结前至少测试：

1. P10-only 83-gene 上限及最终 `m`-gene 结构交集不读取数值、标签或结果；输入顺序变化不改变排序和 hash。
2. P5 lexical-first 256 无法进入科学 runner。
3. 六单元 ID、row join、shape、finite、fold non-empty 与 sparse/no-dense-N×N 检查。
4. 五个空间块确定性、tie/fallback 与 fresh-process 一致。
5. `numpy.linalg.lstsq(rcond=1e-12)` 参考实现与任何批处理/残差化优化在 synthetic 和小型真实切片上误差均不超过 `atol=rtol=1e-10`。
6. 每个 target 正好有 1 个 true link 和 `m-1` 个 exhaustive decoys；tie 必须算作不利于 true link。
7. 64 次 spatial-block bootstrap 固定 seed，fresh process byte-exact 或数值 exact。
8. leave-one-replicate-out 中 discovery 输出在修改 held-out 数值时保持 byte-exact；任何 held-out leakage 均 fail。
9. 四条 score 公式、Fisher pooling、同步 feature bootstrap、strongest-single paired difference 和所有 terminal-state truth table。
10. label/firewall 测试覆盖读文件、列名、指标函数和 runner 参数；禁止项计数预期全为 0。
11. 历史 raw roots 前后不变性。

不要为了形式堆无关测试。上述测试服务于真实输入、数学等价、无泄漏和结论边界。

## 7. FORMAL-FREEZE 与正式执行

在正式数值评价前写 `formal_freeze_manifest.json`，至少哈希：

- 源代码与测试；
- 冻结 contract 和 taskbook；
- 最终 `m`-gene panel、六单元输入 manifest、row-join manifest、fold assignments；
- Python/package/CUDA/GPU/CPU 环境；
- firewall 与 raw immutability baseline。

正式执行严格按 JSON 合约：每个 target 与全部 `m` 个 RNA candidates 做同一套五空间块 cross-fit；保存 baseline 与 linked 的 block SSE、utility、true-link rank tail；再做固定 64 次 block bootstrap。每家族固定三次 leave-one-replicate-out，用两个 discovery repeats 产生 link-only、bootstrap-only、replicate-only 和 combined score，第三个 repeat 只做一次 held-out 评价。

正式冻结后最多允许 1 次全局工程修正。若发生，旧 formal cycle 整体作废并保留，六单元、两家族、三 folds、四 axes 和 2000 次 decision bootstrap 全部重跑。禁止按 family、feature、unit 或 seed 局部补跑；禁止看到数值后改公式、阈值、panel、tie 或 fallback。

## 8. 唯一决策门

对每个家族都必须同时满足：

1. combined pooled Spearman 点估计 `>0`，且三个 held-out folds 各自 `>0`；
2. combined pooled Spearman 的同步 feature bootstrap 95% CI 下界 `>0`；
3. combined 相对每次 bootstrap 中 strongest single axis 的 paired difference 95% CI 下界 `>0`；
4. 三个 held-out folds 各自的 `median p_link <0.5`；
5. 六单元全覆盖、输出 finite、禁区计数全 0。

RNA+ATAC 和 RNA+protein 两家族必须全过。没有“一个家族通过就整体通过”，没有额外 AUROC 或 1% 效应阈值，也没有科学 fallback。

终态只能是：

- 全过：`NIGHT12B_REPLICATE_CALIBRATED_LINK_EVIDENCE_IDENTIFIABLE`，分类 `LOCAL_SIGNAL`；
- 实现正确但任一门失败：`NIGHT12B_LINK_EVIDENCE_NOT_IDENTIFIABLE`，分类 `SCIENTIFIC_NEGATIVE`；
- 输入来源不闭合：`NIGHT12B_INPUT_PROVENANCE_BLOCKED`，分类 `INFRASTRUCTURE_FAILURE`；
- 实现语义错误：`NIGHT12B_IMPLEMENTATION_SEMANTICS_INVALID`，分类 `IMPLEMENTATION_FAILURE`；
- 资源/环境阻塞：`NIGHT12B_INFRASTRUCTURE_BLOCKED`，分类 `INFRASTRUCTURE_FAILURE`。

## 9. 禁区

预期必须为 0：新下载、新外部数据、训练/评价/总标签读取、ARI/NMI/AMI/FMI/Q、annotation-based spatial metrics、聚类 endpoint、候选神经模型训练、第三方 benchmark、MISAR Y、E18.5、GSE263333/GSE213264、QCRD、dataset-name routing、家族专用科学公式或阈值、dense N×N、scientific retry/fallback、跨重复 spot 对齐、历史 raw 修改。

固定 OLS 是本轮统计估计器，不计为候选神经模型训练；但所有矩阵拟合、次数和资源仍需审计。

## 10. 报告与交付

正式输出至少包括：

- `night12b_report.md`
- `night12b_decision.json`
- `bridge_panel.csv` 与构造审计
- `real_unit_preflight.json`
- `row_join_audit.csv`
- `spatial_fold_assignments` 的清单与 hashes
- `per_candidate_crossfit_utility.csv.gz`
- `per_feature_replicate_evidence.csv`
- `leave_one_replicate_out_scores.csv`
- `decision_bootstrap_summary.json` 与可复算的 bootstrap draws
- `formal_freeze_manifest.json`
- `label_and_prohibited_action_firewall.json`
- `raw_immutability_audit.json`
- `resource_audit.json`
- `test_summary.json`
- `correction_cycle_registry.json`
- 完整源代码、配置、环境与 Git bundle。

主报告先用通俗中文给“我现在需要知道的三件事”，第一次出现每个缩写都解释。主表必须给 P10-only 83 上限、最终 `m`、结构删除与 non-identifiable、每家族每 held-out fold 的 median p_link 与四轴 Spearman、两家族 pooled CI 与 paired difference CI、全部 gate、运行时间和峰值资源。因为本轮没有聚类，ARI/NMI 不是“缺失”，而是设计上禁止。

另写 5–8 句“导师汇报版”：研究问题、现有缺口、新增统计对象、最重要数字、局限和下一步。commit、tag、hash、compact 放技术审计附录。

## 11. Git、Windows 独立复算与关机

1. 所有代码、合约、报告与小型可复算结果提交到新分支。
2. final commit 后创建 annotated tag `night12b-final-20260822`，普通 push 分支和 tag，验证远端 tag peel 精确等于 final commit。
3. 生成 root-relative size+SHA-256 compact index；Windows 独立复算 indexed files、missing、size mismatch、SHA mismatch、extras、禁止大工件。
4. bundle 必须能列出唯一 final tag 并验证 peel；报告 bundle SHA-256。
5. 只有 Windows 交付复算和最终回复完成后，才将 `/usr/bin/shutdown` 作为最后一条远端命令派发；之后不得重连，不把命令派发表述为控制面板电源状态证明。

## 12. 执行时的停机原则

凡是缺公式、mask、shape、ID、genome build、mapping、hash 或 lineage 证据，先从权威文件和源码查找。找不到就按对应 blocked/invalid 终态停止；不要猜，不要用“合理默认值”，也不要请求扩大数据范围来救结果。
