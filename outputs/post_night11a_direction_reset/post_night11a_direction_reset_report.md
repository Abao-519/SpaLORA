# Post-Night-11A direction-reset read-only feasibility audit

## 负责人现在需要知道的三件事

1. 现有资产不是“只剩 embedding”。RNA+protein 和 RNA+ATAC 都保留了 feature-level 矩阵；P22 甚至保留 121,068 个 ATAC peak 的原始稀疏矩阵。因此，做 feature bootstrap、稀疏空间稳定性并非算力或代码边界上不可行。
2. 真正的硬缺口在 RNA+ATAC 证据链：P22 的 121,027/121,068 个 peak 名可解析为坐标，但文件内没有权威 genome build，也没有 peak-gene link、gene activity annotation 或可追溯生成脚本。仅凭 `chr-start-end` 不能猜 build，更不能自行造 link。
3. 终态是 `NEW_METHOD_REQUIRES_SCOPE_EXPANSION`。三分对象在概念上可与 gate/shared-private 区分，但当前只有一个已被 Night-11A 用过的 RNA+ATAC 生物单元，而且没有真正 pristine 的确认单元；此时写新模型会把数据缺口误包装成方法贡献。

## 问题、实际动作与论文含义

**问题。** Night-11A 已冻结为 `NIGHT11A_NO_IDENTIFIABLE_TRANSFER_UTILITY / SCIENTIFIC_NEGATIVE`。本审计只回答现有原始资产能否支撑“共享稳定信号、有空间结构的模态特异信号、不稳定技术噪声”这一新对象，不能修 Night-11A gate，也不能重新解释其局部 AUROC/Spearman 信号。

**实际动作。** 从 Night-11A final commit/tag 建立独立审计分支；只读检查 12 个已登记数据/工件根的文件数、总字节、mtime 和 metadata fingerprint；用 HDF5 schema 边界读取 4 个真实配对单元及 GSE198353 两个 10x 切片的矩阵结构、spot ID、feature ID 和坐标；复核 preprocessing、G00/G04、H05、R02/C06/H01 与 Night-11A 调用链；没有打开 label/GT/Y 内容，没有启动训练或评价。

**论文含义。** 这不是方法成功，也不是性能提升。可保留的研究命题是：用“跨重复稳定性 + 空间可复现性 + linked-feature 一致性”把有结构的模态特异信号与技术噪声区分开。该命题若没有 RNA+ATAC 注释证据和独立确认边界，就无法成为比 SpaMV/SpaMode/CANDIES/QMF/ECoLaF/CoRiM 更清楚的新模型问题。

## 资产主表

| 单元 | 家族 | feature-level 原始资产 | 进入旧模型的真实形状 | 对应/注释证据 | 判定 |
|---|---|---|---|---|---|
| A1 lymph node | RNA+protein | 3484×18085 RNA CSR；3484×31 ADT CSR | 3484×3000 + 3484×30；G00/G04 64维 | 29/31 ADT 名称与 RNA symbol 精确对应；2 个不猜 | 可做 discovery，非独立确认 |
| A1 tonsil | RNA+protein | 4337×17950 + 4337×31 | 过滤后 4326×3000 + 4326×30；64维 | 同上 | 旧开发/阈值已使用 |
| D1 lymph node | RNA+protein | 3359×18085 + 3359×31 CSR | 3359×3000 + 3359×30；64维 | 同上 | 已用于后续开发/Q00 |
| P22 mouse brain | RNA+ATAC | 9215×22914 RNA；9215×121068 ATAC CSC | 过滤后 9196×2000 + 9196×50 LSI；128维 | peak 坐标几乎全保留；build/link 缺失 | `MISSING_EVIDENCE` |
| GSE198353 rep1/2 | RNA+protein | 32285 genes + 21 ADT，2653/2768 spots | 未进入当前公式 | 0/21 ADT display name 可与 RNA symbol 直接精确对应；无现成 mapping | 可作下一阶段 discovery/reserve，需 mapping 与治理批准 |

完整字段见 `asset_capability_matrix.csv` 与 `asset_schema_audit.json`。

## 独立验证边界

- RNA+protein 有未参与 Night-11A 公式设计的单元，但 tonsil/D1 早已用于开发或阈值观察；GSE198353 两个切片也经历过注册和 metadata/schema preflight。
- RNA+ATAC 没有第二个已审计生物单元。P22 的多个 seed 不是独立切片，不能把 seed 当 confirmatory unit。
- 因而正式回答为 `NO_PRISTINE_CONFIRMATORY_UNIT`。下一阶段至少缺一个有权威 build/link 的 RNA+ATAC discovery 单元和一个预先封存的确认切片/数据集。RNA+protein 若坚持严格 pristine，也需新增一个 untouched slice；若治理允许已做 metadata preflight 的 rep2 作为 reserve，必须在公式设计前书面冻结。

## 源码依赖与可实现性

Night-1 preprocessing 仍能访问 feature-level counts/IDs；G00/G04 在 forward 前仍有 2,000/3,000 RNA 特征以及 protein PCA/ATAC LSI。H05、R02/C06/H01 和 Night-11A 只接受 64/128 维 embedding 与稀疏 affinity，不再访问 feature IDs。G00/G04 checkpoint 的 reload spec 能沿 base-cache SHA 找回 RNA `selected_genes.tsv`，但 protein target 和 P22 peak 没有被写进 checkpoint；Night-11A arm checkpoint 更只有 embedding/partition。

现有稀疏工具足以实现 feature bootstrap 和空间复现统计，linked-feature consistency 也可用稀疏二部图实现，均不要求 dense N×N。但新方法必须新增 bootstrap draw/stability、稀疏 link map、空间特异残差复现、噪声不稳定性及其原子 provenance/checkpoint；当前代码没有这些数学对象。

## 与已发表工作的边界

SpaMV 已占据 shared/private encoder + measurement + HSIC；SpaMode 已占据 invariant/variant + MoE；CANDIES 已占据高质量模态条件去噪；QMF、ECoLaF、CoRiM 分别覆盖质量置信度、冲突折扣和冲突风险动态融合。因此，“再加一个 gate/阈值/private branch/专家路由”没有防守力。唯一可能清楚的新点，是三个成分都有独立观测证据：跨重复稳定的 linked signal、空间可复现的 modality-specific residual、以及 bootstrap 不稳定且不可复现的 noise。本审计只确认这一定义在概念上有区别，并未确认算法或效果成立。

## 冻结 go/no-go

六项门中，feature-level 数据与无标签稀疏实现边界通过；RNA+protein 对应为部分通过；概念 novelty 通过但尚未实现；RNA+ATAC genome/link、每家族未污染 unit、pristine confirmatory unit 三项失败。因此不能给出 `NEW_METHOD_FEASIBLE_WITH_EXISTING_ASSETS`，也不应把整个对象直接判死；合约终态为：

`NEW_METHOD_REQUIRES_SCOPE_EXPANSION`

## 导师汇报版

Night-11A 的负结果不变，本轮也没有继续调 gate。我们只读盘点后确认，两类数据都保留了 feature-level 原始矩阵，并非只能在 embedding 上拼模块。RNA+protein 的 A1/tonsil/D1 有 29/31 个 ADT target 可用 deposited 名称和 RNA gene symbol 精确对应，未对应的两个没有硬猜。P22 的 ATAC peak 坐标基本保留，但没有权威 genome build，也没有 peak-gene link 或生成脚本，这是当前最大的科学证据缺口。现有代码能稀疏地做 feature bootstrap 和空间复现统计，但 H05、R02 和 Night-11A checkpoint 已经丢失 feature-level 语义。概念上，把共享稳定信号、空间结构化特异信号和技术噪声分开，可以区别于现有 shared/private、MoE 和质量/冲突 gate；但当前 RNA+ATAC 只有已参与公式设计的 P22。项目也没有真正 pristine 的确认单元。因此建议不是立刻写新模型，而是先批准最小的数据/注释扩展并冻结 discovery-confirmation 边界；否则应停止这条模型路线。

## 资源、禁区与不可变性

- GPU：0 MiB；训练/恢复/科学评价：0。
- 标签读取：训练 0、评价 0、总计 0；ARI/NMI/Q/AMI/FMI 与 annotation-based spatial metric 均为 0。
- MISAR Y 0、E18.5 0、新数据 0、新下载 0、第三方 benchmark 0、QCRD 0、dense N×N 0。
- 只对实际读取的非标签 schema/identifiers 和新增交付计算 SHA。历史 raw root 不做双重内容哈希；使用 count/size/mtime 与 metadata fingerprint 前后比对。
- 12/12 raw/artifact roots 的 post-audit metadata fingerprint 必须与 before 快照 byte-exact 一致后才封口。

## 技术附录

父级：`166eaf7c5838a345a2f7c6cf0b213275c8bc5cd6` / `night11a-final-20260822`。审计分支：`audit/post-night11a-direction-reset-feasibility-20260822`。最终 commit、tag、bundle、compact index 与 Windows 独立复算值在交付封口后写入 delivery manifest；不在正文预填未完成事实。
