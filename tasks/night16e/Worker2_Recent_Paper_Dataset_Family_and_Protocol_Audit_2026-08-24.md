# Worker2：近期论文数据家族与评价协议复核（2026-08-24）

## 结论先行

后续不能按数据集名称切换模型，但可以按可解释的实验家族冻结数值参数。建议采用两级分组：

1. 一级是共检测模态家族：RNA+protein 与 RNA+chromatin；
2. 二级是同一研究内部的切片、重复或发育阶段，只用于 discovery/transfer 划分，不改变核心公式。

同一家族保持同一 architecture、loss 和参数语义；允许冻结一组 family numeric profile。任何逐 lane 公开标签 HPO 只属于 development scoreboard，不能冒充 family-frozen transfer。

## RNA+protein

### GSE263617 Human lymph node

- A1：3484 spots，18085 genes，31 ADTs；项目当前细粒度 H&E reference 有 10 类。
- D1：3359 spots，18085 genes，31 ADTs；同一研究的第二张切片。
- SMART 正文的主要展示使用 A1、聚类 K=7，并在 K=5..10 做敏感性；这与项目的 K10 reference protocol 不等价。二者必须分表，不能把 K7 的公开背景线直接当作 K10 胜负线。
- 最自然的 family transfer 是 A1 discovery、D1 frozen transfer，反向审计可作补充。

### Human tonsil multi-section

- 三张 canonical tonsil 切片分别有 4326、4519、4521 spots，属于同一 multi-section study block。
- 当前项目逐切片四类 reference protocol，与 SMART-MS 的跨切片联合整合、K=4..8 敏感性不是同一个评价对象。
- 三张切片不能作为三个完全独立研究重复；可以用 s1 discovery、s2/s3 frozen transfer，也应补一张 joint multi-section board（含 batch conservation），但不能因此改成另一套模型。

### 无逐 spot 专家标签但可用于外部兼容/生物学验证

- GSE213264 Human tonsil：2492 paired spots，RNA 28417 features、protein 283 features；已有真实 P0，作者派生 cluster 不是专家 spatial-domain ground truth。
- GSE198353 SPOTS mouse spleen：rep1/rep2 可作跨重复稳定性、marker/coherence 和缺失视图审计。
- GSE308623 P10S1/S2/S3：RNA+ADT 三重复，真实 feature-level P0 已闭合；适合 family-frozen 无标签 transfer。

## RNA+chromatin

### MISAR-seq developmental mouse brain

- E11.0_S1、E13.5_S1、E15.5_S1、E18.5_S1 是同一研究的四个发育阶段。
- SMART 报告的 reference 分别为 5、7、11、10 类；论文明确说明这些来自原研究的无监督 clustering，并由解剖结构验证，应称 `artificial/reference annotations`，不能称独立病理金标准。
- 项目当前 E15.5 的 K7/K12 与 SMART 报告 E15.5 K11 存在协议差异，必须分别登记 annotation 文件、hash、mask 与用途。
- E18.5_S1 是最优先新增的带 reference transfer unit；之后再补 E11.0/E13.5。它们是跨发育阶段、同研究 transfer，不是外部 study confirmation。

### 必须区分的两个 P22

- 项目 canonical P22：9196 spots，RNA+ATAC，当前主协议 K=9，另有作者 18-state assignment；与 COSMOS/3d-OT 语境有关。
- SMART 效率实验 P22：9752 spots，RNA+H3K27me3，25881 genes、70470 peaks，无 ground-truth label，以内部指标选择 K=12。
- 两者组织年龄简称相同，但物理样本、第二模态、spot 数和评价协议不同。后续文件名必须携带 study/accession、modality 与 N，禁止只写 `P22`。

### 其他无标签/弱标签单元

- GSE308623 P5S1/S2/S3：RNA+ATAC fragments 三重复，已有 clean-room mm10 gene-score/link 与真实 P0；适合跨重复稳定性、模态贡献和资源扩展。
- GSE205055 embryo E13 spatial ATAC-RNA：近期论文多用 SC/DB、空间连续性和 marker/accessibility 解释，不能在没有 canonical annotation 时制造 ARI。

## 对 Night-16D/16E 的直接约束

1. Night-16D 只回答三态边能否形成独立表示贡献，不借数据扩展稀释结论。
2. Night-16E 优先闭合 E18.5_S1 的 processed input、annotation provenance 和真实端到端路径；若本地/远端已有官方资产，先复用并验 hash，避免重复下载 10.7 GB SMART bundle。
3. 若 Night-16D 无表示贡献，主方法候选应回到 Night-15F 的统一稀疏能量，并把 CMBF 三态边写成 relation-specific energy，而不是继续叠加新的神经模块。
4. 若 Night-16D 有贡献，先在 E18.5 frozen transfer；再决定是否补 E11.0/E13.5。
5. 主论文实验建议以 study block 计数：lymph-node、tonsil、MISAR-development、canonical RNA+ATAC P22；无标签单元用于泛化、稳定性、模态保持、空间/生物学解释。

## 主要权威来源

- SMART paper: https://www.nature.com/articles/s41467-026-70821-5
- SMART source: https://github.com/Xubin-s-Lab/SMART-main
- SMART processed-data record: https://doi.org/10.5281/zenodo.17093158
- CoMo paper and data list: https://pmc.ncbi.nlm.nih.gov/articles/PMC13092272/
- ARISE paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC13360277/
- GSE213264: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213264
- GSE198353: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353
- GSE205055: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055

