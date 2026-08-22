# SpaLORA Night-12A 数据结构与真实路径 P0 报告

## 负责人现在需要知道的三件事

1. 本轮想解决的是：两个冻结的 GSE308623 队列是否真的保留了可追溯的 feature-level RNA、ATAC fragments/基因链接和 ADT target，并能进入同一个零步工程模型；没有检验机制是否可识别，也没有检验聚类是否正确。
2. 实际工作位于数据与工程路径层：闭合 accession 到重复和模态的官方文件关系，逐个流式审计六个重复的 shape/方向/ID，冻结 mm10 与 ADT mapping，再让 P5S1 和 P10S1 从真实 feature-level 文件经过预处理、同一模型类、零步 loss、checkpoint、fresh-process reload、等权 fusion 和稀疏 H05-style K=2 endpoint。
3. 对论文的含义只是：可以进入下一轮冻结的 P0-IDENT。它不是 LOCAL SIGNAL，不是科学正负结果，不是聚类提升，也没有证明 SOTA。

终态：`NIGHT12A_SCHEMA_AND_REAL_PATH_P0_READY`。

## 六个真实重复的结构

| unit | RNA obs×feature | RNA 方向 | 第二模态真实结构 | ordered-ID SHA 前12位 |
|---|---:|---|---|---|
| P5S1 | 7794×32285 | features_by_rows_spots_by_columns | ATAC fragments 83593412 rows / 10000 barcodes | d52b05b0f4a9 |
| P5S2 | 28545×32285 | spots_by_rows_features_by_columns | ATAC fragments 136513082 rows / 48397 barcodes | 5759d9ce0e1f |
| P5S3 | 9426×32285 | features_by_rows_spots_by_columns | ATAC fragments 434173558 rows / 10000 barcodes | f8cd2fb4dbfd |
| P10S1 | 7447×32285 | features_by_rows_spots_by_columns | ADT 7447×143 | 03da4d1470da |
| P10S2 | 41289×32285 | spots_by_rows_features_by_columns | ADT 41289×140 | e233cbe38c20 |
| P10S3 | 5845×32285 | features_by_rows_spots_by_columns | ADT 5845×145 | 029d928ae887 |

三份 P5 fragments 的 header 均锁定 Cell Ranger ARC 2.0.2 / mm10 2020-A；clean-room link 使用 Ensembl release 79 GRCm38 的唯一 gene symbol、gene body 加链方向上游 5 kb、fragment midpoint 和完整注册 fragment depth 归一化。它是本轮明确登记的工程 gene-score/link，不声称与 ArchR GeneScoreMatrix 数值等价。

## P10 ADT mapping

| unit | deposited targets | unique | ambiguous | control | unmapped | 工程 linked |
|---|---:|---:|---:|---:|---:|---:|
| P10S1 | 143 | 85 | 9 | 9 | 40 | 85 |
| P10S2 | 140 | 84 | 8 | 9 | 39 | 84 |
| P10S3 | 145 | 86 | 9 | 9 | 41 | 86 |

只有 NCBI Gene Symbol/Synonym 的 exact case-folded token 且单一 GeneID 才进入 unique；alias 不确定项原样进入 ambiguous 或 unmapped，没有按标记物常识人工猜配。

## 2/2 真实零步 smoke

| unit | family | 输入 shape | finite reconstruction loss | fresh-process 数值 | canonical partition | GPU peak MiB |
|---|---|---|---:|---|---|---:|
| P5S1 | RNA+ATAC | [[7794, 256], [7794, 256]] | 0.126021 | PASS | exact | 68.8 |
| P10S1 | RNA+protein | [[7447, 85], [7447, 85]] | 1.28301 | PASS | exact | 30.6 |

两条路径都使用全部真实 observations、同一 `UnifiedZeroStepAutoencoder(view1, view2)` 调用签名、latent 64、seed 20260822、training steps 0、两个 private latent 等权平均和工程 K=2。K=2 只证明 endpoint 调用链，不代表真实生物 cluster 数。

## 限制与边界

- 本轮没有运行 P0-IDENT，因此没有 LOCAL SIGNAL 或 SCIENTIFIC NEGATIVE。
- 没有读取 cluster、cell type、region、GT、Y 等标签；没有计算 ARI/NMI/AMI/FMI/Q 或 annotation-based spatial metric。
- 没有下载 GSE263333/GSE213264、第三方 benchmark、P5 protein、P10 ATAC、FASTQ/SRA 或完整 GSE308623_RAW.tar。
- 没有 QCRD、科学重试、dataset-name model routing、family-specific model branch 或 dense N×N。
- ATAC clean-room gene link 是工程可复算输入合同，不是完整新模型，也不冻结未来 residual preservation 权重。

## 导师汇报版

Night-12A 只做了两个新队列的数据与工程可行性封口。六个重复的 RNA、坐标和对应 ATAC/ADT 文件已从官方 accession 关系逐一闭合，方向与像素 ID 也按真实文件复核。P5 的 mm10 provenance 和可复算的 clean-room gene link 已冻结，P10 的 ADT target 被分为 unique、ambiguous、control、unmapped，未猜 alias。P5S1 与 P10S1 均从真实 feature-level 文件完整走过同一零步模型和稀疏 endpoint。两条 checkpoint 在新进程里都恢复了数值，canonical partition 也 exact。标签和科学指标始终为零。这个结果只说明下一步 P0-IDENT 有合法输入，不说明方法有效、聚类更好或论文已经成立。

## 技术附录

- 下载白名单：23/23，6554457205 bytes。
- formal correction cycle：1；scientific retry/fallback：0。
- peak GPU：68.8 MiB（上限 8192 MiB）。
- 详细 SHA、URL、source commit、mapping 和 frozen inputs 见同目录 JSON/CSV/TSV。
