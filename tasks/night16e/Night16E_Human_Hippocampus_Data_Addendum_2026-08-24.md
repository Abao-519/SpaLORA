# Night-16E 人脑海马独立 RNA+ATAC 数据补充

日期：2026-08-24

## 为什么优先

MultiGATE（Nature Communications 2025）把 adult human hippocampus spatial ATAC–RNA 数据用于空间域识别。作者说明其依据既往海马解剖研究和 marker genes 手工标注 hippocampal layers 与 white matter，并将这些 annotation 作为 ground truth；在经过 SpatialGlue 质量过滤后评价，论文报告 MultiGATE ARI 0.60。该对象属于独立 human RNA+chromatin study，比仅增加 MISAR 同研究发育阶段更能补当前外部 transfer 缺口。

## 官方入口

- 论文：`https://www.nature.com/articles/s41467-025-63418-x`
- 论文 processed data：Figshare article `27978765`，DOI `10.6084/m9.figshare.27978765.v1`
- 论文源码（Apache-2.0）：`https://github.com/cuhklinlab/MultiGATE`
- 人脑海马复现目录：`https://github.com/cuhklinlab/MultiGATE/tree/main/Reproduce/clustering/human_hippData`
- 上游原始门户：`https://brain-spatial-omics.cells.ucsc.edu/`

## 最小 processed 下载

Figshare API `https://api.figshare.com/v2/articles/27978765` 登记：

| 文件 | bytes | MD5 | download |
|---|---:|---|---|
| `Human_ATAC_lsi.h5ad` | 30,697,626 | `f58db1c6e8293663acf54073ae2df630` | `https://ndownloader.figshare.com/files/51021927` |
| `Human_RNA.h5ad` | 11,691,886 | `c6bb8b850ca9bd86ca36e380118044f4` | `https://ndownloader.figshare.com/files/51021930` |

Figshare 另列一个同名、同大小、同 MD5 的 `Human_RNA.h5ad`（file id 51022086），不要重复下载。

## 必须闭合的语义

1. 两个 h5ad 的 spot IDs 是否完全一致；不一致时按字符串交集和明确顺序对齐；
2. `.obsm['spatial']` 的方向处理；官方训练 notebook 对第二坐标乘以 -1；
3. manual annotation 的权威文件/字段、类别数 K、缺失 mask；若不在输入 h5ad，必须沿官方 notebook、results 或论文 source data 追溯；
4. 论文所称 SpatialGlue 质量过滤保留的 pixel set；全量与过滤评价必须分开；
5. `Human_ATAC_lsi.h5ad` 是 processed LSI，不冒充 raw peak matrix。可以作为 retained/processed-view 路径，但需要在方法表中注明输入层级；
6. 作者 annotation 是论文作者基于解剖与 marker 的手工 benchmark reference，不等同独立临床病理金标准；
7. 先做 no-op/evaluator/P0，再作为 RNA+chromatin family 的独立 frozen transfer；未经 discovery profile 冻结不得在主方法板上对它重新 HPO。

若官方 annotation 或过滤 mask 无法闭合，保留为 P0/无标签真实性单元，不制造 ARI。
