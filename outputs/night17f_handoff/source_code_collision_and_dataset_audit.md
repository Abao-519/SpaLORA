# Source collision and dataset audit

This was a read-only audit of formal papers and official repositories; no
third-party code or expression matrix was downloaded or copied.

| Source | Official entry | Closest established object | Night-17F boundary |
|---|---|---|---|
| MultiGATE | Nature Communications 2025, `cuhklinlab/MultiGATE`, Apache-2.0 | two-level graph attention autoencoder; cross-modal regulatory attention and spatial graph embedding | cross-modal graph attention/learned relations are prior art; Night-17F did not train this kind of model |
| SMODEL | Communications Biology 2025, `liying-1028/SMODEL`, archived at Zenodo 15598653 | dual-graph regularized weighted partition ensemble | candidate ensemble/co-clustering evidence is prior art, not a new contribution |
| PRAGA | AAAI 2025, `Xubin-s-Lab/PRAGA`, fixed prior audit commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784`, AGPL-3.0 | prototype-aware adaptive graph aggregation and prototype contrastive structure | prototype/dynamic graph components are prior art; none were copied |
| spaMGCN | Genome Biology 2025, `hongfeiZhang-source/spaMGCN`, MIT | multiscale graph convolution plus autoencoder and adaptive graph handling | multiscale graph aggregation is prior art; Night-17F keeps the inherited sparse cut |
| SEPAR | Communications Biology 2026, `zerovain/SEPAR`, MIT | spatial metagene/pattern decomposition across single- and multi-omics data | metagene decomposition is distinct from the direct pairwise consumer; its paired examples do not create new independent physical studies by themselves |
| SpatialGlue | Nature Methods 2024, `JinmiaoChenLab/SpatialGlue`, AGPL-3.0 | dual-attention GNN over spatial and feature graphs | dual-graph/multimodal integration is prior art |
| AffinityNet / Adaptive Affinity Field / graph-cut literature | CVPR/ECCV 2018 and Boykov–Veksler–Zabih | learned local affinity and pairwise segmentation energies | relation-conditioned graph cuts have strong computer-vision prior art; novelty would require domain-specific evidence and independent gain, which Night-17F lacks |

Official source URLs inspected:

- https://www.nature.com/articles/s41467-025-63418-x
- https://github.com/cuhklinlab/MultiGATE
- https://www.nature.com/articles/s42003-025-08372-6
- https://github.com/liying-1028/SMODEL
- https://github.com/Xubin-s-Lab/PRAGA
- https://github.com/hongfeiZhang-source/spaMGCN
- https://github.com/zerovain/SEPAR
- https://github.com/JinmiaoChenLab/SpatialGlue
- https://openaccess.thecvf.com/content_cvpr_2018/html/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.html
- https://openaccess.thecvf.com/content_ECCV_2018/papers/Jyh-Jing_Hwang_Adaptive_Affinity_Field_ECCV_2018_paper.pdf

## Dataset finding

- `GSE213264` is the spatial-CITE-seq study containing human tonsil RNA and
  protein assets; it is not MISAR.  A SEPAR tutorial link that associates it
  with MISAR is an upstream documentation error.
- The spaMGCN paired multi-omics collection substantially reuses SpatialGlue /
  SpaMosaic / MISAR physical studies (mouse thymus, mouse spleen, human lymph
  node slices, MISAR) plus placenta.  It does not automatically supply an
  independent RNA+ATAC confirmation study.
- Human tonsil has reproducible germinal-center biology, but an authoritative,
  mutually exclusive full-domain K-class annotation has not been closed.
  Therefore it remains a biological/GC-F1 candidate, not an ARI/NMI ground
  truth lane.
- No additional dataset was downloaded in Night-17F.  A genuinely independent,
  full-domain annotated paired RNA+ATAC unit remains an evidence gap.

