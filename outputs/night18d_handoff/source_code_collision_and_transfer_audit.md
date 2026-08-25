# Source-code collision and transfer audit

| Prior work | Audited official source | Relevant prior object | Night-18D boundary |
|---|---|---|---|
| SpatialGlue | https://github.com/JinmiaoChenLab/SpatialGlue, HEAD 7c976d811d27ace51ce47ae0ad94a068a7d222fa | spatial graphs, intra-/cross-omics attention and fusion | fusion/graph integration is prior art |
| spaMGCN | https://github.com/hongfeiZhang-source/spaMGCN, HEAD 77dfe67d4fd80c124722e68a0f71af36d10fa5fa | autoencoder + multi-scale graph adaptation; official human placenta processed asset | neither placenta use nor multi-scale graph is new here |
| BANKSY | https://github.com/prabhakarlab/Banksy, HEAD 5157d9cf8020ec49c99f18d833b143b69a496762 | neighborhood mean/gradient augmentation for spatial clustering | local spatial statistics are prior art |
| PRAGA | https://github.com/Xubin-s-Lab/PRAGA, HEAD 4adb11c96fc7ddad800fa1787eadcc8b91b42784 | dynamic graph, prototype aggregation/contrastive learning | prototypes and adaptive graph semantics are prior art |
| Boykov-Veksler-Zabih / Potts graph cuts | https://cs.uwaterloo.ca/~yboykov/Abstracts/iccv99-abs.html | alpha-expansion large moves for metric pairwise energy | alpha-expansion/Potts cannot be claimed |

The only object under test is the already developed combination as a frozen external consumer. Because that combination fails the matched external gate, Night-18D makes no novelty claim. Human placenta authority is traced to the Nature Medicine article and official analysis repository: https://www.nature.com/articles/s41591-024-03073-9 and https://github.com/jian-shu-lab/hPlacenta-architecture.
