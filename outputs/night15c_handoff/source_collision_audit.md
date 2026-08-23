# Night-15C source and novelty-boundary audit

This audit is attribution and collision control, not an external benchmark reproduction. No third-party source was copied into SpaLORA.

| Source | Pinned source state | License | What it already establishes | Night-15C boundary |
|---|---|---|---|---|
| BANKSY paper and Python implementation | Paper DOI `10.1038/s41588-024-01664-3`; `prabhakarlab/Banksy_py` HEAD `9278996c39e376277d57ef95278000447ba6c57c` | GPL-3.0 | Spatial neighbor means and neighborhood gradients can augment a molecular representation before clustering. | Neighbor means, gradients, and ordinary spatial smoothing are prior art. Night-15C does not claim them. |
| PRAGA paper and implementation | AAAI paper / arXiv `2409.12728`; `Xubin-s-Lab/PRAGA` HEAD `4adb11c96fc7ddad800fa1787eadcc8b91b42784` | AGPL-3.0 | Modality-specific adaptive graphs and prototype-aware contrastive representation learning are prior art. | Dynamic graphs and prototype learning are not claimed. Night-15C operates after retained embeddings and does not copy PRAGA code. |
| Besag ICM / Markov random-field image restoration | DOI `10.1111/j.2517-6161.1986.tb01412.x` | paper | Unary evidence plus local pairwise regularity and iterative conditional updates are classical image-analysis ideas. | Potts/CRF and ICM are not claimed as original. |
| BayesSpace | DOI `10.1038/s41587-021-00935-2` | paper / Bioconductor implementation | A robust multivariate mixture likelihood on low-dimensional molecular features combined with a Potts spatial prior is established spatial-transcriptomics methodology. | Mixture/prototype-like molecular unary plus Potts smoothing is prior art, not a Night-15C novelty claim. |
| BASS | DOI `10.1186/s13059-022-02734-7` | paper / public implementation | Hierarchical Bayesian joint inference of latent cell types and spatial domains, including multi-scale and multi-sample spatial structure, is prior art. | Joint latent clustering and spatial-domain regularization are not claimed. |
| DR.SC | DOI `10.1093/nar/gkac219` | GPL-3 R package | Probabilistic PCA, latent Gaussian mixture clustering, an HMRF/Potts term, EM and ICM are already unified in a spatial clustering model. | Gaussian unary, Potts prior, ICM, and learned smoothness are explicit prior art. |
| SCGP | DOI `10.1016/j.crmeth.2024.100838` | paper / public GitLab implementation | Molecular-feature-weighted spatial graphs followed by Leiden with the Constant Potts Model are prior art across spatial transcriptomics and proteomics. | Feature-weighted graph construction and Constant-Potts community detection are not claimed. |

The tested object is narrower: a clean-room, sparse, post-embedding clustering procedure that alternates a molecular centroid-distance unary with row-normalized neighbor support, while the edge conductance is formed from two modalities by the same registered `max`, `min`, geometric-mean, or spatial-only rule. Its empirical value is only the cross-family public-development signal of this *combination*. Whether that combination is sufficiently novel for a method paper is unresolved and requires a broader next-round collision review plus matched ablations against BayesSpace/DR.SC-style mixture-Potts models, SCGP/Leiden-CPM, BANKSY-like augmentation, graph-cut/CRF solvers, and adaptive-graph multi-omics methods.

Primary links:

- BANKSY paper: https://doi.org/10.1038/s41588-024-01664-3
- BANKSY code: https://github.com/prabhakarlab/Banksy_py
- PRAGA paper: https://ojs.aaai.org/index.php/AAAI/article/view/32010
- PRAGA code: https://github.com/Xubin-s-Lab/PRAGA
- Besag ICM paper: https://doi.org/10.1111/j.2517-6161.1986.tb01412.x
- BayesSpace: https://doi.org/10.1038/s41587-021-00935-2
- BASS: https://doi.org/10.1186/s13059-022-02734-7
- DR.SC: https://doi.org/10.1093/nar/gkac219
- SCGP: https://doi.org/10.1016/j.crmeth.2024.100838
