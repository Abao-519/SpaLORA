# Night-15D source collision and novelty-boundary audit

This is a collision/attribution audit, not an external benchmark reproduction. Night-15D copied no third-party implementation into SpaLORA. The new code is a clean-room implementation built on the frozen local compute kit.

| Prior work | Pinned or authoritative source | What is already established | Consequence for Night-15D claims |
|---|---|---|---|
| Besag ICM | DOI `10.1111/j.2517-6161.1986.tb01412.x` | Unary evidence, local pairwise regularity and iterative conditional updates are classical. | Potts/CRF, unary+pairwise energy and ICM are not original claims. |
| BayesSpace | Nature Biotechnology, DOI `10.1038/s41587-021-00935-2` | A molecular mixture likelihood combined with a Potts spatial prior is established for spatial transcriptomics. | Prototype/Gaussian-like unary plus Potts smoothing is prior art. |
| BASS | Genome Biology, DOI `10.1186/s13059-022-02734-7` | Hierarchical Bayesian joint inference of cell types and spatial domains, including multi-sample structure. | Joint latent clustering and spatial regularization are not claimed. |
| DR.SC | Nucleic Acids Research, DOI `10.1093/nar/gkac219`; GPL-3 R package | Probabilistic PCA, Gaussian mixtures, HMRF/Potts, EM and ICM are already unified. | Learned molecular unary, Potts prior and ICM are explicit prior art. |
| SCGP | Cell Reports Methods, DOI `10.1016/j.crmeth.2024.100838` | Feature-weighted spatial graphs followed by Constant-Potts community detection are established. | Similarity-weighted graph edges and Constant Potts are not claimed. |
| BANKSY | Nature Genetics, DOI `10.1038/s41588-024-01664-3`; `Banksy_py` state previously pinned at `9278996c39e376277d57ef95278000447ba6c57c` (GPL-3.0) | Neighborhood means and spatial gradients augment molecular representations before clustering. | Neighborhood means, gradients, diffusion and multiscale graph features are prior art. |
| PRAGA | AAAI 2025 / arXiv `2409.12728`; implementation state previously pinned at `4adb11c96fc7ddad800fa1787eadcc8b91b42784` (AGPL-3.0) | Modality-specific adaptive graphs and prototype-aware representation learning are prior art. | Dynamic/adaptive multimodal graphs and prototype learning are not claimed. |
| DenseCRF | Krähenbühl and Koltun, NeurIPS 2011 | Feature-dependent Gaussian pairwise terms and mean-field inference are established in dense CRFs. | Feature-dependent conductance and mean-field/CRF terminology are prior art. Night-15D never constructs a dense N×N graph. |
| Graph-cut energy minimization | Boykov, Veksler and Zabih, PAMI 2001 | Discontinuity-preserving pairwise energies and alpha-expansion/alpha-beta-swap approximations are classical. | Boundary-preserving pairwise energy and graph-cut solver families are prior art. Night-15D uses sparse synchronous conditional updates, not a global graph-cut guarantee. |

## What Night-15D actually tests

The tested object is one post-embedding reliability-energy **superset** shared by RNA+ATAC and RNA+protein. It combines four already recognizable primitives in a specific clean-room engineering composition:

1. absolute sparse edge-conductance mass instead of always forcing every row to unit neighbor mass;
2. an optional confidence-preserving self return for rejected conductance mass;
3. modality-specific dynamic prototype unaries fused by label-free local margin;
4. deterministic multiscale sparse graph features and a trust-aware direct clustering update.

Every lane invokes the same implementation and function signatures. However, public-label development HPO chooses the discrete modules and numerical values separately for each lane from that shared superset. This is **not** an automatic gate, a content-learned router or one frozen universal configuration.

The defensible present claim is therefore limited to a clean-room combination mechanism and a reproducible cross-family public-development signal. Whether the composition is sufficiently novel for a methods paper remains unresolved. It still needs frozen selection rules, fair matched comparisons against mixture-Potts/CPM/BANKSY/adaptive-graph baselines, and confirmation outside the development-HPO loop.

Primary links:

- BANKSY: https://www.nature.com/articles/s41588-024-01664-3
- BayesSpace: https://www.nature.com/articles/s41587-021-00935-2
- BASS: https://doi.org/10.1186/s13059-022-02734-7
- DR.SC: https://doi.org/10.1093/nar/gkac219
- SCGP: https://doi.org/10.1016/j.crmeth.2024.100838
- PRAGA: https://ojs.aaai.org/index.php/AAAI/article/view/32010
- DenseCRF: https://graphics.stanford.edu/projects/densecrf/densecrf.pdf
- Graph cuts: https://www.cs.cornell.edu/rdz/Papers/BVZ-pami01-final.pdf

