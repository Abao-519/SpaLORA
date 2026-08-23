# Night-15E source and novelty collision audit

## Audit conclusion

Night-15E does **not** claim novelty for Potts/ICM, prototype or Gaussian unary costs, spatial priors, similarity-weighted graphs, multiscale graph features, dynamic routing, masked graph learning, or cross-modal semantic alignment by themselves. Those objects all have clear precedents. The only defensible claim at this stage is narrower: a clean-room empirical combination of **sparse absolute conductance mass, rejected-mass self-return, continuous cross-modal edge/prototype evidence mixing, and trust-aware local clustering moves**, with reproducible public-benchmark development signal across RNA+ATAC and RNA+protein. Whether that combination is sufficiently novel for a methods paper remains unresolved.

## Source-level comparisons

| Work | Frozen source / license | Source actually inspected | Collision boundary |
|---|---|---|---|
| BANKSY | `9278996c39e376277d57ef95278000447ba6c57c`; GPL-3 | `src/banksy/embed_banksy.py` and neighborhood feature code | Neighborhood means and gradient-like multiscale spatial features are prior art; Night-15E cannot claim those primitives. |
| PRAGA | `4adb11c96fc7ddad800fa1787eadcc8b91b42784`; AGPL-3 | `PRAGA/model.py`, training and graph utilities | Dynamic modality/spatial graphs, GCN reconstruction and prototype structure are prior art. PRAGA source was inspected only and not copied. |
| BayesSpace | `ba8b42211ef1581d31a659766559bd6e42be3acc`; MIT + file | model and spatial prior implementation | Gaussian-like molecular evidence plus spatial neighborhood prior is prior art. |
| BASS | `5c2690fddcdeb12461a69106ff4681777d0a8822`; GPL-3 | model/source files beyond README | Bayesian spatial clustering and Potts-style spatial regularization are prior art. GPL source was not copied. |
| DR.SC | `8c4ddca9240f36cf713aeca888c7265f31bc8622`; GPL-3 | package source and `DESCRIPTION` | Joint dimension reduction and spatial clustering with spatial dependence is prior art. |
| SCGP | `8a081f339ba88c44dfb0c18cefd529a7e53e7431`; license not established | GitLab source snapshot | Feature-weighted spatial graph / constant-Potts-style clustering is prior art; no code was copied. |
| GROVER | AAAI 2026 paper; no verified official repository found | paper method description | Dynamic expert routing is prior art. Night-15E must not describe continuous local mixing as a novel router by itself. |
| SpaMCA | `33319c63350821ae701436c20753a05e87f754f6`; no repository license found | `model.py` and `SpaMCA_Py.py` | Masked spatial/feature graphs, attention, reconstruction, instance/cluster objectives and semantic alignment are prior art. Source was inspected only. |
| SpatialCOC | `40612e6c82368f3c6bae7f61230d78ff9fd3703e`; GPL-3 | `SpatialCOC/model.py`, preprocessing and utility code | Continuous spatial correction/alignment is prior art; no GPL source was copied. |
| SpaMosaic | `cc1336755de8d1ddd1337b3cd67c9d41a61c2cb4`; MIT | `spamosaic/framework.py` and graph utilities | Modality/batch graphs and contrastive GNN integration are prior art. |
| Spatial-CITE-seq author code | `5ea6edef9693af87b5c3c409bd4691d76183ee4e`; no repository license found | `No4_clustering_SCT&CLR.R` and related Seurat scripts | The reported human-tonsil RNA 8-cluster and protein 7-cluster results are author-derived unsupervised clusters, not expert spatial-domain ground truth. |

## What the ablation evidence does and does not support

- Rejected-mass self-return has conspicuous matched importance in protein lanes and parts of MISAR, but is not universally necessary: on P22 K=9 the `NO_REJECTED_MASS_SELF_RETURN` ablation is slightly better than the frozen full profile.
- The trust term is exactly or nearly inactive in A1, D1, both MISAR lanes, P22, and tonsil s2. It therefore cannot be claimed as a generally validated mechanism.
- P22 K=18 is slightly better under a retained-unary-dominant ablation than under the frozen full profile.
- These counterexamples are retained. The evidence supports the continuous combination as a local development signal, not every component as a universal contribution.

## Clean-room and license boundary

The Night-15E implementation was written independently from mathematical descriptions and project-owned Night-15D interfaces. GPL, AGPL and unlicensed repositories were used only for source collision review. No third-party source was copied into SpaLORA or the compact delivery.
