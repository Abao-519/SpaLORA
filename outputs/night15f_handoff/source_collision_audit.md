# Night-15F source and novelty collision audit

## Bottom line

Night-15F does **not** claim novelty for Potts/CRF energies, prototype or Gaussian unaries, graph cuts, alpha-expansion, fusion moves, multiscale graphs, neighborhood features, similarity-weighted edges, dynamic routing, or size balancing by themselves. The defensible result is narrower: a clean-room, sparse implementation that places continuous cross-modal edge evidence, three registered graph scales, absolute accepted conductance mass, explicit rejected-mass stay cost, dynamic prototype unary and large-neighborhood moves in one auditable energy, with public-benchmark development gains across two modality families. Whether this combination is sufficiently novel for a paper remains unresolved.

## New solver collision boundary

| Prior work | Authority inspected | Collision boundary |
|---|---|---|
| Boykov, Veksler & Zabih, *Fast Approximate Energy Minimization via Graph Cuts* | ICCV 1999/PAMI 2001 paper, DOI `10.1109/34.969114` | Alpha-expansion and the statement that a move may assign any subset of nodes to alpha are established prior art. Night-15F cannot claim the large-neighborhood solver itself. |
| Kolmogorov & Zabih, *What Energy Functions Can Be Minimized via Graph Cuts?* | PAMI 2004, DOI `10.1109/TPAMI.2004.1262177` | Binary submodularity/regularity and s-t cut constructions are established prior art. |
| Lempitsky et al., *Fusion Moves for Markov Random Field Optimization* | Microsoft Research publication `MSR-TR-2009-60` / PAMI paper | Proposal fusion by binary graph-cut moves is established prior art. Night-15F implemented alpha expansion only; it does not claim fusion moves as a contribution. |

The project code was independently derived from the Potts energy and binary submodular decomposition. It uses SciPy's public sparse `maximum_flow` primitive. No third-party alpha-expansion/fusion implementation was copied.

## Inherited spatial-method collision boundary

| Work | Frozen source / license | Collision boundary |
|---|---|---|
| BANKSY | `9278996c39e376277d57ef95278000447ba6c57c`; GPL-3 | Neighborhood means, gradients and multiscale spatial feature banks are prior art. |
| PRAGA | `4adb11c96fc7ddad800fa1787eadcc8b91b42784`; AGPL-3 | Dynamic molecular/spatial graphs, reconstruction and prototype structure are prior art. No source was copied. |
| BayesSpace | `ba8b42211ef1581d31a659766559bd6e42be3acc`; MIT + file | Molecular unary evidence plus spatial neighborhood priors are prior art. |
| BASS | `5c2690fddcdeb12461a69106ff4681777d0a8822`; GPL-3 | Bayesian clustering with Potts-style spatial regularization is prior art. |
| DR.SC | `8c4ddca9240f36cf713aeca888c7265f31bc8622`; GPL-3 | Joint dimension reduction and spatial clustering are prior art. |
| SCGP | `8a081f339ba88c44dfb0c18cefd529a7e53e7431`; license not established | Feature-weighted spatial graphs and constant-Potts-style clustering are prior art. |
| GROVER | AAAI 2026 paper; no verified official repository | Dynamic expert routing is prior art. |
| SpaMCA | `33319c63350821ae701436c20753a05e87f754f6`; no repository license found | Masked spatial/feature graphs, attention and semantic alignment are prior art. |
| SpatialCOC | `40612e6c82368f3c6bae7f61230d78ff9fd3703e`; GPL-3 | Continuous spatial correction/alignment is prior art. |
| SpaMosaic | `cc1336755de8d1ddd1337b3cd67c9d41a61c2cb4`; MIT | Modality/batch graphs and contrastive GNN integration are prior art. |

## What the matched ablation actually supports

- The same frozen energy with alpha-expansion beats matched asynchronous single-site descent on MISAR K=12, P22 K=9, P22 K=18, tonsil s1 and tonsil s2. It is identical on A1, MISAR K=7 and tonsil s3. Large-neighborhood optimization is therefore supported on a subset, not universally.
- Three-scale mixing beats registered-scale-only on all nine frozen lanes in the matched table.
- Removing rejected-mass stay cost causes severe degradation in several lanes. This establishes its stabilizing role, but not novelty by itself.
- The tonsil s3 full result is not explained only by size balancing: `NO_SIZE_PRIOR` still exceeds the Night-15E authority on both ARI and NMI, although the full size-prior configuration is materially better.
- On tonsil s3, `PAIRWISE_ZERO_KEEP_STAY` is slightly better than the frozen full result. The explicit pairwise term therefore is not independently supported on every lane; multiscale features, dynamic unary, stay cost and size prior explain much of that lane.
- Dynamic unaries are recomputed between outer cycles. Only energy descent within each frozen-unary cycle is claimed; no single cross-cycle global monotonicity claim is made.

## License boundary

GPL, AGPL and unlicensed repositories were used only for source collision review. No third-party source is present in the SpaLORA code or compact delivery.
