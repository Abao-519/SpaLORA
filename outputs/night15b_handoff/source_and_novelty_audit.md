# Night-15B source and novelty-collision audit

This audit fixes source identity before interpreting SAPR.  No third-party source
was copied into SpaLORA; the implementation is clean-room and uses only the
mathematical descriptions summarized below.

| Method | Official source identity | License observed | Actual mechanism read | Collision boundary for Night-15B |
|---|---|---|---|---|
| PRAGA | AAAI 2025 paper; `Xubin-s-Lab/PRAGA` commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784` | AGPL-3.0 | Omics-specific learnable/dynamic graphs plus Bayesian-GMM-driven dynamic prototype contrastive learning; the public implementation materializes an `n×n` learned graph and contains dataset-type branches. | Dynamic graphs and ordinary prototype contrastive/sharpening are prior art. SAPR cannot claim either. PRAGA code is not copied. |
| Proust | Genome Research 2025 paper; `JianingYao/proust` commit `455466c7d683baabfc5db53cbee9d50607447db1` | GPL-3.0 | Modality graph autoencoders followed by graph-based contrastive self-supervised learning; spatial graph and biological/image views are fused for domain prediction. | Graph autoencoding and cross-view contrast are prior art and are only supporting losses here. Proust code is not copied. |
| SpaMV | Nature Communications 2026 paper; `ericcombiolab/SpaMV` commit `d7105ef70e9276350e8a12bddfbd3d396d1c33d2` | paper code-availability states MIT; repository snapshot did not expose a top-level license in the audited listing | Separate shared/private encoders, MoE integration, omics measurement models, self/cross reconstruction, and HSIC independence. | SAPR is not a shared/private decomposition, MoE, or HSIC method. No SpaMV source was copied. |
| soFusion | 2025 paper; `sunxue-yy/soFusion` commit `aa5fced4f4be182e8ea451928fb1c07bcf5bfca3` | MIT | Spatial GCN encoders, intra/inter-omics feature learning, contrastive integration, and modality-distribution-specific decoders. | Cross-view consistency and modality decoders are established components; SAPR does not claim them. |
| SEPAR | Communications Biology 2026 paper; `zerovain/SEPAR` commit `6d3475fa0bd749d3b1b5592b68323439d473f9fc` | MIT | Interpretable spatial metagene/pattern factorization with graph, sparsity and pattern-orthogonality terms, followed by clustering/refinement. | SAPR is not a metagene factorization or SEPAR refinement. The reported MISAR result is protocol context only. |
| 3d-OT | Nature Methods 2026 paper; `dbjzs/3d-OT` commit `39a7cb02748d83299cd471f172f3b972896e61d8` | Apache-2.0 | PointNet++ geometry encoder plus soft-communication optimal transport for clustering/alignment. | Geometry encoding and OT are not SAPR contributions. The author-supplied P22 18-state assignment is context, not independent truth. |

## Narrow claim that remains testable

The only Night-15B-specific object is the coupling of (i) Hungarian-aligned,
structurally diverse label-free partitions, (ii) a stability-derived interior
trust region, and (iii) a small shared residual whose movement budget is focused
on unstable spatial boundaries.  Prototype initialization, pseudo-label
sharpening, graph consistency, balance regularization, and cross-view consistency
are explicitly treated as prior components.  The module is only interesting if
`SAPR full` improves over both the retained teacher/strong embedding and the
same trained core with the residual disabled under one matched endpoint.

## Primary references

- PRAGA paper: https://ojs.aaai.org/index.php/AAAI/article/view/32010
- Proust paper: https://genome.cshlp.org/content/35/7/1621
- SpaMV paper: https://www.nature.com/articles/s41467-026-74718-1
- soFusion paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12477611/
- SEPAR paper: https://www.nature.com/articles/s42003-025-09340-w
- 3d-OT paper: https://www.nature.com/articles/s41592-026-03034-9

