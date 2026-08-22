# Post-Night-11A source dependency map

## Audited call chain

The actual preprocessing entry point is `SpaLORA/night1_pipeline.py::_load_label_free/prepare_corrected`. It loads the paired H5AD objects, enforces ordered spot and coordinate equality, deletes all observation columns, retains raw RNA counts separately, produces a selected/scaled RNA matrix, and uses CLR/PCA for protein or the deposited `X_lsi` for P22 ATAC. The dataset identifier changes P22 filtering, PCA width, modality-2 preprocessing, embedding width, epoch count and K through registered configuration; it is not a feature-derived router.

`SpaLORA/night6c_pipeline.py::build_graph_data` receives the feature-level preprocessed matrices, RNA PCA, coordinates and sparse supports. G00 and G04 therefore still have access to feature matrices before training. `normalized_views` reduces the model output to six arrays; the private and fused arrays are only 64-dimensional for RNA+protein and 128-dimensional for P22.

`SpaLORA/night6c_pipeline.py::run_head` implements H05 by building three sparse self-tuning affinities from the two private embeddings and their fused embedding, averaging them equally and applying the fixed spectral endpoint. H05 has no access to raw counts, feature identifiers or a feature-link graph.

`SpaLORA/night7b_adaptive.py::AdaptiveFusion` and `scripts/night7b_adapter_stage.py` implement R02-family adapter semantics over six frozen 64/128-dimensional views. The adapter reconstructs those views, uses sparse relation edges where registered, and persists an embedding/checkpoint. It does not accept raw feature matrices or feature identifiers. C06 and H01 likewise operate on sparse affinity/embedding objects rather than raw features.

`SpaLORA/selective_transfer.py`, `SpaLORA/night11a_corruption.py` and `scripts/night11a/run_identifiability.py` consume only G00/G04 private embeddings, ordered spot IDs, coordinates, a sparse spatial graph and frozen K. The cross-fitted ridge pilot predicts held-out embedding dimensions, not biological features. Its checkpoint stores arm embeddings/fusion/partitions; it cannot recover linked genes, proteins or peaks.

## Layer capability table

| Layer | Accepted object | Feature-level access | Identity/metadata dependence | Persistence boundary |
|---|---|---|---|---|
| Night-1 preprocessing | sparse/dense counts, feature IDs, coordinates | Yes | registered dataset config; explicit P22 branch; no label values | base cache, selected genes, model-input SHA |
| G00/G04 graph + encoder | 2,000/3,000 RNA features, 30 protein PCs or 50 ATAC LSI dimensions, sparse graphs | Yes before forward; no peak columns after LSI | graph candidate, shape, dataset config | checkpoint plus reload spec and base-cache SHA |
| H05 | 64/128-dimensional private/fused views | No | K, coordinates/spot IDs; no dataset-name branch in head | sparse affinity and partition SHA |
| R02/C06/H01 | six views and sparse affinities | No | recipe/config, K, shape | adapter checkpoint/embedding/affinity |
| Night-11A | four G00/G04 private views | No | opaque unit ID, direction, shape and frozen K | arm embedding checkpoint; no feature IDs |

The G00/G04 reload specification does preserve a SHA chain to the base cache. For RNA, that cache contains `selected_genes.tsv`, so selected RNA features remain externally traceable. Protein target names and P22 peak identifiers are not embedded into the checkpoint. P22 ATAC is compressed to 50 LSI components before the encoder, so its peak-level provenance is not checkpoint-recoverable without separately binding the original H5AD feature list and LSI generator.

## Sparse implementability of the proposed evidence

- Feature-bootstrap stability can be computed by resampling columns of CSR/CSC count matrices and recomputing low-rank/sparse statistics. It does not require a resident dense N-by-N matrix.
- Spatial reproducibility can reuse the deterministic sparse kNN support and chunked Moran-style sparse matrix products already implemented in Night-1/Night-6C.
- Linked-feature consistency can be sparse if represented as a gene-protein or peak-gene bipartite incidence matrix. That incidence matrix is present only partially for A1/tonsil/D1 and is absent for P22.
- Cross-slice reproducibility needs a registered slice boundary and comparable feature identifiers. Existing GSE198353 replicates can provide an RNA+protein discovery/reserve split, but no independent RNA+ATAC slice exists within the current audited scope.

## Objects a new implementation would have to add

The current code does not define the proposed three-way object. A future preregistered method would need: (1) a feature bootstrap operator and per-feature/per-component stability estimate; (2) a sparse linked-feature incidence matrix with provenance; (3) spatial reproducibility evidence for modality-specific residuals; (4) an instability/noise score that is separately observable rather than inferred from a private branch; (5) a decomposition constraint and identifiable null boundaries; and (6) atomic manifests/checkpoints that bind feature IDs, bootstrap draws, link-map SHA, genome build, spatial graph and replicate membership.

## Novelty boundary

The potentially defensible distinction is not another gate or a renamed shared/private loss. SpaMV already models shared/private representations with measurement and HSIC terms; SpaMode already uses invariant/variant representations and mixture-of-experts routing; CANDIES already uses a higher-quality modality to conditionally denoise a lower-quality one. QMF covers sample-wise quality confidence, while ECoLaF and CoRiM cover conflict-guided discounting/risk. A different claim would require three independently auditable observation classes: cross-replicate stable linked signal, spatially reproducible modality-specific signal, and bootstrap-unstable non-reproducible noise. The current assets cannot instantiate that claim for RNA+ATAC because genome build, peak-gene provenance and an independent family unit are missing.
