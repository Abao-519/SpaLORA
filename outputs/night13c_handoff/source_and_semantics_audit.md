# Night-13C source and semantics audit

## Locked Night-13B interpretation

Night-13B B10 is a deterministic residual applied after standardized concatenation and PCA. The formal path does not execute an optimizer step, the stored threshold offset remains zero, and the registered model seed is not consumed. The 35 rows therefore represent seven deterministic embeddings/partitions copied across five unused seed labels. All 15 registered IDs share the same equation and differ only in threshold, slope, and maximum residual; they are one mechanism family.

## Night-13C implementation boundary

The new module accepts only two numeric modality tensors and a sparse edge list. It has no dataset, tissue, assay-family, path, label, or evaluation-metric argument. Different RNA/protein/ATAC input widths are handled by ordinary learned input adapters; both modality streams then use the same tied latent core, fusion rule, loss components, optimizer schedule, and endpoint protocol.

Three mechanism families are implemented clean-room: symmetric trainable edge reliability, modality-dropout cross-reconstruction, and a shared/private covariance-controlled comparator. The first learns continuous reliability from symmetric edge support in both modalities. The second does not use an edge gate and must reconstruct both modalities when one is dropped. The third is deliberately retained as a high-collision comparator because shared/private reconstruction is already crowded by SpaMV/SpaMode-like work.

No external implementation was copied. SpatialGlue is AGPL-3.0, SMART is GPL-3.0, SpaBalance is AGPL-3.0, and the audited ARISE/SpaMV/SpaMode snapshots had no detected project license in the Night-13B source ledger; these remain mechanism context only. B10 remains an auxiliary head/ablation regardless of its endpoint robustness.

## Evidence firewall

Public benchmark labels are loaded only by the evaluator after the training trajectory and final-step checkpoint are fixed. They may rank different completed candidate/config runs. They do not enter the model API, unsupervised loss, gradient, optimizer, or within-run checkpoint selection. All spatial operations remain sparse edge lists or sparse row-stochastic operators; no dense observation-by-observation matrix is constructed.
