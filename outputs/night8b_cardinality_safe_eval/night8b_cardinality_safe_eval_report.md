# SpaLORA Night-8B Cardinality-Safe Evaluation Recovery Report

## Terminal status

`NIGHT8B_CARDINALITY_SAFE_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST`

This was a post-lock evaluation recovery, not a pristine holdout and not a SOTA benchmark. The 20 fixed `RECOVERY_EIGEN_KMEANS100` partitions remained byte-identical; no training, checkpoint load, forward, adapter, affinity rebuild, or head transform occurred.

## Reference-label contract

- Actual reference K: **7**
- Fixed predicted K for both methods: **12**
- Different reference/predicted cardinalities are valid for ARI and contingency-table information metrics; both methods use the same fixed predicted K.
- Raw Y was read once in this task, the second and final read in the Night-8B lineage. Its per-spot vector was not persisted.

## Uniform-head primary result (HR_F00 - HR_U00)

- mean delta ARI: `0.0137421955988`
- mean delta NMI: `0.0209818647596`
- mean delta Q: `0.0173620301792`
- Q wins: `10/10`
- exact one-sided sign-flip p: `0.0009765625`
- paired bootstrap delta Q 95% CI: `[0.012135163091, 0.0232967838974]`

Science-core gate: `True`; spatial protection: `True`; resource gate: `False`.

## Sensitivity and audit

- Original spectral complete-pair sensitivity: 9 pairs; mean delta Q `0.0161860836997`; direction agrees with uniform head: `True`. It was descriptive only and did not enter the terminal decision.
- Same-process independent recomputation maximum absolute error: `1.78e-15` (required <=1e-12).
- Original raw root: 297/297 unchanged. Head-recovery root: unchanged.
- Evaluation runtime: `6.794` seconds; evaluation GPU use: 0.

## Interpretation boundary

The result tests the frozen RNA+ATAC family policy on MISAR under a fixed K=12 uniform head and the actual reference-K labels. It cannot establish the original H05 endpoint, a pristine external holdout, or SOTA. Seeds measure algorithmic stability rather than independent biological replication.

## Git and delivery

Branch: `revision/q2-night8b-cardinality-safe-eval-20260820`. Planned immutable final tag: `night8b-cardinality-safe-eval-final-20260820`. The exact final commit, tag verification, bundle, compact index, and shutdown dispatch receipt are external post-commit delivery records to avoid self-reference.
