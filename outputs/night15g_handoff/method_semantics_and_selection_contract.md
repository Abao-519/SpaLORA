# Night-15G method semantics and selection contract

- Public benchmark labels define known K, cross-run HPO/profile selection and final metrics.
- Labels do not enter patch extraction, image features, molecular features, optional-view reliability, unary/pairwise energy, GMM/KMeans fitting or move acceptance.
- A1 and tonsil s3 headline rows use the optional-morphology energy from the Night-15F authority partition.
- D1 headline reporting uses the no-microcluster morphology/coordinate head profile selected from the preserved 3432-row partial ledger. The unconstrained maximum and morphology-seeded Stage2 profile are reported separately because they contain singleton clusters.
- D1 Stage2 is a morphology-seeded initialization followed by molecular Night-15F energy; morphology is not an active input in Stage2.
- Missing morphology is represented by a zero presence mask and returns the supplied Night-15F partition exactly; this is a formula-level fallback, not dataset-name routing.
- Dataset/platform-specific numeric HPO is transparent and allowed; the evidence is public-benchmark development, not blind evaluation.
- Deterministic embedding artifacts are registered once. No model-seed replication claim is made.
- Current frozen partitions and metrics replay byte-exactly; full algorithmic recomputation across Windows BLAS schedules is not byte-exact and is recorded as a limitation.
