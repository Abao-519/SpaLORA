# SpaLORA Night-6D final report

## Outcome

Terminal status: `NIGHT6D_D1_P22_BALANCED_CONFIRMED`.

The sole primary treatment was `G04_SP10_F10_EUC_UNION/H05_EQUAL3_AFFINITY_SPECTRAL` against the same-dataset, same-seed fresh `G00_SP18_F20_CORR_UNION/H00_FUSED_PCA20_MCLUST_EEE` reference. No method, graph, head, loss, seed, threshold, epoch, or checkpoint was selected in Night-6D.

- D1: mean ΔARI=+0.031700, ΔNMI=+0.028965, ΔQ=+0.030332; Q wins=10/10; exact p=0.00097656, Holm p=0.00195312; bootstrap ΔQ 95% CI=[+0.024706, +0.036867]; MATERIAL_ACCURACY_CONFIRMED
- P22: mean ΔARI=+0.016361, ΔNMI=+0.046475, ΔQ=+0.031418; Q wins=8/10; exact p=0.00878906, Holm p=0.00878906; bootstrap ΔQ 95% CI=[+0.012356, +0.049276]; MATERIAL_ACCURACY_CONFIRMED

D1 is the primary held-out within-study confirmation. P22 is a pre-locked cross-dataset confirmation but is not a pristine holdout because it participated in earlier Night-3B/Night-5D work. Even a positive result would not establish state of the art before fair external baselines.

## Spatial protection

- D1 gate failed: `False`; mean deltas neighbor=0.0033828284037839664, Moran=-0.007309446997984803, Geary=0.011528641820980217, boundary=-0.0033828284037839664.
- P22 gate failed: `False`; mean deltas neighbor=0.0847773501924134, Moran=0.07493612981086736, Geary=-0.07911778306312107, boundary=-0.0847773501924134.

## Execution and firewall integrity

- Three authority inputs and Night-6C 77/77 internal, 4/4 external, and 3/3 local-post evidence were verified before execution.
- D1 label-free RNA/ADT were built by low-level HDF5 field copying without deserializing original annotation values. The P22 Night-3AF deterministic cache was reused only after every file and canonical hash matched.
- Training 40/40; fresh-process checkpoint/six-view/H00 replay 40/40; terminal transforms 80/80. Formal seeds were 0-9 in the preregistered order.
- D1 and P22 labels were parsed together in one evaluator window only after total lock. There was no return to training, transformation, or clustering afterward.
- Scientific training budget 40/40; retries 0/8; total attempts 40/48; transforms 80/80; corrections 0/8.
- The fixed secondary 2x2 factorial diagnostics were computed only after the primary lock and did not alter method identity.

## Evidence map

Core numerical evidence is in `d1_p22_per_seed_metrics.csv`, `primary_confirmatory_tests.json`, `secondary_factorial_tests.json`, and `spatial_protection.json`. Execution evidence is in `locked_training_manifest.json`, `checkpoint_roundtrip_index.json`, `locked_transform_manifest.json`, `label_window_audit.json`, `resource_accounting.csv`, `failure_and_retry_audit.json`, `budget_and_access_audit.json`, and `tests_and_invariance_audit.json`. Large label-free inputs, caches, affinities, views, raw runs, and checkpoints remain under `/root/autodl-fs` and are protected by absolute paths, sizes, and SHA-256 manifests.
