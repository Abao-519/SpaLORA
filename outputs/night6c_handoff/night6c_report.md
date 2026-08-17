# SpaLORA Night-6C final report

## Outcome

Terminal status: `NIGHT6C_BALANCED_AND_OR_ACCURACY_CANDIDATES_LOCKED`.

Night-6C trained a new, paired C04/B01 reference and every preregistered R1 graph cell on A1 and the firewall-clean tonsil copy. It did not replay or synthesize nonexistent historical checkpoints. All downstream comparisons use the same Night-6C dataset/seed `G00/H00` reference.

Balanced candidate: `G04_SP10_F10_EUC_UNION/H05_EQUAL3_AFFINITY_SPECTRAL`.

Accuracy-frontier candidate: `G04_SP10_F10_EUC_UNION/H05_EQUAL3_AFFINITY_SPECTRAL`. Any spatial trade-off designation is retained in `night6c_decision.json` and `balanced_and_accuracy_frontiers.csv`.

## Execution integrity

- P0 authority, Night-6B 31/31 root-aware evidence verification, ontology, and zero-obs label-free data checks passed.
- C04/B01 training semantics were uniquely reconstructed from the Night-5 registry, runner, and five manifests before science began.
- R1 training: 36/36; R1 head attempts: 432/432 (395 successful, 37 preregistered numerical failures, no fallback).
- R2 training: 12/12; R2 head attempts: 48/48 (48 successful, 0 preregistered numerical failures, no fallback).
- Every one of 48 successful training cells saved a real `model_final.pt`, canonical tensor-state SHA, full provenance, six views, and passed a fresh-process reload with exact H00 cluster labels.
- Scientific training used 48/66 units; implementation/infrastructure retries 2/12; total training attempts 50/78; head transforms 480/552; corrections 0/48. Failed attempts remain preserved and were not mixed into scientific evidence.

## Scientific controls

Labels were unavailable to trainer and transformer processes. R1 and R2 labels were opened only after the corresponding transform manifest was totally locked. A1 and tonsil used the same graph/head rules; seeds 0-4, fixed final epochs, thresholds, and budgets were unchanged. D1, P22, GSE198353, Night-4B, Night-5D metric content, and Night-6A raw/metric evidence remained sealed. Historical Night-5 A1 values appear only in `historical_night5_drift_diagnostic.csv` and were not a parity gate or selection input.

## Evidence map

The principal evidence is in `per_seed_metrics.csv`, `r1_decision.json`, `r2_decision.json`, `graph_head_five_seed_summary.csv`, `balanced_and_accuracy_frontiers.csv`, `checkpoint_roundtrip_index.json`, `raw_artifact_manifest.csv`, `historical_g00_drift_diagnostic.csv`, `tests_and_invariance_audit.json`, and `budget_and_access_audit.json`. Large checkpoints, views, graph caches, affinities, and raw runs remain under `/root/autodl-fs` and are protected by absolute paths, sizes, and SHA-256 values.
