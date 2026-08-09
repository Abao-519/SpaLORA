# SpaLORA Night-3A-R P0A-R Hard Stop

**P0A-R: FAIL; P0B-R: NOT RUN (0/15); main experiment: 0/60; semantic label access: 0; IGE scientific go/no-go: NOT_EVALUATED; architecture ablation: NOT AUTHORIZED.**

`previous_night3a_status = ADMINISTRATIVE_HARD_STOP_SCIENTIFIC_GO_NO_GO_NOT_EVALUATED`. The old Night-3A failure report remains unchanged and is not reinterpreted as a scientific IGE failure.

## Hard-stop reason

Taskbook line 153 requires P0A-R to stop on any old/new preprocessing fingerprint mismatch. All source data SHA-256 values and the exact 60-cell dataset/variant/seed/ordinal order match. Spot order, selected-gene order, shapes, modality-2 model features, both spatial graphs, and the modality-2 feature graph also match on A1, placenta, and P22.

The only differences are the RNA PCA, its derived RNA feature graph, and consequently the combined model-input SHA. The frozen legacy helper constructs `sklearn.decomposition.PCA(n_components=n_comps)` without `random_state` or an explicit solver. At these matrix dimensions sklearn selects randomized SVD, so a fresh process cannot reproduce the old byte fingerprint unless the protocol is changed or a seed is searched. Both are forbidden in this task.

No seed was added or searched, no PCA solver/formula was changed, no tolerance was relaxed, and old Night-3A evidence was not overwritten.

## Required status fields

- Five-seed IGE-C0 and C1-C0 ARI/NMI: NOT AVAILABLE; the 60 runs were prohibited.
- Placenta C1 recovery: NOT EVALUATED.
- Spatial tradeoff: NOT EVALUATED.
- Scalar loss versus weighted-gradient influence: the revised implementation and state-neutral CPU/GPU tests exist, but the scientific gradient-share hard gate was not evaluated because P0A-R failed.
- Architecture ablation authorized: no.
- P0B-R 15-cell old-probe comparison: 0/15, not run.
- Semantic label access: none; no evaluator was started.
- Full test suite: 14 passed, 2 failed. The two expected failures are the P0A-R PASS/source-lock assertions; the pre-P0A implementation subset had 14/14 passed.
- Night-3A protected files: 198/198 match; Night-2C protected files: 913/913 match.

## Evidence

See `night3ar_p0a.json`, `night3ar_p0a_failure.json`, `p0a_randomized_pca_diagnosis.json`, `integrity_read_manifest.json`, `label_flow_audit.md`, `tests_full.log`, and `failure_index.json`. No result table or figure is fabricated for an experiment that did not run.
