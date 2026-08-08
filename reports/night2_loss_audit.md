# SpaLORA Night-2 Loss Causal Audit — P0 hard-stop report

Date: 2026-08-09 (Asia/Shanghai)  
Audience: SpaLORA planning assistant and project owner

## Executive conclusion

Night-2 did **not** proceed to the 60-run loss factorial. The preregistered P0 parity gate failed before training: all three datasets have RNA PCA differences above the declared sign-aligned tolerance, P22 also fails the declared scaled-RNA tolerance, and the RNA feature-graph edge set/normalized adjacency differs for A1 and P22. These are pre-loss differences, so a V3-versus-legacy comparison would not isolate RNA loss semantics.

This is a negative but valid audit outcome. Exactly 0/60 new factorial runs and 0/3 tutorial-2022 reproduction runs were launched. No seed was searched or dropped, no ground truth was used for preprocessing/training/tuning, and ASR was not modified.

## 1. Git and execution identity

- Parent: `fdecb33706ea1fe8429439813eef09e3ea931c86` (`fdecb33`), independently verified before work.
- Branch: `revision/q2-night2-loss-audit-20260808`.
- Final commit reference: annotated tag `night2-p0-final-20260809`, pointing to the commit containing this report. The concrete SHA is also stored in the persistent shutdown manifest and Git bundle verification log.
- Push status: **failed after the one required non-interactive attempt**. `origin` is HTTPS, no credential helper/token is configured, and Git reported that it could not read the GitHub username in a non-interactive session. No credential guessing was attempted.
- Persistence fallback: a verified Git bundle and SHA-256-manifested compact artifact archive were written under `/root/autodl-fs/night2_handoff_20260809/` before shutdown.
- Target environment: Python 3.8.10, NumPy 1.22.3, PyTorch 1.12.1+cu116 in the original `SpaLORA_torch112` environment.

## 2. P0 parity findings

Declared tolerances were `atol=rtol=1e-6` for ordinary arrays, `atol=1e-5` after PCA sign alignment, and `atol=1e-6` for controlled model forward outputs. Observation IDs and ordered HVG names were required to match exactly; graph edge sets were also required to match exactly. Sparse graph comparisons never densified an N×N matrix.

| Dataset | n | HVGs | IDs / HVG order | scaled RNA max abs diff | PCA max abs diff | RNA feature-edge mismatch (legacy-only / corrected-only) | edge Jaccard | normalized-adj max abs diff | P0 dataset result |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| A1 | 3,484 | 3,000 | exact / exact | 1.907e-6 (pass) | 1.742e-3 (fail) | 6 / 6 | 0.999893856 | 0.0387202 | fail |
| Placenta | 1,662 | 3,000 | exact / exact | 1.907e-6 (pass) | 1.875e-3 (fail) | 0 / 0 | 1.000000000 | 0 | fail |
| P22 | 9,196 | 2,000 | exact / exact | 3.815e-6 (fail) | 7.281e-3 (fail) | 16 / 18 | 0.999876057 | 0.0444554 | fail |

Additional findings:

- Modality-2 feature matrices passed for all datasets (maximum absolute differences: A1 0, placenta 2.375e-7, P22 1.579e-6 under the declared combined tolerance).
- Both modality spatial graphs, both normalized spatial adjacencies, and the modality-2 feature graph/adjacency were exactly equal for every dataset.
- The final authoritative audit captured the corrected pipeline's actually used PCA and graphs in-process. An initial diagnostic run recomputed randomized PCA and was therefore not used for the final conclusion; its instrumentation was corrected before producing `reports/night2_parity.json`.
- Ordered gene-name equality is the semantic check. Representation-sensitive array hashes may differ where NumPy string dtypes differ, so the report does not treat those hashes as a contrary biological result.

### Legacy softmax semantics

In PyTorch 1.12.1, a deterministic `[4,2]` input produced exact equality between legacy `torch.nn.functional.softmax(x)` and `torch.softmax(x, dim=1)`: maximum and mean absolute differences were both 0. This component passes.

### Controlled model-forward parity

On placenta, ten corresponding parameter tensors were copied explicitly between the frozen legacy and corrected explicit-attention models. All compared outputs passed `atol=1e-6`: both modality latents, combined latent, cross reconstructions, final reconstructions, and all three attention arrays. The largest observed maximum absolute difference was 7.153e-7. This localizes the P0 failure to preprocessing/feature construction rather than the corrected attention forward implementation.

### Diagnosis

The supported chain is: identical IDs and ordered HVGs → float32-scale RNA matrix differences → larger differences in randomized PCA coordinates → a few changed correlation-neighbor decisions in A1/P22 → non-identical normalized RNA feature adjacency. A plausible cause is the different scale/select operation order plus sparse/dense numerical paths, amplified by randomized PCA and near-tied neighbors. This is a hypothesis, not a fully proven root cause. Because exact feature/graph parity was preregistered, the difference is material for causal interpretation even though the edge Jaccards are very high.

## 3. Tests and failures

- Final target-machine suite: **23 passed, 0 failed, 1 dependency deprecation warning**, in 2.93 s.
- The tests cover the fixed `legacy_bug_weight_vector` mean at d=2000 and d=3000, V3 algebraic identity, V1 uniform shape, explicit attention row sums, sparse adjacency, mandatory loss-log checkpoints, label-access ordering, P0 authorization, stopped-run artifact completeness, and all 260 protected Night-1 checksums.
- One first-pass Night-2 test fixture failed because it paired a 3-gene error vector with a 3,000-gene weight vector. The fixture was corrected to 3,000 genes; no production semantics or tolerance was changed.
- Night-1 checksum result: **260/260 unchanged** relative to `/root/autodl-fs/night2_preexisting_20260809/night1_before.sha256`.

## 4. Frozen loss variants (defined, not run)

Let `MSE_g` be per-gene RNA reconstruction MSE, `w_bad` the exact legacy argsort-bug vector, and `m_bad=mean(w_bad)`.

- V0 `corrected_unweighted`: `mean_g(MSE_g)`; reused only as the Night-1 reference.
- V1 `uniform_legacy_scale`: `m_bad * mean_g(MSE_g)`; uniform gene shape, global-scale test only.
- V2 `legacy_shape_normalized`: `sum_g(w_bad_g*MSE_g) / sum_g(w_bad_g)`; relative legacy shape at unit global scale.
- V3 `legacy_loss_replay`: `mean_g(w_bad_g*MSE_g)`; shape and scale together. Unit tests assert equality to `m_bad * V2`.
- V4 `asr_hvg_legacy_scale_diagnostic`: `m_bad * sum_g(w_asr_g*MSE_g)/sum_g(w_asr_g)`; causal diagnostic only, never a proposed final method.

Mechanical tests reproduce `m_bad=2.290322960` for 2,000 genes and `m_bad=2.289938091` for 3,000 genes within 1e-6. The code consistently calls this vector `legacy_bug_weight_vector`; it is not described as a valid low-expression score.

## 5. Main mean ± sample-SD table

Only immutable Night-1 `legacy_exact` and V0 reference rows are shown. There are no V1–V4 rows because P0 blocked them; `results/night2/summary.csv` carries six reference rows marked `source=night1_fdecb33_reference` and `run_status=existing_reference`.

| Dataset | Variant | ARI | NMI | AMI | FMI | Hungarian macro-F1 | spatial agreement | embedding silhouette | total seconds | GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | legacy_exact | 0.2304 ± 0.0164 | 0.3644 ± 0.0107 | 0.3604 ± 0.0107 | 0.3737 ± 0.0141 | 0.3307 ± 0.0190 | 0.5993 ± 0.0128 | 0.1098 ± 0.0164 | 19.2241 ± 1.9524 | 361.2479 ± 0.0010 |
| A1 | corrected_unweighted | 0.2269 ± 0.0217 | 0.3645 ± 0.0132 | 0.3605 ± 0.0133 | 0.3689 ± 0.0204 | 0.3234 ± 0.0214 | 0.5965 ± 0.0218 | 0.1101 ± 0.0136 | 18.3716 ± 1.2021 | 417.6599 ± 0.3208 |
| Placenta | legacy_exact | 0.6635 ± 0.0323 | 0.7238 ± 0.0179 | 0.7202 ± 0.0181 | 0.7233 ± 0.0267 | 0.6114 ± 0.0212 | 0.4400 ± 0.0086 | 0.1677 ± 0.0161 | 12.7067 ± 0.4016 | 197.9956 ± 0.0010 |
| Placenta | corrected_unweighted | 0.4506 ± 0.0415 | 0.5289 ± 0.0295 | 0.5226 ± 0.0299 | 0.5428 ± 0.0354 | 0.5020 ± 0.0425 | 0.4913 ± 0.0196 | 0.0410 ± 0.0036 | 14.2757 ± 1.6846 | 255.0918 ± 0.0010 |
| P22 | legacy_exact | 0.4144 ± 0.0211 | 0.5538 ± 0.0152 | 0.5531 ± 0.0153 | 0.5066 ± 0.0181 | 0.5340 ± 0.0110 | 0.8262 ± 0.0176 | 0.1416 ± 0.0087 | 72.5928 ± 0.9146 | 723.2081 ± 0.0040 |
| P22 | corrected_unweighted | 0.4138 ± 0.0185 | 0.5527 ± 0.0136 | 0.5519 ± 0.0136 | 0.5065 ± 0.0161 | 0.5330 ± 0.0111 | 0.8263 ± 0.0177 | 0.1453 ± 0.0092 | 66.7334 ± 0.8787 | 763.0175 ± 0.0032 |

## 6. Paired deltas

No V1–V4 embedding, clustering, or label evaluation exists, so paired same-seed deltas versus V0 are **not estimable**. `paired_deltas.csv` is an intentionally header-only stopped-run artifact; no values were imputed from Night-1 and no favorable seed was selected.

## 7. Placenta 2×2 recovery and interaction

The fixed Night-1 reference gap is `0.6635269855 - 0.4506355690 = 0.2128914164` mean ARI.

| Quantity | Value | Status |
|---|---:|---|
| V0 corrected-unweighted mean ARI | 0.4506355690 | Night-1 reference |
| V1 scale recovery | NR | P0 blocked V1 |
| V2 shape recovery | NR | P0 blocked V2 |
| V3 replay recovery | NR | P0 blocked V3 |
| ARI interaction `V3 - V2 - V1 + V0` | NR | P0 blocked factorial |
| NMI interaction | NR | P0 blocked factorial |

No scale-dominant, shape-dominant, or interaction conclusion is permitted from this run.

## 8. Loss dynamics

No optimizer step was executed after P0 failed, so no V1–V4 loss trajectory exists. `loss_components.csv` preserves the preregistered schema and is intentionally header-only. The reusable recorder requires epochs `[0,20,100,199]` for a 200-epoch run and rejects incomplete records; that behavior is unit-tested for the future authorized rerun.

## 9. Attention findings

No V1/V2/V3 attention changes were produced. The fixed Night-1 cross-omics RNA-attention references remain:

| Dataset | legacy_exact | corrected_unweighted | legacy − corrected |
|---|---:|---:|---:|
| A1 | 0.51552 | 0.51512 | +0.00040 |
| Placenta | 0.69608 | 0.56778 | +0.12830 |
| P22 | 0.62299 | 0.59903 | +0.02396 |

These values motivate the scale hypothesis but do not establish causality. `attention_summary.csv` is header-only for the blocked new variants.

## 10. Manuscript reproduction audit

| Dataset | manuscript ARI | manuscript NMI | Night-1 legacy ARI | Night-1 legacy NMI | tutorial seed-2022 ARI/NMI |
|---|---:|---:|---:|---:|---|
| A1 | 0.2443 | 0.3780 | 0.2303937 | 0.3644419 | not run — P0 hard stop |
| Placenta | 0.7226 | 0.7408 | 0.6635270 | 0.7238141 | not run — P0 hard stop |
| P22 | 0.4541 | 0.5747 | 0.4143791 | 0.5538464 | not run — P0 hard stop |

The user explicitly required “only diagnose, save, commit, and shut down” after a P0 failure, so three additional tutorial training runs would have violated the stop rule. `paper_repro_audit.csv` records the immutable manuscript and Night-1 values, blank tutorial fields, and the stop reason. No arbitrary seed search occurred.

## 11. Placenta modality-2 audit

The exact uploaded matrix is `1662 × 63`, CSR, float64, nonnegative, with min 0, max 75.26, mean 0.7832355, SD 1.9312518, zero fraction 0.5028652, negative fraction 0, and integer-like fraction 0.5032281. It contains 63 TF-symbol feature names. `.var['feature_types']` says `Gene Expression`, with `gene_ids`, `gene_name`, and `genome=GRCh38`; this generic label does not prove raw gene counts.

There are no layers, no `.raw`, no `.varm`, and no `.uns`; `.obsm` contains only `spatial (1662×2)`. Neither names nor stored metadata contain `ChromVAR`, motif, deviation, enrichment, activity, gene activity, PCA, or LSI provenance keywords. The 62-dimensional modality-2 model input in the parity audit is the PCA representation of the 63 source features, not evidence that the source has only 62 columns.

Required conservative description: **ATAC-derived / TF-associated regulatory features**. It must not be called “63 raw ATAC peaks.” CLR is numerically defined because values are nonnegative, but the matrix is continuous and only about half integer-like. Without exact provenance showing compositional raw counts, CLR appropriateness is unproven; no preprocessing was changed based on ARI.

## 12. Decision gates

| Gate | Status | Interpretation |
|---|---|---|
| P0 parity | **FIRED / FAIL** | Hard stop before scientific runs |
| P1 legacy replay | not evaluated | V3 was not run |
| S scale-dominant | not evaluated | V1 was not run |
| W legacy-shape dominant | not evaluated | V2 was not run |
| I shape×scale interaction | not evaluated | factorial was not run |
| A ASR scale diagnostic | not evaluated | V4 was not run; ASR unchanged |

## 13. Recommendation to the planning assistant

Do not attribute the placenta regression to global RNA-loss scale yet. First create a label-blind parity revision that makes the non-rescue corrected data path semantically identical to frozen legacy preprocessing while keeping explicit attention and safe label access. Focus on scale/select operation order, sparse-versus-dense arithmetic, deterministic PCA capture, and correlation-neighbor tie behavior. Rerun **P0 only** on all three datasets. If and only if P0 passes under preregistered tolerances, run the unchanged V1–V4, three-dataset, five-seed matrix. Keep `m_bad` mechanical, preserve seeds `[0,1,2,3,4]`, and do not change ASR or inspect labels during this work.

## 14. Files changed or added

- `.gitignore` — ignore Night-2 raw/cache/log/lock artifacts.
- `configs/night2_loss_audit.json` — frozen Night-2 specification and tolerances.
- `SpaLORA/night2_loss_audit.py` — loss decomposition and mandatory logging helpers.
- `scripts/night2_parity_audit.py` — sparse three-dataset P0 and controlled forward audit.
- `scripts/night2_loss_audit.py` — hard-gate finalizer; writes stopped-run artifacts and refuses authorized/full-run mode.
- `scripts/night2_placenta_mod2_audit.py` — exact uploaded-file numerical/provenance audit.
- `tests/test_night2.py` — Night-2 semantics, safety, stopped-run, and Night-1 integrity tests.
- `reports/night2_parity.json`, `reports/night2_p0_diagnosis.json`, `reports/placenta_mod2_audit.json`, `reports/night2_tests.txt`, and this report.
- `results/night2/per_seed_metrics.csv`, `summary.csv`, `paired_deltas.csv`, `factorial_effects.csv`, `loss_components.csv`, `paper_repro_audit.csv`, `attention_summary.csv`, `per_domain_f1.csv`, `p0_parity_summary.csv`, and `gate_status.json`.

The heavy `results/night2/raw`, cache, and logs remain ignored. Compact reference CSVs and reports are committed. All empty factorial-result CSVs are explicitly marked by `gate_status.json` as intentional consequences of the P0 stop, not missing successful runs.
