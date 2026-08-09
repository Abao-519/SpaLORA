# SpaLORA Night-2B Parity-Locked Loss Audit — P0B hard-stop report

Date: 2026-08-09 (Asia/Shanghai)  
Audience: SpaLORA planning assistant and project owner

## Executive conclusion

Night-2B stopped at the new preregistered P0B gate. All three datasets proved exact consumed-input identity, exact initial model-state identity, exact V3 legacy-loss identity, and gradients/Adam state within `1e-7`. However, independently executed frozen-model GPU forwards differed by `3.576e-7–4.768e-7`, above the immutable `1e-7` threshold. Placenta and P22 then amplified sub-threshold gradient differences through Adam into one-step parameter differences of `2.898e-6` and `6.324e-7`; their five-step maximum parameter differences were `4.340e-6` and `1.753e-6`.

The hard gate therefore authorized **0/75 factorial runs and 0/3 tutorial-2022 runs**. No ground truth was accessed by P0B/diagnosis, no seed was searched, no ASR setting was changed, and no frozen legacy module was modified.

## 1. Identity, branch, environment, and preservation

- Exact parent: `16f0cc43673617c73527110962b7ca115c59b4c6`.
- Branch: `revision/q2-night2b-parity-locked-loss-audit-20260809`.
- Harness milestone commit: `f7717429a5e42f6afa4e4c163764309548a26d90`.
- Final commit reference: annotated tag `night2b-p0b-final-20260809`, pointing to the commit containing this report and all compact stopped-run artifacts.
- Runtime: Python 3.8.10, PyTorch 1.12.1+cu116, NumPy 1.22.3, SciPy 1.8.1, scikit-learn 1.1.1, NVIDIA GeForce RTX 4080 SUPER, device `cuda:0`.
- Push status: **failed after exactly one non-interactive branch push attempt** because the HTTPS remote has no configured GitHub username/token and terminal prompts were disabled. No credential guessing or retry was performed.
- Persistence fallback: verified complete Git bundle plus SHA-256-manifested compact/raw artifact archive under `/root/autodl-fs/night2b_handoff_20260809/`, with a second verified copy in the user's established local handoff path.

The pre-run manifest contains 310 repository files. The only preexisting non-Night-2B file intentionally changed is `.gitignore`, solely to add Night-2B raw/cache/log exclusions. The other 309 files match their pre-run SHA-256 values. In particular:

- `reports/night2_loss_audit.md`: `053881bfcc4fafe576578a23b816a004a99b094eebd26cbf14d77ecad93206f8`;
- `reports/night2_parity.json`: `5888b53e3b01e5d5c816c2f79eaaa5709c3761edb8b73e1882d6c8fd3cfd8356`;
- frozen `SpaLORA/model.py`: `0eabca19ae6cec2b3da8a8fbeb811546db0f6ce199c4485705279bf88206c8ea`;
- frozen `SpaLORA/preprocess.py`: `359ddbbc57fcb5704a0de60c83d68aced0ad2df351f383d82d2cb58dc385cb97`;
- frozen `SpaLORA/SpaLORA_pyG.py`: `8c250e4761b21af1166c2afe749a461da447056ea52b6f88fdbb07c9a7ccadb7`.

Night-2's original P0 failure remains intact and is not reinterpreted or bypassed.

## 2. P0B checks and authorization

P0B used seed 0 separately for A1, placenta, and P22. Each dataset called the frozen Night-1 `prepare_legacy` path once, then instantiated a frozen `Train_SpaLORA` reference and the generalized Night-2B trainer from the same prepared AnnData objects. The generalized trainer independently invoked the same legacy container constructor so the audit compared what both trainers actually consumed rather than relying on helper sharing. No label loader was imported or called by the audit path.

Declared hard tolerance for forward, loss, gradient, parameter, and optimizer-state comparison: `atol=rtol=1e-7`. Consumed input and initial state required exact equality.

| Dataset | consumed input | initial state | max forward diff | max loss diff | max gradient diff | one-step parameter diff | max five-step parameter diff | P0B |
|---|---|---|---:|---:|---:|---:|---:|---|
| A1 | exact | exact | 4.768e-7 | 0 | 2.980e-8 | 3.912e-8 | 3.912e-8 | fail — forward |
| Placenta | exact | exact | 4.768e-7 | 0 | 1.490e-8 | 2.898e-6 | 4.340e-6 | fail |
| P22 | exact | exact | 3.576e-7 | 0 | 2.794e-9 | 6.324e-7 | 1.753e-6 | fail |

Additional exact checks for every dataset:

- RNA and modality-2 tensor dtype, shape, value and hash;
- all four coalesced sparse adjacency shapes, indices, values, nnz and hashes;
- ordered HVG names and observation IDs;
- dataset loss factors, epochs, embedding dimension, Adam `lr=0.0001`, and `weight_decay=0`;
- the full legacy bug-weight vector and its mechanical mean (`2.289938211` for 3,000 genes; `2.290322781` for 2,000 genes, both within `1e-6` of preregistration);
- parameter names, shapes and initial values;
- all four raw loss components and total V3 loss (maximum difference exactly 0);
- Adam optimizer states within the hard tolerance.

Authorization result: `p0b_pass=false`, `factorial_authorized=false`.

## 3. Focused diagnosis

The failed gate was not rerun with relaxed tolerances. A separate label-free diagnosis asked whether the same frozen model, with unchanged parameters and exact same input tensors, produces bit-identical outputs when called twice on the same device.

| Dataset | same-model repeated GPU forward max diff | same-model repeated CPU forward max diff |
|---|---:|---:|
| A1 | 4.768e-7 | 0 |
| Placenta | 4.768e-7 | 0 |
| P22 | 2.980e-7 | 0 |

This is strong evidence that CUDA sparse execution/reduction order is numerically sensitive in the exact legacy PyTorch 1.12.1 path. The losses remain exact because the observed output perturbations do not change their float32 reductions in this step; tiny gradient differences can nevertheless be amplified by Adam's normalization for parameters with very small second moments. This diagnosis explains the pattern but **does not undo P0B**, authorize training, or change the threshold.

## 4. Authorized-run counts and failures

| Run family | Planned if authorized | Completed | Failed after launch | Skipped by gate |
|---|---:|---:|---:|---:|
| Five-variant factorial | 75 | 0 | 0 | 75 |
| Public tutorial seed 2022 | 3 | 0 | 0 | 3 |

There are no `results/night2b/raw/*/*/seed_*/metrics.json` files and no tutorial metrics. Required compact CSV paths exist with explicit stopped-run schemas; factorial-dependent files are intentionally header-only and `gate_status.json` records why.

## 5. Exact frozen variant definitions (implemented, not run)

For per-gene `MSE_g`, exact legacy argsort-bug vector `w_bad`, and `m_bad=mean(w_bad)`:

- V0 `locked_unweighted`: `mean_g(MSE_g)`.
- V1 `locked_uniform_legacy_scale`: `m_bad * mean_g(MSE_g)`; uniform gene shape, scale only.
- V2 `locked_legacy_shape_normalized`: `sum_g(w_bad_g*MSE_g) / sum_g(w_bad_g)`; accidental shape at unit scale.
- V3 `locked_legacy_loss_replay`: `mean((diff**2) * w_bad)` using the frozen reduction order. Unit and P0B checks assert `V3 ≈ m_bad * V2`; P0B loss equality to frozen legacy was exact.
- V4 `locked_asr_hvg_legacy_scale_diagnostic`: `m_bad * sum_g(w_asr_g*MSE_g)/sum_g(w_asr_g)`. ASR-v1 is computed label-free from immutable raw RNA counts and the frozen spatial graph, then guarded by exact selected-name/order alignment. It remains diagnostic only.

The code consistently calls `w_bad` `legacy_bug_weight_vector`; it is not described as a biologically valid expression weight.

## 6. P1 V3 full replay

Not evaluated. P1 requires completed V3 runs, which P0B prohibited. `v3_replay_audit.csv` is intentionally header-only. No Night-1 cluster/attention artifact was reconstructed from labels.

## 7. Mean±SD and paired deltas

Not estimable because no authorized factorial cell was run. `per_seed_metrics.csv`, `summary.csv`, and `paired_deltas.csv` contain schemas but no rows. No Night-1 value was copied into a Night-2B cell, no missing result was imputed, and no best seed was reported.

## 8. Placenta recovery and interaction

The fixed external Night-1 legacy mean ARI remains `0.6635269855`, but the new locked V0 does not exist. Therefore the legacy–V0 gap, scale recovery, shape recovery, replay recovery, ARI/NMI interaction, and S/W/I conclusions are all **not estimable**. `factorial_effects.csv` is header-only.

## 9. Loss and attention interpretation

P0B establishes only that V3's initial loss formula exactly replays frozen legacy on exact inputs. It does not produce an optimization trajectory eligible for scientific interpretation. No main-run loss checkpoints or final attention arrays exist, so `loss_components.csv` and `attention_summary.csv` are intentionally header-only.

The fixed Night-1 placenta attention reference (`legacy_exact≈0.69608`) is not compared with any new V0/V1/V3 value. The intended scale mechanism remains untested.

## 10. Per-domain diagnostics

No cluster assignments or labels were produced by an authorized Night-2B run; `per_domain_f1.csv` is header-only. Consequently there is no rare-domain claim, positive or negative.

## 11. Tutorial-2022 reproducibility audit

| Dataset | manuscript ARI / NMI | Night-1 legacy mean ARI / NMI | tutorial-2022 |
|---|---|---|---|
| A1 | 0.2443 / 0.3780 | 0.2303937 / 0.3644419 | not run — P0B stop |
| Placenta | 0.7226 / 0.7408 | 0.6635270 / 0.7238141 | not run — P0B stop |
| P22 | 0.4541 / 0.5747 | 0.4143791 / 0.5538464 | not run — P0B stop |

`paper_repro_audit.csv` preserves these fixed values and blank tutorial fields. No alternative model seed was searched.

## 12. Placenta modality-2 status

Night-2's unchanged audit conclusion is reused: source matrix `1662×63`, CSR float64, nonnegative continuous; generic `feature_types='Gene Expression'` does not prove counts; no stored ChromVAR/motif/deviation/activity provenance. Required wording remains **ATAC-derived / TF-associated regulatory features**. It is not called 63 raw ATAC peaks, and no CLR or label-guided preprocessing experiment was run.

## 13. Decision gates

| Gate | Status | Reason |
|---|---|---|
| P0B | **FIRED / FAIL** | independent GPU forward exceeded `1e-7`; placenta/P22 update trajectory also exceeded |
| P1 | not evaluated | V3 full runs prohibited |
| S | not evaluated | V0/V1 absent |
| W | not evaluated | V2 absent |
| I | not evaluated | factorial absent |
| A | not evaluated | V4 absent and remains diagnostic only |

## 14. Recommendation to the planning assistant

Do not interpret the placenta regression as a loss-scale or loss-shape effect from Night-2B. The loss dispatcher itself passed the strongest local check—exact V3 loss on exact consumed inputs—but the taskbook's independent-GPU-trajectory instrument is stricter than the deterministic behavior delivered by the frozen CUDA sparse path.

For a future, separately preregistered task, choose the causal estimand and deterministic instrument explicitly before running labels:

1. If the estimand is **loss-formula equivalence on one canonical forward graph**, compare legacy and generalized loss dispatch/backprop on the same saved forward tensors/computational graph; this removes duplicate sparse-forward nondeterminism but is a new gate, not a retroactive P0B pass.
2. If independent trajectory identity is mandatory, evaluate a deterministic CPU audit or a documented deterministic sparse backend first, then decide whether its runtime/environment remains scientifically comparable to the target GPU runs.
3. Keep the current `1e-7` P0B result immutable. Do not simply relax it to `1e-6` after observing the failure.

Only after a new gate passes should the unchanged 75+3 design be reconsidered. Preserve seeds `[0,1,2,3,4]`, the exact legacy preparation/model/optimizer, and frozen ASR-v1.

## 15. Tests, changed files, and persistence

Final target-machine suite: **34 passed, 0 failed, 3 warnings** (one Jupyter path deprecation and the two expected frozen legacy tanh/implicit-softmax warnings). Tests cover Night-1, Night-2, and Night-2B; protected checksums, P0B hard authorization, exact consumed input/state checks, V3 loss/gradient/Adam/trajectory evidence, loss identities, V4 order guard, label-access ordering, stopped-run completeness, sparse behavior, attention sums, and required compact outputs.

New/changed files:

- `.gitignore` — Night-2B raw/tutorial/cache/log exclusions only.
- `configs/night2b_parity_locked_loss_audit.json`.
- `SpaLORA/night2b_loss_audit.py`.
- `scripts/night2b_parity_locked_audit.py`.
- `scripts/night2b_loss_audit.py` (authorized-only runner, never launched).
- `scripts/night2b_p0b_diagnose.py`.
- `scripts/night2b_finalize_p0b.py`.
- `tests/test_night2b.py`.
- `reports/night2b_parity_locked.json`, `night2b_p0b_diagnosis.json`, `night2b_tests.txt`, `night2b_completion.json`, and this report.
- `results/night2b/gate_status.json`, `p0b_summary.csv`, and all required stopped-run compact CSVs.

Heavy audit logs are ignored but included in the verified persistent artifact archive. The completion sidecar records the concrete final commit, tag, one push failure, checksums, archive verification, protected-file result, and required final shutdown command.
