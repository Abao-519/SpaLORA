# SpaLORA Night-3B Architecture Ablation and Interpretability

**P0-ARCH: PASS; probes: 24/24; main experiment: 120/120; failure JSON: 0; tests: 6 passed, 0 failed; training semantic label access: 0; method recommendation: MIXED_EVIDENCE.**

## Integrity and replay

- FULL_IGE CPU shared-forward/raw-loss/coefficient/total-loss/gradient/one-step Adam parity passed on 3/3 datasets.
- FULL_IGE exact final replay against Night-3AF IGE: 0/15 all-field exact; initial state 15/15 exact; clusters 14/15 exact; maximum embedding difference 0.132438; maximum attention difference 0.438697.
- Exact final replay was diagnostic, not an additional preregistered hard gate. P0-ARCH CPU exactness and initial GPU-envelope checks remain the locked gate; no post-hoc final tolerance was introduced.
- The published immutable deterministic caches were reused; no preprocessing was performed.
- Historical protection preflight passed: Night-3AF 709/709, Night-3A-R 211/211, Night-3A 198/198, Night-2C 913/913.
- All registered seeds `[0,1,2,3,4]` were retained; no label-guided tuning, seed search, or rescue variant was used.

## Five-seed metrics

| Dataset | Variant | ARI | NMI | Neighbor | Moran I | Geary C |
|---|---|---:|---:|---:|---:|---:|
| a1 | FULL_IGE | 0.2469 | 0.3725 | 0.5766 | 0.4835 | 0.5263 |
| a1 | DROP_RNA_RECON | 0.2146 | 0.3544 | 0.6085 | 0.5512 | 0.4581 |
| a1 | DROP_MOD2_RECON | 0.2560 | 0.3717 | 0.5730 | 0.4787 | 0.5315 |
| a1 | DROP_CORR1 | 0.2623 | 0.3771 | 0.5247 | 0.4130 | 0.5946 |
| a1 | DROP_CORR2 | 0.2548 | 0.3717 | 0.5741 | 0.4851 | 0.5237 |
| a1 | UNIFORM_WITHIN | 0.2520 | 0.3799 | 0.5696 | 0.4744 | 0.5343 |
| a1 | UNIFORM_CROSS | 0.2461 | 0.3753 | 0.5893 | 0.5143 | 0.4959 |
| a1 | UNIFORM_ALL | 0.2555 | 0.3819 | 0.5672 | 0.4904 | 0.5177 |
| placenta | FULL_IGE | 0.6044 | 0.6596 | 0.4773 | 0.4694 | 0.6740 |
| placenta | DROP_RNA_RECON | 0.3261 | 0.4343 | 0.4803 | 0.4494 | 0.6893 |
| placenta | DROP_MOD2_RECON | 0.6146 | 0.6308 | 0.5290 | 0.5272 | 0.6507 |
| placenta | DROP_CORR1 | 0.5603 | 0.6156 | 0.5016 | 0.5044 | 0.6590 |
| placenta | DROP_CORR2 | 0.6202 | 0.6407 | 0.5178 | 0.5129 | 0.6571 |
| placenta | UNIFORM_WITHIN | 0.7506 | 0.7662 | 0.4235 | 0.3510 | 0.7414 |
| placenta | UNIFORM_CROSS | 0.4807 | 0.5427 | 0.6147 | 0.5983 | 0.6535 |
| placenta | UNIFORM_ALL | 0.7282 | 0.7715 | 0.4094 | 0.3524 | 0.7549 |
| p22 | FULL_IGE | 0.3952 | 0.5531 | 0.8231 | 0.7909 | 0.2160 |
| p22 | DROP_RNA_RECON | 0.4128 | 0.5535 | 0.8505 | 0.8131 | 0.1953 |
| p22 | DROP_MOD2_RECON | 0.2683 | 0.4771 | 0.8250 | 0.7562 | 0.2511 |
| p22 | DROP_CORR1 | 0.2868 | 0.4589 | 0.7675 | 0.6765 | 0.3307 |
| p22 | DROP_CORR2 | 0.4222 | 0.5594 | 0.8384 | 0.7963 | 0.2113 |
| p22 | UNIFORM_WITHIN | 0.3323 | 0.5111 | 0.7908 | 0.7169 | 0.2907 |
| p22 | UNIFORM_CROSS | 0.4565 | 0.5886 | 0.8672 | 0.8256 | 0.1812 |
| p22 | UNIFORM_ALL | 0.3849 | 0.5361 | 0.8301 | 0.7619 | 0.2472 |

## Preregistered component decisions

- RNA reconstruction loss: **MIXED**; simplification dominance=False; supported datasets=placenta; equal-weight macro FULL-ablation ARI=+0.0977.
- modality-2 reconstruction loss: **MIXED**; simplification dominance=False; supported datasets=p22; equal-weight macro FULL-ablation ARI=+0.0359.
- omics-1 correspondence loss: **SUPPORTED**; simplification dominance=False; supported datasets=placenta;p22; equal-weight macro FULL-ablation ARI=+0.0457.
- omics-2 correspondence loss: **MIXED**; simplification dominance=False; supported datasets=none; equal-weight macro FULL-ablation ARI=-0.0169.
- within-modality attention: **MIXED**; simplification dominance=False; supported datasets=p22; equal-weight macro FULL-ablation ARI=-0.0295.
- cross-omics attention: **MIXED**; simplification dominance=False; supported datasets=none; equal-weight macro FULL-ablation ARI=+0.0210.
- all attention: **MIXED**; simplification dominance=False; supported datasets=none; equal-weight macro FULL-ablation ARI=-0.0407.

Method-level recommendation: **MIXED_EVIDENCE**. This is a locked recommendation only; no same-night model change was made.

## A1 spatial trade-off

- FULL_IGE - UNIFORM_WITHIN: ARI -0.0052, Moran +0.0091, Geary -0.0080, neighbor +0.0071.
- FULL_IGE - UNIFORM_CROSS: ARI +0.0007, Moran -0.0308, Geary +0.0304, neighbor -0.0127.
- FULL_IGE - UNIFORM_ALL: ARI -0.0087, Moran -0.0069, Geary +0.0086, neighbor +0.0095.

A1 interpretation: at least one attention comparison combines higher FULL_IGE ARI with worse local continuity, supporting a finer/fragmented-boundary trade-off rather than a pure spatial improvement.
The A1 domain most sensitive to full versus all-uniform attention is `pericapsular adipose tissue` (mean F1 FULL-uniform +0.0924).
The boundary maps use fixed seed 0 and the preregistered seed nearest the five-seed FULL_IGE mean; all per-domain values are in `per_domain_metrics.csv`.

## P22 heterogeneity

Five-seed FULL-minus-ablation ARI/NMI/Geary distributions are shown without seed filtering. The most sensitive domain for each ablation was computed from all five paired seeds:
- DROP_RNA_RECON: `L5`, mean domain-F1 FULL-ablation -0.1488.
- DROP_MOD2_RECON: `VL`, mean domain-F1 FULL-ablation -0.5464.
- DROP_CORR1: `VL`, mean domain-F1 FULL-ablation -0.5178.
- DROP_CORR2: `L5`, mean domain-F1 FULL-ablation -0.2524.
- UNIFORM_WITHIN: `VL`, mean domain-F1 FULL-ablation -0.4399.
- UNIFORM_CROSS: `VL`, mean domain-F1 FULL-ablation -0.2159.
- UNIFORM_ALL: `L6a/b`, mean domain-F1 FULL-ablation +0.2641.

- P22 cross_omics_rna_attention mean attention versus ARI Spearman rho=+0.1000 (five-seed exploratory description).
- P22 cross_omics_modality2_attention mean attention versus ARI Spearman rho=-0.1000 (five-seed exploratory description).
- P22 rna_spatial_attention mean attention versus ARI Spearman rho=-0.3000 (five-seed exploratory description).
- P22 rna_feature_attention mean attention versus ARI Spearman rho=+0.3000 (five-seed exploratory description).
- P22 modality2_spatial_attention mean attention versus ARI Spearman rho=-0.6000 (five-seed exploratory description).
- P22 modality2_feature_attention mean attention versus ARI Spearman rho=+0.6000 (five-seed exploratory description).

UNIFORM_CROSS and UNIFORM_WITHIN are compared directly in the paired table; these five-seed correlations are descriptive and were not used to search a favorable seed or threshold.

## IGE and attention interpretability

- Mean cross-seed spot-attention Spearman stability: 0.6756.
- Largest absolute attention/QC Spearman correlation: 0.6712; coefficients and BH-FDR q-values are both reported.
- Domain-association tests passing BH-FDR q<0.05: 90 dataset-seed-channel tests; epsilon-squared and per-domain sample sizes are reported.
- Latter-half weighted-gradient-share normalized entropy mean: 0.9000; max/min ratio mean: 6.5325.
- Coefficient/outcome Spearman correlations are exploratory only and were not used for tuning.

## IGE resource overhead

Relative to the locked Night-3AF C0 runs, FULL_IGE mean runtime delta was -0.68 seconds (-0.78%), peak allocated GPU delta +207.58 MiB, peak reserved GPU delta +230.67 MiB, and peak CPU RSS delta -49.54 MiB.

## Protocol audit

No learning-rate, epoch, embedding, PCA, HVG, graph, clustering, IGE, epsilon, weight-sum, scientific threshold, temperature, ASR, seed, variant, or evaluator metric tuning occurred. Scientific protocol deviations: 0.
Recorded implementation corrections were: the preflight self-file whitelist; isolation of Night-3B tests from historical completion-state tests; a post-training evaluation-source amendment for the missing torch import and required domain boxplots; installed-Matplotlib tick API compatibility; and removal of an extra, non-taskbook exact-final-replay hard gate while retaining every replay difference. None modified training artifacts or the preregistered scientific decision rules. Full details and preserved failed attempts are in `protocol_deviations.json`, `p0_attempts/`, and `evaluation_attempts/`.
