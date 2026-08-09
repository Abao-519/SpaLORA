# SpaLORA Night-3A Label-free Loss Calibration Feasibility

## Strict outcome

P0A: **PASS**. P0B: **FAIL**. The preregistered 60-run factorial was not authorized and **0/60** main runs were executed.

All 15 numerical IGE probe cells (3 datasets x 5 fixed seeds) individually passed their finite-positive gradient/weight, weight range, weight-sum, repeatability, state/RNG preservation, reload-forward envelope, identical-initialization/input, and locked-m_bad checks. However, the P0B runtime file trace recorded reads of:

- `/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv`
- `/root/autodl-fs/P22 mouse brain coronal section/MouseBrain_groundtruth.csv`

The reads came from the P0B lock verifier recomputing SHA-256 over every locked data file. The model did not parse or use semantic label values, and no evaluation module was imported, but byte-reading the two ground-truth files still violates the explicit P0B label-firewall requirement. The failure is therefore retained without weakening the rule, patching the verifier, or restarting the experiment.

## P0A evidence

- Input commit: `49764b6d1b9d16510126a879e58f38973b562ce8`; implementation commit before P0A: `fff05e4b7e5ad4ac9c39121aa14ee87445eb4d13`.
- Config lock SHA-256: `134f21c4d0e80d760a0385857945521d8dab2099b5ca9b4c0f5371fedbf46a66`.
- Prepared observations: A1 3,484; Placenta 1,662; P22 9,196 after the frozen corrected-pipeline QC.
- Data file, matrix, spot, feature, coordinate, PCA, spatial-graph and feature-graph audits passed.
- The fixed random permutation of the exact 60 factorial cells was locked before any semantic label access.

## P0B numerical evidence (diagnostic only)

The values below are initialization diagnostics, not performance results and not grounds for formula changes.

| Dataset | Raw loss | IGE weight mean +/- SD | Range |
|---|---|---:|---:|
| A1 | RNA reconstruction | 3.7746 +/- 0.0447 | 3.7212–3.8367 |
| A1 | Modality-2 reconstruction | 0.04691 +/- 0.01270 | 0.03443–0.06196 |
| A1 | Corr1 | 0.04673 +/- 0.01050 | 0.03917–0.06468 |
| A1 | Corr2 | 0.13180 +/- 0.03043 | 0.08967–0.17222 |
| Placenta | RNA reconstruction | 2.9009 +/- 0.0621 | 2.8310–2.9643 |
| Placenta | Modality-2 reconstruction | 0.12608 +/- 0.01025 | 0.10961–0.13636 |
| Placenta | Corr1 | 0.04544 +/- 0.00340 | 0.04154–0.04926 |
| Placenta | Corr2 | 0.92755 +/- 0.06910 | 0.85975–1.01787 |
| P22 | RNA reconstruction | 3.5431 +/- 0.0439 | 3.4950–3.5920 |
| P22 | Modality-2 reconstruction | 0.11916 +/- 0.00835 | 0.11099–0.13287 |
| P22 | Corr1 | 0.13992 +/- 0.02284 | 0.11779–0.17362 |
| P22 | Corr2 | 0.19781 +/- 0.02261 | 0.16217–0.21468 |

All weights were finite, strictly positive, within `[1e-3, 1e3]`, and summed to four within `1e-6`. Placenta modality 2 is described only as **ATAC-derived / TF-associated regulatory features**.

## Required questions

1. **Did IGE strictly pass go/no-go?** No. P0B failed the ground-truth file-read firewall, so scientific go/no-go was not evaluated.
2. **Did IGE reproduce C1's main placenta benefit?** Not assessed; no training or label evaluation was authorized.
3. **Were A1/P22 harmed, seed by seed?** Not assessed; 0/60 main runs.
4. **Did spatial continuity decline beyond the threshold?** Not assessed.
5. **What were the four IGE weights and were they stable?** The five-seed initialization summaries are reported above and the exact 60 loss-level rows are in `ige_weights.csv`; they passed the numerical range/repeatability checks. No training interpretation is permitted.
6. **How did loss contributions and attention change?** Not assessed; no trajectories exist because training was hard-stopped.
7. **How did ILN perform as a diagnostic?** Not run.
8. **Any leakage, numerical anomaly, protected-file change, or protocol deviation?** The two ground-truth CSV byte reads are the sole protocol violation and the reason for P0B failure. There was no semantic label use, forbidden evaluation import, or numerical probe anomaly. Night-2C protected files matched **913/913**.
9. **May the project enter architecture ablation?** No. Night-3A did not pass and architecture ablation is not authorized.

## Validation and integrity

- Tests: **19 passed, 0 failed**.
- P0B cells: **15/15 numerical cells passed**, overall P0B failed solely on the file-read firewall.
- Main runs: **0/60**; failure artifacts: one stage-level P0B failure, zero run-level failures.
- Night-2C protected files: **913/913 matched**.
- ASR/rescue, scales, seeds, formulas, and thresholds were not changed or searched.
- No ground-truth performance metrics were computed.
- Final Git commit/tag and archive hashes are recorded in the external non-self-referential delivery index.

