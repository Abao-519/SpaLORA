# SpaLORA Night-2C numerical-equivalence gate and loss factorial

## 1. Executive result

P0C **passed and authorized training**. Main/tutorial/technical counts are `75/75`, `3/3`, and `4/4`. Ground truth was not accessed during P0C, no labels selected settings, no seeds were searched, ASR was not modified, and the Night-2/Night-2B failed preregistrations remain unchanged.

## 2. Provenance and immutable runtime

- Required parent: `c283449b188f510e98c2826cbb856f296367aa03` (configuration parent is `c283449b188f510e98c2826cbb856f296367aa03`).
- Branch: `revision/q2-night2c-numerical-equivalence-factorial-20260809`.
- Report-generation HEAD: `63026b31da733533eccb070c9020cd0ad34b9da4`; final handoff resolves through annotated tag `night2c-final-20260809`.
- Taskbook SHA-256: `4225e87d71372eb1257b843327065ffeb8c60271d4bd8dc7af9d2870a11266f2`.
- Environment fingerprint: `7fab7913c37945d5128c4dbf7e12ba789e4bd7501ba354ab36067092f3da17f5`; Python `3.8.10`, PyTorch `1.12.1+cu116`, CUDA `11.6`, GPU `NVIDIA GeForce RTX 4080 SUPER`.
- Critical trainer/config/runner/frozen-module/input hashes are recorded verbatim in `reports/night2c_p0c.json` and atomically copied into `results/night2c/gate_status.json`.

## 3. Why P0C is not a relaxation of P0B

Night-2B remains a valid failure under its fixed `1e-7` independent-GPU threshold. It observed same-model CUDA forward residuals of the same scale as legacy-versus-generalized residuals and Adam amplification, but lacked a complete same-code Adam-trajectory negative control. P0C does not edit that threshold or report. It changes the estimand prospectively: exact shared-graph and CPU identities remain exact gates, while independent GPU cross-code divergence is judged against eight balanced blocks of independently executed legacy-versus-legacy and generalized-versus-generalized trajectories, with fixed float32 floor and fixed twofold margin.

## 4. P0C audit

| Dataset | Consumed state | Shared-forward V3 | CPU one-step | GPU envelope | Failed cells |
|---|---:|---:|---:|---:|---:|
| a1 | True | True | True | True | 0 |
| placenta | True | True | True | True | 0 |
| p22 | True | True | True | True | 0 |

The deterministic-algorithm subprocess is descriptive only. Full named-tensor pair distances are in `results/night2c/p0c_pairwise_distances.csv`; normalized cell summaries and within/cross ratios are in `results/night2c/p0c_summary.csv`. P0C reason: all exact audits and every normalized GPU noise-envelope cell passed.

## 5. Run counts and validation

- Main: `75/75`; tutorial-2022: `3/3`; placenta technical repeats: `4/4`.
- Failure JSON count: `0`.
- Test state: `{"command": "/root/miniconda3/envs/SpaLORA_torch112/bin/python -m pytest -q", "duration_seconds": 6, "environment": "/root/miniconda3/envs/SpaLORA_torch112", "exit_status": 0, "failed": 0, "passed": 49, "preliminary_run": {"failed": 2, "passed": 47, "reason": "The isolated Git worktree lacked 242 Git-ignored historical Night-1 protected files expected by frozen Night-2/Night-2B checksum tests.", "resolution": "Copied only missing files from /root/autodl-fs/SpaLORA-night1 after verifying each source SHA-256 against the protected manifest; no old test or scientific file was edited."}, "stdout_stderr_log": "results/night2c/logs/pytest_full.log", "warnings": 7}`.
- Every authorized process used the gate-recorded critical hashes and environment fingerprint; mismatches are rejected before preparation.
- Embeddings, attention, observation IDs, losses, and clusters were durably written before the first label access in each run.

## 6. Five-seed metrics and prespecified contrasts

All five fixed seeds are represented in `per_seed_metrics.csv`; means use all five and SD is sample SD. Bootstrap intervals use fixed seed `20260809`; paired p-values enumerate all 32 sign flips and are descriptive (`n=5`).

| Dataset | Metric | Contrast | Mean | Sample SD | Exact sign-flip p |
|---|---|---|---:|---:|---:|
| a1 | ari | uniform_scale_effect | -0.003595983113500878 | 0.02142263168315594 | 0.9375 |
| a1 | ari | legacy_shape_effect | 0.0004317625125107083 | 0.002242822900749999 | 0.625 |
| a1 | ari | scale_at_legacy_shape | -0.00968431320985091 | 0.028710151000279842 | 0.5 |
| a1 | ari | legacy_shape_at_full_scale | -0.005656567583839323 | 0.020563516442864718 | 0.625 |
| a1 | ari | factorial_scale_main | -0.006640148161675893 | 0.023481258798770267 | 0.625 |
| a1 | ari | factorial_shape_main | -0.002612402535664307 | 0.011122625003089923 | 0.6875 |
| a1 | ari | interaction | -0.006088330096350031 | 0.018998046696509303 | 0.625 |
| a1 | ari | asr_vs_uniform_same_scale | -0.0072399482547991275 | 0.019200025767355505 | 0.75 |
| a1 | ari | asr_vs_legacy_same_scale | -0.0015833806709598052 | 0.013695231199075626 | 0.8125 |
| a1 | nmi | uniform_scale_effect | 0.0003268644513336638 | 0.011953710666076964 | 1.0 |
| a1 | nmi | legacy_shape_effect | -0.00017396160427781692 | 0.0023319134002050264 | 1.0 |
| a1 | nmi | scale_at_legacy_shape | 7.77733924304025e-05 | 0.014059955981261274 | 1.0 |
| a1 | nmi | legacy_shape_at_full_scale | -0.00042305266318107824 | 0.007094184095093299 | 0.8125 |
| a1 | nmi | factorial_scale_main | 0.00020231892188203316 | 0.012313703423243021 | 1.0 |
| a1 | nmi | factorial_shape_main | -0.0002985071337294476 | 0.0030369632191563924 | 0.9375 |
| a1 | nmi | interaction | -0.00024909105890326134 | 0.008639326002890847 | 0.875 |
| a1 | nmi | asr_vs_uniform_same_scale | -0.0017833340195034131 | 0.005293022453844204 | 0.6875 |
| a1 | nmi | asr_vs_legacy_same_scale | -0.0013602813563223348 | 0.004607220990762668 | 0.5625 |
| placenta | ari | uniform_scale_effect | 0.21049638275100663 | 0.028587593639700055 | 0.0625 |
| placenta | ari | legacy_shape_effect | 0.04241807382939137 | 0.04333523918021155 | 0.1875 |
| placenta | ari | scale_at_legacy_shape | 0.1704733425928394 | 0.03577819557323244 | 0.0625 |
| placenta | ari | legacy_shape_at_full_scale | 0.002395033671224156 | 0.015024044996567674 | 0.625 |
| placenta | ari | factorial_scale_main | 0.190484862671923 | 0.023409601972497838 | 0.0625 |
| placenta | ari | factorial_shape_main | 0.022406553750307763 | 0.023477159360764808 | 0.125 |
| placenta | ari | interaction | -0.04002304015816721 | 0.04475066166133277 | 0.1875 |
| placenta | ari | asr_vs_uniform_same_scale | 0.002537908355317664 | 0.019252412481047296 | 1.0 |
| placenta | ari | asr_vs_legacy_same_scale | 0.000142874684093508 | 0.02471694031515466 | 1.0 |
| placenta | nmi | uniform_scale_effect | 0.19471448986553472 | 0.028891162203613284 | 0.0625 |
| placenta | nmi | legacy_shape_effect | 0.05488341701315117 | 0.03673158152091478 | 0.0625 |
| placenta | nmi | scale_at_legacy_shape | 0.14005405989947822 | 0.033669327535440156 | 0.0625 |
| placenta | nmi | legacy_shape_at_full_scale | 0.00022298704709469152 | 0.013627071602138833 | 1.0 |
| placenta | nmi | factorial_scale_main | 0.16738427488250646 | 0.024331629895041614 | 0.0625 |
| placenta | nmi | factorial_shape_main | 0.02755320203012293 | 0.01937317369472218 | 0.0625 |
| placenta | nmi | interaction | -0.05466042996605648 | 0.03960470788847397 | 0.0625 |
| placenta | nmi | asr_vs_uniform_same_scale | -0.0027257545588273537 | 0.007742004738574036 | 0.4375 |
| placenta | nmi | asr_vs_legacy_same_scale | -0.002948741605922045 | 0.013758587821827763 | 0.6875 |
| p22 | ari | uniform_scale_effect | 0.0010475427373266345 | 0.03118846413134461 | 0.9375 |
| p22 | ari | legacy_shape_effect | -0.01092327180147964 | 0.023661416810234273 | 0.5 |
| p22 | ari | scale_at_legacy_shape | 0.00015011088502925318 | 0.003305281987059737 | 0.9375 |
| p22 | ari | legacy_shape_at_full_scale | -0.01182070365377702 | 0.015171110537595747 | 0.0625 |
| p22 | ari | factorial_scale_main | 0.0005988268111779438 | 0.016068475331051005 | 1.0 |
| p22 | ari | factorial_shape_main | -0.01137198772762833 | 0.012703776120796217 | 0.1875 |
| p22 | ari | interaction | -0.0008974318522973812 | 0.030569703464605653 | 0.8125 |
| p22 | ari | asr_vs_uniform_same_scale | -0.01059419225975693 | 0.01512148466019476 | 0.1875 |
| p22 | ari | asr_vs_legacy_same_scale | 0.001226511394020091 | 0.0008792468088072215 | 0.125 |
| p22 | nmi | uniform_scale_effect | 0.0035332680246008773 | 0.020038752852836364 | 0.8125 |
| p22 | nmi | legacy_shape_effect | -0.002842095265107214 | 0.010939977462685513 | 0.625 |
| p22 | nmi | scale_at_legacy_shape | 0.0005967785356348454 | 0.0022739373637822324 | 1.0 |
| p22 | nmi | legacy_shape_at_full_scale | -0.005778584754073246 | 0.01810475355295775 | 0.5625 |
| p22 | nmi | factorial_scale_main | 0.0020650232801178612 | 0.010237223628257016 | 0.75 |
| p22 | nmi | factorial_shape_main | -0.00431034000959023 | 0.011188035937907695 | 0.4375 |
| p22 | nmi | interaction | -0.002936489488966032 | 0.01985552375544775 | 0.8125 |
| p22 | nmi | asr_vs_uniform_same_scale | -0.004938523142199713 | 0.01722455142520921 | 0.6875 |
| p22 | nmi | asr_vs_legacy_same_scale | 0.0008400616118735327 | 0.001099467994057593 | 0.1875 |

## 7. Loss scale versus shape, especially placenta

ARI scale=+0.190485, shape=+0.022407, interaction=-0.040023; NMI scale=+0.167384, shape=+0.027553, interaction=-0.054660. These are preregistered paired effects, not a tuned setting selection.

No universal winner is inferred by pooling datasets of unequal difficulty.

## 8. V3 replay against Night-1

`v3_replay_audit.csv` records partition ARI, per-metric differences, and fixed warning thresholds. Warnings, if present, were reported without reruns, tuning, or seed replacement.

## 9. Limitations

The target uses float32 CUDA sparse operations and Adam, so low-order scheduling residuals can be amplified. Efficacy summaries have only five fixed model seeds. The frozen public weighting vector is index-misaligned and cannot support a low-expression-gene causal claim. Placenta modality 2 is described only as **ATAC-derived / TF-associated regulatory features** because raw-peak provenance has not been established.

## 10. Recommendation

Use the observed scale/shape/interaction pattern and V4 diagnostic only to choose a separately preregistered next experiment; do not promote a setting from these labels by post-hoc tuning.

## 11. Persistence and shutdown

- GitHub push: `pending final persistence`.
- Bundle: `pending`; SHA-256 `pending`.
- Archive: `pending`; SHA-256 `pending`.
- Protected files: `pending final verification`.
- Local transfer/checksums: `pending`.
- Shutdown: `/usr/bin/shutdown` is required as the last remote command; final confirmation is written after local verification and must not be inferred before execution.
