# SpaLORA Night-5B second-look/rescue report

## Decision

**Final status: `BUDGET_EXHAUSTED`.** P0 passed, S1 and S2 executed, but the final audit found that all 24 B21-B24 Laplacian units consumed `shrink_to_uniform rho=0.25` instead of their registered `uniform_all` attention. Correctly rerunning all affected units would raise attempts from 108 to 132, above the hard maximum 120. The first outputs are retained and invalidated; no rerun, partial repair, parameter change, seed search, or budget expansion was performed.

The draft `s2_decision.json` is retained only as audit evidence. `selected_for_future_locked_p22.json` canonically withdraws every selection. No future P22 candidate is authorized by this run.

## 1. P0 and configuration contract

- 25/25 configurations have unique SHA-256 values.
- Night-5A historical tests: 39/39; dedicated Night-5B tests: 9/9; combined final tests: 48/48.
- P0 reuse audit: 50 locked source records; second-look parity 5/5; new engineering probes 12/12.
- CPU/GPU probes, sparse-graph preservation, label-firewall negatives, withheld-path rejection, latent reliability swap/row-sum/freeze/resume, diffusion alpha=0 parity, and Laplacian edge/gradient tests passed.
- The initial import-path failure and firewall-key initialization failure are retained. Neither started scientific training.

## 2. Reuse and new execution

- B00-B03 reused Night-5A five-seed results without rerun.
- B04-B08 reused Night-5A seed 0 and added seeds 1-2; selected top-ups added seeds 3-4.
- B17-B20 reused locked C09/C10 embeddings and applied one deterministic diffusion step without training.
- S1: 92 new training units, 50 source-reuse records, 40 diffusion records, 0 run failures.
- S2: 16 new training units, 0 failures. Total new attempts: 108.
- No Night-5A artifact was overwritten. Raw runs/checkpoints remain under `/root/autodl-fs/night5b_raw_runs_20260813`.

## 3. Family-cap second look

The second look did reveal a strong candidate previously limited by the family cap: B06/C08 anchor-0.5 reached ΔARI +0.053613, ΔNMI +0.051921, ΔQ +0.052767, worst ΔQ +0.003446, wins 8/10, spatial_fail=True, runtime×0.663, GPU×0.888. It passed the accuracy frontier but retained a spatial trade-off. B15 latent reliability reached ΔARI +0.048722, ΔNMI +0.027108, ΔQ +0.037915, worst ΔQ -0.001742, wins 8/10, spatial_fail=False, runtime×0.902, GPU×0.969; it did not satisfy either final frontier. Thus learned-latent reliability was promising on Placenta but did not establish a cross-dataset advantage over the input-space reliability family.

## 4. Combination and spatial rescue evidence

Unaffected five-seed evidence relative to B00:

- B01 C04 primary: ΔARI +0.040030, ΔNMI +0.025219, ΔQ +0.032624, worst ΔQ +0.013413, wins 8/10, spatial_fail=False, runtime×0.843, GPU×0.968.
- B09 shrink25+anchor0.5: ΔARI +0.044683, ΔNMI +0.025275, ΔQ +0.034979, worst ΔQ +0.012835, wins 9/10, spatial_fail=False, runtime×0.898, GPU×0.967.
- B10 shrink25+anchor1.0: ΔARI +0.042061, ΔNMI +0.026837, ΔQ +0.034449, worst ΔQ +0.013639, wins 9/10, spatial_fail=False, runtime×0.854, GPU×0.967.
- B17 C09+diffusion0.10: ΔARI +0.070756, ΔNMI +0.053021, ΔQ +0.061889, worst ΔQ +0.013592, wins 8/10, spatial_fail=False, runtime×0.000, GPU×0.000.
- B19 C10+diffusion0.10: ΔARI +0.066771, ΔNMI +0.054132, ΔQ +0.060451, worst ΔQ +0.002453, wins 8/10, spatial_fail=False, runtime×0.000, GPU×0.000.

Diffusion 0.10 provided the clearest spatial rescue: B17 and B19 no longer triggered the Night-5A spatial gate while retaining large accuracy gains. C04+anchor combinations were balanced-frontier positive. Laplacian rescue cannot be interpreted because every B21-B24 run was invalidated.

The noncanonical draft frontiers were balanced: B10_SHRINK25_ANCHOR10, B17_C09_DIFFUSE10, B01_C04_SHRINK25, B09_SHRINK25_ANCHOR05, B19_C10_DIFFUSE10, B18_C09_DIFFUSE25; accuracy: B17_C09_DIFFUSE10, B02_C09_RNA_ANCHOR10, B06_SECONDLOOK_RNA_ANCHOR05, B19_C10_DIFFUSE10, B03_C10_MNN_TRIPLET01. Because the candidate pool contained invalid Laplacian units and the budget prevented complete correction, these lists are evidence only, not a canonical selection.

## 5. Label firewall and withheld data

All embeddings/manifests were locked before development labels were read. Training/warmup/reliability/graph/triplet/DGI/Laplacian/diffusion code did not read semantic labels. **P22, D1, GSE198353, and Night-4B were not run or opened.** mclust remained EEE, PCA 20, seed 2020. No per-epoch ARI/NMI, seed search, label-selected checkpoint, or post-hoc parameter expansion occurred.

## 6. Budget, resources, and protection

- Hard limit: 120 attempts; executed: 108; unaffected/valid: 84; invalidated: 24.
- Correct full repair requires 24 attempts; only 12 capacity remained, so the protocol stopped.
- Night-3B 1186/1186 and Night-4A 76/76 historical protection passed.
- The exact per-candidate resource ratios are in `five_seed_summary.csv`; model states are excluded from compact delivery.

## 7. Required-answer summary

1. 25 unique SHAs: yes; P0: pass.
2. Reuse/new execution: documented above; no source rerun or overwrite.
3. Underestimated by family cap: B06 showed an accuracy-frontier signal, with spatial cost.
4. Rescue: diffusion 0.10 and C04+anchor combinations showed positive evidence; Laplacian is uninterpretable.
5. Latent reliability: not consistently superior across A1 and Placenta.
6. Draft frontiers are retained but noncanonical due to final audit.
7. ARI/NMI/Q, wins, spatial, runtime and GPU are in this report and `five_seed_summary.csv`.
8. Future locked P22 candidates: none authorized under `BUDGET_EXHAUSTED`.
9. P22, D1, GSE198353 and Night-4B were not run.
10. Git/bundle/archive and shutdown status are recorded in external non-self-referential delivery metadata.
