# SpaLORA Night-5A metric-driven R&D funnel report

## Decision

**Final status: `R3_CANDIDATES_READY_FOR_LOCKED_P22`**

Selected candidate: **C04_SHRINK25**. The cycle stopped after R3; P22, D1 and Night-4B were not run.

## Git and protection

- Persistent GitHub SSH passed with strict host-key checking under `/root/autodl-fs/.ssh`.
- Night-4A branch/tag were pushed once without force; remote commit is `4a22cfb4afe331e1ca2edcc0b86a01fa3892a452`.
- Historical protection passed: Night-3B 1186/1186 and Night-4A 76/76.
- Night-5A started from exact parent `4a22cfb4afe331e1ca2edcc0b86a01fa3892a452` in an isolated worktree.

## P0 engineering

- C00/C01/C02 CPU forward, loss, gradient coefficients, one-step Adam and final state parity: 3/3 exact.
- 17/17 candidates have unique serialized configuration SHAs and finite/nonzero engineering probes.
- Specialized tests: 39 passed, 0 failed. The retained first attempt was 26 passed/13 failed and is disclosed in `protocol_deviations.json`.
- Label-free preprocessing artifacts were frozen once and reused across model seeds.

## Funnel coverage

- R1: 34/34 success (32 candidate + 2 reference).
- R2: 24/24 success (20 candidate + 4 reference).
- R3: 16/16 success (12 candidate + 4 reference).
- Total: 74 training units; 64 candidate runs (limit 68) and 10 reference runs (limit 10).
- All failures and attempts were retained; no seed, candidate, parameter or budget expansion occurred.

## R1 fate of all registered candidates

The exact lifecycle is in `candidate_lifecycle.csv`. R1 advanced C03, C04, C09, C10 and C13. C02, C11 and C16 had promising Q but failed the spatial protection gate; family caps and numeric gates stopped all other candidates.

## R2 and R3

R2 advanced C04, C09 and C10. C03 passed the numeric gate but ranked fourth under the locked three-candidate cap; C13 failed stability requirements.

Five-seed R3 results relative to C00:

| Candidate | macro ΔARI | macro ΔNMI | macro ΔQ | worst dataset ΔQ | paired Q wins | runtime × | GPU × | spatial gate | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| C04_SHRINK25 | 0.040030 | 0.025219 | 0.032624 | 0.013413 | 8/10 | 0.843 | 0.968 | PASS | SELECT |
| C09_RNA_ANCHOR10 | 0.060776 | 0.053155 | 0.056966 | 0.005087 | 9/10 | 0.645 | 0.887 | FAIL | STOP |
| C10_MNN_TRIPLET01 | 0.069569 | 0.061173 | 0.065371 | -0.002333 | 7/10 | 0.692 | 0.889 | FAIL | STOP |

C09 and C10 achieved larger accuracy gains but were stopped because each triggered the preregistered spatial-protection rule. Their results were not hidden or used to revise the rule.

## Selected method contract

`C04_SHRINK25` retains Corr1, sets Corr2 objective contribution exactly to zero, and freezes active-set initial RMS-gradient equalization over RNA reconstruction, modality-2 reconstruction and Corr1 with coefficient sum exactly 4. For every within- and cross-attention row:

```text
a_final = 0.75 * [0.5, 0.5] + 0.25 * a_learned
```

No entropy regularizer, dataset-specific parameter, early stopping or label-driven graph choice is used. Candidate config SHA is `3ad6c834c5a412760b71916710ebbb2af1cb657a601c93933b1c4a3de7bd184a`; implementation source SHA is `1c72fdf62183be44f091558a6554e8252ff8730c20ce0884a26a83de13b7cb5f`; implementation commit is `9becd07c1d95d6fa62a81df902facf7f940bcdfb`.

## Mechanism interpretation

- Strong shrinkage toward equal weights was the only mechanism that passed all five-seed accuracy, spatial and resource gates.
- RNA-anchor and MNN-triplet were accuracy-positive but spatially unsafe under the locked protection rule.
- Fully uniform fusion, reliability weighting and DGI showed development-set Q signals in R1 but did not survive the locked family/spatial/stability funnel.
- The selected attention and active loss contributions did not collapse (`attention_loss_collapse_audit.json`).

## Scientific scope

A1 and Placenta are development datasets. Five seeds are optimization repeats, not biological replicates. These results do not establish superiority over modern baselines or independent generalization. The historical envelope is a difficulty marker only. P22/D1/Night-4B were not run, and only a planning Worker may authorize the next locked stage.
