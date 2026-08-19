# SpaLORA Night-8A report

Final status: `IMPLEMENTATION_SEMANTICS_INVALID`

## Plain result

Night-8A implemented the planned MF-SPC family router and all six registered mechanism classes, but this run cannot support a scientific gain claim. At the first authorized label window, the RNA-protein B00 reference differed from the locked C00 result by as much as 0.009087943105. The P22 R02 reference remained within 4.441e-16, which localized the defect to the protein-family C00 transform path.

The defect was semantic rather than a historical-table problem: nine A1/tonsil/D1 B00 cells rebuilt an approximate equal-view affinity instead of reusing the SHA-locked C00 G04/H05 affinity and partition. The minimum partition agreement with the locked artifact was ARI 0.879618. Therefore the label window failed closed, no shortlist was created, and R3 stayed at zero.

## Dataset interpretation

- Human lymph (A1 and D1): no valid Night-8A delta is reportable because their reference endpoint was implemented incorrectly.
- Tonsil: no valid Night-8A delta is reportable for the same reason.
- P22: the R02 reference replay itself matched historical metrics, but no P22 candidate is promoted because the preregistered joint selection window was invalidated globally.

## Engineering and audit

- R1/R2 registered cells: 128; real CUDA training/checkpoint round-trips: 116; content-addressed no-op aliases: 12.
- One fixed numerical transform failure was preserved: B03_RR10_PROTO / P22 / seed 1 (`cluster K mismatch`); scientific retry 0, fallback 0.
- The corrected C00 artifact-reuse code and a regression test are delivered for a future clean rerun only. They were not used to backfill, reevaluate, or continue this run.
- External provenance preflight independently locked MISAR_E15_5_S1 (1949 spots, RNA+ATAC) with labels sealed; no external benchmark ran.

## Scientific boundary

There is no unified winner, specialist winner, Pareto claim, or per-dataset improvement claim from this invalid run. Existing C00 and P22 R02 historical conclusions remain unchanged.
