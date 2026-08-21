# SpaLORA Night-10B frozen family-policy integration report

## Locked outcome

`NIGHT10B_FROZEN_FAMILY_POLICY_INTEGRATION_LOCKED`

Night-10B integrated the already frozen modality-family policy. It did not continue QCRD, search candidates, or recalculate scientific performance. Night-10A remains interpreted as: engineering repair succeeded and scientific candidates failed.

## Frozen router and recipes

- Explicit `RNA+PROTEIN` routes to `C00_G04_H05_CONFIRMED`.
- Explicit `RNA+ATAC` routes to `F00_R02_FULL`.
- Resolver inputs are only `primary_assay` and `auxiliary_assay`; dataset-name routes are zero and unknown pairs fail closed.

## Exact replay and engineering smokes

- Active replay cycle 2: 30/30 exact Q00 rows passed in fixed u000-u029 order.
- Fresh-process row verification: 30/30 passed; all replay row hashes are unique.
- Fresh full smokes: A1 seed 0 C00 and P22 seed 0 F00, 2/2 passed.
- Independent checkpoint/endpoint round-trips: 2/2 passed.
- Two global correction cycles were used and prior replay cycles 0 and 1 were explicitly invalidated and preserved.

## Firewalls and resources

All label/science-metric/MISAR-Y/E18.5/new-data/benchmark/QCRD/new-candidate/dataset-name-route/dense-NxN/scientific-retry counters are zero. Historical raw roots remained metadata-exact. Peak GPU use was 1613.52 MiB; wall clock was 2.465 h, within the 6 h contract.

## Claim boundary

This is a reproducibility and engineering integration lock. It is not evidence of a new performance gain, SOTA result, a new scientific candidate, or success of a unified trainable module.
