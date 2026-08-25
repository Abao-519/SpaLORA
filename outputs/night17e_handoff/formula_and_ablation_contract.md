# Night-17E LRCC formula and ablation contract

## Scientific object

Night-17E tests a **frozen learned-representation edge support** signal.  It is
not the direct Night-17C relation posterior.  For each registered undirected
spatial edge and each of three graph scales, cosine similarities from the
strict-reloaded Night-17C seed 0/1/2 representations are converted to within-
scale ranks.  Their mean is the support statistic and their across-seed
standard deviation is the uncertainty statistic.

For edge `e` and graph scale `s`, the primary nonnegative modulation is

`f_es = (1-m) + m * [floor + (1-floor) * (rank_support_es * confidence_es)^p]`

where `confidence_es = clip(1 - uncertainty_scale * uncertainty_es, 0, 1)`.
The inherited Night-15F nonnegative Potts capacity is multiplied by `f_es`
before the three registered graph scales are combined.  The original dynamic
prototype unary, rejected-mass self-return and alpha-expansion solver are kept
unchanged.  Every accepted move lowers the energy for that frozen dynamic-unary
cycle; no cross-cycle single-energy monotonicity is claimed.

The same formula and four numeric configurations are used on P22 K9, MISAR
E15.5 K7 and human hippocampus K7.  There is no lane-name read in the model.
Known K is supplied by each registered public benchmark protocol.

## Frozen grid

| ID | relation mix | power | floor | uncertainty scale |
|---|---:|---:|---:|---:|
| L01_MIX025 | 0.25 | 1 | 0.05 | 1 |
| L02_MIX050 | 0.50 | 1 | 0.05 | 1 |
| L03_MIX075 | 0.75 | 1 | 0.05 | 1 |
| L04_MIX050_POWER2 | 0.50 | 2 | 0.05 | 1 |

Each lane uses six mechanically registered starts: one locked Night-16H method
start and five retained-embedding KMeans robustness starts.  The formal budget
is exactly `6 starts × (1 input + 4 configurations × 5 arms) = 126 rows` per
lane.  The formula and registry were frozen before the independent evaluator
opened public annotations.

## Matched arms

- `RELATION_DISABLED`: factor one; inherited base energy only.
- `ZERO_RELATION`: support from the exact zero-residual representations, then
  per-scale base-weighted pairwise mass matched to the learned arm.
- `UNIFORM_MASS_MATCHED`: constant per-scale attenuation with exactly the same
  base-weighted pairwise mass as the learned arm.
- `PERMUTED_RELATION`: learned factors are deterministically permuted within
  base-edge-weight quartiles, then exactly mass matched.  This preserves the
  coarse edge-difficulty distribution while destroying learned edge location.
- `LEARNED_RELATION`: the primary frozen learned-representation edge support.

All factors are finite and nonnegative.  Runtime assertions require exact edge
set equality with the registered graph and mass-match tolerance `rtol=1e-11,
atol=1e-12`.  The model does not contain a relation-stay parameter; the dead
pre-formal draft field was removed.  Base rejected-mass self-return is inherited
and is not claimed as a new Night-17E contribution.

## Selection and label boundary

- `FIXED_GLOBAL_L02` is the predeclared family-wide numeric profile.
- `DIRECT_HPO_*` is transparent public-benchmark development HPO after all
  candidate partitions were materialized and hashed.
- `STRICT_LOSO` fits one of the four configurations from the two training-study
  evaluation tables in a process that cannot open the held-out table.  A second
  process selects the held-out producer row without labels; a final evaluator
  opens the held-out table and verifies the partition SHA.

The advancement claim gate is evaluated on strict LOSO and requires at least
two of three primary studies to improve both ARI and NMI over the locked
Night-16H start, while at least two studies remain independent of all matched
controls.  The gate failed and therefore melanoma and multi-seed expansion were
not run.
