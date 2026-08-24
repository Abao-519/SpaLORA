# Night-16H method semantics and selection contract

## Frozen computation

Night-16H consumes a locked bank of 89 candidate partitions per study.  Before
any selector or evaluator is called, every candidate is tested against the same
structural admissibility rule:

1. the partition has exactly the registered known `K` and no empty cluster;
2. it has no singleton cluster;
3. every predicted cluster contains at least one internal edge in the smallest
   registered sparse spatial graph.

This rule has no dataset-specific threshold.  `NO_SINGLETON`, any-scale and
degree-derived variants are sensitivity analyses only; they do not replace the
formal `SMALLEST_SCALE_INTERNAL_EDGE` contract.

Within the admissible set, the fixed label-free selector forms a molecular
champion and a spatial-topology champion.  It accepts the molecular champion
when that candidate's topology percentile is at least the empirical median
(`0.5`); otherwise it selects the topology champion.  Candidate IDs resolve
exact ties.  The rule reads neither dataset names nor annotation arrays.

## Strict LOSO boundary

For each held-out study, a first process opens only the other three public
benchmark evaluations and fits three nonnegative rank weights on molecular,
topology and predicted cluster-risk axes.  It writes and hashes the configuration
without opening the held-out evaluation.  A second process applies that config
to the held-out locked bank.  Only a third, independent evaluator joins the
already locked partition to held-out metrics.  All four training folds selected
the same weights: molecular `0.25`, topology `1.0`, risk `0.0`.

## Claim boundary

The candidate bank was developed on public benchmarks, so this is not pristine
blind evidence.  Partition ensemble, medoid and spatial consensus are prior art.
The supported Night-16H object is limited to a universal graph-feasible candidate
set followed by cross-study multi-evidence selection.  It is not claimed as a
new consensus algorithm, SOTA, or paper-ready evidence.
