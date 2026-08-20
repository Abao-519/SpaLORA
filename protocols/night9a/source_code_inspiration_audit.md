# Night-9A source-code inspiration audit

Date: 2026-08-20  
Scope: label-free topology-transfer design only; no third-party benchmark was run.

## Sources inspected

1. Wu et al., *Simplifying Graph Convolutional Networks* (SGC), PMLR 97 (2019):
   https://proceedings.mlr.press/v97/wu19e.html
2. Official SGC implementation, `Tiiiger/SGC`:
   https://github.com/Tiiiger/SGC
3. Rossi et al., *SIGN: Scalable Inception Graph Neural Networks*, arXiv:2004.11198:
   https://arxiv.org/abs/2004.11198
4. Official SIGN implementation, `twitter-research/sign`:
   https://github.com/twitter-research/sign
5. Hu et al. graph pre-training implementation, `snap-stanford/pretrain-gnns`:
   https://github.com/snap-stanford/pretrain-gnns

## Adopted ideas and exact adaptations

- **Fixed sparse propagation.** SGC and SIGN both motivate separating a fixed graph
  propagation operator from the learned predictor. Night-9A adopts only that
  label-free computational pattern for E05--E08: deterministic sparse CSR matrix
  multiplication is applied to the frozen G04 views, followed by row L2
  normalization. The operator, coefficients, number of hops, and delta formulas
  are pre-registered in the Night-9A registry; they are not copied from or tuned
  against a third-party result.
- **Precomputed multi-hop features.** SIGN precomputes graph-operator feature
  products and then combines them downstream. E06 similarly averages the fixed
  zero-, one-, and two-hop P00 views. Night-9A does not import SIGN's architecture,
  train a SIGN classifier, or change the frozen SpaLORA adapter/head.
- **Strict state loading followed by bounded fine-tuning.** The graph pre-training
  repository uses an explicit pre-trained-state load before downstream
  optimization. E00--E04 use the analogous engineering pattern: strictly load the
  already locked G04 tensor state into an identically shaped fresh G00 model; E00
  performs no optimization and E01--E04 run only their registered fixed epoch
  counts with the locked optimizer. This is a state-transfer pattern, not a claim
  that Night-9A reproduces that paper's scientific method.

## Deliberately not adopted

- No labels, validation scores, dataset names, platform names, or old MISAR scores
  are used in routing, training, candidate selection, or checkpoint selection.
- No third-party weights, datasets, graph construction defaults, classifiers,
  losses, hyperparameter searches, fallbacks, best-epoch selection, or formal
  benchmark code are imported.
- No approximation replaces the registered G00/P00 or G04/P04 operators, the
  frozen R02 adapter, the C06 endpoint, or the fixed EIGEN_KMEANS100 head.
- MISAR Y is not opened by Night-9A. MISAR comparisons are partition-fidelity
  checks against the already locked F00 partition only.

## Implementation traceability

- Operator construction and E05--E08 formulas:
  `SpaLORA/night9a_efficient.py`
- Strict state transfer, fixed-epoch E00--E04 execution, unchanged R02 adapter,
  resource accounting, and stage locks: `scripts/night9a_run.py`
- P22-only stage-locked evaluation and MISAR label-free fidelity gates:
  `scripts/night9a_evaluate.py`
- Negative/static/semantic tests: `tests/test_night9a_efficient.py`

This audit records conceptual provenance only. The authoritative candidate space,
budgets, parameters, gates, and run order remain the supplied Night-9A registry
and taskbook.
