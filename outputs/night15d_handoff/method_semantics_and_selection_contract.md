# Night-15D method semantics and selection contract

## Shared computational object

All nine lanes use the same `night15d_reliability_energy.py` core, the same sparse graph interfaces, the same prototype-unary algebra, the same reliability-transition operators and the same synchronous clustering update. The core does not receive dataset names, labels, tissue names or family names.

The common superset contains:

- edge rules: spatial, either-modality similarity, both-modality similarity, geometric mean and agreement;
- reliability transitions: row-normalized, sub-stochastic absolute mass and rejected-mass-to-self;
- feature banks: retained representation, low/high components, view-specific and multiscale sparse graph features;
- unaries: retained dynamic prototype distance or a continuous modality-margin mixture;
- numerical controls: pairwise strength, edge temperature, retained weight, switch trust and update count.

## What varies

Known K, discrete module choices and numerical hyperparameters are selected per lane from the common superset using public benchmark labels. This is ordinary development HPO. The choices are stored in `night15d_frozen_config_registry.json` before the final replay.

Consequently, Night-15D does **not** claim:

- an automatic gate that infers when to select a module;
- a dataset-name router;
- one universal frozen configuration;
- blind or pristine confirmation;
- label-free model selection.

Labels are absent from model inputs, prototype unaries, graph conductance, pairwise energy and update rules. They are present in known-K registration, cross-run HPO and evaluation.

## Numerical and replay contract

- deterministic full-SVD reductions;
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`;
- no dense N×N allocation;
- every final lane must preserve registered K;
- two post-final-source fresh processes must reproduce the frozen partition SHA, metrics, cluster sizes and config exactly;
- the two earlier replay directories are superseded and cannot be cited as final evidence.

