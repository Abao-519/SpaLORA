# Night-16B method semantics and selection contract

## Numeric producer boundary

The core receives ordered numeric views, sparse graphs, an initial partition, K and a JSON configuration. It has no dataset/study identity argument and no reference-label argument. RNA+protein and RNA+ATAC use the same modules and parameter meanings. Morphology is optional; an all-zero presence mask aliases the molecular fallback exactly.

## Candidate and evaluator boundary

Each coarse or refine batch is fully generated, serialized and hashed before its public reference array is opened. The separate evaluator computes ARI, NMI, AMI, FMI, Moran's I and Geary's C. Public metrics may select the next refine neighbourhood and final benchmark profile. This is explicitly `label-assisted benchmark HPO`.

## Headline rule

Candidates must be finite, exact K, and have minimum cluster size at least `max(5, ceil(0.01*N/K))`. The profile maximizes ARI; numerical ties are resolved by NMI, then lower complexity, then larger minimum cluster. Max-NMI, the ARI/NMI Pareto frontier, and 0/1/2/5 percent guard sensitivities are separate profiles.

## Family-default boundary

The default table chooses a modal, then lower-complexity config chain from other studies without reading the held-out lane's metrics. It is applied to fixed Night-16A authority starts. Consequently it measures incremental decoder-parameter transfer, not a complete end-to-end zero-label method.

## Scientific claim boundary

The tuned scoreboard and D1/P22 score advances are valid public benchmark development results. Because the D1 gain requires a strong prior authority start and several lanes select no-op, Night-16B does not claim a new multi-study independent decoder contribution, blind evaluation, SOTA or paper-ready evidence.
