# Night-22A method semantics and selection contract

## Computational object

Night-22A separates two objects.

1. **Geometry ceiling**: locked Night-21C representations are sent to KMeans, full-covariance GMM, self-tuned spectral cuts at two feature scales, graph fusion, density correction, diffusion, spatial-only Ncut and exact-K Leiden. These are label-closed producers and established heads; their scores are transparent, label-assisted ceilings, not a new method.
2. **Direct partition junction**: the trainable variable is the row-stochastic matrix `Q∈R^(N×K)`, not a node embedding and not an index selecting an existing partition. Trainable logits, diagonal cluster centers/scales and cluster-specific graph-mixture weights are updated by Adam. The final hard partition is `argmax(Q)` plus a deterministic exact-K empty-cluster repair if required.

For graph `g`, cluster `k` uses normalized association

`A_gk(Q) = (2 Σ_(i,j∈E_g) w_gij Q_ik Q_jk) / (Σ_i d_gi Q_ik + eps)`.

`FULL` combines an elliptical cluster likelihood on the locked carrier with a cluster-specific simplex over five equal-total-mass graphs: retained k12, retained k24, RNA-view k12, second-view k12 and registered spatial adjacency. Conditional entropy hardens assignments; a one-sided floor penalizes only masses below `0.2/K`; a decaying start penalty is an optimization stabilizer. There is no equal-size target.

## Matched attribution

Every arm shares carrier, representation source, exact-K start, graph bank, seed, steps and optimizer:

- `INPUT_GEOMETRY_START`
- `EMISSION_ONLY`
- `SHARED_GRAPH_ONLY`
- `CLUSTER_GRAPH_ONLY`
- `ADDITIVE_SHARED`
- `FULL`

An independent junction contribution requires `FULL` to change the start partition and strictly beat the input, the strongest pure Stage-A head, and every coordinate-matched atomic arm in both ARI and NMI. This is stronger than merely being the highest `FULL` candidate.

## Label and HPO flow

Known K comes from the registered public protocol. Producers explicitly load numeric IDs, representations, view arrays and sparse graph arrays; no label array is accessed. Candidate banks, checkpoint banks and per-partition SHA-256 values are locked and reload-checked before the evaluator opens labels. Stage-A start choice and Stage-B profile reporting are transparent public benchmark HPO, not label-free model selection. Family-frozen confirmation is authorized only by the taskbook gate; no confirmation lane may be tuned after freezing.

## Prior-art boundary

DEC/prototype assignment, SwAV/P²OT balancing, DeepCut normalized-cut optimization, ordinary multiview graph fusion, BANKSY neighbor geometry/Leiden and spaMGCN/S3RL/SEPAR/CRCT representation or prototype components are prior art. The Night-22A joint object remains a working P0 description unless matched `FULL` evidence is both independent and transferable.
