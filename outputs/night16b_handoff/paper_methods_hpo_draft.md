# Paper methods and HPO draft

## Decoder

The unified reliability-structured decoder consumes ordered reduced RNA, second-modality, retained/fused and coordinate views, three sparse graph scales, an optional morphology view with an explicit presence mask, K, and a numeric JSON configuration. The candidate producer never receives an annotation array. It constructs a common start bank, evaluates prototype unaries and cross-modal edge reliability, preserves rejected conductance as a current-state stay cost, and applies sparse structured optimization followed by a generic non-degeneracy repair. Missing morphology aliases the molecular path exactly.

## Public benchmark tuning protocol

We use label-assisted benchmark HPO. In each coarse batch, all partitions and hashes are written before an independent evaluator opens the public annotation. A mechanical refine batch is then generated around the coarse leaders, again locked before evaluation. The headline profile requires exact K, finite output, and minimum cluster size at least `max(5,ceil(0.01*N/K))`; it maximizes ARI, then NMI, then lower complexity and larger minimum cluster. Max-NMI, ARI/NMI Pareto, and 0/1/2/5% guard profiles are reported separately.

## Family default

For an unannotated new unit, a secondary family-default takes the modal, then lowest-complexity config chain from other studies: A1/D1 mutually held out, each tonsil slice held out as a whole, and P22/MISAR mutually held out. The present audit applies these decoder parameters to fixed Night-16A authority starts; it therefore tests incremental parameter transfer, not a full end-to-end zero-label deployment.

## Fair-comparison requirement

The final paper must give strong external baselines a comparable preprocessing, endpoint, known-K and tuning budget. Different K, annotation, mask or native endpoint numbers remain context rather than formal head-to-head wins.
