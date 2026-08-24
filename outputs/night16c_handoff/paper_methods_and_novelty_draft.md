# Paper methods and novelty draft (Night-16C)

## Common computation graph

For observations `i` and registered sparse spatial edges `(i,j)`, the same API receives two reduced molecular views and a start bank. Robust within-view edge changes and optional directional residuals are converted to empirical ranks. Their agreement produces an interior-support score, joint large change produces a consensus-boundary score, and asymmetric change produces a cross-modal-conflict score. These three non-negative states generate an accepted edge conductance; rejected absolute conductance is not renormalized away and is retained as node self-return.

The trusted-prototype repair (TPR) stage calculates multi-start stability and modality-specific prototype margins. An observation is movable only when all three conditions hold: its start assignments are unstable, its prototype confidence is low, and its CMBF boundary evidence exceeds the frozen floor. High-trust cores are anchors. Candidate moves use the same prototype unary and sparse pairwise semantics for RNA+protein and RNA+chromatin. The parameter schema and code path are identical; only one externally frozen numerical configuration per family differs.

## Parameter freezing and evaluation

Public annotations from A1 plus tonsil slice 1 selected the RNA+protein configuration; P22 K9 plus MISAR E15.5 K7 selected the RNA+chromatin configuration. Candidate partitions were materialized and hashed before a separate evaluator loaded annotations. Ranking was mechanical: number of discovery studies with simultaneous ARI/NMI gain, worst-study delta ARI, study-balanced mean delta ARI, mean delta NMI, then lower complexity. D1 and tonsil slices 2/3 were not used to select the protein-family configuration. Seven added SPOTS/GSE308623 units had no per-spot annotation read.

## Contribution and novelty boundary

The edge states provide a directly auditable representation of agreement, common boundaries and conflict. TPR is deliberately conservative: the frozen configurations moved only 1--10 observations in labeled units and zero in added unlabeled P0 units. BANKSY-like neighborhood features, direction weights, prototype learning, graph intersection and shared/private representations all have precedents. The present claim is therefore the combined family-frozen sparse edge-state/self-return/conjunctive-trust mechanism and its cross-family development signal, not novelty of the individual ingredients.
