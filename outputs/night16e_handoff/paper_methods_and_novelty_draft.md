# Methods and novelty draft — TSRE

## Computation graph

For each modality, the input adapter produces a reduced observation-by-feature view. Three registered sparse spatial graphs form fine, registered and broad scales. The same computation is used for RNA+protein and RNA+chromatin; only one numeric profile is frozen per family.

For edge `(i,j)`, modality-specific feature differences are robustly rank transformed. Their soft product decomposition yields support, consensus-boundary and two private-conflict masses summing to one. The support mass nonnegatively modulates the inherited Potts capacity. Consensus-boundary mass contributes a unary cost against adopting the neighbor's frozen-cycle label. Private masses weight the two modality prototype costs by local prototype margin. Rejected base conductance and low-confidence relation mass add a current-label stay cost. Dynamic prototype unaries are refreshed between outer cycles; accepted alpha moves decrease only the current frozen-cycle energy.

## Selection protocol

Public annotation is allowed only in the independent evaluator and across-run family/profile selection. The RNA+protein profile uses A1 and tonsil s1 discovery. The chromatin role was frozen before human-hippocampus evaluation as P22+MISAR discovery, because its `C_GEO` base includes both historical configs. Existing starts are historical per-lane benchmark-HPO assets, so those results are only frozen-operator transfer. The human-hippocampus start is selected without reference labels by partition-consensus adjusted Rand centrality and a microcluster penalty; the producer output is hashed before the evaluator reads `true_label`.

## Novelty boundary supported by data

Potts/CRF clustering, alpha expansion, multiscale graph features, prototype unary, dynamic graphs and shared/private representations are established prior art. Optional modality views alone are also prior art. The only empirically supported Night-16E object is the combination of cross-modal support modulation with absolute rejected-mass self-return in a common sparse direct-clustering energy. Boundary-exclusion and private-conflict unaries remain hypotheses because matched independent-transfer ablations show no added score.

## Evidence boundary

The independent human-hippocampus full result is a frozen RNA+chromatin operator transfer on a new unlabeled producer path. It does not establish a complete cross-family model. The reported per-lane board is transparent label-assisted development HPO. A second independent chromatin study, a corrected relation carrier, biological marker analysis and matched external baselines are required before a paper-ready claim.
