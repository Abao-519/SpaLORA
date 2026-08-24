# Night-16H source collision and transfer audit

Night-16H inherited the Night-16G audit of STCC, SACCELERATOR, PHD-MS, SCALE and
SMODEL and added the Worker2 collision check for SpatialESD.  No third-party
source was copied into SpaLORA; the implementation uses the project's licensed
NumPy/SciPy/scikit-learn stack.

| Prior work | Collision relevant to Night-16H | Claim boundary |
|---|---|---|
| STCC | Consensus over multiple base clusterings | Plain consensus and medoid are controls. |
| SACCELERATOR | Multi-method consensus and uncertainty | Candidate aggregation is not claimed as new. |
| PHD-MS / SCALE | Persistent or multiscale clustering paths | Night-16G persistence received zero formal weight and is not the Night-16H story. |
| SMODEL | Weighted spatial multi-omics ensemble | Weighted partition ensemble is established prior art. |
| SpatialESD | Multiple base partitions with spatial graph constraints/consensus | A spatial feasibility filter alone is at most a safety/head signal. |

Night-16H therefore makes no novelty claim for partition ensembles, spatial
consensus, inertia, Calinski-Harabasz, Moran-style topology evidence or medoids.
The only supported empirical combination is: a dataset-name-free structural
admissible set plus fixed/LOSO cross-evidence selection that removes the known
singleton failure and transfers across four RNA+chromatin studies.  Whether this
combination is sufficiently novel remains unresolved and requires a broader
formal literature comparison before a paper claim.
