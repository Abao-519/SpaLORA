# Night-15E method semantics and selection contract

## Core object

All nine evaluated lanes call the same `continuous_reliability_energy(initial, k, evidence, config)` core. The core does not read dataset, tissue, family, path, label, ARI or NMI identifiers. It operates on sparse graph evidence, retained and modality-private reduced views, an initial partition, a numeric configuration and K.

The continuous transition keeps absolute conductance mass instead of forcing every nonzero neighborhood to unit neighbor mass. Rejected mass returns to the node's self state. Edge union/intersection evidence, prototype margins, retained/view-specific unary evidence, multiscale low/high-frequency features and trust-aware move strength are mixed continuously. The deterministic local conditional solver does not claim a global MAP optimum.

## Score-ceiling selection

The frozen balanced profile merges the formal, adaptive and multi-initial ledgers. Per lane, it first requires both ARI and NMI to exceed the Night-15D authority, then maximizes `min(delta_ari, delta_nmi)`, then `absolute_ari + 0.35 * absolute_nmi`. Public reference labels are used only outside the energy core for cross-run numeric HPO and evaluation. This is a public-label development score ceiling, not blind evaluation, automatic configuration selection or end-to-end zero-label deployment.

## Incremental policy transfer

The global, family-numeric and leave-one-study-out policies begin from fixed Night-15D authority partitions. Those initial partitions were themselves selected by per-lane public-label HPO in Night-15D. Held-out labels are excluded only from selection of the **incremental Night-15E numeric layer**; that cannot erase the initial partition's label history. The frozen conclusion is `NO_MEANINGFUL_INCREMENTAL_POLICY_TRANSFER`.

## New engineering unit

GSE213264 Human tonsil has no per-spot manual/expert reference partition in this delivery. K=8 is the public author-reported RNA cluster count used as a known engineering K, and K=7 is the corresponding protein-cluster-count sensitivity context. Neither is treated as ground truth; no ARI/NMI is computed.
