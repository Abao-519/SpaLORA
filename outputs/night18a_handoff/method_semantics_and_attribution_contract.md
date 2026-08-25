# Night-18A method and attribution contract

The numeric producer uses two modality adapters, registered sparse spatial propagation, per-modality sparse cosine-kNN propagation, masked row reconstruction, modality-drop consistency, a small DEC-style soft assignment loss, and a bounded residual around the retained embedding. `C02_NO_GRAPH`, `C03_NO_ANCHOR`, and `FROZEN_RETAINED` are matched controls. No historical partition is used as a training target.

Every locked embedding produces the same 15 candidates: common KMeans, six one-init KMeans starts, two diagonal GMM starts, and two KMeans starts at each of three sparse diffusion strengths. The universal feasible set requires exact K, no singleton, and at least one real smallest-scale internal edge per cluster. `FEASIBLE_MEDOID` is the partition medoid. `NIGHT16H_FIXED_STRUCTURED` applies the frozen Night-16H rule: use the molecular champion only when its topology percentile is at least 0.5; otherwise use the topology champion.

Representation contribution is measured as learned common KMeans minus frozen-retained common KMeans. Decoder contribution is measured on the identical learned embedding as structured minus common KMeans and structured minus feasible medoid. Labels are loaded only by the independent evaluator after candidate partitions and hashes are written.
