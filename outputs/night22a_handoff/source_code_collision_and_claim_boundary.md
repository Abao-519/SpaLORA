# Night-22A source collision and claim boundary

The source audit closes a narrow but testable boundary. Deep Embedded Clustering (DEC) already covers trainable prototype assignment; SwAV and P²OT cover balanced or imbalanced soft assignment; DeepCut directly optimizes normalized-cut and correlation-clustering losses; broad multiview graph clustering covers graph fusion; BANKSY covers neighbor-derived geometry plus Leiden; spaMGCN, S3RL, SEPAR and CRCT cover spatial/multiview representations, prototypes or adaptive graphs. None of those ingredients is claimed as new here.

The only working object left for P0 is the *joint* direct partition variable: an `N×K` assignment is optimized against a locked representation through an elliptical cluster likelihood and against four equal-mass sparse graphs through cluster-conditioned graph-mixture weights. A one-sided cluster-volume floor prevents empty collapse without an equal-size target. The decaying start term is an optimizer stabilizer, not a contribution.

The matched arms are therefore decisive: `EMISSION_ONLY`, `SHARED_GRAPH_ONLY`, `CLUSTER_GRAPH_ONLY`, `ADDITIVE_SHARED`, and `FULL` use the same source, start, seed and budget. If `FULL` does not produce a new exact-K partition and strictly outperform the geometry start and all matched atomic arms, this revision must be reported as `NO_PARTITION_JUNCTION_SIGNAL`; neither the word junction nor the combination itself is evidence of novelty.

No third-party implementation was copied. P²OT's repository is research/personal-use only and BANKSY is GPL-3.0, so both were treated strictly as readable prior art. DeepCut and spaMGCN have permissive licenses, but the Night-22A implementation remains clean-room to keep algorithm attribution explicit.
