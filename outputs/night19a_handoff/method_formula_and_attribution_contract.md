# Night-19A method and attribution contract

The zero-start anchor is the **registered relation-smoothed carrier**, not the raw retained embedding.  Let `g_p` be the gradient of the weighted anchor, masked-view consistency, and variance protection losses.  Relation loss is split into six registered sparse groups: spatial/feature-neighbour edge type crossed with low/mid/high evidence support.  For group `s`, if `g_s · g_p < 0` after the two-update zero-start warm-up,

`g'_s = g_s - rho_s (g_s · g_p / ||g_p||^2) g_p`, with `rho=(5/6,1/2,1/6)` for low/mid/high support.

The composite adjusted relation gradient is L2-matched to the unadjusted composite relation gradient before adding `g_p`.  Hence the primary comparison changes direction, not the overall relation-gradient norm.  The permuted arm performs a SHA-256 endpoint-stable bijection within spatial/feature edge type and exactly matches the base-relation-weighted projection-strength mass within each type.  The only possibly distinct object is this registered sparse edge-evidence conditioning of relation-gradient strata against strong-start protection.  Generic confidence scaling, block/layer projection, PCGrad, minimum-norm aggregation, and shared/private multimodal learning are prior art.

Stage A used one preregistered seed and one parameter profile.  Labels were not used to alter the formula, optimizer, checkpoint, representation, or partition.  Because full failed the 2/3 matched-control gate, no multi-seed stage and no HPO rescue were run.
