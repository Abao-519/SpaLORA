# Night-18E methods and proof semantics

## CCSR object

Confidence-Certified Self-Return (CCSR) starts from the fixed input partition. For each node, retained, RNA and second-modality prototype costs vote whether the current label is supported; a trusted node requires a registered number of view votes and a sufficient **rank of its raw prototype-unary margin**. This is rank-based and dataset-scale invariant. Dividing every margin by one positive global robust scale would not change its order, so the robust scale is diagnostic only and is not claimed as amplitude calibration.

For trusted node i, target label y0, dynamic unary margin m_i and total incident Potts capacity d_i, CCSR adds to every leave label

`g_i = max(0, d_i - m_i + epsilon_i)`.

The target label receives no added gap. The conservative epsilon includes the registered relative unary scale and an integer-rounding guard `2*(degree_i+4)/capacity_scale + 1e-12`. Thus `m_i+g_i>d_i`; switching i cannot gain enough pairwise energy to compensate its unary loss in an alpha move. The target stays fixed across cycles, while base unary and the required gap are recomputed each cycle. Since trusted nodes begin at their target and the premise is checked before every cycle, the per-move statement extends by induction across the run.

## Quantized implementation boundary

The test and formal solver share one round-to-int64 capacity map. Deterministic edge cases and 40 random tiny graphs enumerate every binary alpha subspace and assert both: (1) the formal min-cut integer energy equals the exhaustive optimum and (2) no exhaustive optimum or formal cut changes a trusted node. The certificate does not prove global optimality of the final dynamic multi-label problem.

## Matched-control boundary

The original rejected-mass stay cost remains only on untrusted nodes when enabled; trusted nodes receive the certificate leave penalty. Therefore FULL versus disabled/random/unary/view-support are not total-stay-mass-matched comparisons. `CCSR_CERTIFICATE_DISABLED` and `NIGHT15F_ORIGINAL_SELF_RETURN` isolate useful inherited paths; `CCSR_PROTECT_ALL` is an exact no-op sensitivity. The observed negative result prevents any component claim.
