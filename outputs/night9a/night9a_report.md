# SpaLORA Night-9A report

## Terminal decision

`NIGHT9A_NO_EFFICIENT_R02_REPLACEMENT_KEEP_FULL_F00`

Night-9A completed all 54 registered R1 chains successfully, but zero of nine
candidates passed every pre-registered gate. R2 and R3 were therefore not run.
The locked recommendation is to keep the full F00/R02 route and not claim an
efficient topology-transfer replacement.

## Plain-language scientific result

The old F00 is slow because it trains both a full G04 backbone and a second full
G00 backbone before the fixed R02 relation adapter. Night-9A removed that second
from-scratch backbone: E00--E04 transferred the G04 state into G00 with 0--320
fixed epochs, while E05--E08 used deterministic sparse topology projections.

The closest candidate was `E03_WARM_FULL_E160`. Its mean P22 delta Q versus
full F00 was -0.000945, its delta Q versus U00
was +0.023945, MISAR mean partition ARI was
0.901296, and runtime remained
1.9968x. The fastest candidate,
`E05_SGC1_RESIDUAL50`, still required 1.8858x
U00, above the 1.50x gate. The highest P22-Q-preserving candidate,
`E08_DELTA_TOPOLOGY_RESIDUAL100`, changed Q versus full F00 by
+0.000640, but failed these gates:
runtime, p22_q_vs_u00, p22_nmi_vs_full, p22_spatial, misar_mean_ari, misar_mean_nmi.

Thus the registered transfer idea provides useful negative engineering evidence,
but it cannot currently be promoted as the paper's efficient topology-transfer
innovation. MISAR results are label-free partition-fidelity diagnostics only;
raw MISAR Y was not read and no metric inheritance claim is made.

## Audit summary

- Authority/base/tag and 275 teacher artifact rows: PASS; before/after exact.
- P0 strict transfer, projection determinism, same-head parity: PASS.
- P0 real P22 seed-0 construction smoke: 9/9 PASS.
- Formal science: R1 54/54 success; R2 0; R3 0.
- Independent metric/resource/fidelity maximum absolute error:
  5.55e-16 (tolerance 1e-12).
- Scientific retry: 0; fallback: 0; timeouts: 0.
- Infrastructure corrections: 3/4, all before formal science and preserved.
- P22 label windows: one locked R1 window; MISAR Y reads this task: 0;
  lineage total remains 2 and a third read remains forbidden.
- Third-party benchmark: 0; SOTA claim: false; AutoDL API: unused.
