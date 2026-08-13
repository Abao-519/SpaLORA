# Night-5D evaluation-only recovery report

Terminal status: `P22_PARTIAL_OR_MIXED_EVIDENCE`

This recovery consumed only the frozen 50-row metric table. It performed 0 training units, 0 diffusion transforms, used 0 GPUs, and did not re-read P22 labels or run artifacts.

Canonical boundary is deterministically `1 - spatial_neighbor_agreement`; it is not independent evidence. The frozen symmetric-union boundary is retained only as a diagnostic and never enters inference or protection gates.

## Primary locked contrasts

- B01_C04_SHRINK25-B00_C00_FULL_IGE: ΔARI=0.0247043951, ΔNMI=0.0024338477, ΔQ=0.0135691214, wins=8/10, exact p=0.1328125000, Holm p=0.3662109375, Q CI=[-0.0097155922, 0.0317985888], material=False, spatial=True, interpretation=positive_mean_but_statistically_inconclusive.
- B10_SHRINK25_ANCHOR10-B00_C00_FULL_IGE: ΔARI=0.0236542313, ΔNMI=-0.0066990058, ΔQ=0.0084776128, wins=6/10, exact p=0.2294921875, Holm p=0.3662109375, Q CI=[-0.0112893007, 0.0287919856], material=False, spatial=True, interpretation=positive_mean_but_statistically_inconclusive.
- B17_C09_DIFFUSE10-B00_C00_FULL_IGE: ΔARI=0.0274536974, ΔNMI=-0.0071950383, ΔQ=0.0101293295, wins=7/10, exact p=0.1220703125, Holm p=0.3662109375, Q CI=[-0.0051123793, 0.0245342868], material=False, spatial=False, interpretation=positive_mean_inconclusive_spatial_fail.

Positive means are not described as significant gains. B01 and B10 are directionally positive but statistically inconclusive; B17 is directionally positive, statistically inconclusive, and fails the preregistered spatial protection gate.

## Secondary mechanisms

- B10_SHRINK25_ANCHOR10-B01_C04_SHRINK25: ΔQ=-0.0050915086, Holm p=0.9101562500, Δneighbor=-0.0071512070, ΔMoran=-0.0035305548, ΔGeary=0.0034054819.
- B17_C09_DIFFUSE10-C09_RNA_ANCHOR10: ΔQ=0.0009033753, Holm p=0.9101562500, Δneighbor=0.0090209058, ΔMoran=0.0124426502, ΔGeary=-0.0123699384.

P22 participated in earlier Night-3B architecture analysis, so this is not a pristine external test. Night-5 candidate ranking used A1/Placenta only; methods must not be altered after this evaluation.

Original `outputs/night5d_handoff` remains byte-identical and retains its valid `BASELINE_REPLAY_MISMATCH` record for the first attempt.
