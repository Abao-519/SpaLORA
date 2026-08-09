# SpaLORA Night-3A-F Deterministic PCA Run

**P0D: PASS (3/3 independent builds byte-exact); P0B-F: PASS (15/15); main experiment: 60/60; semantic label access during training: 0; IGE scientific go/no-go: PASS; architecture ablation: AUTHORIZED.**

`previous_night3ar_status = DETERMINISTIC_PREPROCESSING_BUG_FOUND_SCIENTIFIC_GO_NO_GO_NOT_EVALUATED`.
The old random PCA hash is diagnostic only. All four variants and five seeds consumed one immutable deterministic cache per dataset.

## Five-seed summary

| Dataset | Variant | ARI mean (SD) | NMI mean (SD) | Neighbor mean (SD) | Moran mean (SD) |
|---|---|---:|---:|---:|---:|
| a1 | C0 | 0.2121 (0.0222) | 0.3610 (0.0117) | 0.5779 (0.0278) | 0.5292 (0.0240) |
| a1 | C1 | 0.2271 (0.0274) | 0.3600 (0.0112) | 0.6052 (0.0544) | 0.5478 (0.0406) |
| a1 | IGE | 0.2469 (0.0096) | 0.3725 (0.0062) | 0.5766 (0.0147) | 0.4835 (0.0104) |
| a1 | ILN | 0.2049 (0.0163) | 0.3380 (0.0125) | 0.5936 (0.0103) | 0.5349 (0.0260) |
| placenta | C0 | 0.4670 (0.0186) | 0.5469 (0.0222) | 0.4798 (0.0152) | 0.4494 (0.0350) |
| placenta | C1 | 0.6825 (0.0289) | 0.7337 (0.0085) | 0.4308 (0.0022) | 0.3753 (0.0087) |
| placenta | IGE | 0.6044 (0.0361) | 0.6596 (0.0216) | 0.4773 (0.0175) | 0.4694 (0.0233) |
| placenta | ILN | 0.2296 (0.0619) | 0.3434 (0.0604) | 0.4644 (0.0315) | 0.4145 (0.0302) |
| p22 | C0 | 0.4065 (0.0125) | 0.5488 (0.0051) | 0.8262 (0.0159) | 0.7813 (0.0198) |
| p22 | C1 | 0.4142 (0.0187) | 0.5534 (0.0111) | 0.8271 (0.0166) | 0.7821 (0.0211) |
| p22 | IGE | 0.3881 (0.0333) | 0.5523 (0.0116) | 0.8192 (0.0090) | 0.7885 (0.0035) |
| p22 | ILN | 0.4635 (0.0263) | 0.5864 (0.0223) | 0.8453 (0.0061) | 0.7952 (0.0026) |

## Five-seed preregistered contrasts

| Dataset | Seed | IGE-C0 ARI | IGE-C0 NMI | C1-C0 ARI | C1-C0 NMI | IGE-C0 Neighbor | IGE-C0 Moran |
|---|---:|---:|---:|---:|---:|---:|---:|
| a1 | 0 | +0.0074 | +0.0103 | +0.0011 | +0.0033 | -0.0061 | -0.0373 |
| a1 | 1 | +0.0822 | +0.0256 | +0.0331 | +0.0137 | +0.0384 | -0.0224 |
| a1 | 2 | +0.0182 | -0.0099 | +0.0419 | -0.0096 | -0.0072 | -0.0530 |
| a1 | 3 | +0.0323 | +0.0036 | +0.0033 | -0.0106 | -0.0107 | -0.0567 |
| a1 | 4 | +0.0336 | +0.0276 | -0.0044 | -0.0020 | -0.0208 | -0.0590 |
| placenta | 0 | +0.1824 | +0.1412 | +0.2116 | +0.1931 | -0.0042 | -0.0082 |
| placenta | 1 | +0.1469 | +0.1200 | +0.2630 | +0.1925 | -0.0343 | -0.0222 |
| placenta | 2 | +0.1887 | +0.1499 | +0.2019 | +0.2020 | -0.0144 | -0.0088 |
| placenta | 3 | +0.0889 | +0.1095 | +0.2273 | +0.2041 | -0.0066 | +0.0704 |
| placenta | 4 | +0.0802 | +0.0430 | +0.1739 | +0.1423 | +0.0469 | +0.0689 |
| p22 | 0 | -0.0706 | -0.0085 | +0.0028 | +0.0022 | +0.0178 | +0.0446 |
| p22 | 1 | -0.0263 | +0.0021 | -0.0031 | -0.0037 | -0.0092 | +0.0018 |
| p22 | 2 | +0.0372 | +0.0161 | +0.0342 | +0.0205 | +0.0012 | +0.0002 |
| p22 | 3 | -0.0354 | +0.0007 | +0.0020 | +0.0021 | -0.0152 | +0.0013 |
| p22 | 4 | +0.0029 | +0.0074 | +0.0023 | +0.0021 | -0.0297 | -0.0122 |

## Night-2C bridge (diagnostic, different preprocessing)

- a1 C0 vs V0, ari: -0.0213.
- a1 C0 vs V0, nmi: -0.0044.
- a1 C0 vs V0, spatial_cluster_moran_mean: -0.0167.
- a1 C0 vs V0, spatial_neighbor_agreement: -0.0290.
- a1 C1 vs V1, ari: -0.0027.
- a1 C1 vs V1, nmi: -0.0057.
- a1 C1 vs V1, spatial_cluster_moran_mean: +0.0114.
- a1 C1 vs V1, spatial_neighbor_agreement: +0.0095.
- p22 C0 vs V0, ari: -0.0152.
- p22 C0 vs V0, nmi: -0.0096.
- p22 C0 vs V0, spatial_cluster_moran_mean: +0.0012.
- p22 C0 vs V0, spatial_neighbor_agreement: -0.0014.
- p22 C1 vs V1, ari: -0.0086.
- p22 C1 vs V1, nmi: -0.0085.
- p22 C1 vs V1, spatial_cluster_moran_mean: -0.0010.
- p22 C1 vs V1, spatial_neighbor_agreement: -0.0024.
- placenta C0 vs V0, ari: +0.0163.
- placenta C0 vs V0, nmi: +0.0181.
- placenta C0 vs V0, spatial_cluster_moran_mean: -0.0218.
- placenta C0 vs V0, spatial_neighbor_agreement: -0.0116.
- placenta C1 vs V1, ari: +0.0214.
- placenta C1 vs V1, nmi: +0.0101.
- placenta C1 vs V1, spatial_cluster_moran_mean: -0.0089.
- placenta C1 vs V1, spatial_neighbor_agreement: +0.0011.

These bridge differences are not a paired test under identical preprocessing and are not a failure gate.

## Scientific gates

- Placenta C1 recovery fraction: 0.6376.
- Spatial joint-decline failures: none.
- Weighted-gradient influence collapse flags: 0.
- Attention saturation flags: 0.
- Strict reasons: none.
- Weighted-gradient influence, not scalar loss fraction, is the mechanism hard gate.
- Architecture ablation recommended: yes.

No PCA/training seed, scale, solver, tau, formula, ASR, evaluator, or threshold search occurred.
