# SpaLORA Night-9B report

## Plain-language outcome

Night-9B completed the fair COSMOS calibration and the preregistered MF-RACF funnel. The final terminal status is `NIGHT9B_COSMOS_GAP_CALIBRATED_NO_RACF_CANDIDATE`: no MF-RACF candidate survived the five-seed cross-dataset gates, and no SOTA claim is made.

On P22, the locked F00/R02 reference remained stronger than COSMOS under this project's identical K/input/seed protocol. F00 mean ARI/NMI/Q were 0.467740/0.633373/0.550556. COSMOS common-head mean ARI/NMI/Q were 0.449967/0.622008/0.535988, with delta Q -0.014569. Its native fixed endpoint was substantially weaker (delta Q -0.152672). The paper-reported COSMOS ARI=0.63 remains context only and is not claimed to be the same protocol.

The only R1 candidate promoted was `N02_HIER_ONLY`, as a provisional brain specialist. Across five seeds it improved P22 by delta ARI +0.038517, delta NMI +0.022814, and delta Q +0.030665, with 5/5 Q wins and a passing spatial-protection gate. However, on A1 it changed ARI -0.005915, NMI -0.005930, and Q -0.005923, with only 1/5 Q wins. Its A1 Q decline crossed the locked brain-specialist protection bound, so it was correctly rejected after R2.

The useful mechanistic signal is therefore narrow: hierarchical fusion helps the P22 development setting, but RNA-anchor, reliability gating and DGI combinations did not yield a transferable A1+P22 candidate under the locked gates. The prudent next step is to retain the family baselines (C00 for RNA+protein and F00/R02 for RNA+epigenome), treat hierarchy as a platform-specific development clue rather than a final method, and design any future round prospectively without returning to these labels for tuning.

## R1 frozen three-seed deltas

| Candidate | A1 ΔARI | A1 ΔNMI | A1 ΔQ | P22 ΔARI | P22 ΔNMI | P22 ΔQ |
|---|---:|---:|---:|---:|---:|---:|
| N01_ANCHOR_K10 | -0.000370 | -0.000197 | -0.000284 | +0.020137 | +0.009877 | +0.015007 |
| N02_HIER_ONLY | -0.003142 | -0.005653 | -0.004398 | +0.050316 | +0.031122 | +0.040719 |
| N03_ANCHOR_K10_HIER | +0.000721 | +0.000698 | +0.000709 | +0.021265 | +0.010531 | +0.015898 |
| N04_ANCHOR_K15_HIER | -0.001126 | +0.000569 | -0.000279 | +0.022240 | +0.010665 | +0.016452 |
| N05_ANCHOR_K10_HIER_DGI | +0.000174 | +0.000348 | +0.000261 | +0.020755 | +0.010455 | +0.015605 |
| N06_ANCHOR_K10_HIER_REL | -0.004391 | -0.004149 | -0.004270 | +0.014965 | +0.006237 | +0.010601 |
| N07_ANCHOR_K10_HIER_REL_DGI | -0.003270 | -0.003303 | -0.003287 | +0.015665 | +0.006901 | +0.011283 |

## Five-seed finalist audit

| Dataset | Delta ARI | Delta NMI | Delta Q | Q wins | Worst-seed delta Q | Spatial gate |
|---|---:|---:|---:|---:|---:|---|
| A1 | -0.005915 | -0.005930 | -0.005923 | 1/5 | -0.010215 | True |
| P22 | +0.038517 | +0.022814 | +0.030665 | 5/5 | +0.000821 | True |

## Protocol and integrity audit

- P0 authority: independently accepted commit `397638728debfe2c9aef0b6425776b471bfe4977`; P0 was not rerun in REV1.
- P1: semantic/gradient/real-construction/checkpoint tests passed; 7 implementation and 2 environment-only correction cycles were retained before formal science.
- Formal computation: 5 COSMOS training units, 42 R1 units, and 4 conditional R2 units. Scientific retry=0 and fallback=0.
- COSMOS: one training per seed shared by native/common endpoint lanes; 5/5 training outputs and 10/10 endpoint partitions locked. Seeds 0/1 required environment-only endpoint completion without retraining after R discovery failed post-save.
- Label firewall: A1 and P22 each read once in one evaluator process after total lock; MISAR Y read count remained 0. No return to structure, checkpoint, seed, epoch, hyperparameter or clustering selection followed.
- Independent contingency-table evaluator: PASS, 56 rows, maximum ARI/NMI absolute error 5.551e-16 against tolerance 1.0e-12.
- P22 is development data, not a pristine holdout. No SOTA claim is authorized.
- Large raw runs/checkpoints/embeddings/affinities remain at `/root/autodl-fs/night9b_racf_20260820`.

## Git and delivery

- Authoritative parent: `aa933b8fc11fc05470287a21f80facd03d0acfb9`.
- Accepted P0 commit: `397638728debfe2c9aef0b6425776b471bfe4977`.
- Pre-final evidence commit: `b377dee510e5c65abc87b2316f6b8dcfae8cf7e7`.
- Branch: `revision/q2-night9b-rna-anchor-cooperative-fusion-rnd-20260820`.
- Planned one-time annotated tag: `night9b-final-20260820` (created only after the tracked delivery-index commit).
