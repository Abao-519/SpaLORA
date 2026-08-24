# Night-16E source-code collision and transfer audit

Night-16E is a clean-room implementation. No third-party implementation was copied into SpaLORA or the compact.

| Prior work | Frozen authority | Established prior art / collision boundary |
|---|---|---|
| Boykov–Veksler–Zabih graph cuts | ICCV 1999 / PAMI 2001, DOI `10.1109/34.969114` | Potts energy and alpha-expansion large-neighborhood moves are prior art. |
| BANKSY | commit `9278996c39e376277d57ef95278000447ba6c57c`; GPL-3 | Neighborhood means, directional gradients and multiscale feature banks are prior art. |
| BayesSpace | commit `ba8b42211ef1581d31a659766559bd6e42be3acc`; MIT + file | Molecular unary plus spatial prior is prior art. |
| BASS | commit `5c2690fddcdeb12461a69106ff4681777d0a8822`; GPL-3 | Bayesian clustering with Potts regularization is prior art. |
| DR.SC | commit `8c4ddca9240f36cf713aeca888c7265f31bc8622`; GPL-3 | Joint dimension reduction and spatial clustering is prior art. |
| SCGP | commit `8a081f339ba88c44dfb0c18cefd529a7e53e7431`; license unresolved | Feature-weighted spatial graph / constant-Potts clustering is prior art; no source copied. |
| PRAGA | commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784`; AGPL-3 | Dynamic graphs, prototype aggregation and prototype contrastive structure are prior art. |
| SpaMV | commit `d7105ef70e9276350e8a12bddfbd3d396d1c33d2`; MIT | Shared/private latent decomposition and private preservation are prior art. |
| MultiGATE | official `cuhklinlab/MultiGATE`, Apache-2.0 | Cross-modal graph attention and multimodal spatial clustering are prior art; its human-hippocampus processed/reference artifacts are used only as an independent benchmark carrier. |

## What Night-16E actually tests

The common sparse energy gives three soft relations different legal roles: support modulates a nonnegative Potts capacity, consensus boundary adds a frozen-cycle exclusion unary, and conflict mixes private-modality prototype costs by numeric reliability. Absolute rejected capacity is not renormalized away; it contributes a current-state stay cost. `support_mix<1` means the full operator is **base Potts plus tri-state modulation**, not pure support-only smoothing.

The independent human-hippocampus ablation supports only a narrower combination: support modulation plus rejected-mass self-return. `BOUNDARY_OFF` and `CONFLICT_PRIVATE_OFF` are numerically identical to full, while `SUPPORT_MODULATION_ONLY` is better than full. Therefore Night-16E makes no positive claim for the complete three-state mechanism. It also records that boundary/private relation carriers inherit the base smoothing edge, which may suppress the very relations they are intended to carry; this scientific formula is frozen and deferred rather than changed after evaluation.

The defensible empirical claim is a family-frozen RNA+chromatin operator signal on one independent study, plus transparent public-benchmark score-frontier gains. It is not a two-family unified success, external SOTA, confirmed milestone, or paper-ready evidence.
