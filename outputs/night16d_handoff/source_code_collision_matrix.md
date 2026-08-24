# Night-16D source-code collision matrix

No third-party implementation was copied. The module is a clean-room implementation; repositories/papers were inspected for attribution and collision risk.

| Method | Fixed official source | License observed | Prior art that cannot be claimed | Night-16D boundary |
|---|---|---|---|---|
| PRAGA | `Xubin-s-Lab/PRAGA` commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784` | AGPL-3.0 | dynamic graph, prototype aggregation and prototype contrastive learning | prototypes and adaptive aggregation are prior art |
| SpaMV | `ericcombiolab/SpaMV` commit `d7105ef70e9276350e8a12bddfbd3d396d1c33d2` | MIT | shared/private latent decomposition, cross reconstruction and HSIC | private preservation alone is prior art |
| SpaBalance | `nudt-bioinfo/SpaBalance` official repository, inspected 2026-08-24 | AGPL-3.0 | shared/private dual learning, cross-modal attention and balance learning | balancing modalities alone is prior art |
| ARISE | `XiangxiangWang-code/ARISE` commit `fefdd849494c0d08e755052a7a31b20169945e40` | no LICENSE observed in fixed snapshot | RNA-anchored intersection topology and hierarchical fusion | anchored graph intersection alone is prior art; code not copied |
| CoMo | Bioinformatics 2026 article `bbag192`; official source was not unambiguously resolved in this audit | not resolved | graph autoencoder, cross-attention, neighbor/cluster contrastive objectives | cross-modal graph contrastive learning is prior art; no source transferred |
| PRESENT | `lizhen18THU/PRESENT`, official repository inspected 2026-08-24 | repository LICENSE present; exact SPDX not relied on here | contrastive cross-modality representation and multi-sample integration | cross-modal contrastive alignment is prior art |
| SpatialMOSI | Genome Research 2026 article, DOI `10.1101/gr.281568.125` | source/license not resolved in bounded audit | hierarchical graph contrastive cross-omic/cross-slice integration | hierarchical graph contrastive integration is prior art |

The only tested Night-16D object is the *combination* of a three-state edge field with relation-specific trainable operations: support low-pass, consensus-boundary high-pass/separation, and conflict-private preservation, constrained by teacher anchoring and rejected-mass/trust statistics. Even this combination did not produce independent cross-study gains, so Night-16D makes no positive novelty claim. Exact retained embeddings were used only as an audited plug-in anchor; this is not raw-feature end-to-end training.
