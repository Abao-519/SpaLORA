# Night-16C source-code collision and transfer audit

This audit uses papers and official repositories as attribution/context only. No third-party implementation was copied into SpaLORA.

| Method | Fixed official source | License observed | Existing idea that must not be renamed | Night-16C boundary |
|---|---|---|---|---|
| BANKSY | `prabhakarlab/Banksy_py`, release v1.3.4 / commit prefix `6f22bb0`, https://github.com/prabhakarlab/Banksy_py | GPL-3.0 | neighbor-mean and azimuthal-gradient feature augmentation; spatially weighted sparse neighbors | directional residuals alone are not novel |
| stLVG | `YikaiLou/stLVG`, official repository inspected 2026-08-24, https://github.com/YikaiLou/stLVG | MIT | angle/direction-weighted graph views and multi-view contrastive learning | directional weighting alone is not novel |
| ARISE | `XiangxiangWang-code/ARISE`, local fixed commit `fefdd849494c0d08e755052a7a31b20169945e40`, https://github.com/XiangxiangWang-code/ARISE | no LICENSE observed in fixed snapshot | RNA-anchored intersection of feature and spatial graphs | graph intersection/anchoring alone is not novel; code not copied |
| PRAGA | `Xubin-s-Lab/PRAGA`, commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784`, https://github.com/Xubin-s-Lab/PRAGA | AGPL-3.0 | dynamic prototype aggregation and prototype contrastive learning | prototype margins/aggregation alone are not novel; code not copied |
| SpatialCOC | `xjtu-omics/SpatialCOC`, release v0.1.0 / Zenodo 10.5281/zenodo.18591935, https://github.com/xjtu-omics/SpatialCOC | GPL-3.0 | spatial continuous mapping and cross-omics correction | cross-omics correction alone is not novel |
| SpaMV | `ericcombiolab/SpaMV`, local fixed commit `d7105ef70e9276350e8a12bddfbd3d396d1c33d2`, https://github.com/ericcombiolab/SpaMV | MIT in current official repository | shared/private VAE, cross reconstruction, HSIC separation | shared/private decomposition alone is not novel |

GitHub rate limiting prevented resolving a single current `main` commit for BANKSY/stLVG during final audit; fixed releases/repository URLs and licenses are recorded rather than inventing hashes.

The only provisional Night-16C method claim is the clean-room combination of (i) an explicit tri-state sparse cross-modal edge field (`support`, `boundary`, `conflict`), (ii) absolute rejected-conductance mass retained as self-return, and (iii) a conjunctive trust rule requiring unstable starts, weak prototype margin and boundary evidence before moving a spot, under one frozen numerical configuration per modality family. The matched ablation is mixed, so universal necessity of every subterm is not claimed. Novelty remains subject to a broader literature review.
