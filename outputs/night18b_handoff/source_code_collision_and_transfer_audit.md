# Night-18B source collision and transfer audit


| Object | Official source | Prior-art boundary | Night-18B decision |
|---|---|---|---|
| SMGFM (2026) | https://arxiv.org/abs/2606.12867 | Chebyshev graph-frequency bands, band semantic roles, coupling reliability, consensus/private routes | Original frequency-coherence proposal rejected before execution; never claimed. |
| GatorPrism | https://github.com/Gator-Group/GatorPrism | Joint coalition expert, modality-private experts, prototype-conditioned spot router | Expert routing/shared-private scaffold not novel and not implemented here. |
| ARISE | https://github.com/XiangxiangWang-code/ARISE | RNA-anchored graph intersection and hierarchical fusion | RNA anchor/graph intersection/fusion are prior art. |
| DRIFT | https://github.com/rsinghlab/DRIFT | Heat-kernel low-pass preprocessing before a downstream model | Generic graph diffusion/low-pass is prior art and is a matched control. |
| SpaDDM | https://github.com/WHY-17/SpaDDM | Directional graph diffusion for spatial multi-omics | Directional diffusion is prior art; no such novelty claim is made. |
| SpaGFT / DeepGFT | official paper/code entrances audited in taskbook | Graph Fourier representation and filtering | Spectral filtering itself is prior art. |
| SpatialCOC / FOCUS | official papers/code entrances audited in taskbook | Continuous spatial functions and cross-resolution mapping | Continuous/cross-resolution correction is prior art. |
| Night-18B bounded response calibration | this clean-room source | Robust input-derived modality roughness, bounded rational inverse/forward response to a common target, same residual consumer | Minimal distinct hypothesis only; empirical independent-contribution gate failed, so novelty remains unsupported. |

No third-party code was copied.  Official repositories were used only for semantic collision review.
