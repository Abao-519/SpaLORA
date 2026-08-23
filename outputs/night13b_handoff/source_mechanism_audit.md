# Night-13B source and mechanism audit

Night-13B did not run or copy external methods. The unified residual was implemented clean-room from the mathematical objects already present in this project: a shared embedding, two row-stochastic sparse spatial operators, continuous content-derived gating, and a fixed common endpoint. No external source code appears in `SpaLORA/night13b_unified.py`.

| Source | Frozen reference | License / copying boundary | What was used |
|---|---|---|---|
| SpaLORA C00/G04+H05 | historical manifests and artifacts audited row-by-row | project history | strong RNA+protein reference and graph/fusion semantics only |
| SpaLORA F00/R02 | Night-7B/Night-10B frozen artifacts | project history | strong RNA+ATAC reference only |
| SpaLORA N02 | Night-9B `N02_HIER_ONLY` checkpoints/artifacts | project history | hierarchical residual as a comparison and design warning; no direct code reuse |
| SpatialGlue | `7c976d811d27ace51ce47ae0ad94a068a7d222fa` | AGPL-3.0; no code copied | paper/source mechanism context only |
| SMART | `75676546a66d48a7ebfd7c5f3a2758a4c538fcfc` | GPL-3.0; no code copied | paper/source mechanism context only |
| ARISE | `fefdd849494c0d08e755052a7a31b20169945e440` | no LICENSE detected; no code copied | paper/source mechanism context only |
| SpaMV | `d7105ef70e9276350e8a12bddfbd3d396d1c33d2` | no LICENSE detected; no code copied | shared/private and cross-reconstruction context only |
| SpaMode | `d8d8e2b70c6ad47ef12aa1a5d9a65cf4fb226c00` | no LICENSE detected; no code copied | invariant/variant MoE context only |
| SpaBalance | `c3610a638c98c2d525c247ed62eeb41bda4330e2d` | AGPL-3.0; no code copied | gradient-balancing context only |

The candidate formula is therefore a project clean-room synthesis, not a port of any listed method. External board entries remain protocol context and are explicitly not fair head-to-head results.
