# Night-14A source and mechanism audit

## Backbone decision

Night-14A fixed and read the actual source trees, not only their README files.

| lane | fixed commit | license | source-level finding | decision |
|---|---|---|---|---|
| SMART | `75676546a66d48a7ebfd7c5f3a2758a4c538fcfc` | GPL-3.0 | The official triplet construction calls a full pairwise distance matrix; the inspected model also keeps encoder/decoder modules in ordinary Python lists rather than `ModuleList`, so those modules are not registered parameters in that revision. | Audit/reference only; no official lane was represented as a valid Night-14A reproduction. |
| SpaBalance | `c3610a638c98c2d525c247ed62eeb41bda430e2d` | AGPL-3.0 | The source requests dense adjacency, contains cross-observation attention with an `N×N` footprint, branches by data type, and the inspected cross-attention constructor/call signatures do not close without changing source semantics. | Audit/reference only; source was not copied into SpaLORA. |
| clean-room sparse balanced cross-reconstruction graph autoencoder | Night-14A project source | project code | Independently written modality projections, one registered shared sparse graph-residual core, self/cross reconstruction, learned continuous fusion, and homoscedastic loss balancing. It accepts only tensors and sparse edge lists. | Selected traceable clean-room lane. |

The clean-room choice is not presented as a faithful SMART or SpaBalance reproduction. It is the contract-authorized fallback built from standard mature graph-autoencoder primitives after both official lanes failed the sparse, registered-parameter, and unchanged-semantics boundaries.

## TCF semantics and collision boundary

TCF is evaluated offline from one frozen trained checkpoint. For each observed sparse spatial edge, it measures cosine support in both private modality representations, their joint support, and their conflict. A global gate combines cross-modal topology disagreement and the fused representation's sparse-graph roughness. Below the frozen confidence floor, the operation returns the original fused tensor byte-for-byte. Otherwise it applies a convex mixture of ordinary spatial support and cross-modal evidence-weighted trust; the selected main lane uses only identity and low-pass channels. High-pass, support-only, fixed-low-pass, no-floor, and global-gate-only variants remain explicit ablations.

This differs from Night-13B B10 because B10 operated on one deterministic PCA embedding and had no two-private-view edge evidence or exact rejection channel. It differs from generic graph attention because TCF is an auditable sparse frequency response with an explicit identity fallback rather than a learned opaque attention layer. It differs from generic dynamic-graph replacement because the registered spatial graph is not replaced: cross-modal evidence only attenuates its edge response. These distinctions describe the implemented object; they are not a claim that the broad idea of conflict-aware graph filtering is already publication-novel.

## License and copying statement

No SMART GPL-3.0 or SpaBalance AGPL-3.0 source was copied into the project repository or compact. The source trees were read in their existing external audit roots. The Night-14A implementation is independently written and records the two projects as methodological and engineering references.
