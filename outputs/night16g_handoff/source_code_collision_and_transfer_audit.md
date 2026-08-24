# Night-16G source-code collision and transfer audit

## Scope and conclusion

The audit covered the formal papers and official project entry points for STCC, SACCELERATOR, PHD-MS, SCALE and SMODEL. The purpose was collision checking, not third-party benchmark reproduction. No third-party source was copied into SpaLORA.

The collision is substantial: consensus over base partitions, persistence across multiscale clusterings, entropy/rank-based model choice and weighted spatial multi-omics ensembles all have clear prior art. In addition, Night-16G's fitted persistence coefficient is exactly zero. Consequently Night-16G does **not** claim a new persistent-basin or consensus method. Its supported object is a clean-room sparse engineering combination of molecular separation, categorical spatial topology and predicted small-cluster risk for selecting a locked candidate partition; novelty remains unestablished.

## Representative precedents

| Work | Official paper/source entry | Relevant prior object | Night-16G boundary |
|---|---|---|---|
| STCC | Genome Research, DOI `10.1101/gr.280031.124`; `github.com/hucongcong97/STCC` | Consensus of base clusterings through hypergraph/factorization machinery | Ordinary partition consensus or medoid is a control, not a contribution claim. |
| SACCELERATOR | Nature Methods 2026; `github.com/SpatialHackathon/SACCELERATOR` | Consensus-guided multi-method workflow and cross-method uncertainty | Night-16G does not claim that aggregating several clusterers is new. |
| PHD-MS | Cell Reports Methods 2026, DOI `10.1016/j.crmeth.2026.101376` | Persistent-homology reasoning across clustering resolutions | Ordered path persistence is directly collision-sensitive and was unsupported by the fitted result. |
| SCALE | 2025/2026 official article (PMC12774663) | Spatial clustering at multiple levels with automated scale search | A multiscale path alone cannot support novelty. |
| SMODEL | Communications Biology 2025; `github.com/liying-1028/SMODEL` | Element-wise weighted ensemble and dual-graph integration | Night-16G avoids dense observation-level co-association, but weighted ensemble is established prior art. |

## Transfer/implementation note

Only mathematical ideas described in the audit were implemented clean-room. The formal selector uses NumPy/SciPy/scikit-learn primitives already licensed in the project environment. It constructs no dense `N×N` observation matrix; the only dense similarity object is candidate×candidate (89×89).

