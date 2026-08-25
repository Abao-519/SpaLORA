# Source-code collision and novelty audit

## Decision

Every primitive in the Stage-A object has strong prior art. Graph total variation, robust/Huber fidelity, group-sparse residuals, shared/private multi-view decomposition, and trainable algorithm unrolling cannot be claimed as new. The only not-yet-exactly-matched object found in this bounded audit is their same-spot spatial multi-omics combination: one shared tissue graph trend plus spot-level modality-private group-sparse residuals, consumed by clustering. Because Stage A failed 0/3, Night-18C makes no novelty claim for that combination and does not open an unrolled network.

| Prior work | Audited object | Collision with Night-18C | Consequence |
|---|---|---|---|
| Trend Filtering on Graphs, JMLR 2016 | graph fused lasso / graph trend filtering | Direct collision with graph-TV prior | graph-TV is mature scaffold |
| Vector-valued graph trend filtering | vector graph signals and ADMM solvers | Direct collision with vector-valued trend solver | vector extension is not new |
| Graph Unrolling Networks, IEEE TSP / arXiv:2006.01301 | unsupervised trainable graph trend-filtering unrolling | Direct collision with proposed Stage-B solver unrolling | unrolling is not new |
| Unrolling Nonconvex Graph Total Variation, arXiv:2506.02381 | Huber-related graph-TV and ADMM unrolling | Strong collision with robust graph-TV/unrolling | robust unrolling is not new |
| Robust multi-view shared/private factorization (IJCAI/AAAI line) | shared and private latent structure with structured sparsity | Strong component collision | shared/private wording is prior art |
| SpaMV, Nature Communications 2026 | spatial multi-omics shared/private encoders and mixture-of-experts | Strong application-domain collision | ordinary shared/private spatial fusion is not new |

Audited primary entry points:
- https://www.jmlr.org/beta/papers/v17/15-147.html
- https://arxiv.org/abs/2006.01301
- https://arxiv.org/abs/2506.02381
- https://www.nature.com/articles/s41467-026-74718-1

This was a bounded collision audit, not a proof of global novelty.
