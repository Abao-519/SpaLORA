# Method semantics and formula contract

Night-18C Stage A tests a deterministic **robust shared/private graph-trend decomposition**. After per-view robust scaling and PCA, the second view's score coordinates are aligned to the first by same-spot orthogonal Procrustes; merely matching dimensions is forbidden.

For coordinated views H1 and H2, shared tissue trend Z and spot-private residuals R1,R2 solve the registered objective

`0.5 * sum_m sum_i Huber_delta(||Z_i + R_mi - H_mi||_2) + lambda * sum_(i,j) a_ij ||Z_i-Z_j||_2 + 0.5*gamma*sum_m sum_i ||R_mi||_2`.

Monotone backtracked block updates optimize Z; group shrinkage updates R. The clustering representation concatenates the unchanged retained anchor with a fixed-weight shared trend block. Every arm uses the same `KMeans(K, n_init=20, random_state=0)` endpoint. Labels are not an input to alignment, decomposition, representation, or clustering.

Matched arms are retained-only, identity fused, ordinary L2 low-pass, graph-TV-only (`R=0`), private-only (`lambda=0`), full, and an ID-stable permutation of private residual locations. Trainable unrolling is disallowed unless full has unexplained ARI/NMI gains on at least two of three lanes.
