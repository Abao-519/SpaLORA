# AMCF module contract

1. Inputs are two finite numeric matrices, a retained carrier `Z0`, three nonnegative sparse registered graphs, and known `K`.
2. Each modality is robustly standardized. At each sparse scale `s`, `P_s` is row-normalized and produces `P_s X` and `X-P_s X`.
3. One adapter per modality is shared across that modality's channels. Node-wise softmax weights compose channels within each modality.
4. A second node-wise softmax composes the two modality-specific states. No dataset identifier is accepted by the core API.
5. The output is `Z0 + b sigmoid(g_i) tanh(R_i)`, with the residual projection initialized to exactly zero and `b=0.12` in the frozen base profile.
6. The minimal frozen loss is two-view reconstruction, weak cross-modal alignment, carrier trust, and a low-weight unlabeled prototype compactness term.
7. The primary matched endpoint is KMeans with known `K`, `n_init=20`, and seed 0. Labels are opened only by a separate evaluator after the representation and partition hashes are locked.
8. Sparse operator cost is `O(sum_s nnz(P_s) * (d1+d2))`; neural cost is linear in observations and channels. No dense observation-by-observation matrix is formed.
