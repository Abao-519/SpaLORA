# Unified Reliability-Structured Decoder: common computation graph

All lanes call the same numeric producer API. Dataset identity is absent from the core.

`registered reduced views -> common start bank -> prototype unary -> sparse multiscale reliability energy -> optional morphology(presence mask) -> rejected-mass self-return -> alpha-expansion authority start -> generic split/merge/boundary repair -> guard -> independent evaluator`

For observation i and cluster c, the inherited structured stage minimizes a sparse Potts energy

`E(z)=sum_i U_i(z_i)+sum_(i,j in G) beta*w_ij*[z_i!=z_j]+sum_i rho_i*[z_i=z_i(current)]+size(z)`.

Here `U` is a prototype-distance unary; `w_ij` combines two-modality agreement/conflict over fine, registered and broad sparse graphs; rejected conductance mass yields the current-state self-return `rho`; morphology is another numeric view and has exact zero-mask fallback. Night-16B then applies one generic repair bank. Small predicted clusters are merged by prototype distance plus sparse boundary support; the most dispersed surviving cluster is split by a deterministic PCA quantile or two-means proposal; boundary moves minimize prototype unary plus sparse neighbour disagreement while preserving the registered minimum size. No dense N x N matrix is formed.

The tuned benchmark profile selects an already locked partition using public ARI, then NMI, lower complexity and larger minimum cluster. This selection is label-assisted benchmark HPO, not an automatic no-label selector.
