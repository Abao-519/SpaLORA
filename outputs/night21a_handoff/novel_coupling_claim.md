# Night-21A coupling boundary

AMCF does **not** claim neighbourhood means, graph high-pass features, attention, graph convolution, autoencoding, contrastive learning, prototype clustering, or residual connections as new. BANKSY already supplies the closest texture primitives; SpatialGlue supplies a clear within-/cross-modality attention precedent; spaMGCN supplies adaptive multi-order graph fusion; m2ST and GraphST-like methods supply multiscale graph-autoencoding precedents.

The only candidate contribution left after the collision audit is the following joint computation: for each modality, form a registered sparse multiscale bank containing self, low-pass mean, and graph high-pass channels; learn node-wise channel weights, then node-wise cross-modality weights; use the composed result only through a bounded residual whose final projection is exactly zero at initialization around an independently auditable retained carrier. This is a narrow compositional hypothesis, not an originality claim established by architecture inspection.

The empirical claim is therefore fail-closed. It requires the full arm, under the same carrier, endpoint, seeds, loss budget and training steps, to beat mean-only, no-gradient, no-hierarchy, no-anchor, fixed low-pass, and carrier-only controls. If this does not occur on both modality families, the correct conclusion is `NO_COMPOSITIONAL_METHOD_SIGNAL`, regardless of whether an individual run gives a visually plausible or high absolute partition.

Implementation is clean-room. No third-party source file is imported into the model. The registered-graph high-pass channel is `X - P_s X`; it is axis-free after graph registration and is not a copy of BANKSY's azimuthal Gabor implementation.

