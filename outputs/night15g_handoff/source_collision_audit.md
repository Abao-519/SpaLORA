# Night-15G source and novelty collision audit

Night-15G does not claim that adding H&E as an optional third view is novel. MISO, Proust, STESH, stGCL, SpatialEx and COSIE already establish nearby image-plus-molecular graph, contrastive or fusion precedents. BANKSY and PRAGA also cover neighborhood statistics and dynamic graph construction, while BayesSpace, BASS, DR.SC and SCGP establish spatial clustering priors and graph-aware clustering context.

The only defensible object left by this sprint is narrower: a clean-room combination of a numeric `views[] + presence mask` interface, local morphology consistency/conflict reliability, and exact molecular fallback when the image is absent. Even that object has only public-benchmark development evidence here. The strongest D1 nonmicro result uses a PCA plus tied-covariance GMM consumer, whereas A1 and tonsil s3 use the optional morphology energy, so Night-15G has not yet shown one final unified consumer. No new trainable representation loss was implemented.

Implementation provenance:

- ImageNet ResNet18 was used in evaluation mode through torchvision; weight SHA-256 is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; torchvision is BSD-3-Clause.
- The optional morphology energy and runners are project clean-room code. No third-party method source was copied into the project.
- External papers/source were used only to delimit novelty; no complete external benchmark method was run in Night-15G.

Conclusion: `LOCAL_SIGNAL`, not a novelty lock, SOTA claim, confirmed milestone or paper-ready method.
