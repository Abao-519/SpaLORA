# Official spaMGCN source and compatibility audit

- Fixed upstream: `hongfeiZhang-source/spaMGCN@77dfe67d4fd80c124722e68a0f71af36d10fa5fa`, MIT License.
- The immutable snapshot contains the audited `model/`, `train/`, `utils/`, config and relevant notebooks. The compatibility runner imports the upstream model classes without editing them.
- Preserved math: two-view AE and multi-order NGNN, sigma fusion, dense fused-adjacency BCE, dense all-pair cosine NCE after 10% epochs, reconstruction, graph-feature and AE-GNN consistency terms.
- Compatibility only: sanitized numeric carrier input, upstream support normalization, fixed final epoch without label monitoring, strict checkpoint reload, and an encoder-only replay subpath algebraically tested against upstream `forward`.
- Night-21B used sparse positives plus sampled negatives and is therefore a source-faithful engineering port, not official numerical reproduction.
- Snapshot irregularities are preserved rather than repaired inside upstream: missing `spaMGCN_ZINB.py` imported by `Creat_model.py`, and `train.py` stored under `train/__pycache__`; the runner uses the complete `spaMGCN.py` and `train3.py` paths directly.
