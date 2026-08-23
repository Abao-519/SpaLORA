# Night-15A source and protocol audit

- 3d-OT official source: `dbjzs/3d-OT` commit `39a7cb02748d83299cd471f172f3b972896e61d8`, Apache-2.0. Only protocol/artifact semantics were used; no third-party source was copied into SpaLORA.
- SEPAR official source: `zerovain/SEPAR` commit `6d3475fa0bd749d3b1b5592b68323439d473f9fc`, MIT. Tutorial4 was audited for the K=12 lane; external context execution is kept separate from own-method evidence.
- GSE213264 source: NCBI GEO processed Spatial-CITE-seq tonsil members. Only RNA/protein identifiers and matrix metadata were opened for deduplication; annotation values were not opened.
- MCDF code is an independent project implementation. Its four experts and masked-modality objective are evaluated as a development hypothesis, not claimed as established novelty.
