# Night-17E source collision and external-data audit

This was a metadata/source audit only.  No expression archive was downloaded.

## Source collision boundary

| Method | Official source authority | Pinned repository HEAD | License | Relevant prior object | Night-17E boundary |
|---|---|---|---|---|---|
| SpatialGlue | Nature Methods 2024; `JinmiaoChenLab/SpatialGlue` | `7c976d811d27ace51ce47ae0ad94a068a7d222fa` | AGPL-3.0 | modality-specific spatial graphs and dual attention | learned multimodal graph integration is prior art |
| PRAGA | AAAI 2025; `Xubin-s-Lab/PRAGA` | `4adb11c96fc7ddad800fa1787eadcc8b91b42784` | AGPL-3.0 | learned dynamic graph plus prototype-aware contrastive aggregation | learned semantic edges and prototypes are prior art |
| spaMGCN | Genome Biology 2025; `hongfeiZhang-source/spaMGCN` | `77dfe67d4fd80c124722e68a0f71af36d10fa5fa` | MIT | autoencoder plus multi-scale adaptive graph convolution | multi-scale learned spatial graph representation is prior art |
| SEPAR | Communications Biology 2026; `zerovain/SEPAR` | `6d3475fa0bd749d3b1b5592b68323439d473f9fc` | MIT | graph-regularized metagene patterns and multi-omics correlation analysis | graph regularization and multi-omics pattern discovery are prior art |

The narrow Night-17E test object is therefore not “learned graph weighting” or
“graph cut” alone.  It is the combination of a frozen cross-seed learned-
representation edge-support statistic, uncertainty attenuation, exact
base-weighted mass controls and nonnegative submodular alpha-expansion on
pre-existing strong molecular starts.  Because the learned arm did not separate
from ZERO/control arms across studies, this experiment supplies no defensible
new-method claim.

## Dataset/source corrections

1. **GSE213264 is spatial-CITE-seq, not MISAR.**  GEO sample GSM6578062 is
   `Human_tonsil_RNA`; the paired protein sample belongs to the same spatial-
   CITE-seq series.  The SEPAR Tutorial 4 link that labels GSE213264 as MISAR
   E15.5 is an upstream tutorial source error and is not propagated.
2. SEPAR's paired multi-omics examples are substantively MISAR-seq and spatial-
   CITE-seq.  DLPFC, olfactory bulb, osmFISH, MERFISH and CRC examples are not
   silently counted as paired spatial multi-omics benchmark units.
3. spaMGCN's six paired units are inherited from SpatialGlue/SpaMosaic/spaVAE:
   mouse thymus, mouse spleen, three human lymph-node slices and MISAR, with an
   additional placenta example.  These are not automatically six independent
   studies and overlap existing project assets.
4. Zenodo 10362607 exposes one large `Data_SpatialGlue.zip` (registered size
   671,921,238 bytes); Zenodo 7480069 is a MISAR source.  Neither was downloaded
   in Night-17E.
5. GSE213264 human tonsil is a useful physically independent RNA+protein unit,
   but the current authority closes germinal-center/pathology semantics rather
   than a mutually exclusive expert K-class whole-tissue label.  It is eligible
   for germinal-center F1 and biological validation, not full-domain ARI/NMI,
   until a spot-level reference protocol is closed.

## Next external unit

The best nonduplicate, low-ambiguity next unit is GSE213264 human tonsil as an
independent **biological/GC-region validation** unit.  It is not presently a
full-domain clustering confirmation.  Placenta remains a possible full-domain
candidate only after annotation provenance, physical-section identity and
evaluation mask are closed.  No unsupported labels or mappings were created.

## Official links inspected

- https://www.nature.com/articles/s41592-024-02316-4
- https://github.com/JinmiaoChenLab/SpatialGlue
- https://ojs.aaai.org/index.php/AAAI/article/view/32010
- https://github.com/Xubin-s-Lab/PRAGA
- https://pubmed.ncbi.nlm.nih.gov/40495225/
- https://github.com/hongfeiZhang-source/spaMGCN
- https://pubmed.ncbi.nlm.nih.gov/41372421/
- https://github.com/zerovain/SEPAR
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6578062
