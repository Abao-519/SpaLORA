# SpaLORA Night-4A report

**Gate: BLOCKED_PREFLIGHT**

- Night-3B original results changed: **NO**.
- Night-3B decision changed: **NO - MIXED_EVIDENCE**.
- Formal benchmark run: **NO**.
- Semantic labels accessed only after preprocessing lock: **YES**.

## Evidence repair
Lower-is-better counts were regenerated without overwriting Night-3B. Exact round-trip recomputation preserved 120 keys, summaries, paired deltas and MIXED_EVIDENCE. Ten PNG/PDF figure pairs were redrawn; they remain diagnostic until local PDF visual QA is complete.

## Data preflight
D1 lymph node is accuracy-eligible with paired RNA/protein, coordinates and ten-domain manual ground truth. A1/D1 tonsil have no auditable semantic label column. Independent GSE198353 provides two aligned RNA+protein spleen replicates with coordinates but no official manual domain labels, so only the label-free layer is eligible. GSE213264 lacks reproducible paired spatial coordinates. The data gate fails.

## Baselines
All five official repositories and commits were locked. Attempt 1 and at most one compatibility-only retry were retained. SpatialGlue, Seurat/WNN, COSMOS, ARISE and SMART are UNREPRODUCIBLE in this bounded target-machine preflight; 0/5 reached integrated embedding and official clustering. No scientific formula or parameter was changed.

## Decision
Night-3B evidence repair preserved all locked numerical results and the preregistered `MIXED_EVIDENCE` decision. Candidate external datasets and baseline implementations were evaluated for provenance, task compatibility, and end-to-end reproducibility before any formal benchmark outcome was generated. The project is `BLOCKED_PREFLIGHT`. Night-4B was not started.
