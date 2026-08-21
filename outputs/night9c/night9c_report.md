# SpaLORA Night-9C report

## Terminal status

`BLOCKED_DATA_PROVENANCE`

Night-9C stopped at P0-DATA. No scientific training, embedding, clustering, label evaluation, SMART run, or PRESENT run was started.

## Why the external confirmation did not run

The preregistered input contract requires a paired E18.5 S1 source with 2,129 spots, 32,285 RNA features, 294,734 ATAC features, and `obs/Annotation_for_Combined` in both source H5AD files. All accessible public candidates failed at least one of these immutable checks:

| Public source | RNA shape | ATAC shape | Required annotation key | Decision |
|---|---:|---:|---|---|
| SpaDDM Git/Zenodo | 2129 x 32285 | 2129 x 117473 | absent | reject |
| SpaMosaic Zenodo | 2129 x 25740 | 2129 x 191034 | absent | reject |
| SMART Zenodo | 2129 x 32285 | 2129 x 117473 | absent; sidecar only | reject |
| MISAR official share | E18_5-S1.arrow only | E18_5-S1.arrow only | no paired tutorial H5AD | reject |

The PRESENT tutorial documents the expected full schema, but the repository does not distribute those H5AD files; its maintainer directs data requests to OEP003285. Reconstructing the full ATAC matrix, renaming `Combined_Clusters_annotation`, or injecting a sidecar label would change the source contract and was not authorized. The run therefore failed closed rather than producing a scientifically incomparable result.

## Provenance and firewall

- Raw lineage: MISAR-seq, OEP003285 / SRP491963.
- Reference tier: `TIER_B_PUBLISHED_REFERENCE_CLUSTER`; it must not be described as manual ground truth.
- Historical role remains `PRISTINE_EXTERNAL_CONFIRMATION`; no earlier E18.5 model/selection evidence was found before download.
- Label value reads: 0.
- Annotation value hashes: 0.
- Training / embedding / partition / evaluation: 0 / 0 / 0 / 0.
- K remained preregistered at 10 from the published tutorial, never inferred from local labels.

## What would be required to resume

A planning erratum must explicitly authorize one of the following: (a) provide the exact two PRESENT-style source H5AD files; or (b) define and hash a reproducible OEP/Arrow-to-H5AD conversion, including the exact 294,734-peak feature universe and an explicit, provenance-backed mapping to `Annotation_for_Combined`. Without that authority, Night-9C must remain blocked.

## Technical authority

- Parent commit: `e9bd62e2bb07c58f58c219956b8aaf090527c471`
  - Parent tag peel: `e9bd62e2bb07c58f58c219956b8aaf090527c471`
- Branch: `revision/q2-night9c-frozen-hierarchy-e18-5-confirmation-20260821`
- N02 implementation SHA-256: `5204e1cd9320336a424ccffd867269e64397c4a6a6210efe09ae614b0a58ab7f`
- Host: `autodl-pro-78617928975f`
