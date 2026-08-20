# SpaLORA Night-8B blocked prelabel report

## Terminal state

`INFRASTRUCTURE_BLOCKED` with reason `FIXED_H05_NUMERICAL_HEAD_FAILURE_K_NOT_12`. This is not a scientific comparison result.

## Completed work

- Authority, provenance, 1,949/1,949 mapping, immutable cache, P22 code-only parity and CUDA P0: PASS.
- Formal base training: 10/10, fresh-process checkpoint reload: 10/10.
- Formal R02 adapter training and reload: 10/10.
- Fixed transforms: 19/20 success. U00/H05 seed 6 failed because the frozen spectral endpoint did not produce K=12.
- Scientific retry: 0; fallback: 0; post-hoc K/solver/seed changes: 0.
- MISAR Y values read: 0. No ARI/NMI/Q/spatial or external-family conclusion was computed.

## Interpretation

The frozen training routes were deployable on MISAR, but the universal comparator failed its fixed partition contract for one seed. The protocol requires all 20 partitions to be locked and ordinarily pushed before label access. Retrying only seed 6 or changing the solver would violate the preregistration, so evaluation was not opened. This result neither confirms nor rejects R02 generalization.

## Preservation

Raw runs, checkpoints, affinities and logs remain under `/root/autodl-fs/night8b_raw_runs_20260820` and are indexed by SHA-256 in `raw_artifact_manifest.csv`. The failing log SHA is `03a646046f17b0f75adae452001642b21b9ea31c22a53c30fc808df3f5d6023e`. GSE213264 remains explicitly excluded from MISAR provenance; valid identifiers are OEP003285, SRP491963 and Zenodo 7480069.
