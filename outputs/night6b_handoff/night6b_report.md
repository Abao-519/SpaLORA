# SpaLORA Night-6B report

## Authoritative terminal status

`IMPLEMENTATION_SEMANTICS_INVALID`

Night-6B stopped at P0-SEMANTIC before any formal training or cluster-head transform. The taskbook requires A1 G00/H00 seeds 0-4 to be created by loading the valid historical Night-5 C04/B01 final checkpoints, replaying forward on the valid cache, reproducing fused clusters and ARI/NMI exactly, and deriving the previously unsaved private views.

That historical model state does not exist. All five authoritative C04 seed manifests and their 30 declared artifacts rehashed without mismatch, but each run contains only `embedding.npz`, `attention.npz`, `clusters.csv`, `observation_ids.csv`, `loss_trajectory.csv`, and `coefficient_probe.json`. The Night-5A runner's own required-artifact list excluded model state and contains no `torch.save`. Remote raw storage and the local Night-5A compact/full archives contain no C04/B01 `.pt`, `.pth`, `.ckpt`, `model_final`, or checkpoint artifact. Historical Night-3AF model states are semantically different and were not substituted.

Retraining C04/B01 would be new evidence, not read-only replay, and is explicitly forbidden by this taskbook. Consequently exact private-view derivation is impossible and the hard gate correctly fails.

## Completed valid preflight evidence

- Four authoritative input SHA-256 values matched locally and remotely.
- Night-6A compact delivery index independently verified 16/16; its status remains `IMPLEMENTATION_SEMANTICS_INVALID` and none of its checkpoints, embeddings, or candidate metrics was used.
- Git parent/tag/protection tag matched `7f204a56690768f22bd06e0dac1b5785c97c4c70`.
- The independent data steward verified the official tonsil RNA/ADT source hashes.
- The preregistered `final_annot` column has actual nonmissing `K=4`: connective & epithelial tissue 731, germinal center 183, lymphoid follicle 834, tonsillar parenchyma 2578. RNA and ADT annotations match spot-by-spot.
- New label-free RNA/ADT files contain zero `obs` columns and share exact ordered barcodes.
- Data-steward label access was authorized and explicitly recorded as deserialized and observed, but never used for training or selection.
- The trainer/transformer and evaluator never started. D1, P22, GSE198353, Night-4B, Night-5D metric content, and Night-6A raw runs were not accessed.
- Nine fail-closed firewall tests passed.

## Budget and scientific interpretation

- Formal scientific training: 0/61.
- Training retry attempts: 0/12.
- Head transforms: 0/552.
- Transform corrections: 0/48.
- No ARI/NMI, graph comparison, head comparison, balanced frontier, or accuracy frontier was produced.

This is an infrastructure-of-evidence/implementation-contract failure, not a negative scientific result. It cannot be reported as `NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE`.

## Required next decision

A future authority document must choose one of two auditable routes: provide genuinely SHA-verified historical C04/B01 checkpoints, or preregister a clean baseline retraining and count it as new Night-6B evidence. The present taskbook does not authorize either substitution, so this run stops without guessing.
