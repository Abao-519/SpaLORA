# SpaLORA Night-8B Uniform-Head Evaluation Recovery Report

## Terminal status

`RECOVERY_BLOCKED_INPUT_INTEGRITY`

Reason: `LOCKED_ANNOTATION_K_CONTRACT_MISMATCH_AT_SINGLE_AUTHORIZED_WINDOW`.

The original Night-8B remains `INFRASTRUCTURE_BLOCKED` and its raw root is byte-identical before versus after this recovery.

## What completed before labels

- Authority taskbook and registry SHA matched.
- Original Windows compact independently verified 38/38.
- Original remote raw artifact manifest rehashed 297/297.
- Base 10/10, adapter 10/10, reload 20/20, s04 10/10, F00 affinity 10/10.
- Historical U00 affinity versus s04 canonical parity 9/9.
- Recovery input manifest locked 20/20.
- RECOVERY_EIGEN_KMEANS100 produced 20/20 exact K=12 partitions.
- Same-process determinism and partition SHAs passed 20/20.
- Partition total-lock and ordinary push completed before label access.
- Training=0, adapter=0, affinity rebuild=0, scientific retry=0, fallback=0.

## Authorized label window and hard stop

MISAR Y was read exactly once after total lock and ordinary push. Before any metric was computed, the evaluator found that the observed annotation cardinality did not match the preregistered K=12 contract and failed closed. The process did not persist the actual category count before exiting. The protocol forbids rereading Y, changing K or annotation granularity, returning to mapping/head/affinity/code, or running another evaluation. Therefore ARI, NMI, Q, paired statistics, spatial gates, resource gates, and the historical 9-pair descriptive sensitivity were not computed.

This is an input-authority semantic conflict, not evidence that F00 or U00 won. It neither supports nor refutes the RNA+ATAC R02 family policy.

## Invariance and interpretation

Original Night-8B files remained byte-identical. All recovery partitions and failure evidence are preserved. No third-party benchmark was run and no SOTA claim is made.
