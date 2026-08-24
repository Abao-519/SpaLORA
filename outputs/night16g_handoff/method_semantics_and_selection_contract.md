# Night-16G method semantics and selection contract

## Frozen computation boundary

For every protocol, the producer consumes only ordered observation IDs, reduced RNA/ATAC/retained representations, sparse registered spatial graphs, known `K`, and a mechanically generated bank of partitions. It does not accept annotations, dataset names, ARI/NMI, or an observation-by-observation dense matrix.

The locked bank contains 47 inherited Night-16F start×operator candidates and 42 new candidates from a seven-level ordered pairwise-strength continuation applied to six starts. Every ordered-path artifact is reopened after serialization and its IDs and partitions are compared exactly before `artifact_reload: PASS` is written.

## Evidence axes

Each locked candidate receives four possible label-free ranks:

1. **Molecular separation**: robustly aggregated cluster separation in retained, RNA and ATAC reduced views.
2. **Sparse topology**: categorical neighbor agreement/excess on the three registered sparse graph scales.
3. **Cluster-size risk**: a predicted-partition regularity term based only on its smallest cluster, not on reference class sizes.
4. **Persistence**: ARI continuity inside candidate components and, for the ordered paths, adjacent-level agreement plus contiguous plateau span.

The formal family-global fit selected weights `molecular=0.25`, `topology=1.0`, `cluster-risk=0.25`, `persistence=0.0`. Therefore the formal Night-16G consumer is a **multi-evidence candidate selector**. Ordered-path persistence was implemented and tested, but it is not an active contribution in the winning rule.

## Label-assisted fit and strict LOSO

Candidate partitions and evidence features are saved and hashed before labels are opened. The family-global profile uses the four public benchmark evaluation tables to tune the four scalar evidence weights; it is transparently a label-assisted public-benchmark HPO result.

For strict leave-one-study-out (LOSO), a fit process receives only the other three studies' feature banks and evaluation tables. It writes and hashes the held-out configuration with `heldout_evaluation_opened=false`. A separate producer applies that configuration to the held-out locked bank, after which a separate evaluator reads the held-out labels. This establishes a clean parameter-transfer boundary, but it does not erase historical provenance of inherited candidate starts.

## Exactness and resource boundary

- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `threadpool_limits(1)` are active at numerical production.
- Dense work is restricted to the 89×89 candidate-partition similarity matrix; dense observation×observation allocation is zero.
- The four family-global outputs were regenerated in two fresh processes after the final core source edit and matched on IDs, candidate ID, selector-config SHA and partition SHA (4/4).
- Reference labels are used only by independent evaluators and public-benchmark configuration fitting.

