# Night-15F method semantics and selection contract

## Frozen model object

For every lane, the core receives numeric retained/RNA/second-modality views, three registered sparse graphs (fine k=4, registered k=6, broad k=18), an initial partition and K. It does not receive a dataset/study string or reference labels.

The continuous energy combines:

1. retained, RNA and second-modality dynamic prototype unaries mixed by content margins;
2. multiscale low/two-hop/high graph features;
3. a nonnegative simplex mixture of the three registered graph scales;
4. cross-modal agreement/conflict edge conductance with absolute accepted mass;
5. rejected conductance mass as an explicit current-state stay unary refreshed per outer cycle;
6. an optional cluster-size cost computed only from the current predicted counts and K;
7. Potts pairwise disagreement cost;
8. sparse alpha-expansion moves accepted only after the original floating-point frozen-cycle energy strictly decreases.

## Selection boundary

- Public reference labels are used for known K, cross-run per-lane numeric HPO and evaluation.
- Labels do not enter features, prototype unary, graph edges, pairwise energy, size prior, min-cut construction or move acceptance.
- Balanced selection first requires positive ARI and NMI deltas versus Night-15E, then maximizes `min(delta_ari, delta_nmi)`, then the sum of deltas.
- Max-ARI and max-NMI profiles are retained separately and are not merged into a fictitious joint best.
- Different lanes may select different numeric values from the common formula. This is public benchmark development HPO, not an automatic content gate or deployable policy.
- The initial Night-15E partitions were themselves selected by prior public-label per-lane HPO; Night-15F does not erase that history.

## Energy guarantee boundary

For each outer cycle, dynamic unaries are frozen. Every accepted alpha move decreases that cycle's original floating-point energy, and the cycle end is no larger than its start. Dynamic unaries are recomputed between cycles, so energies from different cycles are not treated as one fixed objective. No global optimum or cross-cycle global monotonicity is claimed.

## Engineering boundary

- Sparse graphs only; dense N×N count is zero.
- SciPy sparse max-flow supplies the s-t cut primitive; no third-party alpha-expansion code is copied.
- The same formula and code path serve RNA+protein and RNA+ATAC.
