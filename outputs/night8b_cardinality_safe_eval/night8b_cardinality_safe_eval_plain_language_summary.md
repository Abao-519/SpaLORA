# Night-8B cardinality-safe recovery: plain-language summary

The frozen predictions contain 12 clusters, while the official reference labels contain 7 categories. This is not an error: clustering agreement metrics can compare partitions with different category counts, and both competing methods still use the same fixed prediction K.

F00 minus U00 had mean delta ARI 0.013742, mean delta NMI 0.020982, and mean delta Q 0.017362, with 10/10 Q wins. The preregistered terminal status is `NIGHT8B_CARDINALITY_SAFE_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST`.

This is a post-lock recovery because the labels had already been opened once by the failed evaluator. It is neither a pristine holdout nor a SOTA claim.
