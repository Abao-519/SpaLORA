# Night-13B search decision ledger

- Registered search space: B00 identity plus B01-B14, spanning thresholds 0.62-0.70, slopes 20-40, and maximum residual weights 0.50-0.70. Every executed formula/config has a distinct candidate ID in `candidate_registry.json`.
- First pass: all 15 candidates ran on A1, canonical tonsil s1, and P22 with seed 0 under the same fixed evaluator. No row was deleted.
- Early interpretation: lower thresholds over-smoothed RNA+protein, while identity-like configurations failed to close the large P22 gap. The useful region combined a conservative threshold with a steep gate.
- Selection: B10 (`threshold=.66`, `slope=40`, `max_residual=.70`) maximized the weak-family-first rule while leaving tonsil s1 effectively unchanged. Public labels drove this cross-run choice and the result is explicitly a development experiment.
- Freeze: B10, the global config, seeds 0-4, and the seed-0 common endpoint were frozen before D1, tonsil s2/s3, and MISAR confirmation.
- Post-confirmation rule: MISAR's negative result was retained. No return to discovery, formula change, threshold adjustment, or seed search was allowed in this terminal.
