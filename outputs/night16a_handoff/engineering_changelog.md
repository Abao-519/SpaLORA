# Night-16A engineering changelog

- Built one dataset-name-blind observable-statistics-to-energy API with seven global/family controls.
- Corrected optional edge coherence: the median of `exp(-d/median(d))` is nearly constant, so the implementation now uses mean similarity plus interquartile persistence.
- Found and fixed the arena writer metadata bug: partitions and metrics were correct, but per-row constants/statistics were not stored with each generated tuple. Strict and fixed-start arenas were rerun as `v2`; earlier outputs remain preserved and marked superseded.
- Reconstructed the internal development ledger's parameter metadata from its authoritative calibration registry; scientific partitions and metrics were not changed.
- Added physical descriptor/metric separation and grouped hold-out files for lymph-node, tonsil, P22 and MISAR studies.
- Added a frozen Ridge-checkpoint replay; two new processes recovered all 9/9 selected candidate IDs without opening current-study labels.
- Added a measured-resource replay (9/9 exact; 377.79 MiB peak RSS). No GPU computation or new download was required.
- Pytest was unavailable locally; the dependency-free targeted runner passed 5/5 tests.
- The first final test-runner invocation omitted its required `--test-file` CLI argument and exited before executing tests; it was corrected immediately, and the post-source run passed 5/5.
- The remote login shell had no unqualified `python`; the delivery AST/JSON audit was rerun with `/root/miniconda3/bin/python` and passed (21 Python files, 16 JSON files, zero banned payloads).
- The first remote precommit whitespace check rejected Windows CRLF line endings before a commit was created; all Night-16A text payloads were normalized to LF, restaged and rechecked.
