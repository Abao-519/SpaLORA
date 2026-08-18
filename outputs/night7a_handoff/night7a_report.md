# SpaLORA Night-7A CPU consensus and benchmark preflight report

## Terminal result

`KEEP_CONFIRMED_G04_H05_FOR_EXTERNAL_VALIDATION`

- Selected structure: `C00_G04_H05_CONFIRMED`.
- Confirmatory status: retains Night-6D confirmation.
- Dual-graph candidates passing both preregistered gates: `none`.
- This round used CPU only: scientific training `0`, checkpoint forward `0`, diffusion `0`, GPU use `0`, formal benchmark runs `0`.
- Night-6D remains authoritative and is not rewritten by this development selection.

## Authority and source reuse

- Night-6C local evidence independently verified before remote work: internal `77/77`, external `4/4`, post-dispatch `3/3`.
- Night-6D local evidence independently verified before remote work: internal `68/68`, external `5/5`, post-dispatch `3/3`.
- Remote views were resolved only through authoritative raw manifests: `60/60`; G00/G04 observation and coordinate parity: `30/30`; historical prediction files: `120/120`.
- Exact real H05 parity: `30/30 x 2`. C02 six-view arithmetic identity tolerance was `1e-12`.
- Three pre-science infrastructure attempts are preserved: the first had zero completed units under an oversubscribed 0.5-CPU quota; the next two were terminated at the 2 GiB memory boundary after 21 and 1 completed P0 units. All had zero formal transforms, zero label reads, and zero GPU use. The final P0 used non-overlapping source-verifier, per-cell, and aggregator processes without changing solver, tolerance, data, order, or candidate semantics.
- One later pre-label source-preflight infrastructure attempt is also preserved: GitHub HTTPS pack transfer stalled and the old presence check could have accepted a residual incomplete `.git` directory. It was interrupted before labels, quarantined, and never reused; it modified zero formal consensus cells and used zero GPU. The correction retained the same canonical repository and resolved commit while requiring a complete official codeload snapshot.
- Post-lock historical H00/H05 metric replay: `90/90`, maximum absolute error `1.11e-16` (tolerance `1e-12`).

## Consensus execution

- Formal cells: `360/360`.
- Successes: `354`; preserved scientific numerical failures: `6`.
- Formal transform implementation/infrastructure correction attempts: `0`; total formal-plus-correction transform attempts: `360/372`. The three pre-science P0 infrastructure attempts are reported separately because each used zero formal candidate transforms.
- Labels were opened only after transform and benchmark/data preflight locks. No affinity, clustering, retry, or registry operation followed label access.

## Four-dataset results

| candidate_id | dataset | mean_ari | mean_nmi | mean_q | mean_delta_ari | mean_delta_nmi | mean_delta_q | wins_delta_q |
|---|---|---|---|---|---|---|---|---|
| C00_G04_H05_CONFIRMED | a1 | 0.269205 | 0.408674 | 0.338940 | -0.002390 | 0.018931 | 0.008270 | 5.000000 |
| C00_G04_H05_CONFIRMED | tonsil | 0.152954 | 0.272723 | 0.212839 | 0.067377 | 0.099076 | 0.083227 | 5.000000 |
| C00_G04_H05_CONFIRMED | d1 | 0.241201 | 0.377656 | 0.309428 | 0.031700 | 0.028965 | 0.030332 | 10.000000 |
| C00_G04_H05_CONFIRMED | p22 | 0.420274 | 0.590306 | 0.505290 | 0.016361 | 0.046475 | 0.031418 | 8.000000 |
| C01_G00_H05 | a1 | 0.259626 | 0.391003 | 0.325315 | -0.011969 | 0.001260 | -0.005354 | 1.000000 |
| C01_G00_H05 | tonsil | 0.148902 | 0.266539 | 0.207720 | 0.063324 | 0.092893 | 0.078108 | 5.000000 |
| C01_G00_H05 | d1 | 0.257631 | 0.383706 | 0.320669 | 0.048130 | 0.035015 | 0.041572 | 10.000000 |
| C01_G00_H05 | p22 | 0.446145 | 0.614224 | 0.530184 | 0.042233 | 0.070392 | 0.056313 | 10.000000 |
| C02_DUAL_ARITHMETIC_MEAN | a1 | 0.266680 | 0.403813 | 0.335247 | -0.004915 | 0.014070 | 0.004577 | 4.000000 |
| C02_DUAL_ARITHMETIC_MEAN | tonsil | 0.149122 | 0.270751 | 0.209937 | 0.063545 | 0.097105 | 0.080325 | 5.000000 |
| C02_DUAL_ARITHMETIC_MEAN | d1 | 0.246457 | 0.381094 | 0.313776 | 0.036956 | 0.032403 | 0.034680 | 10.000000 |
| C02_DUAL_ARITHMETIC_MEAN | p22 | 0.456576 | 0.619658 | 0.538117 | 0.052664 | 0.075827 | 0.064245 | 10.000000 |
| C03_DUAL_ELEMENTWISE_MAX | a1 | 0.263601 | 0.402035 | 0.332818 | -0.007995 | 0.012292 | 0.002149 | 3.000000 |
| C03_DUAL_ELEMENTWISE_MAX | tonsil | 0.152363 | 0.272319 | 0.212341 | 0.066785 | 0.098673 | 0.082729 | 5.000000 |
| C03_DUAL_ELEMENTWISE_MAX | d1 | 0.241867 | 0.379973 | 0.310920 | 0.032366 | 0.031282 | 0.031824 | 10.000000 |
| C03_DUAL_ELEMENTWISE_MAX | p22 | 0.463860 | 0.619711 | 0.541785 | 0.059948 | 0.075880 | 0.067914 | 10.000000 |
| C04_DUAL_ELEMENTWISE_MIN | a1 | 0.258559 | 0.389653 | 0.324106 | -0.013036 | -0.000090 | -0.006563 | 0.000000 |
| C04_DUAL_ELEMENTWISE_MIN | tonsil | 0.144152 | 0.265555 | 0.204853 | 0.058575 | 0.091909 | 0.075242 | 5.000000 |
| C04_DUAL_ELEMENTWISE_MIN | d1 | nan | nan | nan | nan | nan | nan | nan |
| C04_DUAL_ELEMENTWISE_MIN | p22 | 0.429394 | 0.604087 | 0.516741 | 0.025482 | 0.060256 | 0.042869 | 10.000000 |
| C05_DUAL_HARMONIC_INTERSECTION | a1 | 0.256578 | 0.387403 | 0.321990 | -0.015017 | -0.002340 | -0.008679 | 0.000000 |
| C05_DUAL_HARMONIC_INTERSECTION | tonsil | 0.143483 | 0.264657 | 0.204070 | 0.057906 | 0.091010 | 0.074458 | 4.000000 |
| C05_DUAL_HARMONIC_INTERSECTION | d1 | nan | nan | nan | nan | nan | nan | nan |
| C05_DUAL_HARMONIC_INTERSECTION | p22 | 0.428733 | 0.603956 | 0.516345 | 0.024821 | 0.060125 | 0.042473 | 10.000000 |
| C06_DUAL_ROW_STOCHASTIC_MEAN | a1 | 0.266709 | 0.404767 | 0.335738 | -0.004887 | 0.015024 | 0.005069 | 4.000000 |
| C06_DUAL_ROW_STOCHASTIC_MEAN | tonsil | 0.149335 | 0.269692 | 0.209513 | 0.063758 | 0.096045 | 0.079902 | 5.000000 |
| C06_DUAL_ROW_STOCHASTIC_MEAN | d1 | 0.248244 | 0.381516 | 0.314880 | 0.038743 | 0.032825 | 0.035784 | 10.000000 |
| C06_DUAL_ROW_STOCHASTIC_MEAN | p22 | 0.472255 | 0.628435 | 0.550345 | 0.068342 | 0.084604 | 0.076473 | 10.000000 |
| C07_DUAL_LOCAL_RELIABILITY | a1 | 0.266272 | 0.403344 | 0.334808 | -0.005324 | 0.013601 | 0.004139 | 4.000000 |
| C07_DUAL_LOCAL_RELIABILITY | tonsil | 0.148934 | 0.270386 | 0.209660 | 0.063356 | 0.096740 | 0.080048 | 5.000000 |
| C07_DUAL_LOCAL_RELIABILITY | d1 | 0.247218 | 0.381651 | 0.314434 | 0.037717 | 0.032959 | 0.035338 | 10.000000 |
| C07_DUAL_LOCAL_RELIABILITY | p22 | 0.451889 | 0.616570 | 0.534230 | 0.047977 | 0.072738 | 0.060358 | 10.000000 |
| C08_SIX_VIEW_SUPPORT_MEDIAN | a1 | 0.266479 | 0.403677 | 0.335078 | -0.005117 | 0.013934 | 0.004409 | 4.000000 |
| C08_SIX_VIEW_SUPPORT_MEDIAN | tonsil | 0.149369 | 0.271016 | 0.210193 | 0.063792 | 0.097370 | 0.080581 | 5.000000 |
| C08_SIX_VIEW_SUPPORT_MEDIAN | d1 | 0.246395 | 0.380913 | 0.313654 | 0.036894 | 0.032222 | 0.034558 | 10.000000 |
| C08_SIX_VIEW_SUPPORT_MEDIAN | p22 | 0.456713 | 0.619473 | 0.538093 | 0.052800 | 0.075642 | 0.064221 | 10.000000 |
| C09_DUAL_SPARSE_SNF10 | a1 | 0.218475 | 0.370226 | 0.294350 | -0.053121 | -0.019517 | -0.036319 | 0.000000 |
| C09_DUAL_SPARSE_SNF10 | tonsil | 0.142783 | 0.264753 | 0.203768 | 0.057205 | 0.091106 | 0.074156 | 5.000000 |
| C09_DUAL_SPARSE_SNF10 | d1 | 0.219846 | 0.338579 | 0.279213 | 0.010345 | -0.010112 | 0.000117 | 4.000000 |
| C09_DUAL_SPARSE_SNF10 | p22 | 0.338224 | 0.504147 | 0.421185 | -0.065689 | -0.039685 | -0.052687 | 0.000000 |
| C10_DUAL_MEAN_SPATIAL05 | a1 | 0.261543 | 0.402286 | 0.331914 | -0.010052 | 0.012543 | 0.001245 | 2.000000 |
| C10_DUAL_MEAN_SPATIAL05 | tonsil | 0.148755 | 0.266053 | 0.207404 | 0.063178 | 0.092407 | 0.077792 | 5.000000 |
| C10_DUAL_MEAN_SPATIAL05 | d1 | 0.246292 | 0.380733 | 0.313512 | 0.036791 | 0.032041 | 0.034416 | 10.000000 |
| C10_DUAL_MEAN_SPATIAL05 | p22 | 0.477701 | 0.633080 | 0.555390 | 0.073788 | 0.089248 | 0.081518 | 10.000000 |
| C11_DUAL_MEAN_SPATIAL10 | a1 | 0.255214 | 0.398200 | 0.326707 | -0.016382 | 0.008457 | -0.003962 | 1.000000 |
| C11_DUAL_MEAN_SPATIAL10 | tonsil | 0.164878 | 0.251583 | 0.208231 | 0.079301 | 0.077937 | 0.078619 | 5.000000 |
| C11_DUAL_MEAN_SPATIAL10 | d1 | 0.240461 | 0.377393 | 0.308927 | 0.030960 | 0.028702 | 0.029831 | 10.000000 |
| C11_DUAL_MEAN_SPATIAL10 | p22 | 0.480695 | 0.636724 | 0.558709 | 0.076783 | 0.092892 | 0.084838 | 10.000000 |

## Preregistered gates

| candidate_id | generalization_gate_pass | complexity_gate_pass | spatial_protection_all_pass | eligible | worst_dataset_mean_delta_q | dataset_balanced_macro_mean_delta_q | total_paired_q_wins |
|---|---|---|---|---|---|---|---|
| C00_G04_H05_CONFIRMED | True | True | True | True | 0.008270 | 0.038312 | 28 |
| C01_G00_H05 | False | True | True | False | -0.005354 | 0.042660 | 26 |
| C02_DUAL_ARITHMETIC_MEAN | False | False | True | False | 0.004577 | 0.045957 | 29 |
| C03_DUAL_ELEMENTWISE_MAX | False | False | True | False | 0.002149 | 0.046154 | 28 |
| C04_DUAL_ELEMENTWISE_MIN | False | False | False | False | nan | nan | 0 |
| C05_DUAL_HARMONIC_INTERSECTION | False | False | False | False | nan | nan | 0 |
| C06_DUAL_ROW_STOCHASTIC_MEAN | True | False | True | False | 0.005069 | 0.049307 | 29 |
| C07_DUAL_LOCAL_RELIABILITY | False | False | True | False | 0.004139 | 0.044971 | 29 |
| C08_SIX_VIEW_SUPPORT_MEDIAN | False | False | True | False | 0.004409 | 0.045942 | 29 |
| C09_DUAL_SPARSE_SNF10 | False | False | True | False | -0.052687 | -0.003683 | 9 |
| C10_DUAL_MEAN_SPATIAL05 | False | False | True | False | 0.001245 | 0.048743 | 27 |
| C11_DUAL_MEAN_SPATIAL10 | False | False | True | False | -0.003962 | 0.047331 | 26 |

The fixed ranking was worst-dataset mean Delta-Q, dataset-balanced macro Delta-Q, total paired Q wins, future complexity, and registry order. A dual-graph structure was allowed to replace C00 only if it also paid the preregistered double-encoder complexity cost. The independent table-level implementation reproduced Q, paired deltas, spatial directions, gates, and the selected structure within `1e-12`.

## External source-code readiness

| method | commit | license | label_hits | selection_hits | readiness |
|---|---|---|---|---|---|
| SpatialGlue | 7c976d811d27ace51ce47ae0ad94a068a7d222fa | GPL | 7.000000 | 0.000000 | READY_WITH_FIXED_ENDPOINT_ADAPTER |
| Seurat_WNN | 17d6f4fc6f57a9092871b85c62f9524b173b09df | UNRESOLVED,MIT | 40.000000 | 0.000000 | READY_WITH_FIXED_ENDPOINT_ADAPTER |
| COSMOS | 56ea355be51e64d9253e2871b8bd447fdfd0d230 | MIT | 40.000000 | 4.000000 | READY_WITH_FIXED_ENDPOINT_ADAPTER |
| SMART | 75676546a66d48a7ebfd7c5f3a2758a4c538fcfc | GPL | 40.000000 | 6.000000 | READY_WITH_FIXED_ENDPOINT_ADAPTER |
| PRESENT | c88a609b34aae9b84c2c8a7ffb6824bae5c78f23 | MIT | 40.000000 | 4.000000 | READY_WITH_FIXED_ENDPOINT_ADAPTER |
| MultiGATE | None | None | nan | nan | BLOCKED_ENVIRONMENT |
| SpatialCOC | 40612e6c82368f3c6bae7f61230d78ff9fd3703e | GPL | 40.000000 | 40.000000 | READY_WITH_FIXED_ENDPOINT_ADAPTER |
| SpaMode | d8d8e2b70c6ad47ef12aa1a5d9a65cf4fb226c00 | NO_LICENSE_FILE | 40.000000 | 14.000000 | SOURCE_ONLY_LICENSE_BLOCKED |
| SpaMCA | 33319c63350821ae701436c20753a05e87f754f6 | NO_LICENSE_FILE | 40.000000 | 16.000000 | SOURCE_ONLY_LICENSE_BLOCKED |
| ARISE | None | None | nan | nan | BLOCKED_ENVIRONMENT |
| GROVER | None | None | nan | nan | BLOCKED_ENVIRONMENT |
| MultiSP | None | None | nan | nan | BLOCKED_ENVIRONMENT |
| SpatialEx | c3e1e069cc364d9d256cd50126bf36c7bb2af450 | MIT | 40.000000 | 0.000000 | BLOCKED_PRIVATE_ASSET_OR_THIRD_MODALITY |

This is a source audit, not an accuracy comparison. Successfully acquired official GitHub source snapshots were independently resolved to exact commits, while any source-acquisition failure is preserved as `BLOCKED_ENVIRONMENT`; executable source and tutorials were scanned for label access, best-ARI/NMI or best-epoch selection, K handling, endpoints, environment files, and licenses. Missing licenses remain source-only; methods requiring labels for checkpoint selection require a disclosed fixed-final label-free adapter; private weights or a mandatory third modality remain task-mismatch blockers. No upstream no-license source was copied.

## Fresh-data metadata preflight

| dataset | status | annotation |
|---|---|---|
| SPAMODE_HUMAN_TONSIL_THREE_SECTIONS | NEEDS_MANUAL_PROVENANCE_REVIEW | None |
| GSE205055 | NEEDS_MANUAL_PROVENANCE_REVIEW | independent manual/histology annotation not established from accession metadata |
| GSE198353 | READY_LABEL_FREE_REPLICATION | no official auditable manual domain labels established in metadata-only preflight |
| SIMULATED_GROUND_TRUTH_PANEL | NOT_SUITABLE | controlled robustness and mechanism only; never replaces real external validation |

The Zenodo tonsil record's section 1 overlaps the current Night-6C development tonsil; sections 2/3 may be section-level fresh but are not study-independent until donor/annotation provenance is manually audited. MISAR remains exploratory until exact same-section pairing, coordinates, and independent annotation provenance are verified. GSE198353 remains label-free replication. No fresh per-spot labels or large archives were opened.

## Firewall and statistics

- One evaluator process opened all four already-used development labels only after the `360/360` transform lock and preflight SHA lock.
- No `anndata.read_h5ad` call was made; tonsil `final_annot` was read post-lock through a low-level HDF5 column reader.
- Four datasets were weighted equally at 0.25; no spot-count or seed-count pooling was used.
- Per-dataset mean, median, SD, wins, full exact sign-flip tests, 100,000 paired bootstrap replicates (seed `20260818`), spatial protection, and an exploratory 48-test Holm table are delivered.
- Fresh external label reads: `0`.

## Next bounded GPU round

Do not resume graph/head/loss development on A1, tonsil, D1, or P22. The next GPU round should first resolve the fresh-data provenance blockers, freeze one truly fresh annotated human RNA+protein section and one auditable mouse RNA+ATAC section, then run the selected structure and source-audited modern baselines under a common fixed-final, label-free protocol. A conservative starting estimate is one 24-32 GB GPU, at least 64 GB system RAM, roughly 80-120 GB persistent disk, and a one-seed-per-method/dataset infrastructure pilot used only to measure resource envelopes before preregistering the full fixed-seed matrix; Night-7A itself starts none of that work.

## Evidence locations

- Remote raw consensus: `/root/autodl-fs/night7a_consensus_20260818` (affinities and clusters remain remote only).
- External exact-commit source snapshots: `/root/autodl-fs/night7a_external_sources_20260818`.
- Metadata cache: `/root/autodl-fs/night7a_dataset_metadata_20260818`.
- Compact output: `outputs/night7a_handoff`; final Windows root: `D:/文档/ChatGPT/博士第一篇科研论文项目/night7a_handoff_20260818/official_compact` (independently verified after final Git persistence).
- Git branch: `revision/q2-night7a-cpu-consensus-preflight-20260818`; planned immutable final tag: `night7a-final-20260818`. The final commit, bundle SHA, compact indexes, Windows verification, and shutdown dispatch status are recorded in the non-self-referential external/post-dispatch indexes created after this report's delivery-index commit.
