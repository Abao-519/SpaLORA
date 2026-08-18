# SpaLORA Night-7B adaptive relational score R&D report

## Plain-language outcome

Night-7B reused the 60 immutable Night-6C/Night-6D views, compared 18 clustering heads, then trained the ten registered adaptive/relational recipes under a strict no-label worker contract. Labels were opened only after each stage was completely locked.

Terminal status: `NIGHT7B_P22_FRONTIER_ONLY_NO_UNIFIED_WINNER`.

Unified locked candidates: none.

The top all-seed configuration `R02__E1_ADAPTER_C06_MEAN__H01` did not improve both the equal-weight human-lymph aggregate and P22 relative to C00. Its priority-weighted delta versus C00 was ARI +0.02568, NMI +0.01902, Q +0.02235; versus C06 it was ARI +0.00689, NMI +0.00614, Q +0.00652.

Across the locked recipe panel, the strongest descriptive loss-family association was `MNN` (+0.00396 present-minus-absent priority delta Q) and the weakest was `MASK` (-0.02588). This is not a causal ablation claim.

The quantitative tables below report the same-seed differences against confirmed C00 and the Night-7A C06 accuracy frontier; no seed, epoch, recipe, threshold, or dataset-specific setting was selected after viewing labels.

## Locked stage decisions

- H promoted heads: H01, H02
- R1 promoted configurations: R02__E1_ADAPTER_C06_MEAN__H01, R02__E1_ADAPTER_C06_MEAN__H02, R08__E1_ADAPTER_C06_MEAN__H01, R08__E1_ADAPTER_C06_MEAN__H02
- Accuracy frontier order: R02__E1_ADAPTER_C06_MEAN__H01, R02__E1_ADAPTER_C06_MEAN__H02, R08__E1_ADAPTER_C06_MEAN__H01, R08__E1_ADAPTER_C06_MEAN__H02
- Balanced frontier order: R02__E1_ADAPTER_C06_MEAN__H01, R02__E1_ADAPTER_C06_MEAN__H02, R08__E1_ADAPTER_C06_MEAN__H01, R08__E1_ADAPTER_C06_MEAN__H02

## Final all-seed comparison versus C00

| config_id | priority_weighted_delta_q | balanced_macro_delta_q | worst_dataset_delta_q | a1_mean_delta_q | d1_mean_delta_q | p22_mean_delta_q | tonsil_mean_delta_q | total_q_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R02__E1_ADAPTER_C06_MEAN__H01 | 0.02235 | 0.01467 | -0.00647 | -0.00647 | -0.00146 | 0.07171 | -0.00511 | 15 |
| R02__E1_ADAPTER_C06_MEAN__H02 | 0.02235 | 0.01467 | -0.00647 | -0.00647 | -0.00146 | 0.07171 | -0.00511 | 15 |
| R08__E1_ADAPTER_C06_MEAN__H01 | 0.01676 | 0.01127 | -0.00970 | -0.00970 | 0.00803 | 0.05080 | -0.00405 | 20 |
| R08__E1_ADAPTER_C06_MEAN__H02 | 0.01676 | 0.01127 | -0.00970 | -0.00970 | 0.00803 | 0.05080 | -0.00405 | 20 |

## Final all-seed comparison versus C06

| config_id | priority_weighted_delta_q | balanced_macro_delta_q | a1_mean_delta_q | d1_mean_delta_q | p22_mean_delta_q | tonsil_mean_delta_q |
| --- | --- | --- | --- | --- | --- | --- |
| R02__E1_ADAPTER_C06_MEAN__H01 | 0.00652 | 0.00367 | -0.00327 | -0.00692 | 0.02666 | -0.00178 |
| R02__E1_ADAPTER_C06_MEAN__H02 | 0.00652 | 0.00367 | -0.00327 | -0.00692 | 0.02666 | -0.00178 |
| R08__E1_ADAPTER_C06_MEAN__H01 | 0.00092 | 0.00028 | -0.00650 | 0.00258 | 0.00574 | -0.00072 |
| R08__E1_ADAPTER_C06_MEAN__H02 | 0.00092 | 0.00028 | -0.00650 | 0.00258 | 0.00574 | -0.00072 |

## H and R1 rankings

| config_id | priority_weighted_delta_q | balanced_macro_delta_q | worst_dataset_delta_q | total_q_wins |
| --- | --- | --- | --- | --- |
| H01 | 0.01583 | 0.01099 | -0.00333 | 19 |
| H02 | 0.01565 | 0.01082 | -0.00458 | 21 |
| H07 | 0.01560 | 0.01074 | -0.00396 | 19 |
| H06 | 0.01537 | 0.01038 | -0.00444 | 19 |
| H03 | 0.01412 | 0.00981 | -0.00405 | 20 |
| H04 | 0.01277 | 0.00886 | -0.00367 | 19 |
| H15 | 0.01139 | 0.00623 | -0.01429 | 12 |
| H05 | 0.01009 | 0.00690 | -0.00318 | 20 |
| H14 | 0.00540 | 0.00209 | -0.01701 | 12 |
| H00 | 0.00000 | 0.00000 | 0.00000 | 0 |

| config_id | priority_weighted_delta_q | balanced_macro_delta_q | worst_dataset_delta_q | total_q_wins |
| --- | --- | --- | --- | --- |
| R02__E1_ADAPTER_C06_MEAN__H01 | 0.03385 | 0.02243 | -0.01024 | 3 |
| R02__E1_ADAPTER_C06_MEAN__H02 | 0.03385 | 0.02243 | -0.01024 | 3 |
| R08__E1_ADAPTER_C06_MEAN__H01 | 0.03179 | 0.02381 | -0.00973 | 5 |
| R08__E1_ADAPTER_C06_MEAN__H02 | 0.03179 | 0.02381 | -0.00973 | 5 |
| R04__E1_ADAPTER_C06_MEAN__H01 | 0.03103 | 0.02318 | -0.00993 | 5 |
| R04__E1_ADAPTER_C06_MEAN__H02 | 0.03103 | 0.02318 | -0.00993 | 5 |
| R07__E1_ADAPTER_C06_MEAN__H01 | 0.02780 | 0.02091 | -0.01349 | 5 |
| R07__E1_ADAPTER_C06_MEAN__H02 | 0.02780 | 0.02091 | -0.01349 | 5 |

## Mechanism and resource diagnostics

Loss diagnostics contain 124 successful training rows. Gate diagnostics contain 124 rows; 0 learned-MoE rows met the preregistered descriptive collapse flag (minimum expert mean below 0.05). These are mechanism diagnostics, not post-hoc selection criteria.

`loss_family_effects_descriptive.csv` reports registered present-versus-absent recipe associations for every loss family; these comparisons are explicitly descriptive rather than causal. `gate_unlabeled_quality_correlations.csv` relates MoE behavior only to label-free training losses.

## Failures and integrity

There were 34 preserved numerical/upstream failures. No formal scientific retry or fallback was used. P0 passed 30/30 real C00/C06 parity and 10/10 invalid smoke checkpoint reloads after eight preserved global pre-label corrections.

All checkpoints, affinities and large raw arrays remain under `/root/autodl-fs`; the compact handoff contains hashes and small evidence only.
