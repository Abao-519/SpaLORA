# SpaLORA Night-10A report

## Terminal decision

`IMPLEMENTATION_SEMANTICS_INVALID`

The authority and resource checks passed, the server reference suite passed 10/10 tests, historical A1 ARI/NMI parity was exact, and the implemented subset passed real-view probes. A subsequent field-by-field coverage audit found that this subset did not implement the complete registered QCRD semantics. The earlier partial P0 PASS is preserved verbatim and is explicitly superseded for the go/no-go decision by `p0_semantic_coverage_reaudit.json`.

## Blocking differences

- Global quality omits registered spatial local consistency and a global graph-local residual feature.
- Per-spot quality omits registered local-neighbor entropy.
- The authority does not uniquely specify the masking fraction/recipe or the numerical loss weights for correction magnitude, original anchor, boundary preservation, and Q07 MNN supervision.

These are definition-level differences, not ordinary implementation details. No values were guessed.

## Execution counts

- Stage M: not started.
- Formal training: 0.
- Formal transforms: 0.
- R1/R2 evaluation: 0.
- Scientific retry: 0; fallback: 0.
- New third-party benchmark: 0; external-data access: 0.

## Label firewall

A1 was read four times by isolated P0 evaluator/diagnostic processes while fixing evaluator-only serialization/alignment issues. The computed values were used only to establish metric parity and never entered training, loss, candidate selection, or hyperparameter definition. tonsil/D1/P22 label reads were 0; MISAR Y and E18.5 reads were 0.

## Required recovery authority

A Night-10A REV1 must define the omitted feature formulas and all fixed numerical training weights/masking semantics, and must add negative/real-construction tests that cover those definitions. The existing partial implementation must not be promoted as a scientific negative result.
