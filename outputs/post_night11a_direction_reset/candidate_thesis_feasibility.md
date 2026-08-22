# Candidate thesis feasibility

## Decision

`NEW_METHOD_REQUIRES_SCOPE_EXPANSION`

The scientific object is potentially distinguishable from existing shared/private or gating methods, but the present assets do not support the full two-family claim or a clean confirmatory boundary.

## What existing assets support

- Both audited families retain feature-level matrices. A1/tonsil/D1 preserve RNA counts and 31 protein targets; P22 preserves 22,914 RNA features and 121,068 ATAC peaks, not merely embeddings.
- P22 peak names preserve genomic intervals for 121,027 of 121,068 features. Existing code can compute bootstrap and spatial evidence with sparse operations.
- A1/tonsil/D1 provide a deposited, auditable exact-name correspondence for 29 of 31 protein targets. Two targets remain unresolved and are not guessed.
- GSE198353 contains two existing independent RNA+protein spleen slices with raw 10x matrices and spatial archives; they were not used to define the Night-11A formula.

## Blocking evidence gaps

1. P22 has no authoritative genome build in H5AD metadata, no peak-gene link, no gene-activity annotation and no traceable generator for either. Genomic-looking strings alone do not authorize a build or link.
2. The only audited RNA+ATAC biological unit is P22, which was used in development, Q00 references and Night-11A formula design. Thus RNA+ATAC has no unused discovery unit and no confirmatory unit.
3. All current RNA+protein biological units have at least been registered, preflighted or used in development. GSE198353 can be a next discovery asset, but under the strict governance definition there is `NO_PRISTINE_CONFIRMATORY_UNIT`.
4. Current checkpoints preserve embeddings and a cache-SHA chain, not a feature-link object. P22 LSI specifically severs peak-level recovery at the model boundary unless a new provenance manifest is added.

## Minimum scope expansion

- RNA+ATAC: obtain one existing-authority or newly approved slice/dataset with raw peak counts, paired spot IDs, coordinates, explicit genome build, and deposited or reproducibly generated peak-gene/gene-activity annotation. A second untouched slice or dataset must be reserved before formula design for confirmation.
- RNA+protein: approve an authoritative ADT-target-to-gene map for GSE198353 and freeze one replicate for discovery while defining whether the other replicate's prior metadata preflight is acceptable as confirmation. If strict pristine status is required, add one untouched slice.
- Baselines/data: a later scientific phase would need a preregistered modern comparison set. This audit does not download, train or benchmark anything.

## Go/no-go interpretation

This is not a finding that a new method works. It is a bounded feasibility result: the mathematical object is worth considering only after the missing annotation and validation boundaries are supplied. Implementing another threshold, hard gate, temperature rule, shared/private loss or expert router with the present embeddings would be `NO_DEFENSIBLE_NEW_METHOD_FROM_EXISTING_ASSETS` for that narrower proposal.
