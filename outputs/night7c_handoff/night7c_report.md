# SpaLORA Night-7C report

## Outcome

Terminal status: `IMPLEMENTATION_SEMANTICS_INVALID`.

Night-7C stopped at the preregistered P1 semantic gate. Five R02 units completed
fresh-process checkpoint, endpoint-affinity, partition, and initial-MNN parity.
The next unit, `u005`, produced an initial MNN value of
`0.09747149795293808` twice, including after restoring the checkpoint's exact
pre-forward RNG state. The locked Night-7B loss curve contains
`0.09749911725521079`; absolute error `2.7619302272713364e-05` exceeds the registered
`1e-7` tolerance.

## Integrity interpretation

P0 passed all authority inputs, Night-7B compact 67/67 and root SHA, 30 locked
input units with 60 six-view archives, 124/124 historical CUDA training cells,
120 R02/R08 specialist transforms, checkpoint hashes, and the deny-by-default
label firewall. Historical files report `NVIDIA GeForce RTX 4080`; the current
instance reports `NVIDIA GeForce RTX 4080 SUPER`. This hardware difference is a
plausible explanation for the replay deviation, but it is an inference and is
not used to waive the exact tolerance.

## Work not authorized after the hard stop

- P2 serial/parallel runtime experiment: not run.
- Stage T routing: 0/240 transforms.
- Stage W weighted-MNN pilot: 0/48 training and 0/48 transforms.
- Label window and evaluation: never opened.
- Router or weighted-MNN scientific conclusion: none.

## Budgets and failures

Scientific retry was 0. No seed search, fallback, threshold change, H02 repeat,
or label access occurred. Two preformal implementation corrections are retained:
the observation-id SHA audit semantic correction and the saved RNG tensor-device
correction. Partial P1 files are marked invalid and remain under `/root/autodl-fs/night7c_conflict_rnd_20260818`.

## Tests

Night-7C directed tests: 9 passed. Full repository: 277 passed and 7 legacy
failures caused by old raw/result locks absent or intentionally not materialized
in the isolated worktree; no Night-7C touched-function test failed.

## Scientific limitation

This terminal state is an implementation/replay validity result, not evidence
for or against conflict routing, weighted MNN, P22 performance, or unified
cross-dataset improvement.
