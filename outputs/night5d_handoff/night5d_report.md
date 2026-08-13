# SpaLORA Night-5D locked P22 confirmation report

## Final status

`BASELINE_REPLAY_MISMATCH` (preregistered hard stop). No locked-candidate win/loss conclusion was produced.

## Execution completeness before the hard stop

- Authority parent: `f8bba65ff3f1a3dacf310dd12f81f8b379315b22`.
- P0-CONFIRM passed: REV1 historical evidence 5/5, historical initial-state/no-label probes 5/5, runtime semantic contracts 4/4.
- New training: 35/35; deterministic B17 transforms: 10/10; failures: 0.
- Historical B00 seeds 0-4 were not retrained or overwritten. Their absent final weight files remain expected under the original Night-3B contract.
- All new training units saved a final `model_state.pt`, a 10-row checkpoint index, and SHA-protected outputs.
- The label firewall passed. P22 labels were opened once only after the 35+10 total lock.

## Hard-stop evidence

Historical B00 replay reproduced ARI, NMI, neighbor agreement, Moran's I and Geary C exactly for seeds 0-4. Derived Q differed by at most `5.55e-17`. Boundary disagreement alone differed for every historical seed:

| seed | absolute boundary difference |
|---:|---:|
| 0 | 0.007510632768202685 |
| 1 | 0.007736672640913322 |
| 2 | 0.007802936093173812 |
| 3 | 0.007813460288532692 |
| 4 | 0.007532340249943204 |

The mismatch is consistent with a boundary edge-counting definition difference between the historical table and the current evaluator, but the locked protocol did not authorize substituting a metric definition after labels opened. Therefore the `1e-12` replay gate failed exactly as written.

## Statistical consequence

The three primary exact sign-flip tests, Holm correction, 100,000-replicate paired bootstrap, spatial protection decisions, and two secondary mechanistic contrasts were **not run**. The frozen 50-row metric table is retained as diagnostic evidence only and was not used to rank or modify candidates.

## Semantic and reproducibility audit

- C09/B17 source forward used exact `[0.5, 0.5]` fusion weights. Attention gradients were zero/None; parameter perturbation stayed within the measured CUDA sparse-repeat envelope.
- B17 used one alpha=0.10 diffusion step from each corresponding frozen C09 embedding and immutable normalized sparse P22 adjacency.
- D1, GSE198353, Night-4B and formal external benchmark were not opened or run.
- Night-3B final weights remain unavailable and no Night-3AF weights were substituted. Historical weight-level analysis remains forbidden.

## Scientific interpretation

Night-5D cannot support a confirmatory method claim because the preregistered baseline replay gate failed before confirmatory statistics. This is not evidence that a candidate won or lost; it is a metric-provenance inconsistency requiring planner review. No post-hoc repair or candidate tuning was attempted.
