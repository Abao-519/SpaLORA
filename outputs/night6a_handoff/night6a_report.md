# SpaLORA Night-6A report

## Authoritative terminal status

`IMPLEMENTATION_SEMANTICS_INVALID`.

The final audit found that the P0 numeric-audit script called `anndata.read_h5ad` on the original tonsil files before the formal lock. Although the script never indexed, printed, exported, or used annotation values, eager AnnData loading deserialized `obs` values into memory. This violates the taskbook's strict pre-lock firewall. Formal training used label-free copies with zero `obs` columns, no label informed any training, checkpoint, seed, parameter, or candidate decision, and P22/D1/GSE198353/MISAR training/Night-4B remained untouched. Nevertheless, this run is not certified firewall-clean and no candidate may be promoted from it.

## Execution and evidence preservation

- P0 semantic tests: 8/8 passed; formal runtime contracts: 45/45 matched.
- Formal units: 45 success, 0 failure, 0 retry, within the cap of 96.
- R1 completed 32/32 fixed A1 units before evaluation. No candidate had positive mean delta-Q while passing spatial protection; the advancement set was empty.
- R2 and R3 therefore ran only preregistered N00 reference coverage: 4/4 and 9/9 units, respectively. No negative candidate was back-filled.
- Raw artifacts remain under `/root/autodl-fs/night6a_raw_runs_20260814`. 270 files were independently rehashed; mismatches: 0.

## Diagnostic results (not an authoritative scientific promotion)

The least-negative R1 delta-Q was N10 (-0.000879); among spatial-protection-passing candidates it was N05 (-0.002516). N00 A1 five-seed means were ARI=0.262235, NMI=0.383919, Q=0.323077; these do not meet the competitiveness marker ARI >= 0.316 and NMI >= 0.406.

## Required scientific questions

1. **Real pruning versus intersection-edge gain:** diagnostically, no registered pruning candidate improved mean A1 Q over N00; some pruning variants also failed spatial protection. This cannot support a promotion claim.
2. **Hard versus soft pruning:** the hard and soft variants did not show an actionable monotone rescue. Stronger graph changes tended to impair neighbor/Moran behavior; there is no defensible dose trend.
3. **Gradient conflict after IGE:** PCGrad/MinNorm semantics passed analytic and runtime probes, but neither converted the diagnostic A1 conflict intervention into positive mean delta-Q. The run cannot establish an externally valid causal conclusion.
4. **Barlow/neighbor alignment across A1 and tonsil:** no alignment candidate advanced from R1, so the preregistered funnel correctly did not spend tonsil candidate runs. Cross-dataset benefit was not demonstrated.
5. **Combinations versus modules:** every combination had negative mean R1 delta-Q. Complexity stacking did not outperform the single modules under the locked screen.
6. **Spatial trade-offs:** several candidates failed the locked neighbor/Moran/Geary protection rule. Candidates that passed it still had negative mean delta-Q, so no accuracy-spatial win was identified.
7. **Recent-method ranges:** direct competitiveness claims are not made. Public-method numbers can differ in data processing, label use, checkpoint selection, and evaluation protocol; those risks prevent a fair numerical ranking here.

## Interpretation and next action

The preregistered numeric funnel would diagnostically end as `NO_STRUCTURAL_RESCUE_CANDIDATE`, but the stricter authoritative result is `IMPLEMENTATION_SEMANTICS_INVALID`. Do not open D1/P22 for these candidates and do not use this run as confirmatory evidence. Preserve it as an auditable failed implementation attempt. A future rerun, if separately authorized, must perform all pre-lock numeric inspection through a low-level label-excluding reader from the first byte access.
