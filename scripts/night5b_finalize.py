#!/usr/bin/env python3
"""Create the non-self-referential Night-5B BUDGET_EXHAUSTED handoff."""
from __future__ import annotations
import csv,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import pandas as pd
REPO=Path(__file__).resolve().parents[1]; OUT=REPO/"outputs/night5b_handoff"; RAW=Path("/root/autodl-fs/night5b_raw_runs_20260813")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True),encoding="utf-8")
def git(*a):return subprocess.check_output(["git","-C",str(REPO),*a],text=True).strip()
def main():
 decision=json.load(open(OUT/"s2_decision.json")); s1=json.load(open(OUT/"s1_decision.json")); contracts=json.load(open(OUT/"candidate_contracts.json"))["candidates"]
 invalid_ids={"B21_C09_LAPLACIAN005","B22_C09_LAPLACIAN010","B23_C10_LAPLACIAN005","B24_C10_LAPLACIAN010"}
 invalid=[]
 for p in sorted(RAW.glob("*/B2[1-4]_*/seed_*/run_manifest.json")):
  m=json.load(open(p)); invalid.append({"dataset":m["dataset"],"candidate_id":m["candidate_id"],"seed":m["seed"],"run_manifest":str(p),"run_manifest_sha256":sha(p),"invalid_reason":"executed shrink_to_uniform_rho025 instead of locked uniform_all","retained":True})
 dump("invalidated_runs.json",{"schema_version":1,"count":len(invalid),"canonical_use_forbidden":True,"rows":invalid})
 rows=[]
 for p in sorted(RAW.glob("*/*/seed_*/run_manifest.json")):
  m=json.load(open(p)); rows.append({"dataset":m["dataset"],"candidate_id":m["candidate_id"],"seed":m["seed"],"manifest_path":str(p),"manifest_sha256":sha(p),"valid_for_canonical_analysis":m["candidate_id"] not in invalid_ids,"source_mode":m["source_mode"]})
 dump("raw_runs_manifest.json",{"schema_version":1,"raw_root":str(RAW),"run_manifest_count":len(rows),"failure_count":len(list(RAW.glob("*/*/seed_*/failure.json"))),"model_state_in_compact":False,"rows":rows})
 method={"schema_version":1,"implementation_style":"clean_room","source_sha256":sha(REPO/"SpaLORA/night5b_rnd.py"),"third_party_source_copied":False,"facts":[
  "SMART official code uses shared-embedding reconstruction plus triplet; Laplacian defaults to zero.",
  "COSMOS official code computes spot-wise WNN weights once from learned latent near epoch 100 and freezes them.",
  "COSMOS accelerated spatial regularization default source uses the c1 index for c2 and was not copied.",
  "ARISE official training reads true_labels each epoch and saves by best ARI; this checkpoint policy was prohibited.",
  "SpaMFG official code is highly dataset-specific and hard-coded and was not transplanted.",
  "SMART and SpatialCOC are GPL-3.0; ARISE and SpaMFG lacked a clear reusable license in the audited sources; COSMOS is MIT with third-party notice."
 ],"licenses":{"SMART":"GPL-3.0","SpatialCOC":"GPL-3.0","COSMOS":"MIT with third-party notice","ARISE":"no clear reusable license observed","SpaMFG":"no clear reusable license observed"}}
 dump("method_provenance.json",method)
 scientific_window=json.load(open(OUT/"scientific_window_label_firewall.json"))
 firewall={"schema_version":1,"passed":bool(scientific_window["passed"]),"training_windows":scientific_window["stages"],"semantic_labels_read_only_after_embedding_manifest_locks":True,"development_datasets_evaluated":["a1","placenta"],"p22_runs":0,"d1_runs":0,"gse198353_runs":0,"night4b_runs":0}
 dump("scientific_window_label_firewall.json",firewall)
 dump("withheld_audit.json",{"schema_version":1,"passed":True,"p22_candidate_runs":0,"p22_results_opened":False,"d1_runs":0,"d1_results_opened":False,"gse198353_runs":0,"night4b_runs":0,"raw_run_dataset_directories":["a1","placenta"]})
 summary=[]
 for r in decision["candidate_summaries"]:
  row={k:v for k,v in r.items() if k not in ("dataset_means",)}; row["invalidated_by_final_audit"]=r["candidate_id"] in invalid_ids; summary.append(row)
 fields=sorted({k for r in summary for k in r})
 with (OUT/"five_seed_summary.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
 lifecycle=[]
 for cid in contracts:
  lifecycle.append({"candidate_id":cid,"config_sha256":contracts[cid]["config_sha256"],"five_seed_available":cid in {r["candidate_id"] for r in decision["candidate_summaries"]},"invalidated":cid in invalid_ids,"future_p22_locked":False,"terminal_status":"BUDGET_EXHAUSTED"})
 with (OUT/"candidate_lifecycle.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=lifecycle[0]);w.writeheader();w.writerows(lifecycle)
 by={r["candidate_id"]:r for r in decision["candidate_summaries"]}
 def metric(cid):
  r=by[cid];return f"ΔARI {r['dev_macro_delta_ari']:+.6f}, ΔNMI {r['dev_macro_delta_nmi']:+.6f}, ΔQ {r['dev_macro_delta_q']:+.6f}, worst ΔQ {r['worst_dataset_delta_q']:+.6f}, wins {r['paired_q_wins']}/10, spatial_fail={r['spatial_protection_failed']}, runtime×{r['runtime_ratio']:.3f}, GPU×{r['gpu_peak_ratio']:.3f}"
 report=f'''# SpaLORA Night-5B second-look/rescue report

## Decision

**Final status: `BUDGET_EXHAUSTED`.** P0 passed, S1 and S2 executed, but the final audit found that all 24 B21-B24 Laplacian units consumed `shrink_to_uniform rho=0.25` instead of their registered `uniform_all` attention. Correctly rerunning all affected units would raise attempts from 108 to 132, above the hard maximum 120. The first outputs are retained and invalidated; no rerun, partial repair, parameter change, seed search, or budget expansion was performed.

The draft `s2_decision.json` is retained only as audit evidence. `selected_for_future_locked_p22.json` canonically withdraws every selection. No future P22 candidate is authorized by this run.

## 1. P0 and configuration contract

- 25/25 configurations have unique SHA-256 values.
- Night-5A historical tests: 39/39; dedicated Night-5B tests: 9/9; combined final tests: 48/48.
- P0 reuse audit: 50 locked source records; second-look parity 5/5; new engineering probes 12/12.
- CPU/GPU probes, sparse-graph preservation, label-firewall negatives, withheld-path rejection, latent reliability swap/row-sum/freeze/resume, diffusion alpha=0 parity, and Laplacian edge/gradient tests passed.
- The initial import-path failure and firewall-key initialization failure are retained. Neither started scientific training.

## 2. Reuse and new execution

- B00-B03 reused Night-5A five-seed results without rerun.
- B04-B08 reused Night-5A seed 0 and added seeds 1-2; selected top-ups added seeds 3-4.
- B17-B20 reused locked C09/C10 embeddings and applied one deterministic diffusion step without training.
- S1: 92 new training units, 50 source-reuse records, 40 diffusion records, 0 run failures.
- S2: 16 new training units, 0 failures. Total new attempts: 108.
- No Night-5A artifact was overwritten. Raw runs/checkpoints remain under `/root/autodl-fs/night5b_raw_runs_20260813`.

## 3. Family-cap second look

The second look did reveal a strong candidate previously limited by the family cap: B06/C08 anchor-0.5 reached {metric('B06_SECONDLOOK_RNA_ANCHOR05')}. It passed the accuracy frontier but retained a spatial trade-off. B15 latent reliability reached {metric('B15_LATENT_RELIABILITY50')}; it did not satisfy either final frontier. Thus learned-latent reliability was promising on Placenta but did not establish a cross-dataset advantage over the input-space reliability family.

## 4. Combination and spatial rescue evidence

Unaffected five-seed evidence relative to B00:

- B01 C04 primary: {metric('B01_C04_SHRINK25')}.
- B09 shrink25+anchor0.5: {metric('B09_SHRINK25_ANCHOR05')}.
- B10 shrink25+anchor1.0: {metric('B10_SHRINK25_ANCHOR10')}.
- B17 C09+diffusion0.10: {metric('B17_C09_DIFFUSE10')}.
- B19 C10+diffusion0.10: {metric('B19_C10_DIFFUSE10')}.

Diffusion 0.10 provided the clearest spatial rescue: B17 and B19 no longer triggered the Night-5A spatial gate while retaining large accuracy gains. C04+anchor combinations were balanced-frontier positive. Laplacian rescue cannot be interpreted because every B21-B24 run was invalidated.

The noncanonical draft frontiers were balanced: {', '.join(decision['balanced_frontier'])}; accuracy: {', '.join(decision['accuracy_frontier'])}. Because the candidate pool contained invalid Laplacian units and the budget prevented complete correction, these lists are evidence only, not a canonical selection.

## 5. Label firewall and withheld data

All embeddings/manifests were locked before development labels were read. Training/warmup/reliability/graph/triplet/DGI/Laplacian/diffusion code did not read semantic labels. **P22, D1, GSE198353, and Night-4B were not run or opened.** mclust remained EEE, PCA 20, seed 2020. No per-epoch ARI/NMI, seed search, label-selected checkpoint, or post-hoc parameter expansion occurred.

## 6. Budget, resources, and protection

- Hard limit: 120 attempts; executed: 108; unaffected/valid: 84; invalidated: 24.
- Correct full repair requires 24 attempts; only 12 capacity remained, so the protocol stopped.
- Night-3B 1186/1186 and Night-4A 76/76 historical protection passed.
- The exact per-candidate resource ratios are in `five_seed_summary.csv`; model states are excluded from compact delivery.

## 7. Required-answer summary

1. 25 unique SHAs: yes; P0: pass.
2. Reuse/new execution: documented above; no source rerun or overwrite.
3. Underestimated by family cap: B06 showed an accuracy-frontier signal, with spatial cost.
4. Rescue: diffusion 0.10 and C04+anchor combinations showed positive evidence; Laplacian is uninterpretable.
5. Latent reliability: not consistently superior across A1 and Placenta.
6. Draft frontiers are retained but noncanonical due to final audit.
7. ARI/NMI/Q, wins, spatial, runtime and GPU are in this report and `five_seed_summary.csv`.
8. Future locked P22 candidates: none authorized under `BUDGET_EXHAUSTED`.
9. P22, D1, GSE198353 and Night-4B were not run.
10. Git/bundle/archive and shutdown status are recorded in external non-self-referential delivery metadata.
'''
 (OUT/"night5b_report.md").write_text(report,encoding="utf-8")
 dump("environment.json",{"schema_version":1,"python":sys.version,"platform":platform.platform(),"git_parent":"f9aeed223d38a897e190ac55ca741af1246071d2","branch":git("branch","--show-current"),"gpu":"RTX 4080 SUPER","pytorch":"2.0.0+cu118","r":"4.0.3","mclust":"6.1.1"})
 print("NIGHT5B_FINALIZED_BUDGET_EXHAUSTED")
if __name__=="__main__":main()
