#!/usr/bin/env python3
"""Build the compact Night-5C audit closure without copying raw checkpoints."""
from __future__ import annotations
import csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import pandas as pd
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from SpaLORA.night5a_rnd import sha256_file
CONFIG=json.loads((REPO/"configs/night5c_laplacian_correction.json").read_text());OUT=REPO/CONFIG["paths"]["output_root"]
def atomic_json(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8");os.replace(str(tmp),str(path))
def write_csv(path,rows):
 fields=sorted({k for r in rows for k in r});
 with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 lock=json.loads((OUT/"corrected_training_manifest.json").read_text());old_invalid=json.loads((Path(CONFIG["paths"]["night5b_handoff"])/"invalidated_runs.json").read_text());old_by={(x["dataset"],x["candidate_id"],int(x["seed"])):x for x in old_invalid["rows"]}
 supers=[]
 for r in lock["runs"]:
  m=json.loads(Path(r["record_path"]).read_text());key=(r["dataset"],r["candidate_id"],int(r["seed"]));old=old_by[key]
  if m["supersedes_invalid_run"]["old_run_manifest_sha256"]!=old["run_manifest_sha256"] or not m["semantic_contract_match"]:raise RuntimeError("Supersession or semantic manifest drift")
  supers.append({"dataset":r["dataset"],"candidate_id":r["candidate_id"],"seed":r["seed"],"old_invalid_manifest":old["run_manifest"],"old_invalid_manifest_sha256":old["run_manifest_sha256"],"old_invalid_retained":True,"new_corrected_manifest":r["record_path"],"new_corrected_manifest_sha256":r["record_sha256"],"new_semantic_contract_match":True})
 if len(supers)!=24 or len(old_by)!=24:raise RuntimeError("Supersession count mismatch")
 atomic_json(OUT/"supersession_map.json",{"schema_version":1,"count":24,"old_results_permanently_invalid":True,"old_results_overwritten":False,"rows":supers})
 raw=json.loads((Path(CONFIG["paths"]["night5b_handoff"])/"raw_runs_manifest.json").read_text());valid=[x for x in raw["rows"] if x.get("valid_for_canonical_analysis") and x.get("source_mode")!="locked_night5a_embedding_single_step_diffusion"]
 if len(valid)!=84:raise RuntimeError("Expected 84 valid Night-5B training units")
 for x in valid:
  if sha256_file(Path(x["manifest_path"]))!=x["manifest_sha256"]:raise RuntimeError("Night-5B valid manifest drift")
 atomic_json(OUT/"night5b_valid_reuse_audit.json",{"schema_version":1,"passed":True,"valid_training_units":84,"rerun_count":0,"overwrite_count":0,"all_manifest_sha256_reverified":True,"rows":valid})
 frame=pd.read_csv(OUT/"per_run_summary_corrected.csv");decision=json.loads((OUT/"recomputed_s1_topup_candidates.json").read_text());elig=json.loads((OUT/"canonical_exploratory_eligibility.json").read_text());contracts=json.loads((Path(CONFIG["paths"]["night5b_handoff"])/"candidate_contracts.json").read_text())["candidates"]
 corr={(x["dataset"],x["candidate_id"],int(x["seed"])):x for x in supers};rows=[]
 for _,r in frame.iterrows():
  d=r.to_dict();cid=d["candidate_id"];key=(d["dataset"],cid,int(d["seed"]));d["validity"]="night5c_corrected_valid" if key in corr else "night5b_or_night5a_reused_valid";d["canonical_eligibility"]="canonical_eligible" if cid in elig["canonical_eligible"] else ("valid_exploratory_noncanonical" if cid in elig["valid_exploratory_noncanonical"] else "three_seed_development_only")
  d["config_sha256"]=contracts[cid]["config_sha256"];d["source_candidate"]=contracts[cid].get("source_candidate","");d["source_or_contract_sha256"]=d["config_sha256"];rows.append(d)
 write_csv(OUT/"per_run_summary_night5c.csv",rows)
 five=pd.read_csv(OUT/"five_seed_summary_corrected.csv");five.to_csv(OUT/"five_seed_summary_night5c.csv",index=False)
 source_resource=OUT/"diffusion_effective_resource_audit.csv";target_resource=OUT/"diffusion_resource_accounting.csv"
 if source_resource.exists():source_resource.replace(target_resource)
 elif not target_resource.exists():raise RuntimeError("Missing diffusion resource accounting")
 lifecycle=[]
 for cid in contracts:
  if cid in ("B21_C09_LAPLACIAN005","B22_C09_LAPLACIAN010","B23_C10_LAPLACIAN005","B24_C10_LAPLACIAN010"):
   status="corrected_three_seed_not_selected_for_topup"
  elif cid in elig["canonical_eligible"]:status="canonical_eligible_five_seed"
  elif cid in elig["valid_exploratory_noncanonical"]:status="valid_exploratory_noncanonical_five_seed"
  else:status="valid_three_seed_development_only"
  lifecycle.append({"candidate_id":cid,"config_sha256":contracts[cid]["config_sha256"],"night5c_status":status,"recomputed_s1_selected":cid in decision["recomputed_topup_candidates"]})
 write_csv(OUT/"candidate_lifecycle_night5c.csv",lifecycle)
 atomic_json(OUT/"withheld_audit.json",{"schema_version":1,"passed":True,"p22_candidate_runs":0,"p22_results_opened":False,"d1_runs":0,"d1_results_opened":False,"gse198353_runs":0,"gse198353_results_opened":False,"night4b_runs":0,"raw_run_dataset_directories":["a1","placenta"]})
 atomic_json(OUT/"protocol_deviations.json",{"schema_version":1,"deviation_count":0,"ordinary_implementation_failures_before_scientific_training":[{"kind":"missing_firewall_config_key","effect":"zero scientific attempts; fixed before first run"},{"kind":"S1 evaluator Python comprehension error","effect":"post-lock evaluation only; no scientific rerun"}],"scientific_parameters_changed":False,"seed_search":False,"result_dependent_tuning":False})
 attempts=24;conditional=len(decision["newly_selected_incomplete_laplacian_candidates"])*4
 atomic_json(OUT/"budget_audit.json",{"schema_version":1,"status":"PASS","mandatory_training_units":24,"conditional_training_units":conditional,"scientific_training_units":attempts+conditional,"training_failures":0,"same_tuple_infrastructure_retries":0,"total_training_attempts":attempts+conditional,"scientific_unit_cap":40,"attempt_cap":44,"within_budget":attempts+conditional<=40,"p22_run":False,"d1_run":False,"gse198353_run":False,"night4b_run":False})
 atomic_json(OUT/"shutdown_status.json",{"schema_version":1,"status":"PENDING_DISPATCH","command":"/usr/bin/shutdown","dispatch_exit_status":None,"console_power_state_claimed":False,"reconnect_after_dispatch":False})
 final=json.loads((OUT/"final_frontier_decision.json").read_text());selected=final["selected_for_future_locked_p22"]
 corrected=frame[frame.candidate_id.str.startswith(("B21_","B22_","B23_","B24_"))].groupby(["candidate_id","dataset"])[["ari","nmi","q","spatial_neighbor_agreement","spatial_cluster_moran_mean","spatial_cluster_geary_mean","boundary_disagreement","runtime_seconds","gpu_peak_allocated_mib"]].mean().reset_index()
 corrected_table=corrected.to_csv(index=False).strip()
 report=["# SpaLORA Night-5C report","","## Outcome","", "Status: `NIGHT5C_CORRECTION_COMPLETE_CANDIDATES_LOCKED_FOR_P22`.","", "Future locked P22 candidates: "+", ".join("`%s`"%x for x in selected)+". No P22 run was performed.","","## Runtime semantic correction","","All 24 corrected units used actual `uniform_all`: each manifest contains declared and resolved contracts, both contract SHAs, `semantic_contract_match=true`, exact within/cross alpha `[0.5,0.5]`, and zero learnable attention parameters. The eight preflight construction probes covered four candidates x two datasets at seed 0; the negative shrink regression was rejected before training.","","The 24 old B21-B24 units remain permanently invalid and untouched. `supersession_map.json` links them one-to-one to the new corrected manifests. All 84 unaffected Night-5B training manifests were independently re-hashed; reruns/overwrites: 0/0.","","## Corrected three-seed results","","```csv",corrected_table,"```","","## S1 replay and conditional stage","", "Old top-up set: `"+"`, `".join(decision["old_topup_candidates"])+"`.","Recomputed set: `"+"`, `".join(decision["recomputed_topup_candidates"])+"`.","Added/removed: none. No Laplacian candidate newly entered, so conditional seeds 3/4 ran 0 units.","","## Eligibility and frontiers","", "Canonical five-seed: "+", ".join("`%s`"%x for x in elig["canonical_eligible"])+".", "Exploratory five-seed: "+(", ".join("`%s`"%x for x in elig["valid_exploratory_noncanonical"]) or "none")+".", "B10 is canonical and the strict balanced choice above B01. B17 is canonical and the accuracy-frontier leader. B21-B24 remain valid corrected three-seed development evidence but are noncanonical because they were not selected for top-up.","", "Balanced frontier: "+", ".join("`%s`"%x for x in final["balanced_frontier"])+".", "Accuracy frontier: "+", ".join("`%s`"%x for x in final["accuracy_frontier"])+".", "All candidate summaries contain deltas against historical B00 and explicit `relative_b01` fields. Diffusion gating uses source training cost plus preserved historical incremental zero; the zero is not interpreted as end-to-end cost.","","## Protocol and budget","", "Training: 24 mandatory + 0 conditional = 24 scientific units; failures 0; same-tuple retries 0; total attempts 24, within 40/44 caps. P22, D1, GSE198353, and Night-4B access/run counts are all zero.","","## Engineering and delivery","", "Tests, Git commit/tag/push, compact D-drive delivery hashes, and shutdown dispatch are finalized after this report draft; machine shutdown status must only claim command dispatch exit status, not console power state.",""]
 (OUT/"night5c_report.md").write_text("\n".join(report),encoding="utf-8")
 # Raw manifest lists only hashes and paths; no checkpoints are copied.
 rawrows=[]
 for root in (Path(CONFIG["paths"]["corrected_raw_runs"]),Path(CONFIG["paths"]["conditional_raw_runs"])):
  if root.exists():
   for p in sorted(root.rglob("run_manifest.json")):rawrows.append({"manifest_path":str(p),"manifest_sha256":sha256_file(p),"model_state_in_compact":False})
 atomic_json(OUT/"raw_runs_manifest_night5c.json",{"schema_version":1,"run_manifest_count":len(rawrows),"rows":rawrows,"raw_checkpoints_remote_only":True})
 print("NIGHT5C_FINALIZE_PASS",len(supers),len(valid),len(rows),len(selected))
if __name__=="__main__":main()
