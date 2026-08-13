#!/usr/bin/env python3
"""Fail-closed compact-delivery validation for Night-5C."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from SpaLORA.night5a_rnd import sha256_file
CONFIG=json.loads((REPO/"configs/night5c_laplacian_correction.json").read_text());OUT=REPO/CONFIG["paths"]["output_root"]
def require(condition,message):
 if not condition:raise RuntimeError(message)
def main():
 required=["night5c_report.md","p0_semantic_contract.json","corrected_run_order.json","corrected_training_manifest.json","supersession_map.json","recomputed_s1_topup_candidates.json","candidate_lifecycle_night5c.csv","per_run_summary_night5c.csv","five_seed_summary_night5c.csv","diffusion_resource_accounting.csv","selected_for_future_locked_p22.json","withheld_audit.json","budget_audit.json","protocol_deviations.json","shutdown_status.json"]
 for name in required:require((OUT/name).is_file(),"Missing required output "+name)
 p0=json.loads((OUT/"p0_semantic_contract.json").read_text());require(p0["status"]=="P0_SEMANTIC_PASS" and len(p0["resolved_runtime_contract"])==25 and len(p0["corrective_runtime_probes"])==8,"P0 semantic closure failed")
 require(all(x["semantic_contract_match"] and x["resolved_contract"]["learnable_attention_parameter_count"]==0 and max(x["uniform_forward_max_deviation"].values())==0 for x in p0["corrective_runtime_probes"]),"Corrective semantic probe failed")
 lock=json.loads((OUT/"corrected_training_manifest.json").read_text());require(lock["row_count"]==24 and lock["success_count"]==24 and lock["failure_count"]==0,"Corrected run count failed")
 for row in lock["runs"]:
  path=Path(row["record_path"]);require(sha256_file(path)==row["record_sha256"],"Corrected manifest drift");m=json.loads(path.read_text());require(m["semantic_contract_match"] and m["resolved_runtime_contract"]["actual_attention_policy"]=="uniform_all" and m["resolved_runtime_contract"]["actual_learned_fraction"] is None and m["resolved_runtime_contract"]["learnable_attention_parameter_count"]==0,"Corrected manifest semantic failure")
  for name,expected in m["artifact_sha256"].items():require(sha256_file(path.parent/name)==expected,"Corrected artifact drift")
 supers=json.loads((OUT/"supersession_map.json").read_text());require(supers["count"]==24 and not supers["old_results_overwritten"],"Supersession closure failed")
 reuse=json.loads((OUT/"night5b_valid_reuse_audit.json").read_text());require(reuse["passed"] and reuse["valid_training_units"]==84 and reuse["rerun_count"]==0 and reuse["overwrite_count"]==0,"84-run reuse audit failed")
 for row in reuse["rows"]:require(sha256_file(Path(row["manifest_path"]))==row["manifest_sha256"],"Reused manifest drift")
 replay=json.loads((OUT/"recomputed_s1_topup_candidates.json").read_text());require(replay["recomputed_topup_candidates"]==replay["old_topup_candidates"] and replay["newly_selected_incomplete_laplacian_candidates"]==[],"S1 replay unexpected")
 budget=json.loads((OUT/"budget_audit.json").read_text());require(budget["within_budget"] and budget["total_training_attempts"]==24 and budget["training_failures"]==0,"Budget failure")
 withheld=json.loads((OUT/"withheld_audit.json").read_text());require(withheld["passed"] and sum(withheld[k] for k in ("p22_candidate_runs","d1_runs","gse198353_runs","night4b_runs"))==0,"Withheld failure")
 selected=json.loads((OUT/"selected_for_future_locked_p22.json").read_text());require(len(selected["selected_candidates"])<=3 and not any(selected[k] for k in ("p22_run","d1_run","gse198353_run","night4b_run")),"Future selection failure")
 require(len(list(Path(CONFIG["paths"]["conditional_raw_runs"]).rglob("run_manifest.json")))==0,"Unexpected conditional runs")
 diff=subprocess.run(["git","diff","--check"],cwd=REPO,text=True,capture_output=True);require(diff.returncode==0,"git diff --check failed: "+diff.stdout+diff.stderr)
 files=[]
 for p in sorted(OUT.rglob("*")):
  if p.is_file() and p.name not in ("local_verification.json","delivery_index.json"):files.append({"path":str(p.relative_to(OUT)),"size":p.stat().st_size,"sha256":sha256_file(p)})
 payload={"schema_version":1,"status":"PASS","required_output_count":len(required),"corrected_runs_verified":24,"corrected_artifact_hashes_verified":True,"unaffected_training_runs_reverified":84,"old_invalid_runs_retained":24,"conditional_runs":0,"withheld_runs":0,"selected_candidate_count":len(selected["selected_candidates"]),"files_indexed":len(files),"files":files}
 (OUT/"local_verification.json").write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
 print("NIGHT5C_LOCAL_VERIFICATION_PASS files=%d"%len(files))
if __name__=="__main__":main()
