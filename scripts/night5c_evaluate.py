#!/usr/bin/env python3
"""Post-lock Night-5C S1 replay and final locked selection evaluator."""
from __future__ import annotations
import argparse,csv,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary,symmetric_knn_adjacency
from SpaLORA.night5a_rnd import sha256_file
from SpaLORA.night5b_rnd import DIFFUSION_MAP
from SpaLORA.night5c_semantic import CORRECTIVE_IDS
CONFIG_PATH=REPO/"configs/night5c_laplacian_correction.json"
METRICS=("ari","nmi","spatial_neighbor_agreement","spatial_cluster_moran_mean","spatial_cluster_geary_mean","boundary_disagreement","runtime_seconds","gpu_peak_allocated_mib")

def atomic_json(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8");os.replace(str(tmp),str(path))
def write_csv(path,rows):
 fields=sorted({k for r in rows for k in r});path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def spatial_failed(m):
 n=float(m.delta_spatial_neighbor_agreement.mean());mo=float(m.delta_spatial_cluster_moran_mean.mean());g=float(m.delta_spatial_cluster_geary_mean.mean())
 return bool((n<-.03 and mo<-.03) or (g>.03 and (n<-.03 or mo<-.03)))
def summarize(frame,cid,seeds,reference="B00_C00_FULL_IGE"):
 c=frame[(frame.candidate_id==cid)&frame.seed.isin(seeds)];r=frame[(frame.candidate_id==reference)&frame.seed.isin(seeds)];expected=2*len(seeds)
 if len(c)!=expected or len(r)!=expected:return {"candidate_id":cid,"complete":False,"observed_rows":len(c),"expected_rows":expected}
 m=c.merge(r,on=["dataset","seed"],suffixes=("","_reference"),validate="one_to_one")
 for x in METRICS:m["delta_"+x]=m[x]-m[x+"_reference"]
 m["q"]=(m.ari+m.nmi)/2;m["q_reference"]=(m.ari_reference+m.nmi_reference)/2;m["delta_q"]=m.q-m.q_reference
 dq={d:float(g.delta_q.mean()) for d,g in m.groupby("dataset")}
 result={"candidate_id":cid,"complete":True,"dev_macro_delta_ari":float(m.groupby("dataset").delta_ari.mean().mean()),
  "dev_macro_delta_nmi":float(m.groupby("dataset").delta_nmi.mean().mean()),"dev_macro_delta_q":float(np.mean(list(dq.values()))),
  "worst_dataset_delta_q":float(min(dq.values())),"dataset_delta_q":dq,"paired_q_wins":int((m.delta_q>0).sum()),"paired_q_total":len(m),
  "spatial_protection_failed":spatial_failed(m),"runtime_ratio":float(c.runtime_seconds.mean()/r.runtime_seconds.mean()),
  "gpu_peak_ratio":float(c.gpu_peak_allocated_mib.max()/r.gpu_peak_allocated_mib.max()),"runtime_seconds_mean":float(c.runtime_seconds.mean()),
  "dataset_means":{d:{x:float(group[x].mean()) for x in ("ari","nmi","q","spatial_neighbor_agreement","spatial_cluster_moran_mean","spatial_cluster_geary_mean","boundary_disagreement")} for d,group in c.groupby("dataset")}}
 # Complete deltas against fixed B01 as additionally required.
 b=frame[(frame.candidate_id=="B01_C04_SHRINK25")&frame.seed.isin(seeds)]
 if len(b)==expected:
  mb=c.merge(b,on=["dataset","seed"],suffixes=("","_b01"));mb["q_b01"]=(mb.ari_b01+mb.nmi_b01)/2
  result["relative_b01"]={"macro_delta_ari":float((mb.ari-mb.ari_b01).groupby(mb.dataset).mean().mean()),
   "macro_delta_nmi":float((mb.nmi-mb.nmi_b01).groupby(mb.dataset).mean().mean()),"macro_delta_q":float((mb.q-mb.q_b01).groupby(mb.dataset).mean().mean())}
 return result
def rank(rows):return sorted(rows,key=lambda x:(-x.get("worst_dataset_delta_q",-999),-x.get("dev_macro_delta_q",-999),-x.get("paired_q_wins",-1),x.get("spatial_protection_failed",True),x.get("runtime_seconds_mean",999999),x["candidate_id"]))

def evaluate_manifest(config,lock_path,stage):
 lock=json.loads(lock_path.read_text());
 if not lock.get("locked_before_semantic_label_access") or lock.get("failure_count")!=0:raise RuntimeError("Training lock invalid")
 from SpaLORA.night1_evaluation import evaluate,load_evaluation_labels
 from scripts.night3a_evaluate import coordinates_for_ids
 context={}
 for d in ("a1","placenta"):
  first=next(r for r in lock["runs"] if r["dataset"]==d);directory=Path(first["record_path"]).parent
  ids=pd.Index(pd.read_csv(directory/"observation_ids.csv")["observation_id"].astype(str));pos,labels=load_evaluation_labels(d,config["datasets"][d],ids);coords=coordinates_for_ids(config["datasets"][d],ids);graph=symmetric_knn_adjacency(coords,config["datasets"][d]["spatial_neighbors"]);context[d]=(ids,pos,labels,coords,graph)
 rows=[]
 for record in lock["runs"]:
  d,cid,seed=record["dataset"],record["candidate_id"],int(record["seed"]);directory=Path(record["record_path"]).parent;ids,pos,labels,coords,graph=context[d]
  run_ids=pd.Index(pd.read_csv(directory/"observation_ids.csv")["observation_id"].astype(str));
  if not run_ids.equals(ids):raise RuntimeError("Observation ID mismatch")
  clusters=pd.read_csv(directory/"clusters.csv")["cluster"].to_numpy();embedding=np.load(directory/"embedding.npz")["SpaLORA"]
  metrics=evaluate(labels,clusters[pos],clusters,embedding,coords,config["datasets"][d]["spatial_neighbors"]);geary,_=mean_one_vs_rest_geary(clusters,graph);edges=np.transpose(graph.nonzero());boundary=float(np.mean(clusters[edges[:,0]]!=clusters[edges[:,1]])) if len(edges) else 0.0
  manifest=json.loads((directory/"run_manifest.json").read_text())
  rows.append({"dataset":d,"candidate_id":cid,"seed":seed,"stage_evaluated":stage,"ari":float(metrics["ari"]),"nmi":float(metrics["nmi"]),"q":float((metrics["ari"]+metrics["nmi"])/2),
   "spatial_neighbor_agreement":float(metrics["spatial_neighbor_agreement"]),"spatial_cluster_moran_mean":float(metrics["spatial_cluster_moran_mean"]),"spatial_cluster_geary_mean":float(geary),"boundary_disagreement":boundary,
   "runtime_seconds":float(manifest["timings"]["training_seconds"]),"gpu_peak_allocated_mib":float(manifest["resources"]["gpu_peak_allocated_mib"]),"run_manifest_sha256":sha256_file(directory/"run_manifest.json")})
 return rows

def build_combined(config,output,include_conditional):
 old=pd.read_csv(Path(config["paths"]["night5b_handoff"])/"per_run_summary.csv")
 old=old[~old.candidate_id.isin(CORRECTIVE_IDS)]
 corrected=pd.read_csv(output/"corrected_per_run_summary.csv")
 parts=[old,corrected]
 if include_conditional and (output/"conditional_per_run_summary.csv").exists():parts.append(pd.read_csv(output/"conditional_per_run_summary.csv"))
 frame=pd.concat(parts,ignore_index=True);frame=frame.sort_values(["dataset","candidate_id","seed"])
 return frame

def resource_correction(config,frame):
 rows=[];fixed=frame.copy()
 for i,r in fixed.iterrows():
  cid,d,seed=r.candidate_id,r.dataset,int(r.seed)
  if cid in DIFFUSION_MAP:
   source=DIFFUSION_MAP[cid];src=Path(config["paths"]["night5a_raw_runs"])/d/source/("seed_%d"%seed)/"run_manifest.json";sm=json.loads(src.read_text())
   old=Path(config["paths"]["night5b_raw_runs"])/d/cid/("seed_%d"%seed)/"run_manifest.json";dm=json.loads(old.read_text())
   inc_s=float(dm["timings"]["training_seconds"]);inc_g=float(dm["resources"]["gpu_peak_allocated_mib"]);src_s=float(sm["timings"]["training_seconds"]);src_g=float(sm["resources"]["gpu_peak_allocated_mib"])
   eff_s=src_s+inc_s;eff_g=max(src_g,inc_g);fixed.at[i,"runtime_seconds"]=eff_s;fixed.at[i,"gpu_peak_allocated_mib"]=eff_g
   rows.append({"dataset":d,"candidate_id":cid,"seed":seed,"source_candidate_id":source,"source_manifest_sha256":sha256_file(src),"historical_diffusion_manifest_sha256":sha256_file(old),
    "historical_incremental_diffusion_seconds":inc_s,"historical_incremental_diffusion_gpu_mib":inc_g,"source_training_seconds":src_s,"source_peak_gpu_mib":src_g,
    "effective_end_to_end_seconds":eff_s,"effective_peak_gpu_mib":eff_g,"historical_zero_preserved":inc_s==0 and inc_g==0,
    "interpretation":"historical zero is incremental post-hoc cost only; source cost is added for frontier gating"})
 write_csv(REPO/config["paths"]["output_root"]/"diffusion_effective_resource_audit.csv",rows)
 return fixed

def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",choices=("S1_REPLAY","FINAL"),required=True);a=p.parse_args();config=json.loads(CONFIG_PATH.read_text());output=REPO/config["paths"]["output_root"]
 if a.stage=="S1_REPLAY":
  rows=evaluate_manifest(config,output/"corrected_training_manifest.json","S1_CORRECTED");write_csv(output/"corrected_per_run_summary.csv",rows);frame=build_combined(config,output,False)
  all_ids=sorted(set(frame.candidate_id));excluded={"B00_C00_FULL_IGE","B01_C04_SHRINK25","B02_C09_RNA_ANCHOR10","B03_C10_MNN_TRIPLET01","B17_C09_DIFFUSE10","B18_C09_DIFFUSE25","B19_C10_DIFFUSE10","B20_C10_DIFFUSE25"}
  three=[x for x in all_ids if x not in excluded];summaries=[summarize(frame,c,[0,1,2]) for c in three];balanced=rank([x for x in summaries if x.get("complete") and not x["spatial_protection_failed"]])[:3];accuracy=rank([x for x in summaries if x.get("complete")])[:2];selected=[]
  for x in balanced+accuracy:
   if x["candidate_id"] not in selected:selected.append(x["candidate_id"])
  latent=[x for x in summaries if x["candidate_id"] in ("B14_LATENT_RELIABILITY25","B15_LATENT_RELIABILITY50") and x.get("dev_macro_delta_q",-1)>0]
  if latent:
   best=rank(latent)[0]["candidate_id"]
   if best not in selected:selected.append(best)
  selected=selected[:6];old=json.loads((Path(config["paths"]["night5b_handoff"])/"s1_decision.json").read_text())["topup_candidates"]
  new_lap=[c for c in selected if c in CORRECTIVE_IDS and len(frame[(frame.candidate_id==c)&frame.seed.isin([3,4])])<4]
  payload={"schema_version":1,"status":"S1_REPLAY_LOCKED","old_topup_candidates":old,"recomputed_topup_candidates":selected,"added_candidates":[x for x in selected if x not in old],"removed_candidates":[x for x in old if x not in selected],
   "newly_selected_incomplete_laplacian_candidates":new_lap,"balanced_frontier_top3":[x["candidate_id"] for x in balanced],"accuracy_frontier_top2":[x["candidate_id"] for x in accuracy],"candidate_summaries":summaries,
   "selection_rule":"exact Night-5B balanced top3 + accuracy top2 + positive latent reserve, stable rank, cap6","parameter_tuning":False,"seed_search":False,"withheld_results_opened":False,"labels_read_only_after_training_manifest_lock":True}
  atomic_json(output/"recomputed_s1_topup_candidates.json",payload);atomic_json(output/"s1_replay_semantic_label_access.json",{"occurred":True,"after_training_manifest_lock":True,"development_datasets_only":["a1","placenta"],"withheld_results_opened":False});print("S1_REPLAY_LOCKED",selected,"new_lap",new_lap)
 else:
  decision=json.loads((output/"recomputed_s1_topup_candidates.json").read_text());new_lap=decision["newly_selected_incomplete_laplacian_candidates"]
  if new_lap:
   rows=evaluate_manifest(config,output/"conditional_training_manifest.json","S2_CONDITIONAL");write_csv(output/"conditional_per_run_summary.csv",rows)
  frame=build_combined(config,output,True);write_csv(output/"per_run_summary_corrected.csv",frame.to_dict("records"));effective=resource_correction(config,frame)
  topups=decision["recomputed_topup_candidates"];canonical=["B01_C04_SHRINK25","B02_C09_RNA_ANCHOR10","B03_C10_MNN_TRIPLET01","B17_C09_DIFFUSE10","B18_C09_DIFFUSE25","B19_C10_DIFFUSE10","B20_C10_DIFFUSE25"]+[x for x in topups if x not in ("B01_C04_SHRINK25","B02_C09_RNA_ANCHOR10","B03_C10_MNN_TRIPLET01")]
  canonical=[x for x in canonical if len(effective[effective.candidate_id==x])==10];five=sorted([x for x,n in effective.groupby("candidate_id").size().items() if n>=10 and x!="B00_C00_FULL_IGE"]);exploratory=[x for x in five if x not in canonical]
  summaries=[]
  for cid in five:
   x=summarize(effective,cid,list(range(5)));x["eligibility_tier"]="canonical_eligible" if cid in canonical else "valid_exploratory_noncanonical";resource_ok=x.get("complete") and x["runtime_ratio"]<=2 and x["gpu_peak_ratio"]<=1.5;x["effective_resource_gate_pass"]=bool(resource_ok)
   x["balanced_frontier"]=bool(cid in canonical and resource_ok and x["dev_macro_delta_ari"]>=.01 and x["dev_macro_delta_nmi"]>=.01 and x["dev_macro_delta_q"]>=.02 and x["worst_dataset_delta_q"]>=0 and x["paired_q_wins"]>=7 and not x["spatial_protection_failed"])
   x["accuracy_frontier"]=bool(cid in canonical and resource_ok and x["dev_macro_delta_ari"]>=.04 and x["dev_macro_delta_nmi"]>=.04 and x["dev_macro_delta_q"]>=.05 and x["worst_dataset_delta_q"]>=-.005 and x["paired_q_wins"]>=7);summaries.append(x)
  bal=rank([x for x in summaries if x["balanced_frontier"]]);acc=rank([x for x in summaries if x["accuracy_frontier"]]);selected=["B01_C04_SHRINK25"]
  b01=next(x for x in summaries if x["candidate_id"]=="B01_C04_SHRINK25");ordered=rank([b01]+[x for x in bal if x["candidate_id"]!="B01_C04_SHRINK25"])
  better=[x for x in ordered if x["candidate_id"]!="B01_C04_SHRINK25" and ordered.index(x)<ordered.index(b01)]
  if better:selected.append(better[0]["candidate_id"])
  if acc and acc[0]["candidate_id"] not in selected:selected.append(acc[0]["candidate_id"])
  selected=selected[:3];status="NIGHT5C_CANDIDATES_LOCKED_FOR_FUTURE_P22" if len(selected)>1 else "NO_ADDITIONAL_CANDIDATE_C04_REMAINS_PRIMARY"
  write_csv(output/"five_seed_summary_corrected.csv",summaries);atomic_json(output/"canonical_exploratory_eligibility.json",{"canonical_eligible":canonical,"valid_exploratory_noncanonical":exploratory,"rules_locked":True})
  atomic_json(output/"final_frontier_decision.json",{"schema_version":1,"status":status,"candidate_summaries":summaries,"balanced_frontier":[x["candidate_id"] for x in bal],"accuracy_frontier":[x["candidate_id"] for x in acc],"selected_for_future_locked_p22":selected,"strictly_better_than_b01_balanced":[x["candidate_id"] for x in better],"withheld_results_opened":False,"parameter_tuning":False,"seed_search":False})
  contracts=json.loads((Path(config["paths"]["night5b_handoff"])/"candidate_contracts.json").read_text())["candidates"]
  atomic_json(output/"selected_for_future_locked_p22.json",{"schema_version":1,"status":status,"selected_candidates":[{"candidate_id":x,"config_sha256":contracts[x]["config_sha256"],"formula":contracts[x]} for x in selected],"p22_run":False,"d1_run":False,"gse198353_run":False,"night4b_run":False});print(status,selected)
if __name__=="__main__":main()
