#!/usr/bin/env python3
"""Post-lock Night-5B evaluator, S1 top-up lock and final dual-frontier decision."""
from __future__ import annotations
import argparse,csv,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary,symmetric_knn_adjacency
from SpaLORA.night5a_rnd import sha256_file
from SpaLORA.night5b_rnd import load_registry,registry_contracts
CONFIG_PATH=REPO/"configs/night5b_secondlook_rescue.json"
METRICS=("ari","nmi","spatial_neighbor_agreement","spatial_cluster_moran_mean","spatial_cluster_geary_mean","boundary_disagreement","runtime_seconds","gpu_peak_allocated_mib")
def atomic_json(path,payload):
 tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); os.replace(str(tmp),str(path))
def write_csv(path,rows):
 fields=sorted({k for r in rows for k in r})
 with path.open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
def spatial_failed(m):
 n=float(m.delta_spatial_neighbor_agreement.mean()); mo=float(m.delta_spatial_cluster_moran_mean.mean()); g=float(m.delta_spatial_cluster_geary_mean.mean())
 return bool((n<-.03 and mo<-.03) or (g>.03 and (n<-.03 or mo<-.03)))
def summarize(frame,cid,seeds,reference="B00_C00_FULL_IGE"):
 c=frame[(frame.candidate_id==cid)&frame.seed.isin(seeds)]; r=frame[(frame.candidate_id==reference)&frame.seed.isin(seeds)]
 expected=2*len(seeds)
 if len(c)!=expected or len(r)!=expected: return {"candidate_id":cid,"complete":False,"observed_rows":len(c),"expected_rows":expected}
 m=c.merge(r,on=["dataset","seed"],suffixes=("","_reference"),validate="one_to_one")
 for x in METRICS: m["delta_"+x]=m[x]-m[x+"_reference"]
 m["q"]=(m.ari+m.nmi)/2; m["q_reference"]=(m.ari_reference+m.nmi_reference)/2; m["delta_q"]=m.q-m.q_reference
 dq={d:float(g.delta_q.mean()) for d,g in m.groupby("dataset")}
 return {"candidate_id":cid,"complete":True,"dev_macro_delta_ari":float(m.groupby("dataset").delta_ari.mean().mean()),
  "dev_macro_delta_nmi":float(m.groupby("dataset").delta_nmi.mean().mean()),"dev_macro_delta_q":float(np.mean(list(dq.values()))),
  "worst_dataset_delta_q":float(min(dq.values())),"dataset_delta_q":dq,"paired_q_wins":int((m.delta_q>0).sum()),"paired_q_total":len(m),
  "spatial_protection_failed":spatial_failed(m),"runtime_ratio":float(c.runtime_seconds.mean()/r.runtime_seconds.mean()),
  "gpu_peak_ratio":float(c.gpu_peak_allocated_mib.max()/r.gpu_peak_allocated_mib.max()),"runtime_seconds_mean":float(c.runtime_seconds.mean()),
  "dataset_means":{d:{x:float(g[x].mean()) for x in ("ari","nmi","q","spatial_neighbor_agreement","spatial_cluster_moran_mean","spatial_cluster_geary_mean","boundary_disagreement")} for d,g in c.groupby("dataset")}}
def rank(rows): return sorted(rows,key=lambda x:(-x.get("worst_dataset_delta_q",-999),-x.get("dev_macro_delta_q",-999),-x.get("paired_q_wins",-1),x.get("spatial_protection_failed",True),x.get("runtime_seconds_mean",999999),x["candidate_id"]))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("S1","S2"),required=True); a=p.parse_args(); config=json.loads(CONFIG_PATH.read_text()); output=REPO/config["paths"]["output_root"]
 lock=output/(a.stage.lower()+"_training_manifest.json"); complete=output/(a.stage.lower()+"_training_complete.json"); payload=json.loads(lock.read_text()); comp=json.loads(complete.read_text())
 if not payload["locked_before_semantic_label_access"] or comp["training_manifest_sha256"]!=sha256_file(lock): raise RuntimeError("Training lock invalid")
 from SpaLORA.night1_evaluation import evaluate,load_evaluation_labels
 from scripts.night3a_evaluate import coordinates_for_ids
 context={}
 for d in ("a1","placenta"):
  first=next(r for r in payload["runs"] if r["dataset"]==d and r["status"]=="success"); directory=Path(first["record_path"]).parent
  ids=pd.Index(pd.read_csv(directory/"observation_ids.csv")["observation_id"].astype(str)); pos,labels=load_evaluation_labels(d,config["datasets"][d],ids); coords=coordinates_for_ids(config["datasets"][d],ids); graph=symmetric_knn_adjacency(coords,config["datasets"][d]["spatial_neighbors"]); context[d]=(ids,pos,labels,coords,graph)
 metric_rows=[]
 for record in payload["runs"]:
  if record["status"]!="success": continue
  d,cid,seed=record["dataset"],record["candidate_id"],int(record["seed"]); directory=Path(record["record_path"]).parent; ids,pos,labels,coords,graph=context[d]
  run_ids=pd.Index(pd.read_csv(directory/"observation_ids.csv")["observation_id"].astype(str));
  if not run_ids.equals(ids): raise RuntimeError("Observation ID mismatch")
  clusters=pd.read_csv(directory/"clusters.csv")["cluster"].to_numpy(); embedding=np.load(directory/"embedding.npz")["SpaLORA"]
  metrics=evaluate(labels,clusters[pos],clusters,embedding,coords,config["datasets"][d]["spatial_neighbors"]); geary,_=mean_one_vs_rest_geary(clusters,graph)
  edges=np.transpose(graph.nonzero()); boundary=float(np.mean(clusters[edges[:,0]]!=clusters[edges[:,1]])) if len(edges) else 0.0
  manifest=json.loads((directory/"run_manifest.json").read_text()); metric_rows.append({"dataset":d,"candidate_id":cid,"seed":seed,"stage_evaluated":a.stage,
   "ari":float(metrics["ari"]),"nmi":float(metrics["nmi"]),"q":float((metrics["ari"]+metrics["nmi"])/2),"spatial_neighbor_agreement":float(metrics["spatial_neighbor_agreement"]),
   "spatial_cluster_moran_mean":float(metrics["spatial_cluster_moran_mean"]),"spatial_cluster_geary_mean":float(geary),"boundary_disagreement":boundary,
   "runtime_seconds":float(manifest["timings"]["training_seconds"]),"gpu_peak_allocated_mib":float(manifest["resources"]["gpu_peak_allocated_mib"]),"run_manifest_sha256":sha256_file(directory/"run_manifest.json")})
 cumulative=output/"per_run_summary.csv"; old=pd.read_csv(cumulative).to_dict("records") if cumulative.exists() else []; keyed={(r["dataset"],r["candidate_id"],int(r["seed"])):r for r in old}
 for r in metric_rows:keyed[(r["dataset"],r["candidate_id"],int(r["seed"]))]=r
 rows=list(keyed.values()); rows.sort(key=lambda r:(r["dataset"],r["candidate_id"],int(r["seed"]))); write_csv(cumulative,rows); frame=pd.DataFrame(rows)
 registry=load_registry(REPO/config["candidate_registry"]); contracts=registry_contracts(registry)
 if a.stage=="S1":
  three=[x["id"] for x in registry["candidates"] if x["id"] not in ("B00_C00_FULL_IGE","B01_C04_SHRINK25","B02_C09_RNA_ANCHOR10","B03_C10_MNN_TRIPLET01") and not x["id"].startswith(("B17_","B18_","B19_","B20_"))]
  summaries=[summarize(frame,c,[0,1,2]) for c in three]; balanced=rank([x for x in summaries if x.get("complete") and not x["spatial_protection_failed"]])[:3]; accuracy=rank([x for x in summaries if x.get("complete")])[:2]
  selected=[]
  for x in balanced+accuracy:
   if x["candidate_id"] not in selected:selected.append(x["candidate_id"])
  latent=[x for x in summaries if x["candidate_id"] in ("B14_LATENT_RELIABILITY25","B15_LATENT_RELIABILITY50") and x.get("dev_macro_delta_q",-1)>0]
  if latent:
   best=rank(latent)[0]["candidate_id"]
   if best not in selected:selected.append(best)
  selected=selected[:6]
  atomic_json(output/"s1_decision.json",{"schema_version":1,"stage":"S1","labels_read_only_after_training_manifest_lock":True,"candidate_summaries":summaries,"balanced_frontier_top3":[x["candidate_id"] for x in balanced],"accuracy_frontier_top2":[x["candidate_id"] for x in accuracy],"topup_candidates":selected,"withheld_results_opened":False,"parameter_tuning":False,"seed_search":False})
  atomic_json(output/"s1_semantic_label_access.json",{"occurred":True,"after_training_manifest_lock":True,"development_datasets_only":["a1","placenta"],"withheld_results_opened":False})
  print("S1_EVALUATED topup="+",".join(selected))
 else:
  five=sorted(set(frame.groupby("candidate_id").size()[lambda x:x>=10].index)); summaries=[summarize(frame,c,list(range(5))) for c in five if c!="B00_C00_FULL_IGE"]
  for x in summaries:
   resource=x.get("complete") and x["runtime_ratio"]<=2 and x["gpu_peak_ratio"]<=1.5
   x["balanced_frontier"]=bool(resource and x["dev_macro_delta_ari"]>=.01 and x["dev_macro_delta_nmi"]>=.01 and x["dev_macro_delta_q"]>=.02 and x["worst_dataset_delta_q"]>=0 and x["paired_q_wins"]>=7 and not x["spatial_protection_failed"])
   x["accuracy_frontier"]=bool(resource and x["dev_macro_delta_ari"]>=.04 and x["dev_macro_delta_nmi"]>=.04 and x["dev_macro_delta_q"]>=.05 and x["worst_dataset_delta_q"]>=-.005 and x["paired_q_wins"]>=7)
  bal=rank([x for x in summaries if x["balanced_frontier"]]); acc=rank([x for x in summaries if x["accuracy_frontier"]]); selected=["B01_C04_SHRINK25"]
  better=[x for x in bal if x["candidate_id"]!="B01_C04_SHRINK25"]
  if better:selected.append(better[0]["candidate_id"])
  if acc and acc[0]["candidate_id"] not in selected:selected.append(acc[0]["candidate_id"])
  selected=selected[:3]; status="NIGHT5B_CANDIDATES_LOCKED_FOR_FUTURE_P22" if len(selected)>1 else "NO_ADDITIONAL_CANDIDATE_C04_REMAINS_PRIMARY"
  decision={"schema_version":1,"status":status,"labels_read_only_after_training_manifest_lock":True,"candidate_summaries":summaries,"balanced_frontier":[x["candidate_id"] for x in bal],"accuracy_frontier":[x["candidate_id"] for x in acc],"selected_for_future_locked_p22":selected,"withheld_results_opened":False,"parameter_tuning":False,"seed_search":False}
  atomic_json(output/"s2_decision.json",decision); atomic_json(output/"selected_for_future_locked_p22.json",{"schema_version":1,"status":status,"selected_candidates":[{"candidate_id":x,"config_sha256":contracts[x]["config_sha256"],"formula":contracts[x]} for x in selected],"p22_run":False,"d1_run":False,"gse198353_run":False,"night4b_run":False})
  atomic_json(output/"s2_semantic_label_access.json",{"occurred":True,"after_training_manifest_lock":True,"development_datasets_only":["a1","placenta"],"withheld_results_opened":False}); print(status,selected)
if __name__=="__main__":main()
