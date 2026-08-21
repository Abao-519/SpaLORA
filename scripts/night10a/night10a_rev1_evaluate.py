from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pathlib
import statistics
import time

import numpy as np
import pandas as pd

from SpaLORA.night10a_qcrd import row_normalize
from scripts.night10a.metric_expansion_reference import (
    embedding_cluster_metrics, paired_cross_modal_metrics,
    supervised_clustering_metrics,
)
from scripts.night7b_evaluate import load_coordinates, load_labels, metric, read_clusters


REPO=pathlib.Path("/root/autodl-fs/SpaLORA-night10a-rev1")
RAW=pathlib.Path("/root/autodl-fs/night10a_rev1_qcrd_20260821")
SOURCE=pathlib.Path("/root/autodl-fs/night7b_score_rnd_20260818")
OUT=REPO/"outputs/night10a_rev1_handoff"; OUT.mkdir(parents=True,exist_ok=True)
CANDIDATES=["Q01_GLOBAL_QUALITY_BLEND","Q02_SPOT_QUALITY_BLEND","Q03_MASKED_RESIDUAL","Q04_SPOT_GATED_MASKED_RESIDUAL","Q05_BOUNDARY_GATED_RESIDUAL","Q06_COORDINATE_PRIOR_RESIDUAL","Q07_CONFIDENCE_MNN_RESIDUAL"]
DATA={"a1":{"base":0,"K":10,"r1":range(3),"r2":range(5)},"tonsil":{"base":5,"K":4,"r1":range(3),"r2":range(5)},"d1":{"base":10,"K":10,"r1":range(3),"r2":range(10)},"p22":{"base":20,"K":9,"r1":range(3),"r2":range(10)}}
METRICS=("ari","nmi","q","mi","ami","fmi","homogeneity","completeness","v_measure","neighbor_agreement","moran_i","geary_c","boundary_disagreement","silhouette","davies_bouldin","calinski_harabasz","foscttm_mean","recall_at_1","recall_at_5","recall_at_10","paired_median_rank")


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(tmp,path)


def unit(d,s): return f"u{DATA[d]['base']+s:03d}"


def source_views(u):
    x=np.load(SOURCE/f"source/{u}/g04_views.npz"); return tuple(np.asarray(x[k]) for k in ("emb_latent_omics1","emb_latent_omics2","SpaLORA_fused"))


def r02_root(u):
    for stage in ("R1","R2"):
        path=SOURCE/f"adapter_stage/{stage}/formal/R02/{u}/attempt_001"
        if (path/"worker/embedding.npy").exists() and (path/"transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv").exists(): return path
    raise FileNotFoundError(f"no complete authoritative R02 endpoint for {u}")


def read_csv_clusters(path,expected): return read_clusters(path,np.asarray(expected))


def candidate_cell(stage,d,s,c): return RAW/f"{stage}/{d}/seed_{s}/{c}"


def reference_partition(d,s,expected_ids):
    u=unit(d,s)
    if d!="p22": return np.load(SOURCE/f"source/{u}/c00_partition.npy")
    return read_csv_clusters(r02_root(u)/"transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv",expected_ids)


def reference_embedding(d,s):
    u=unit(d,s)
    if d!="p22": return source_views(u)[2]
    return np.load(r02_root(u)/"worker/embedding.npy")


def one_metrics(true,pred,coords,embedding,z1,z2):
    result=metric(true,pred,coords); result.update(supervised_clustering_metrics(true,pred))
    result["q"]=(result["ari"]+result["nmi"])/2
    result.update(embedding_cluster_metrics(embedding,pred,silhouette_sample_size=5000,random_seed=0))
    cross=paired_cross_modal_metrics(z1,z2,block_size=512)
    result.update({
        "foscttm_mean":cross["foscttm_mean"], "recall_at_1":cross["recall_at_1_mean"],
        "recall_at_5":cross["recall_at_5_mean"], "recall_at_10":cross["recall_at_10_mean"],
        "paired_median_rank":.5*(cross["median_paired_rank_mod1_to_mod2"]+cross["median_paired_rank_mod2_to_mod1"]),
    })
    return result


def independent_ari_nmi(true, pred):
    _, ti=np.unique(np.asarray(true),return_inverse=True); _, pi=np.unique(np.asarray(pred),return_inverse=True)
    table=np.zeros((ti.max()+1,pi.max()+1),dtype=np.int64); np.add.at(table,(ti,pi),1)
    n=int(table.sum()); choose=lambda x:x*(x-1)/2
    nij=float(np.sum(choose(table))); ai=float(np.sum(choose(table.sum(1)))); bj=float(np.sum(choose(table.sum(0)))); total=choose(n)
    expected=ai*bj/total; ari=(nij-expected)/(.5*(ai+bj)-expected)
    pxy=table/n; px=pxy.sum(1); py=pxy.sum(0); nz=np.nonzero(table)
    mi=float(np.sum(pxy[nz]*np.log(pxy[nz]/(px[nz[0]]*py[nz[1]]))))
    hx=float(-np.sum(px[px>0]*np.log(px[px>0]))); hy=float(-np.sum(py[py>0]*np.log(py[py>0]))); nmi=mi/((hx+hy)/2)
    return float(ari),float(nmi)


def verify_lock(stage,candidates,seeds_by_dataset):
    expected=sum(len(list(seeds_by_dataset[d]))*len(candidates) for d in DATA); locked=0; rows=[]
    for d in DATA:
        for s in seeds_by_dataset[d]:
            for c in candidates:
                cell=candidate_cell(stage,d,s,c); tm=cell/"training_manifest.json"; fm=cell/"transform_manifest.json"
                ok=tm.exists() and fm.exists() and json.loads(tm.read_text()).get("status")=="CHECKPOINT_ROUNDTRIP_PASS" and json.loads(fm.read_text()).get("status")=="PASS"
                if ok: locked+=1
                rows.append({"dataset":d,"seed":s,"candidate":c,"locked":ok,"training_manifest_sha256":sha(tm) if tm.exists() else None,"transform_manifest_sha256":sha(fm) if fm.exists() else None})
    audit={"stage":stage,"expected_trainable_outputs":expected,"locked_trainable_outputs":locked,"all_locked":locked==expected,"label_reads_before_lock":0,"rows":rows}
    atomic_json(OUT/f"{stage}_prelabel_lock_audit.json",audit)
    if locked!=expected: raise RuntimeError(f"{stage} pre-label lock incomplete {locked}/{expected}")
    return audit


def evaluate(stage,candidates,seeds_by_dataset,include_r1=False):
    verify_lock(stage,candidates,{d:[s for s in seeds_by_dataset[d] if not(include_r1 and s in DATA[d]["r1"])] for d in DATA}) if include_r1 else verify_lock(stage,candidates,seeds_by_dataset)
    # One evaluator window: each locked snapshot is opened once in this process.
    labels={}; label_audit={}
    for d in DATA:
        label_ids,true,digest,keys=load_labels(d); coords=load_coordinates(d,label_ids)
        labels[d]=(label_ids.tolist(),true,coords); label_audit[d]={"read_count_this_stage":1,"snapshot_sha256":digest,"keys":keys,"authorized_role":f"night10a_rev1_{stage}_evaluator","used_for_training_or_selection_before_lock":False}
    rows=[]
    for d in DATA:
        names,true,coords=labels[d]
        for s in seeds_by_dataset[d]:
            u=unit(d,s); z1,z2,_=source_views(u); refp=reference_partition(d,s,names); refz=reference_embedding(d,s)
            rows.append({"stage":stage,"candidate":"Q00_REFERENCE_ALIAS","dataset":d,"family":"RNA+ATAC" if d=="p22" else "RNA+protein","seed":s,"success":True,**one_metrics(true,refp,coords,refz,z1,z2)})
            for c in candidates:
                cell=candidate_cell("r1" if include_r1 and s in DATA[d]["r1"] else stage,d,s,c)
                try:
                    pred=read_csv_clusters(cell/"clusters.csv",names); x=np.load(cell/"corrected_views.npz")
                    row={"stage":stage,"candidate":c,"dataset":d,"family":"RNA+ATAC" if d=="p22" else "RNA+protein","seed":s,"success":True,**one_metrics(true,pred,coords,x["zc"],x["z1c"],x["z2c"])}
                    tm=json.loads((cell/"training_manifest.json").read_text()); row.update({"training_runtime_seconds":tm["runtime_seconds"],"peak_gpu_mib":tm["peak_gpu_bytes"]/(1024**2)})
                except Exception as exc:
                    row={"stage":stage,"candidate":c,"dataset":d,"family":"RNA+ATAC" if d=="p22" else "RNA+protein","seed":s,"success":False,"error":repr(exc)}
                    row.update({m:np.nan for m in METRICS})
                rows.append(row)
    frame=pd.DataFrame(rows); frame.to_csv(OUT/f"{stage}_per_seed_metrics.csv",index=False)
    atomic_json(OUT/f"{stage}_label_window_audit.json",label_audit)
    independent=[]
    for _,r in frame[frame.success].iterrows():
        names,true,_=labels[r.dataset]; u=unit(r.dataset,int(r.seed))
        pred=reference_partition(r.dataset,int(r.seed),names) if r.candidate=="Q00_REFERENCE_ALIAS" else read_csv_clusters(candidate_cell("r1" if include_r1 and int(r.seed) in DATA[r.dataset]["r1"] else stage,r.dataset,int(r.seed),r.candidate)/"clusters.csv",names)
        ari2,nmi2=independent_ari_nmi(true,pred); err=max(abs(ari2-r.ari),abs(nmi2-r.nmi),abs((ari2+nmi2)/2-r.q))
        independent.append({"candidate":r.candidate,"dataset":r.dataset,"seed":int(r.seed),"ari_recomputed":ari2,"nmi_recomputed":nmi2,"q_recomputed":(ari2+nmi2)/2,"max_abs_error":float(err)})
    if max(x["max_abs_error"] for x in independent)>1e-12: raise AssertionError("independent contingency recomputation mismatch")
    pd.DataFrame(independent).to_csv(OUT/f"{stage}_independent_recompute.csv",index=False)
    return frame


def summary(frame,candidates):
    ref=frame[frame.candidate=="Q00_REFERENCE_ALIAS"].set_index(["dataset","seed"]); rows=[]
    for order,c in enumerate(candidates):
        g=frame[frame.candidate==c]; row={"candidate":c,"registry_order":order,"complete":bool(len(g)==sum(len(DATA[d]["r1"]) if frame.stage.iloc[0]=="r1" else len(DATA[d]["r2"]) for d in DATA) and g.success.all())}
        for d in DATA:
            dg=g[g.dataset==d].sort_values("seed"); row[f"{d}_count"]=int(dg.success.sum())
            for m in METRICS:
                vals=dg[m].to_numpy(float); refs=np.asarray([ref.loc[(d,int(s)),m] for s in dg.seed],float); delta=vals-refs
                row[f"{d}_mean_{m}"]=float(np.mean(vals)); row[f"{d}_mean_delta_{m}"]=float(np.mean(delta)); row[f"{d}_wins_{m}"]=int(np.sum(delta>0))
        protein=np.mean([row[f"{d}_mean_delta_q"] for d in ("a1","tonsil","d1")]); atac=row["p22_mean_delta_q"]
        row["protein_family_mean_delta_q"]=float(protein); row["atac_family_mean_delta_q"]=float(atac); row["macro_delta_q"]=float(np.mean([row[f"{d}_mean_delta_q"] for d in DATA])); row["worst_dataset_delta_q"]=float(min(row[f"{d}_mean_delta_q"] for d in DATA)); row["worst_family_delta_q"]=float(min(protein,atac))
        spatial_ok=all(row[f"{d}_mean_delta_neighbor_agreement"]>=-.03 and row[f"{d}_mean_delta_moran_i"]>=-.03 and row[f"{d}_mean_delta_geary_c"]<=.03 and row[f"{d}_mean_delta_boundary_disagreement"]<=.03 for d in DATA)
        row["spatial_protected_0_03"]=bool(spatial_ok); row["any_family_positive_q"]=bool(max(protein,atac)>0); rows.append(row)
    return pd.DataFrame(rows)


def choose_frontiers(table):
    eligible=table[table.complete]
    if eligible.empty: return [],{"entry":False,"reason":"no complete candidate"}
    acc=eligible.sort_values(["macro_delta_q","registry_order"],ascending=[False,True]).iloc[0].candidate
    bal=eligible.sort_values(["worst_family_delta_q","worst_dataset_delta_q","macro_delta_q","registry_order"],ascending=[False,False,False,True]).iloc[0].candidate
    protected=eligible[eligible.spatial_protected_0_03]
    spa=(protected.sort_values(["macro_delta_q","registry_order"],ascending=[False,True]).iloc[0].candidate if len(protected) else None)
    selected=[]
    for c in (acc,bal,spa):
        if c and c not in selected: selected.append(c)
    entry=any(bool(r.any_family_positive_q and r.spatial_protected_0_03) for _,r in eligible.iterrows())
    return selected,{"entry":entry,"accuracy":acc,"balanced":bal,"spatial":spa,"spatial_material_threshold":.03}


def stage_m():
    scoreboard=REPO/"protocols/night10a/authoritative_scoreboard_20260821.csv"
    if not scoreboard.exists():
        atomic_json(OUT/"stage_m_audit.json",{"status":"MISSING_SOURCE_ARTIFACT","path":str(scoreboard),"training":0})
        pd.DataFrame(columns=["dataset","method","metric","value","status"]).to_csv(OUT/"metric_backfill_long.csv",index=False); return
    source=pd.read_csv(scoreboard); rows=[]
    desired=["ari","nmi","q","ami","fmi","homogeneity","completeness","v_measure","neighbor_agreement","moran_i","geary_c","boundary_disagreement","runtime_seconds","peak_gpu_mib"]
    for _,r in source.iterrows():
        for m in desired:
            value=r.get(m,np.nan); missing=pd.isna(value) or value==""
            rows.append({"dataset":r.dataset,"method":r.method,"metric":m,"value":np.nan if missing else value,"status":"MISSING_SOURCE_ARTIFACT" if missing else "REUSED_AUTHORITY"})
    pd.DataFrame(rows).to_csv(OUT/"metric_backfill_long.csv",index=False); atomic_json(OUT/"stage_m_audit.json",{"status":"COMPLETE_WITH_EXPLICIT_MISSING","rows":len(rows),"training":0,"source_sha256":sha(scoreboard)})


def r1():
    frame=evaluate("r1",CANDIDATES,{d:list(DATA[d]["r1"]) for d in DATA}); stage_m(); table=summary(frame,CANDIDATES); table.to_csv(OUT/"r1_candidate_summary.csv",index=False); selected,decision=choose_frontiers(table)
    decision.update({"stage":"r1","selected":selected,"maximum":3,"q00_excluded":True,"label_reads_per_dataset":1}); atomic_json(OUT/"r1_frontier_decision.json",decision); print(json.dumps(decision))


def exact_sign_flip(values):
    values=np.asarray(values,float); observed=float(values.mean()); stats=[float(np.mean(values*np.asarray(sign))) for sign in itertools.product((-1,1),repeat=len(values))]; return float(sum(x>=observed-1e-15 for x in stats)/len(stats))


def bootstrap(values,seed=20260821):
    values=np.asarray(values,float); rng=np.random.default_rng(seed); draws=np.mean(values[rng.integers(0,len(values),size=(100000,len(values)))],axis=1); return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def r2(candidates):
    frame=evaluate("r2",candidates,{d:list(DATA[d]["r2"]) for d in DATA},include_r1=True); table=summary(frame,candidates); table.to_csv(OUT/"r2_candidate_summary.csv",index=False); selected,decision=choose_frontiers(table); ref=frame[frame.candidate=="Q00_REFERENCE_ALIAS"].set_index(["dataset","seed"]); stats=[]
    for c in candidates:
        for d in DATA:
            g=frame[(frame.candidate==c)&(frame.dataset==d)].sort_values("seed"); delta=np.asarray([r.q-ref.loc[(d,int(r.seed)),"q"] for _,r in g.iterrows()]); stats.append({"candidate":c,"dataset":d,"mean_delta_q":float(delta.mean()),"bootstrap_ci":bootstrap(delta),"exact_sign_flip_p_one_sided":exact_sign_flip(delta)})
    atomic_json(OUT/"r2_preregistered_statistics.json",stats); decision.update({"stage":"r2","selected":selected,"label_reads_per_dataset":1}); atomic_json(OUT/"frontier_registry.json",decision)


def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["r1","r2"]); p.add_argument("--candidates"); a=p.parse_args()
    if a.stage=="r1": r1()
    else: r2(json.loads(a.candidates))


if __name__=="__main__": main()
