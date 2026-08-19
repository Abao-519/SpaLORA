#!/usr/bin/env python3
"""Single-window evaluation and locked Night-7C recovery decisions."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402
from scripts.night7b_evaluate import load_coordinates, load_labels, metric, read_clusters  # noqa: E402

RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
HANDOFF7B = RAW7B / "official_compact/handoff"
OUT = REPO / "outputs/night7c_replay_recovery_handoff"
DATASETS = ("a1", "tonsil", "d1", "p22")
WEIGHTS = {"a1": .25, "tonsil": .15, "d1": .25, "p22": .35}
METRICS = ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement")
TCANDS = ["T00_C00_REFERENCE", "T01_R02_REFERENCE", "T02_GLOBAL_WIDE", "T03_GLOBAL_CONSERVATIVE",
          "T04_HARD_CONFLICT_030", "T05_LOCAL_CONFLICT", "T06_LOCAL_CONFLICT_SQUARED",
          "T07_LOCAL_QUALITY_CONFLICT", "T08_LOCAL_SHARED_SUPPORT", "T09_THREE_SPECIALIST_EXPLORATORY"]
WCANDS = ["W00_FILTER75", "W01_QUALITY_SOFT", "W02_CONFLICT_RANK", "W03_QUALITY_CONFLICT",
          "W04_QUALITY_SHARED", "W05_QUALITY_CONFLICT_SHARED"]


def require(ok: bool, message: str) -> None:
    if not ok: raise RuntimeError(message)


def source_rows() -> list[dict]:
    return list(csv.DictReader((HANDOFF7B / "source_unit_index.csv").open(newline="")))


def maps():
    c00={}; r02={}
    for row in csv.DictReader((HANDOFF7B / "historical_reference_partition_index.csv").open(newline="")):
        if row["reference"] == "C00": c00[row["unit_id"]] = Path(row["clusters_path"])
    for stage in ("R1","R2"):
        locked=json.loads((HANDOFF7B/f"locked_{stage}_manifest.json").read_text())
        for x in locked["transforms"]:
            if x["recipe_id"]=="R02" and x["endpoint"]=="E1_ADAPTER_C06_MEAN" and x["head_id"]=="H01":
                r02[x["unit_id"]]=RAW7B/"adapter_stage"/stage/"formal"/"R02"/x["unit_id"]/"attempt_001/transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv"
    require(len(c00)==len(r02)==30,"reference cluster maps are not 30/30"); return c00,r02


def labels_once():
    labels={}; audit={}
    for d in DATASETS:
        ids,true,digest,keys=load_labels(d); coords=load_coordinates(d,ids)
        labels[d]=(ids,true,coords); audit[d]={"snapshot_sha256":digest,"snapshot_keys":keys,
          "authorized_role":"night7c_recovery_single_evaluator","used_for_training_or_transform":False}
    return labels,audit


def spatial_ok(row: dict) -> tuple[bool,dict]:
    detail={}
    for d in DATASETS:
        detail[d]={"neighbor":row[f"{d}_mean_delta_neighbor_agreement"]>=-.01,
                   "moran":row[f"{d}_mean_delta_moran_i"]>=-.02,
                   "geary":row[f"{d}_mean_delta_geary_c"]<=.02,
                   "boundary":row[f"{d}_mean_delta_boundary_disagreement"]<=.01}
    return all(all(v.values()) for v in detail.values()),detail


def summarize(frame: pd.DataFrame, candidates: list[str], reference: str) -> pd.DataFrame:
    ref=frame[frame.candidate_id==reference].set_index(["dataset","seed"]); rows=[]
    for order,c in enumerate(candidates):
        g=frame[frame.candidate_id==c].copy(); row={"candidate_id":c,"registry_order":order,
          "success_cells":int(g.success.sum()),"failure_cells":int((~g.success).sum()),
          "mean_runtime_seconds":float(g.runtime_seconds.mean())}
        for d in DATASETS:
            dg=g[g.dataset==d].sort_values("seed"); row[f"{d}_success"]=int(dg.success.sum())
            for m in METRICS:
                vals=dg[m].dropna(); complete=len(vals)==len(dg) and len(dg)>0
                row[f"{d}_mean_{m}"]=float(vals.mean()) if complete else np.nan
                if complete:
                    rv=np.asarray([ref.loc[(d,int(seed)),m] for seed in dg.seed],dtype=float)
                    delta=dg[m].to_numpy(dtype=float)-rv
                    row[f"{d}_mean_delta_{m}"]=float(delta.mean()); row[f"{d}_wins_{m}"]=int(np.sum(delta>0))
                else: row[f"{d}_mean_delta_{m}"]=np.nan; row[f"{d}_wins_{m}"]=0
        complete=row["failure_cells"]==0
        row["complete_eligible"]=complete
        row["priority_weighted_delta_q"]=sum(WEIGHTS[d]*row[f"{d}_mean_delta_q"] for d in DATASETS) if complete else np.nan
        row["balanced_macro_delta_q"]=float(np.mean([row[f"{d}_mean_delta_q"] for d in DATASETS])) if complete else np.nan
        row["human_lymph_equal_mean_delta_q"]=float((row["a1_mean_delta_q"]+row["d1_mean_delta_q"])/2) if complete else np.nan
        row["worst_dataset_delta_q"]=min(row[f"{d}_mean_delta_q"] for d in DATASETS) if complete else np.nan
        row["total_q_wins"]=sum(row[f"{d}_wins_q"] for d in DATASETS) if complete else 0
        rows.append(row)
    return pd.DataFrame(rows)


def independent_summary_check(frame: pd.DataFrame, summary: pd.DataFrame) -> dict:
    maxerr=0.; compared=0
    for _,s in summary.iterrows():
        for d in DATASETS:
            g=frame[(frame.candidate_id==s.candidate_id)&(frame.dataset==d)]
            for m in METRICS:
                vals=[float(x) for x in g[m] if np.isfinite(x)]
                if len(vals)==len(g) and vals:
                    value=sum(vals)/len(vals); maxerr=max(maxerr,abs(value-float(s[f"{d}_mean_{m}"]))); compared+=1
    return {"status":"PASS" if maxerr<=1e-12 else "FAIL","compared_aggregates":compared,"maximum_absolute_error":maxerr,"tolerance":1e-12}


def main() -> None:
    tlock=json.loads((OUT/"routing_transform_manifest.json").read_text()); wtrain=json.loads((OUT/"weighted_mnn_training_manifest.json").read_text()); wlock=json.loads((OUT/"weighted_mnn_transform_manifest.json").read_text())
    total_lock=json.loads((OUT/"total_prelabel_lock.json").read_text())
    plan=json.loads((OUT/"stagew_resource_bounded_plan_and_eligibility.json").read_text())
    require(tlock["status"]==wtrain["status"]==wlock["status"]=="LOCKED_PRE_LABEL","pre-label locks missing")
    require(total_lock.get("status")=="LOCKED_PRE_LABEL" and total_lock.get("label_access") is False and total_lock.get("label_window_authorized") is True,"total pre-label lock missing")
    require(len(tlock["transforms"])==240 and len(wtrain["training_cells"])==48 and len(wlock["transforms"])==48,"locked cardinality mismatch")
    eligible_w=list(plan.get("eligible_weighted_mnn_candidates",[]))
    require(eligible_w==["W01_QUALITY_SOFT","W02_CONFLICT_RANK","W03_QUALITY_CONFLICT","W04_QUALITY_SHARED","W05_QUALITY_CONFLICT_SHARED"],"eligible weighted-MNN set mismatch")
    require(wlock.get("eligible_weighted_mnn_candidates")==eligible_w,"weighted manifest eligibility mismatch")
    labels,audit=labels_once(); c00,r02=maps(); units=source_rows(); byid={x["unit_id"]:x for x in units}
    tlookup={(x["candidate_id"],x["unit_id"]):x for x in tlock["transforms"]}; trows=[]
    for candidate in TCANDS:
        for u in units:
            uid=u["unit_id"]; d=u["dataset"]; seed=int(u["seed"])
            if candidate=="T00_C00_REFERENCE": path=c00[uid]; status="success"; runtime=0.
            elif candidate=="T01_R02_REFERENCE": path=r02[uid]; status="success"; runtime=0.
            else:
                x=tlookup[(candidate,uid)]; path=RAW/"stage_t/formal"/candidate/uid/"clusters.csv"; status=x["status"]; runtime=float(x["runtime_seconds"])
            row={"stage":"T","candidate_id":candidate,"unit_id":uid,"dataset":d,"seed":seed,"status":status,"success":status=="success","runtime_seconds":runtime}
            if row["success"]:
                ids,true,coords=labels[d]; row.update(metric(true,read_clusters(path,ids),coords))
            else: row.update({m:np.nan for m in METRICS})
            trows.append(row)
    tf=pd.DataFrame(trows); tf.to_csv(OUT/"routing_per_seed_metrics.csv",index=False)
    ts=summarize(tf,TCANDS,"T00_C00_REFERENCE")
    gate=[]; passing=[]; strong=[]
    for _,s in ts.iterrows():
        row=s.to_dict(); spok,spdetail=spatial_ok(row)
        nonloss=sum(int(x>=0) for x in tf[tf.candidate_id==s.candidate_id].merge(tf[tf.candidate_id=="T00_C00_REFERENCE"],on=["dataset","seed"],suffixes=("","_ref")).eval("q-q_ref")) if s.candidate_id!="T00_C00_REFERENCE" else 30
        safe=(bool(s.complete_eligible) and row["a1_mean_delta_q"]>=-.001 and row["tonsil_mean_delta_q"]>=-.001 and row["d1_mean_delta_q"]>=-.001 and row["p22_mean_delta_q"]>=.04 and row["p22_wins_q"]>=8 and row["priority_weighted_delta_q"]>=.015 and nonloss>=26 and spok)
        strongok=(bool(s.complete_eligible) and all(row[f"{d}_mean_delta_q"]>0 for d in DATASETS) and row["priority_weighted_delta_q"]>=.018 and row["total_q_wins"]>=20 and spok)
        gate.append({"candidate_id":s.candidate_id,"safe_router_gate":safe,"strong_unified_gate":strongok,"nonloss_q_cells":nonloss,"spatial_ok":spok,"spatial":spdetail})
        if safe: passing.append(s.candidate_id)
        if strongok: strong.append(s.candidate_id)
    eligible=ts[ts.candidate_id.isin(set(passing)|set(strong))].copy()
    eligible["strong_priority"]=eligible.candidate_id.isin(strong).astype(int)
    eligible=eligible.sort_values(["strong_priority","priority_weighted_delta_q","worst_dataset_delta_q","total_q_wins","mean_runtime_seconds","registry_order"],ascending=[False,False,False,False,True,True])
    selected_router=eligible.candidate_id.iloc[0] if len(eligible) else None
    ts.to_csv(OUT/"routing_candidate_summary.csv",index=False)

    trlookup={(x["candidate_id"],x["unit_id"]):x for x in wtrain["training_cells"]}; wx={(x["candidate_id"],x["unit_id"]):x for x in wlock["transforms"]}; wrows=[]
    # Include the same-seed R02 reference for all eight pilot units.
    # Resource-censored W00 is not evaluated and never enters a scientific mean.
    for candidate in ["R02_REFERENCE",*eligible_w]:
        for u in [x for x in units if int(x["seed"])<2]:
            uid=u["unit_id"]; d=u["dataset"]; seed=int(u["seed"])
            if candidate=="R02_REFERENCE": path=r02[uid]; status="success"; runtime=0.
            else:
                x=wx[(candidate,uid)]; path=RAW/"stage_w/formal"/candidate/uid/"attempt_001/transform/clusters.csv"; status=x["status"]; runtime=float(trlookup[(candidate,uid)].get("training_manifest",{}).get("runtime_seconds",np.nan))
            row={"stage":"W","candidate_id":candidate,"unit_id":uid,"dataset":d,"seed":seed,"status":status,"success":status=="success","runtime_seconds":runtime}
            if row["success"]:
                ids,true,coords=labels[d]; row.update(metric(true,read_clusters(path,ids),coords))
            else: row.update({m:np.nan for m in METRICS})
            wrows.append(row)
    wf=pd.DataFrame(wrows); wf.to_csv(OUT/"weighted_mnn_per_seed_metrics.csv",index=False)
    ws=summarize(wf,["R02_REFERENCE",*eligible_w],"R02_REFERENCE"); promotions=[]; wgate=[]
    for _,s in ws[ws.candidate_id!="R02_REFERENCE"].iterrows():
        row=s.to_dict(); spok,spdetail=spatial_ok(row)
        material=row["p22_mean_delta_q"]>=.005 or row["p22_mean_delta_ari"]>=.01
        ok=bool(s.complete_eligible) and material and spok
        wgate.append({"candidate_id":s.candidate_id,"pass":ok,"material_or_gate":material,"spatial_ok":spok,"spatial":spdetail})
        if ok: promotions.append(s.candidate_id)
    ranked=ws[ws.candidate_id.isin(promotions)].sort_values(["p22_mean_delta_q","p22_mean_delta_ari","human_lymph_equal_mean_delta_q","mean_runtime_seconds","registry_order"],ascending=[False,False,False,True,True])
    promoted=ranked.candidate_id.head(2).tolist(); ws.to_csv(OUT/"weighted_mnn_candidate_summary.csv",index=False)

    independent={"routing":independent_summary_check(tf,ts),"weighted":independent_summary_check(wf,ws)}
    require(independent["routing"]["status"]==independent["weighted"]["status"]=="PASS","independent summary mismatch")
    if selected_router and promoted: terminal="NIGHT7C_RECOVERY_SAFE_ROUTER_AND_WEIGHTED_MNN_CANDIDATES_READY"
    elif selected_router in strong: terminal="NIGHT7C_RECOVERY_STRONG_UNIFIED_CONFLICT_ROUTER_LOCKED"
    elif selected_router: terminal="NIGHT7C_RECOVERY_SAFE_P22_CONFLICT_ROUTER_LOCKED"
    elif promoted: terminal="NIGHT7C_RECOVERY_WEIGHTED_MNN_CANDIDATES_READY_FOR_FULL_SEEDS"
    else: terminal="NIGHT7C_RECOVERY_NO_SAFE_ROUTER_OR_WEIGHTED_MNN_CANDIDATE"
    result={"schema_version":1,"terminal_status":terminal,"selected_router":selected_router,"strong_router_candidates":strong,"safe_router_candidates":passing,"promoted_weighted_mnn_candidates":promoted,
            "evaluated_weighted_mnn_candidates":eligible_w,
            "resource_ineligible_weighted_mnn_candidates":[{"candidate_id":"W00_FILTER75","reason":"RESOURCE_CENSORED_USER_STOP_AFTER_EXTREME_LONGTAIL","scientific_metrics_computed":False}],
            "routing_gates":gate,"weighted_mnn_gates":wgate,"single_label_window":True,"return_to_training_or_transform":False,"independent_recalculation":independent,
            "routing_metrics_sha256":sha256_file(OUT/"routing_per_seed_metrics.csv"),"weighted_metrics_sha256":sha256_file(OUT/"weighted_mnn_per_seed_metrics.csv")}
    atomic_json(OUT/"label_window_audit.json",{"schema_version":1,"window_count":1,"opened_after_total_lock":True,"datasets":audit,"return_to_training_or_transform":False})
    atomic_json(OUT/"gate_audit.json",result); print(json.dumps({"terminal_status":terminal,"selected_router":selected_router,"promoted_weighted_mnn_candidates":promoted},sort_keys=True))


if __name__=="__main__": main()
