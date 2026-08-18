#!/usr/bin/env python3
"""Stage-locked Night-7B evaluator; the only module allowed to load labels."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran  # noqa: E402
from SpaLORA.night6c_pipeline import array_sha  # noqa: E402
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, canonical_partition, sha256_file  # noqa: E402

OUT = REPO / "outputs/night7b_handoff"
RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
LABEL_ROOT = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots")
LABEL_AUDIT = REPO / "outputs/night7a_handoff/label_window_audit.json"
DATASETS = ("a1", "tonsil", "d1", "p22")
WEIGHTS = {"a1":.25, "tonsil":.15, "d1":.25, "p22":.35}


def load_labels(dataset: str):
    path = LABEL_ROOT / (dataset + "_labels_locked.npz")
    authority = json.loads(LABEL_AUDIT.read_text())
    expected = authority["datasets"][dataset]
    observed_sha = sha256_file(path)
    if observed_sha != expected["snapshot_sha256"]:
        raise RuntimeError("locked label snapshot byte SHA mismatch: %s" % dataset)
    with np.load(path, allow_pickle=False) as value:
        keys = set(value.files)
        if keys != {"observation_id", "label"}:
            raise RuntimeError("locked label snapshot schema mismatch: %s" % sorted(keys))
        ids = np.asarray(value["observation_id"]).astype(str)
        labels = np.asarray(value["label"]).astype(str)
        if len(ids) != expected["aligned_rows"] or len(np.unique(labels)) != expected["known_k"]:
            raise RuntimeError("locked label snapshot cardinality mismatch: %s" % dataset)
        if array_sha(labels) != expected["ordered_label_vector_sha256"]:
            raise RuntimeError("locked ordered label vector SHA mismatch: %s" % dataset)
        return ids, labels, observed_sha, sorted(keys)


def load_coordinates(dataset: str, expected_ids: np.ndarray) -> np.ndarray:
    units = list(csv.DictReader((OUT / "source_unit_index.csv").open(newline="")))
    selected = [x for x in units if x["dataset"] == dataset]
    if not selected:
        raise RuntimeError("missing anonymous source unit for coordinates: %s" % dataset)
    unit_dir = RAW / "source" / selected[0]["unit_id"]
    ids = np.asarray([x.strip() for x in (unit_dir / "observation_ids.txt").read_text().splitlines() if x.strip()])
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError("coordinate/source observation order mismatch: %s" % dataset)
    return np.load(unit_dir / "coordinates.npy", allow_pickle=False)


def metric(true, pred, coords):
    graph = symmetric_knn_adjacency(coords, 18); rows, cols = graph.nonzero()
    neighbor = float(np.mean(pred[rows] == pred[cols]))
    moran = float(_mean_cluster_moran(pred, graph)); geary, _ = mean_one_vs_rest_geary(pred, graph)
    ari = float(adjusted_rand_score(true, pred)); nmi = float(normalized_mutual_info_score(true, pred))
    return {"ari":ari, "nmi":nmi, "q":(ari+nmi)/2,
            "neighbor_agreement":neighbor, "moran_i":moran,
            "geary_c":float(geary), "boundary_disagreement":1-neighbor}


def read_clusters(path: Path, expected_ids):
    table = pd.read_csv(path)
    ids = table["observation_id"].astype(str).to_numpy()
    if not np.array_equal(ids, expected_ids): raise RuntimeError("cluster observation order mismatch")
    return canonical_partition(table["cluster"].to_numpy())


def summarize(frame: pd.DataFrame, ids, reference="H00") -> pd.DataFrame:
    metrics = ["ari","nmi","q","neighbor_agreement","moran_i","geary_c","boundary_disagreement"]
    ref = frame[frame.config_id == reference].set_index(["dataset","seed"])
    rows=[]
    for config_id in ids:
        group=frame[frame.config_id == config_id]
        row={"config_id":config_id, "success_cells":int(group.success.sum()), "failure_cells":int((~group.success).sum())}
        for dataset in DATASETS:
            dg=group[group.dataset==dataset]
            row[dataset+"_success"]=int(dg.success.sum())
            for metric_name in metrics:
                values=dg[metric_name].dropna(); row[dataset+"_mean_"+metric_name]=float(values.mean()) if len(values)==len(dg) else np.nan
                if config_id != reference and len(values)==len(dg):
                    rv=np.asarray([ref.loc[(dataset,int(seed)),metric_name] for seed in dg.seed])
                    delta=dg[metric_name].to_numpy()-rv
                    row[dataset+"_mean_delta_"+metric_name]=float(delta.mean())
                    row[dataset+"_wins_"+metric_name]=int(np.sum(delta>0))
                else:
                    row[dataset+"_mean_delta_"+metric_name]=0.0 if config_id==reference else np.nan
                    row[dataset+"_wins_"+metric_name]=0
        complete=row["failure_cells"]==0
        row["complete_eligible"]=complete
        row["priority_weighted_q"]=sum(WEIGHTS[d]*row[d+"_mean_q"] for d in DATASETS) if complete else np.nan
        row["priority_weighted_delta_q"]=sum(WEIGHTS[d]*row[d+"_mean_delta_q"] for d in DATASETS) if complete else np.nan
        row["balanced_macro_delta_q"]=float(np.mean([row[d+"_mean_delta_q"] for d in DATASETS])) if complete else np.nan
        row["worst_dataset_delta_q"]=min(row[d+"_mean_delta_q"] for d in DATASETS) if complete else np.nan
        row["total_q_wins"]=sum(row[d+"_wins_q"] for d in DATASETS) if complete else 0
        row["mean_runtime_seconds"] = float(group.runtime_seconds.mean())
        row["mean_peak_rss_mib"] = float(group.peak_rss_mib.mean())
        row["mean_gpu_allocation_mib"] = float(group.gpu_allocation_mib.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_h():
    locked_path=OUT/"locked_head_transform_manifest.json"; locked=json.loads(locked_path.read_text())
    if len(locked["transforms"])!=540 or not locked["locked_before_evaluation"]: raise RuntimeError("H not locked")
    labels={}; label_audit={}
    for dataset in DATASETS:
        ids,true,digest,keys=load_labels(dataset); coords=load_coordinates(dataset, ids); labels[dataset]=(ids,true,coords)
        label_audit[dataset]={"snapshot_sha256":digest,"keys":keys,"authorized_role":"stage_H_evaluator"}
    rows=[]
    for cell in locked["transforms"]:
        row={"stage":"H","config_id":cell["head_id"],"dataset":cell["dataset"],"seed":cell["seed"],
             "success":cell["status"]=="success","status":cell["status"],
             "runtime_seconds":cell["runtime_seconds"], "peak_rss_mib":cell["peak_rss_mib"],
             "gpu_allocation_mib":cell["gpu_allocation_mib"]}
        if row["success"]:
            ids,true,coords=labels[cell["dataset"]]; pred=read_clusters(Path(cell["clusters_artifact"]["path"]),ids)
            row.update(metric(true,pred,coords))
        else:
            row.update({k:np.nan for k in ("ari","nmi","q","neighbor_agreement","moran_i","geary_c","boundary_disagreement")})
        rows.append(row)
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"H_per_seed_metrics.csv",index=False)
    summary=summarize(frame,["H%02d"%i for i in range(18)],"H00")
    summary["registry_order"] = summary.config_id.str.slice(1).astype(int)
    summary=summary.sort_values(["complete_eligible","priority_weighted_delta_q","balanced_macro_delta_q","worst_dataset_delta_q","total_q_wins","mean_runtime_seconds","mean_peak_rss_mib","mean_gpu_allocation_mib","registry_order"],ascending=[False,False,False,False,False,True,True,True,True])
    summary.to_csv(OUT/"H_candidate_summary.csv",index=False)
    eligible=summary[summary.complete_eligible].config_id.tolist()
    promoted=eligible[:2]
    if len(promoted)<2: raise RuntimeError("fewer than two complete H templates")
    contract={"stage":"H","status":"PASS","locked_manifest_sha256":sha256_file(locked_path),
              "label_window":label_audit,"label_opened_after_lock":True,
              "promoted_head_ids":promoted,"fallback_comparator":"H00",
              "ranking_table_sha256":sha256_file(OUT/"H_candidate_summary.csv"),
              "aggregate_only_message":{"config_ids":promoted,"ranking_sha256":sha256_file(OUT/"H_candidate_summary.csv")}}
    atomic_json(OUT/"H_to_R1_contract.json",contract)
    atomic_json(OUT/"H_label_window_audit.json",label_audit)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("H",)); args=parser.parse_args()
    if args.stage=="H": evaluate_h()


if __name__=="__main__": main()
