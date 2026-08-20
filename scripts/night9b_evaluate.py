#!/usr/bin/env python3
"""Single-window A1/P22 evaluator and conditional R2 supervisor for Night-9B."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran  # noqa: E402
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency  # noqa: E402
from SpaLORA.night6c_pipeline import array_sha  # noqa: E402
from SpaLORA.night7a_consensus import canonical_partition  # noqa: E402

OUT = REPO / "outputs/night9b"
RAW = Path("/root/autodl-fs/night9b_racf_20260820")
LABEL_ROOT = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots")
LABEL_AUTHORITY = REPO / "outputs/night7a_handoff/label_window_audit.json"
PYTHON = "/root/miniconda3/envs/SpaLORA/bin/python"
THRESHOLDS = {"neighbor_min": -.01, "moran_min": -.02, "geary_max": .02, "boundary_max": .01}
METRICS = ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(tmp, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prelabel_gate() -> dict:
    lock_path = OUT / "prelabel_total_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["status"] != "LOCKED_PRE_LABEL": raise RuntimeError("prelabel total lock missing")
    if git("status", "--porcelain"): raise RuntimeError("repository is not clean before label window")
    head, upstream = git("rev-parse", "HEAD"), git("rev-parse", "@{u}")
    if head != upstream: raise RuntimeError("prelabel lock commit is not pushed")
    if sha(REPO / "scripts/night9b_evaluate.py") != lock["source_sha256"]["scripts/night9b_evaluate.py"]:
        raise RuntimeError("locked evaluator SHA drift")
    return {"status": "PASS", "head": head, "upstream": upstream,
            "prelabel_lock_sha256": sha(lock_path)}


def load_labels_once() -> tuple[dict, dict]:
    authority = json.loads(LABEL_AUTHORITY.read_text())["datasets"]
    labels = {}; audit = {}
    for dataset in ("a1", "p22"):
        path = LABEL_ROOT / f"{dataset}_labels_locked.npz"
        expected = authority[dataset]
        if sha(path) != expected["snapshot_sha256"]: raise RuntimeError(f"{dataset} label snapshot SHA drift")
        with np.load(path, allow_pickle=False) as value:
            if set(value.files) != {"observation_id", "label"}: raise RuntimeError("label snapshot schema drift")
            ids = np.asarray(value["observation_id"]).astype(str); true = np.asarray(value["label"]).astype(str)
        if len(ids) != expected["aligned_rows"] or len(np.unique(true)) != expected["known_k"]:
            raise RuntimeError(f"{dataset} label authority count/K drift")
        if array_sha(true) != expected["ordered_label_vector_sha256"]:
            raise RuntimeError(f"{dataset} label vector SHA drift")
        labels[dataset] = (ids, true)
        audit[dataset] = {"read_count_this_round": 1, "path": str(path), "sha256": sha(path),
                          "rows": len(ids), "reference_K": len(np.unique(true)),
                          "authorized_role": "single_night9b_evaluator_process",
                          "used_for_training_checkpoint_epoch_or_hyperparameter": False}
    return labels, audit


def independent_ari_nmi(true, pred) -> tuple[float, float]:
    _, ti = np.unique(np.asarray(true), return_inverse=True); _, pi = np.unique(np.asarray(pred), return_inverse=True)
    table = np.zeros((ti.max()+1, pi.max()+1), dtype=np.int64); np.add.at(table, (ti, pi), 1)
    n = int(table.sum()); comb = lambda x: x * (x - 1.0) / 2.0
    cells, rows, cols, total = map(float, (comb(table).sum(), comb(table.sum(1)).sum(),
                                           comb(table.sum(0)).sum(), comb(n)))
    expected = rows * cols / total; ari = (cells - expected) / (.5 * (rows + cols) - expected)
    pxy = table / n; px = pxy.sum(1); py = pxy.sum(0); ii, jj = np.nonzero(table)
    mi = float(sum(pxy[i,j]*math.log(pxy[i,j]/(px[i]*py[j])) for i,j in zip(ii,jj)))
    hx = float(-sum(x*math.log(x) for x in px if x>0)); hy = float(-sum(x*math.log(x) for x in py if x>0))
    return float(ari), float(mi / ((hx + hy) / 2.0))


def metric(true: np.ndarray, pred: np.ndarray, graph) -> tuple[dict, dict]:
    rows, cols = graph.nonzero(); ari = float(adjusted_rand_score(true, pred)); nmi = float(normalized_mutual_info_score(true, pred))
    ia, inn = independent_ari_nmi(true, pred); error = {"ari_abs_error": abs(ari-ia), "nmi_abs_error": abs(nmi-inn)}
    if max(error.values()) > 1e-12: raise RuntimeError(f"independent evaluator mismatch: {error}")
    neighbor = float(np.mean(pred[rows] == pred[cols])); geary, _ = mean_one_vs_rest_geary(pred, graph)
    return {"ari": ari, "nmi": nmi, "q": (ari+nmi)/2,
            "neighbor_agreement": neighbor, "moran_i": float(_mean_cluster_moran(pred, graph)),
            "geary_c": float(geary), "boundary_disagreement": 1-neighbor}, error


def load_clusters(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["observation_id", "cluster"] or not np.array_equal(frame.observation_id.astype(str), ids):
        raise RuntimeError(f"cluster ID/schema mismatch: {path}")
    pred = canonical_partition(frame.cluster.to_numpy())
    return pred


def a1_reference(seed: int) -> Path:
    rows = list(csv.DictReader((REPO / "outputs/night7a_handoff/source_prediction_index.csv").open(newline="")))
    match = [r for r in rows if r["dataset"] == "a1" and int(r["seed"]) == seed
             and r["graph_id"] == "G04_SP10_F10_EUC_UNION" and r["head_id"] == "H05_EQUAL3_AFFINITY_SPECTRAL"]
    if len(match) != 1: raise RuntimeError("A1 C00 reference not unique")
    return Path(match[0]["path"])


def p22_reference(seed: int) -> Path:
    return Path(f"/root/autodl-fs/night9a_efficient_topology_transfer_20260820/p0/references/p22/seed_{seed}/FULL_F00/clusters.csv")


def evaluate_stage(stage: str, labels: dict, seeds: list[int], candidate_ids: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    lock = json.loads((OUT / f"{stage.lower()}_lock_manifest.json").read_text())
    by_key = {(r["candidate_id"], r["dataset"].lower(), int(r["seed"])): r for r in lock["rows"]}
    metric_rows = []; independent = []
    coords = {"a1": np.load("/root/autodl-fs/night6c_cache_20260817/base/a1/coordinates.npy", allow_pickle=False),
              "p22": np.load("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22/coordinates.npy", allow_pickle=False)}
    graphs = {key: symmetric_knn_adjacency(value, 18) for key, value in coords.items()}
    for dataset in ("a1", "p22"):
        ids, true = labels[dataset]
        for seed in seeds:
            ref_path = a1_reference(seed) if dataset == "a1" else p22_reference(seed)
            values, errors = metric(true, load_clusters(ref_path, ids), graphs[dataset])
            metric_rows.append({"stage": stage, "candidate_id": "REFERENCE", "dataset": dataset,
                                "seed": seed, "status": "success", **values})
            independent.append({"stage": stage, "candidate_id": "REFERENCE", "dataset": dataset, "seed": seed, **errors})
            for cid in candidate_ids:
                row = by_key.get((cid, dataset, seed))
                if not row or row["status"] != "success":
                    metric_rows.append({"stage": stage, "candidate_id": cid, "dataset": dataset,
                                        "seed": seed, "status": "failed", **{k: np.nan for k in METRICS}}); continue
                manifest = json.loads((Path(row["output_dir"]) / "training_manifest.json").read_text())
                path = Path(manifest["artifacts"]["clusters.csv"]["path"])
                values, errors = metric(true, load_clusters(path, ids), graphs[dataset])
                metric_rows.append({"stage": stage, "candidate_id": cid, "dataset": dataset,
                                    "seed": seed, "status": "success", **values,
                                    "total_runtime_seconds": manifest["runtime"]["total_seconds"]})
                independent.append({"stage": stage, "candidate_id": cid, "dataset": dataset, "seed": seed, **errors})
    return pd.DataFrame(metric_rows), independent


def summarize(metrics: pd.DataFrame, candidate_ids: list[str], seeds: list[int]) -> list[dict]:
    rows = []
    for cid in candidate_ids:
        item = {"candidate_id": cid, "seeds": seeds, "complete": True, "datasets": {}}
        for dataset in ("a1", "p22"):
            ref = metrics[(metrics.dataset == dataset) & (metrics.candidate_id == "REFERENCE")].set_index("seed").loc[seeds]
            cand = metrics[(metrics.dataset == dataset) & (metrics.candidate_id == cid)].set_index("seed").reindex(seeds)
            complete = bool(len(cand) == len(seeds) and cand.status.eq("success").all())
            item["complete"] &= complete
            if not complete: item["datasets"][dataset] = {"complete": False}; continue
            delta = {m: cand[m].to_numpy(float) - ref[m].to_numpy(float) for m in METRICS}
            spatial = {"neighbor": float(delta["neighbor_agreement"].mean()) >= THRESHOLDS["neighbor_min"],
                       "moran": float(delta["moran_i"].mean()) >= THRESHOLDS["moran_min"],
                       "geary": float(delta["geary_c"].mean()) <= THRESHOLDS["geary_max"],
                       "boundary": float(delta["boundary_disagreement"].mean()) <= THRESHOLDS["boundary_max"]}
            item["datasets"][dataset] = {
                "complete": True, **{f"mean_delta_{m}": float(v.mean()) for m,v in delta.items()},
                "q_wins": int(np.sum(delta["q"] > 0)), "worst_seed_delta_q": float(delta["q"].min()),
                "mean_runtime_seconds": float(cand.total_runtime_seconds.mean()),
                "spatial_protection": {"pass": all(spatial.values()), "components": spatial, "thresholds": THRESHOLDS}}
        rows.append(item)
    return rows


def shortlist(summary: list[dict], wins_min: int) -> dict:
    slots = {"BALANCED": [], "LYMPH_SPECIALIST": [], "BRAIN_SPECIALIST": []}
    for row in summary:
        if not row["complete"]: continue
        a, p = row["datasets"]["a1"], row["datasets"]["p22"]
        spatial = a["spatial_protection"]["pass"] and p["spatial_protection"]["pass"]
        if spatial and a["mean_delta_q"] >= .01 and p["mean_delta_q"] >= .01 and a["q_wins"] >= wins_min and p["q_wins"] >= wins_min:
            slots["BALANCED"].append(row)
        if spatial and a["mean_delta_q"] >= .02 and p["mean_delta_q"] >= -.005:
            slots["LYMPH_SPECIALIST"].append(row)
        if spatial and p["mean_delta_q"] >= .03 and a["mean_delta_q"] >= -.005:
            slots["BRAIN_SPECIALIST"].append(row)
    ranked = {}; union = []
    for name, values in slots.items():
        def key(row):
            a, p = row["datasets"]["a1"], row["datasets"]["p22"]
            primary = (a["mean_delta_q"] + p["mean_delta_q"]) / 2 if name == "BALANCED" else a["mean_delta_q"] if name == "LYMPH_SPECIALIST" else p["mean_delta_q"]
            worst = min(a["worst_seed_delta_q"], p["worst_seed_delta_q"])
            wins = a["q_wins"] + p["q_wins"]; runtime = a["mean_runtime_seconds"] + p["mean_runtime_seconds"]
            return (-primary, -worst, -wins, runtime, row["candidate_id"])
        selected = [r["candidate_id"] for r in sorted(values, key=key)[:2]]
        ranked[name] = selected
        for cid in selected:
            if cid not in union and len(union) < 4: union.append(cid)
    return {"slots": ranked, "shortlist_candidate_ids": union, "max_per_slot": 2,
            "max_unique_r2": 4, "selection_order": ["BALANCED", "LYMPH_SPECIALIST", "BRAIN_SPECIALIST"]}


def evaluate_cosmos(labels: dict) -> tuple[pd.DataFrame, dict]:
    ids, true = labels["p22"]; coords = np.load("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22/coordinates.npy")
    graph = symmetric_knn_adjacency(coords, 18); lock = json.loads((OUT / "p2_cosmos_lock_manifest.json").read_text())
    rows = []
    for seed in range(5):
        ref, _ = metric(true, load_clusters(p22_reference(seed), ids), graph)
        rows.append({"seed": seed, "lane": "F00_R02_REFERENCE", **ref})
        unit = [x for x in lock["rows"] if x["seed"] == seed][0]; out = Path(unit["output_dir"])
        for lane in ("COSMOS_NATIVE_FIXED", "COSMOS_COMMON_HEAD"):
            values, _ = metric(true, load_clusters(out / f"{lane}_clusters.csv", ids), graph)
            rows.append({"seed": seed, "lane": lane, **values})
    frame = pd.DataFrame(rows); ref = frame[frame.lane == "F00_R02_REFERENCE"].set_index("seed")
    summary = {"protocol_role": "P22 same locked K and project inputs; common-head lane is direct numeric comparison",
               "paper_reported_ARI_context_only": .63, "not_claimed_same_protocol_as_paper": True, "lanes": {}}
    for lane in ("COSMOS_NATIVE_FIXED", "COSMOS_COMMON_HEAD"):
        cand = frame[frame.lane == lane].set_index("seed")
        summary["lanes"][lane] = {"mean_ari": float(cand.ari.mean()), "mean_nmi": float(cand.nmi.mean()),
                                   "mean_q": float(cand.q.mean()), "mean_delta_ari_vs_F00": float((cand.ari-ref.ari).mean()),
                                   "mean_delta_nmi_vs_F00": float((cand.nmi-ref.nmi).mean()),
                                   "mean_delta_q_vs_F00": float((cand.q-ref.q).mean())}
    return frame, summary


def commit_push(paths: list[Path], message: str) -> str:
    subprocess.run(["git", "add", "--", *[str(p.relative_to(REPO)) for p in paths]], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=REPO, check=True)
    subprocess.run(["git", "push"], cwd=REPO, check=True)
    return git("rev-parse", "HEAD")


def main() -> None:
    gate = prelabel_gate(); labels, label_audit = load_labels_once()
    candidate_ids = [x["id"] for x in json.loads((REPO / "protocols/night9b/SpaLORA_Night9B_RACF_Registry_2026-08-20.json").read_text())["candidates"]]
    r1_metrics, independent = evaluate_stage("R1", labels, [0,1,2], candidate_ids)
    r1_metrics_path = OUT / "r1_per_seed_metrics.csv"; r1_metrics.to_csv(r1_metrics_path, index=False)
    r1_summary = summarize(r1_metrics, candidate_ids, [0,1,2]); atomic_json(OUT / "r1_candidate_summary.json", r1_summary)
    selected = shortlist(r1_summary, 2); atomic_json(OUT / "r1_shortlist.json", selected)
    cosmos_frame, cosmos_summary = evaluate_cosmos(labels); cosmos_frame.to_csv(OUT / "cosmos_p22_per_seed_metrics.csv", index=False)
    atomic_json(OUT / "cosmos_gap_calibration.json", cosmos_summary)
    atomic_json(OUT / "label_window_in_progress.json", {"status": "OPEN_SINGLE_PROCESS", "prelabel_gate": gate,
                                                         "datasets": label_audit, "MISAR_Y_read_count": 0,
                                                         "raw_snapshot_deserializations": {"A1": 1, "P22": 1}})
    first_commit = commit_push([r1_metrics_path, OUT / "r1_candidate_summary.json", OUT / "r1_shortlist.json",
                                OUT / "cosmos_p22_per_seed_metrics.csv", OUT / "cosmos_gap_calibration.json",
                                OUT / "label_window_in_progress.json"], "Night-9B: lock R1 evaluation and shortlist")
    r2_ids = selected["shortlist_candidate_ids"]
    r2_summary = []; r2_commit = None; final_summary = r1_summary; final_slots = selected
    if r2_ids:
        subprocess.run([PYTHON, "scripts/night9b_orchestrate.py", "stage", "--stage", "R2",
                        "--candidates", ",".join(r2_ids), "--seeds", "3,4"], cwd=REPO, check=True)
        r2_lock = json.loads((OUT / "r2_lock_manifest.json").read_text())
        if r2_lock["planned_units"] != len(r2_ids)*4 or r2_lock["success_units"] != len(r2_ids)*4:
            raise RuntimeError("conditional R2 total lock incomplete")
        r2_commit = commit_push([OUT / "r2_lock_manifest.json"], "Night-9B: lock conditional R2 outputs")
        r2_metrics, r2_independent = evaluate_stage("R2", labels, [3,4], r2_ids); independent += r2_independent
        all_metrics = pd.concat([r1_metrics, r2_metrics], ignore_index=True)
        all_metrics.to_csv(OUT / "r1_r2_per_seed_metrics.csv", index=False)
        final_summary = summarize(all_metrics, r2_ids, [0,1,2,3,4]); atomic_json(OUT / "r1_r2_candidate_summary.json", final_summary)
        final_slots = shortlist(final_summary, 3); atomic_json(OUT / "final_candidate_slots.json", final_slots)
    else:
        r1_metrics.to_csv(OUT / "r1_r2_per_seed_metrics.csv", index=False)
        atomic_json(OUT / "r1_r2_candidate_summary.json", final_summary)
        atomic_json(OUT / "final_candidate_slots.json", final_slots)
    max_error = max(max(x["ari_abs_error"], x["nmi_abs_error"]) for x in independent)
    atomic_json(OUT / "independent_metric_recompute.json", {"status": "PASS", "rows": len(independent),
                                                             "max_abs_error": max_error, "tolerance": 1e-12})
    if max_error > 1e-12: raise RuntimeError("independent metric recompute failed")
    terminal = ("NIGHT9B_RACF_CANDIDATES_LOCKED" if final_slots["shortlist_candidate_ids"] else
                "NIGHT9B_COSMOS_GAP_CALIBRATED_NO_RACF_CANDIDATE")
    atomic_json(OUT / "label_window_audit.json", {"status": "CLOSED", "single_process_window": True,
                                                  "datasets": label_audit, "MISAR_Y_read_count": 0,
                                                  "first_label_derived_commit": first_commit,
                                                  "conditional_R2_lock_commit": r2_commit,
                                                  "return_to_structure_evaluator_or_hyperparameter_change": False,
                                                  "scientific_retry": 0, "fallback": 0})
    atomic_json(OUT / "night9b_decision.json", {"terminal_status": terminal,
                                                "final_slots": final_slots,
                                                "P22_role": "development data; not pristine holdout",
                                                "SOTA_claim": False})
    print(json.dumps({"terminal_status": terminal, "R1_shortlist": r2_ids,
                      "final_candidates": final_slots["shortlist_candidate_ids"]}, sort_keys=True))


if __name__ == "__main__": main()
