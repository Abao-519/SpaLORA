#!/usr/bin/env python3
"""Post-lock evaluator and preregistered Night-6C R1/R2 selector."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran, load_evaluation_labels
from SpaLORA.night3af_cache import load_cache, sha256_file
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency
from SpaLORA.night6c_firewall import guard_path
from SpaLORA.night6c_pipeline import atomic_json, load_views, parse_registry

OUT = REPO / "outputs/night6c_handoff"
CACHE = Path("/root/autodl-fs/night6c_cache_20260817/base")
REG_PATH = REPO / "protocols/night6c/SpaLORA_Night6B_Candidate_Registry_2026-08-17.json"
TONSIL_SOURCE = Path("/root/autodl-fs/datasets/human_tonsil_official/section1/s1_adata_rna.h5ad")
A1_CFG = {"ground_truth": "/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv",
          "ground_truth_id_column": "Barcode", "ground_truth_label_column": "manual-anno",
          "ground_truth_id_rule": "strip_s1_prefix"}


def labels_for(dataset: str, ids: pd.Index, stage_locked: bool) -> tuple[np.ndarray, np.ndarray]:
    if not stage_locked:
        raise RuntimeError("evaluator cannot start before transform manifest total lock")
    if dataset == "a1":
        return load_evaluation_labels("a1", A1_CFG, ids)
    guard_path(TONSIL_SOURCE, role="evaluator", operation="read final_annot",
               phase_locked=True, audit_log=OUT / "firewall/evaluator_access.jsonl")
    src = ad.read_h5ad(TONSIL_SOURCE, backed="r")
    series = src.obs["final_annot"]
    mapping = pd.Series(series.astype(str).values, index=src.obs_names.astype(str))
    src.file.close()
    if not ids.isin(mapping.index).all():
        raise RuntimeError("tonsil label alignment incomplete")
    aligned = mapping.reindex(ids)
    if aligned.isna().any() or aligned.nunique() != 4:
        raise RuntimeError("tonsil ontology replay mismatch")
    return np.arange(len(ids)), aligned.to_numpy(dtype=str)


def metrics(true: np.ndarray, pred: np.ndarray, coords: np.ndarray, k: int = 18) -> dict:
    graph = symmetric_knn_adjacency(coords, k)
    rows, cols = graph.nonzero()
    neighbor = float(np.mean(pred[rows] == pred[cols]))
    moran = _mean_cluster_moran(pred, graph)
    geary, _ = mean_one_vs_rest_geary(pred, graph)
    ari = float(adjusted_rand_score(true, pred))
    nmi = float(normalized_mutual_info_score(true, pred))
    return {"ari": ari, "nmi": nmi, "q": (ari + nmi) / 2.0,
            "neighbor_agreement": neighbor, "moran_i": moran,
            "geary_c": geary, "boundary_disagreement": 1.0 - neighbor}


def spatial_fail(delta: dict) -> bool:
    return bool((delta["delta_neighbor"] < -0.03 and delta["delta_moran"] < -0.03) or
                (delta["delta_geary"] > 0.03 and
                 (delta["delta_neighbor"] < -0.03 or delta["delta_moran"] < -0.03)))


def stage_evaluate(stage: str, registry: dict) -> pd.DataFrame:
    transform_path = OUT / f"{stage.lower()}_transform_manifest.json"
    manifest = json.loads(transform_path.read_text(encoding="utf-8"))
    if manifest["status"] != "LOCKED" or not manifest["locked_before_label_access"]:
        raise RuntimeError("transform manifest is not totally locked")
    labels = {}; prepared = {}
    for dataset in ("a1", "tonsil"):
        base = CACHE / dataset
        prepared[dataset] = load_cache(base, sha256_file(base / "manifest.json"))
        ids = prepared[dataset].obs_names.astype(str)
        labels[dataset] = labels_for(dataset, ids, True)
    rows = []
    for item in manifest["transforms"]:
        dataset = item["dataset"]
        run = Path(item["run_dir"])
        cluster_file = Path(item["head_dir"]) / "clusters.csv"
        table = pd.read_csv(cluster_file)
        ids = prepared[dataset].obs_names.astype(str)
        if not np.array_equal(table["observation_id"].astype(str).to_numpy(), ids.to_numpy()):
            raise RuntimeError("cluster observation order mismatch")
        pred = table["cluster"].to_numpy(dtype=np.int64)
        pos, true = labels[dataset]
        observed = metrics(true, pred[pos], prepared[dataset].coordinates, 18)
        training = json.loads((run / "run_manifest.json").read_text())
        rows.append({"stage": stage, "dataset": dataset, "graph_id": item["graph_id"],
                     "seed": int(item["seed"]), "head_id": item["head_id"],
                     **observed,
                     "training_runtime_seconds": training["runtime_seconds"],
                     "head_runtime_seconds": item["runtime_seconds"],
                     "effective_runtime_seconds": training["runtime_seconds"] + item["runtime_seconds"],
                     "peak_gpu_allocated_mib": training["peak_gpu_allocated_mib"],
                     "process_peak_rss_mib": max(training["process_peak_rss_mib"], item["process_peak_rss_mib"]),
                     "checkpoint_file_sha256": training["checkpoint_file_sha256"],
                     "cluster_file_sha256": item["cluster_file_sha256"]})
    current_path = OUT / "per_seed_metrics.csv"
    previous = pd.read_csv(current_path) if current_path.exists() else pd.DataFrame()
    current = pd.DataFrame(rows)
    if not previous.empty:
        keys = ["dataset", "graph_id", "seed", "head_id"]
        overlap = previous.merge(current, on=keys)
        if len(overlap): raise RuntimeError("stage metric primary-key overlap")
        current = pd.concat([previous, current], ignore_index=True)
    current.sort_values(["dataset", "graph_id", "seed", "head_id"]).to_csv(current_path, index=False)
    atomic_json(OUT / f"firewall/{stage.lower()}_label_access_audit.json", {
        "stage": stage, "transform_manifest_sha256": sha256_file(transform_path),
        "manifest_locked_before_access": True,
        "evaluator": {"deserialized_into_memory": True, "explicitly_indexed_or_observed": True,
                      "used_for_training_or_selection": False, "authorized_role": "evaluator"},
        "trainer_transformer_original_tonsil_access": 0,
        "protected_dataset_access": 0,
    })
    return current


def _reference(frame: pd.DataFrame, dataset: str, seed: int) -> pd.Series:
    row = frame[(frame.dataset == dataset) & (frame.seed == seed) &
                frame.graph_id.str.startswith("G00_") & frame.head_id.str.startswith("H00_")]
    if len(row) != 1: raise RuntimeError("fresh reference cardinality mismatch")
    return row.iloc[0]


def delta_row(candidate: pd.Series, reference: pd.Series) -> dict:
    return {"delta_ari": float(candidate.ari - reference.ari),
            "delta_nmi": float(candidate.nmi - reference.nmi),
            "delta_q": float(candidate.q - reference.q),
            "delta_neighbor": float(candidate.neighbor_agreement - reference.neighbor_agreement),
            "delta_moran": float(candidate.moran_i - reference.moran_i),
            "delta_geary": float(candidate.geary_c - reference.geary_c),
            "delta_boundary": float(candidate.boundary_disagreement - reference.boundary_disagreement)}


def summarize_combo(frame: pd.DataFrame, graph_id: str, head_id: str,
                    seeds: list[int]) -> dict:
    cells = []
    for dataset in ("a1", "tonsil"):
        for seed in seeds:
            selected = frame[(frame.dataset == dataset) & (frame.seed == seed) &
                             (frame.graph_id == graph_id) & (frame.head_id == head_id)]
            if len(selected) != 1: raise RuntimeError(f"candidate cell cardinality mismatch {graph_id}/{head_id}/{dataset}/{seed}")
            candidate = selected.iloc[0]; reference = _reference(frame, dataset, seed)
            cells.append({"dataset": dataset, "seed": seed, **delta_row(candidate, reference),
                          "ari": float(candidate.ari), "nmi": float(candidate.nmi), "q": float(candidate.q),
                          "effective_runtime_seconds": float(candidate.effective_runtime_seconds),
                          "peak_gpu_allocated_mib": float(candidate.peak_gpu_allocated_mib),
                          "process_peak_rss_mib": float(candidate.process_peak_rss_mib)})
    z = pd.DataFrame(cells)
    dataset_means = {d: float(g.delta_q.mean()) for d, g in z.groupby("dataset")}
    spatial = {}
    for d, g in z.groupby("dataset"):
        delta = {"delta_neighbor": float(g.delta_neighbor.mean()),
                 "delta_moran": float(g.delta_moran.mean()),
                 "delta_geary": float(g.delta_geary.mean()),
                 "delta_boundary": float(g.delta_boundary.mean())}
        delta["spatial_gate_failed"] = spatial_fail(delta); spatial[d] = delta
    return {"graph_id": graph_id, "head_id": head_id,
            "macro_delta_ari": float(z.delta_ari.mean()), "macro_delta_nmi": float(z.delta_nmi.mean()),
            "macro_delta_q": float(z.delta_q.mean()), "worst_dataset_mean_delta_q": min(dataset_means.values()),
            "dataset_mean_delta_q": dataset_means,
            "paired_q_wins": int((z.delta_q > 0).sum()), "paired_q_total": len(z),
            "delta_q_std": float(z.delta_q.std(ddof=0)),
            "spatial_by_dataset": spatial,
            "spatial_gate_pass_both": not any(x["spatial_gate_failed"] for x in spatial.values()),
            "mean_effective_runtime_seconds": float(z.effective_runtime_seconds.mean()),
            "peak_gpu_allocated_mib": float(z.peak_gpu_allocated_mib.max()),
            "peak_rss_mib": float(z.process_peak_rss_mib.max()),
            "complete_cells": len(z), "cells": cells}


def r1_decision(frame: pd.DataFrame, registry: dict) -> dict:
    graphs, heads = parse_registry(registry)
    g00, h00 = list(graphs)[0], list(heads)[0]
    graph_summaries = [summarize_combo(frame, g, h00, [0, 1]) for g in list(graphs)[1:]]
    balanced = [x for x in graph_summaries if x["macro_delta_q"] >= .005 and
                x["worst_dataset_mean_delta_q"] >= -.005 and x["paired_q_wins"] >= 3 and
                x["spatial_gate_pass_both"]]
    accuracy = [x for x in graph_summaries if x["macro_delta_q"] >= .015 and
                x["worst_dataset_mean_delta_q"] >= -.010 and x["paired_q_wins"] >= 3]
    balanced.sort(key=lambda x: (-x["macro_delta_q"], x["delta_q_std"], x["mean_effective_runtime_seconds"], x["graph_id"]))
    accuracy.sort(key=lambda x: (-x["macro_delta_q"], x["delta_q_std"], x["mean_effective_runtime_seconds"], x["graph_id"]))
    balanced_ids = [x["graph_id"] for x in balanced[:2]]
    accuracy_ids = [x["graph_id"] for x in accuracy[:2]]
    advanced_graphs = []
    for value in balanced_ids + accuracy_ids:
        if value not in advanced_graphs: advanced_graphs.append(value)

    head_summaries = []
    safe_graphs = [g00] + advanced_graphs
    for head_id in list(heads)[1:]:
        marginal = []
        for graph_id in graphs:
            for dataset in ("a1", "tonsil"):
                for seed in (0, 1):
                    candidate = frame[(frame.graph_id == graph_id) & (frame.head_id == head_id) &
                                      (frame.dataset == dataset) & (frame.seed == seed)].iloc[0]
                    within = frame[(frame.graph_id == graph_id) & (frame.head_id == h00) &
                                   (frame.dataset == dataset) & (frame.seed == seed)].iloc[0]
                    marginal.append(float(candidate.q - within.q))
        combos = [summarize_combo(frame, graph_id, head_id, [0, 1]) for graph_id in safe_graphs]
        best = max(combos, key=lambda x: x["macro_delta_q"])
        head_summaries.append({"head_id": head_id, "family": heads[head_id]["family"],
                               "marginal_delta_q": float(np.mean(marginal)),
                               "marginal_paired_wins": int(np.sum(np.asarray(marginal) > 0)),
                               "heterogeneous": bool(np.any(np.asarray(marginal) > 0) and np.any(np.asarray(marginal) < 0)),
                               "best_valid_graph_id": best["graph_id"],
                               "best_valid_combination_delta_q": best["macro_delta_q"],
                               "mean_runtime_seconds": float(frame[frame.head_id == head_id].head_runtime_seconds.mean())})
    selected = []
    positive_marginal = sorted([x for x in head_summaries if x["marginal_delta_q"] > 0],
                               key=lambda x: (-x["marginal_delta_q"], x["mean_runtime_seconds"], x["head_id"]))
    if positive_marginal: selected.append(positive_marginal[0]["head_id"])
    positive_best = sorted([x for x in head_summaries if x["best_valid_combination_delta_q"] > 0],
                           key=lambda x: (-x["best_valid_combination_delta_q"], x["mean_runtime_seconds"], x["head_id"]))
    if positive_best and positive_best[0]["head_id"] not in selected:
        selected.append(positive_best[0]["head_id"])
    eligible_family = {"sparse_affinity", "sparse_affinity_spatial", "partition_ensemble"}
    family_rows = sorted([x for x in head_summaries if x["family"] in eligible_family and
                          max(x["marginal_delta_q"], x["best_valid_combination_delta_q"]) > 0],
                         key=lambda x: (-max(x["marginal_delta_q"], x["best_valid_combination_delta_q"]),
                                        x["mean_runtime_seconds"], x["head_id"]))
    if family_rows and not any(x["head_id"] in selected for x in family_rows) and len(selected) < 3:
        selected.append(family_rows[0]["head_id"])
    for row in sorted(head_summaries,
                      key=lambda x: (-max(x["marginal_delta_q"], x["best_valid_combination_delta_q"]),
                                     x["mean_runtime_seconds"], x["head_id"])):
        if max(row["marginal_delta_q"], row["best_valid_combination_delta_q"]) > 0 and row["head_id"] not in selected and len(selected) < 3:
            selected.append(row["head_id"])
    return {"stage": "R1", "status": "LOCKED", "label_access_after_output_lock": True,
            "graph_summaries": graph_summaries, "balanced_graph_pool": balanced_ids,
            "accuracy_graph_pool": accuracy_ids, "advanced_graphs": advanced_graphs,
            "head_summaries": head_summaries, "advanced_heads": selected,
            "reference": {"graph_id": g00, "head_id": h00},
            "parameter_tuning": False, "seed_search": False,
            "protected_dataset_access": False}


def historical_drift(frame: pd.DataFrame) -> None:
    path = REPO / "outputs/night5a_handoff/per_seed_metrics.csv"
    if not path.exists():
        return
    old = pd.read_csv(path)
    cid = "candidate_id" if "candidate_id" in old.columns else "candidate"
    old = old[(old[cid] == "C04_SHRINK25") & (old["dataset"] == "a1")]
    rows = []
    for seed in range(5):
        new = _reference(frame, "a1", seed)
        prior = old[old.seed == seed]
        if len(prior) == 1:
            prior = prior.iloc[0]
            old_ari = float(prior["ari"]); old_nmi = float(prior["nmi"])
            rows.append({"status": "HISTORICAL_DRIFT_DIAGNOSTIC_ONLY", "dataset": "a1", "seed": seed,
                         "fresh_ari": new.ari, "historical_ari": old_ari, "delta_ari": new.ari-old_ari,
                         "fresh_nmi": new.nmi, "historical_nmi": old_nmi, "delta_nmi": new.nmi-old_nmi,
                         "used_for_selection": False, "parity_hard_gate": False})
    pd.DataFrame(rows).to_csv(OUT / "historical_night5_drift_diagnostic.csv", index=False)


def r2_decision(frame: pd.DataFrame, registry: dict) -> dict:
    graphs, heads = parse_registry(registry)
    r1 = json.loads((OUT / "r1_decision.json").read_text())
    graph_ids = [list(graphs)[0]] + list(r1["advanced_graphs"])
    head_ids = [list(heads)[0]] + list(r1["advanced_heads"])
    summaries = []
    for graph_id in graph_ids:
        for head_id in head_ids:
            if graph_id == list(graphs)[0] and head_id == list(heads)[0]: continue
            summaries.append(summarize_combo(frame, graph_id, head_id, [0, 1, 2, 3, 4]))
    balanced = [x for x in summaries if x["complete_cells"] == 10 and x["macro_delta_q"] >= .020 and
                x["worst_dataset_mean_delta_q"] >= 0 and x["paired_q_wins"] >= 7 and
                x["spatial_gate_pass_both"]]
    accuracy = [x for x in summaries if x["complete_cells"] == 10 and x["macro_delta_q"] >= .030 and
                x["worst_dataset_mean_delta_q"] >= -.005 and x["paired_q_wins"] >= 7]
    key = lambda x: (-x["macro_delta_q"], x["delta_q_std"], x["mean_effective_runtime_seconds"], x["graph_id"], x["head_id"])
    balanced.sort(key=key); accuracy.sort(key=key)
    balanced_locked = balanced[0] if balanced else None
    accuracy_locked = accuracy[0] if accuracy else None
    if accuracy_locked and not accuracy_locked["spatial_gate_pass_both"]:
        accuracy_locked = dict(accuracy_locked)
        accuracy_locked["designation"] = "ACCURACY_FRONTIER_SPATIAL_TRADEOFF"
    terminal = "NIGHT6C_BALANCED_AND_OR_ACCURACY_CANDIDATES_LOCKED" if (balanced_locked or accuracy_locked) else "NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE"
    compact = []
    for row in summaries:
        compact.append({k: v for k, v in row.items() if k != "cells"})
    pd.DataFrame([{k: v for k, v in x.items() if k not in {"cells", "dataset_mean_delta_q", "spatial_by_dataset"}}
                  for x in summaries]).to_csv(OUT / "graph_head_five_seed_summary.csv", index=False)
    frontier_rows = []
    for track, rows in (("balanced", balanced), ("accuracy_frontier", accuracy)):
        for rank, row in enumerate(rows, 1):
            frontier_rows.append({"track": track, "rank": rank, "locked": rank == 1,
                                  "graph_id": row["graph_id"], "head_id": row["head_id"],
                                  "macro_delta_q": row["macro_delta_q"],
                                  "worst_dataset_mean_delta_q": row["worst_dataset_mean_delta_q"],
                                  "paired_q_wins": row["paired_q_wins"],
                                  "spatial_gate_pass_both": row["spatial_gate_pass_both"],
                                  "designation": "BALANCED" if track == "balanced" else
                                      ("ACCURACY_FRONTIER" if row["spatial_gate_pass_both"] else "ACCURACY_FRONTIER_SPATIAL_TRADEOFF")})
    pd.DataFrame(frontier_rows, columns=["track","rank","locked","graph_id","head_id","macro_delta_q",
                                        "worst_dataset_mean_delta_q","paired_q_wins","spatial_gate_pass_both","designation"]).to_csv(
                                            OUT / "balanced_and_accuracy_frontiers.csv", index=False)
    return {"stage": "R2", "status": "LOCKED", "terminal_status": terminal,
            "candidate_summaries": compact, "locked_balanced_candidate": balanced_locked,
            "locked_accuracy_frontier_candidate": accuracy_locked,
            "reference_five_seed_complete": True, "label_access_after_output_lock": True,
            "parameter_tuning": False, "seed_search": False, "protected_dataset_access": False}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=("R1", "R2"), required=True)
    args = ap.parse_args()
    registry = json.loads(REG_PATH.read_text())
    frame = stage_evaluate(args.stage, registry)
    if args.stage == "R1":
        decision = r1_decision(frame, registry)
        atomic_json(OUT / "r1_decision.json", decision)
    else:
        historical_drift(frame)
        decision = r2_decision(frame, registry)
        atomic_json(OUT / "r2_decision.json", decision)
    print(json.dumps({"stage": args.stage, "status": decision["status"],
                      "advanced_graphs": decision.get("advanced_graphs", []),
                      "advanced_heads": decision.get("advanced_heads", []),
                      "terminal_status": decision.get("terminal_status")}, sort_keys=True))


if __name__ == "__main__":
    main()
