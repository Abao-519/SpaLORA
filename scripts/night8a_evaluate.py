#!/usr/bin/env python3
"""Single-window post-lock evaluator for Night-8A development stages."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score,
    completeness_score, davies_bouldin_score, fowlkes_mallows_score,
    homogeneity_score, normalized_mutual_info_score, silhouette_score,
    v_measure_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night8a_mfspc import array_sha, file_sha  # noqa: E402
from SpaLORA.night1_evaluation import _mean_cluster_moran  # noqa: E402
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency  # noqa: E402

OUT = REPO / "outputs/night8a_handoff"
REGISTRY = REPO / "protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json"
LABEL_ROOT = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots")
SOURCE_INDEX = REPO / "outputs/night7b_handoff/source_unit_index.csv"
CONFIGS = {}
COORDINATE_PATHS = {
    "a1": Path("/root/autodl-fs/night6c_cache_20260817/base/a1/coordinates.npy"),
    "tonsil": Path("/root/autodl-fs/night6c_cache_20260817/base/tonsil/coordinates.npy"),
    "d1": Path("/root/autodl-fs/night6d_cache_20260817/base/d1/coordinates.npy"),
    "p22": Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22/coordinates.npy"),
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify_stage(stage: str) -> dict:
    path = OUT / f"{stage.lower()}_stage_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["status"] != "LOCKED_PRE_LABEL" or not manifest["locked_before_label_access"] or manifest["label_access"]:
        raise RuntimeError(f"{stage} not locked before label access")
    if manifest["terminal_cells"] != manifest["planned_registered_cells"]:
        raise RuntimeError(f"{stage} coverage incomplete")
    for row in manifest["cells"]:
        if row["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL":
            if file_sha(row["alias_target_transform_manifest"]) != row["alias_target_transform_manifest_sha256"]:
                raise RuntimeError("alias transform manifest SHA mismatch")
            continue
        if row.get("transform_status") != "SUCCESS_PRE_LABEL":
            continue
        tm = row["transform_manifest"]; directory = Path(row["transform_dir"])
        if file_sha(directory / "clusters.csv") != tm["cluster_file_sha256"]:
            raise RuntimeError("cluster SHA mismatch before evaluator")
        training = row["training_manifest"]
        for artifact in training["artifacts"].values():
            if file_sha(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("training artifact SHA mismatch before evaluator")
    return manifest


def load_labels() -> tuple[dict, dict]:
    labels = {}; audit = {}
    for dataset in ("a1", "tonsil", "d1", "p22"):
        path = LABEL_ROOT / f"{dataset}_labels_locked.npz"
        observed = file_sha(path); value = np.load(path, allow_pickle=False)
        ids = value["observation_id"].astype(str); y = value["label"].astype(str)
        labels[dataset] = (ids, y)
        audit[dataset] = {"authorized_role": "night8a_single_evaluator", "snapshot_path": str(path),
                          "snapshot_sha256": observed, "rows": len(ids), "K": len(np.unique(y)),
                          "label_vector_sha256": array_sha(y), "used_for_training_or_transform": False}
    return labels, audit


def source_map() -> dict:
    return {x["unit_id"]: x for x in csv.DictReader(SOURCE_INDEX.open(newline=""))}


def no_diag(value: sp.spmatrix) -> sp.csr_matrix:
    out = value.tocsr().astype(np.float64); out.setdiag(0); out.eliminate_zeros(); out.sum_duplicates(); out.sort_indices()
    return out


def spatial_metrics(labels: np.ndarray, adjacency: sp.spmatrix) -> dict:
    a = no_diag(adjacency); rows, cols = a.nonzero()
    neighbor = float(np.mean(labels[rows] == labels[cols]))
    geary, _ = mean_one_vs_rest_geary(labels, a)
    return {"neighbor_agreement": neighbor, "moran_i": float(_mean_cluster_moran(labels, a)),
            "geary_c": float(geary), "boundary_disagreement": 1.0 - neighbor}


def graph_modularity(labels: np.ndarray, adjacency: sp.spmatrix) -> float:
    a = no_diag(adjacency); degree = np.asarray(a.sum(axis=1)).ravel(); total = degree.sum()
    if total <= 0:
        return float("nan")
    result = 0.0
    for group in np.unique(labels):
        idx = np.flatnonzero(labels == group); internal = float(a[idx][:, idx].sum())
        volume = float(degree[idx].sum()); result += internal / total - (volume / total) ** 2
    return float(result)


def label_free_metrics(embedding: np.ndarray, clusters: np.ndarray, adjacency: sp.spmatrix) -> dict:
    x = np.asarray(embedding, dtype=np.float64)
    unique = np.unique(clusters)
    if len(unique) < 2:
        return {k: float("nan") for k in ("silhouette", "davies_bouldin", "calinski_harabasz",
                                            "graph_connectivity", "modularity")}
    n = len(x); sample = min(n, 2000)
    silhouette = silhouette_score(x, clusters, metric="euclidean", sample_size=sample, random_state=20260820)
    return {"silhouette": float(silhouette), "davies_bouldin": float(davies_bouldin_score(x, clusters)),
            "calinski_harabasz": float(calinski_harabasz_score(x, clusters)),
            "graph_connectivity": spatial_metrics(clusters, adjacency)["neighbor_agreement"],
            "modularity": graph_modularity(clusters, adjacency)}


def evaluate_stage(stage: str, manifest: dict, labels: dict, sources: dict) -> pd.DataFrame:
    rows = []
    for cell in manifest["cells"]:
        base = {"stage": stage, "config_id": cell["config_id"], "dataset": cell["dataset"],
                "unit_id": cell["unit_id"], "seed": int(cell["seed"]), "status": cell["status"]}
        transform_status = cell.get("transform_status", cell["status"])
        if transform_status not in {"SUCCESS_PRE_LABEL", "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL"}:
            rows.append({**base, "evaluation_status": "INELIGIBLE_UPSTREAM_FAILURE"}); continue
        transform_dir = Path(cell["transform_dir"])
        clusters_table = pd.read_csv(transform_dir / "clusters.csv")
        label_ids, y = labels[cell["dataset"]]
        ids = clusters_table["observation_id"].astype(str).to_numpy()
        if not np.array_equal(ids, label_ids):
            raise RuntimeError(f"label alignment mismatch: {cell['unit_id']}")
        pred = clusters_table["cluster"].to_numpy(dtype=np.int64)
        if len(np.unique(pred)) != len(np.unique(y)):
            raise RuntimeError("locked K mismatch")
        primary = {"ari": adjusted_rand_score(y, pred), "nmi": normalized_mutual_info_score(y, pred)}
        primary["q"] = .5 * (primary["ari"] + primary["nmi"])
        secondary = {"ami": adjusted_mutual_info_score(y, pred), "fmi": fowlkes_mallows_score(y, pred),
                     "homogeneity": homogeneity_score(y, pred), "completeness": completeness_score(y, pred),
                     "v_measure": v_measure_score(y, pred)}
        coords = np.load(COORDINATE_PATHS[cell["dataset"]], allow_pickle=False)
        adjacency = symmetric_knn_adjacency(coords, 18)
        spatial = spatial_metrics(pred, adjacency)
        training_dir = Path(cell["root"]) / "training"
        if cell["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL":
            target = Path(cell["alias_target_training_manifest"]).parent
            embedding_path = target / "embeddings.npz"
        else:
            embedding_path = training_dir / "embeddings.npz"
        embedding = np.load(embedding_path, allow_pickle=False)["SpaLORA_fused"]
        descriptive = label_free_metrics(embedding, pred, adjacency)
        rows.append({**base, "evaluation_status": "SUCCESS", **primary, **spatial, **secondary, **descriptive,
                     "cluster_file_sha256": file_sha(transform_dir / "clusters.csv")})
    return pd.DataFrame(rows)


def historical_references() -> pd.DataFrame:
    n7 = pd.read_csv(REPO / "outputs/night7a_handoff/per_seed_metrics.csv")
    protein = n7[n7.candidate_id == "C00_G04_H05_CONFIRMED"].copy()
    protein = protein[protein.dataset.isin(["a1", "tonsil", "d1"])]
    protein["reference_id"] = "C00_G04_H05_CONFIRMED"
    n7b = pd.read_csv(REPO / "outputs/night7b_handoff/R2_full_per_seed_metrics.csv")
    epi = n7b[(n7b.config_id == "R02__E1_ADAPTER_C06_MEAN__H01") & (n7b.dataset == "p22")].copy()
    epi["reference_id"] = "R02_P22_FRONTIER_DEVELOPMENT_REFERENCE"
    cols = ["dataset", "seed", "reference_id", "ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement"]
    result = pd.concat((protein[cols], epi[cols]), ignore_index=True)
    if len(result) != 30 or result.duplicated(["dataset", "seed"]).any():
        raise RuntimeError("historical family reference coverage mismatch")
    return result


def add_deltas(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(reference, on=["dataset", "seed"], how="left", suffixes=("", "_reference"), validate="many_to_one")
    if merged.reference_id.isna().any():
        raise RuntimeError("missing family reference")
    for metric in ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement"):
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_reference"]
    return merged


def spatial_pass(group: pd.DataFrame) -> tuple[bool, list[str]]:
    failures = []
    for dataset, rows in group.groupby("dataset"):
        dn = rows.delta_neighbor_agreement.mean(); dm = rows.delta_moran_i.mean()
        dg = rows.delta_geary_c.mean(); db = rows.delta_boundary_disagreement.mean()
        if (dn < -.03 and dm < -.03) or (dg > .03 and db > .03):
            failures.append(str(dataset))
    return not failures, failures


def summaries(frame: pd.DataFrame, registry: dict) -> pd.DataFrame:
    output = []
    complexity = {x["id"]: len(x["modules"]) for x in registry["R2_configs"]}
    for config_id, rows in frame.groupby("config_id"):
        success = rows[rows.evaluation_status == "SUCCESS"]
        item = {"config_id": config_id, "success_cells": len(success), "failure_cells": len(rows) - len(success),
                "complexity": complexity.get(config_id, 99)}
        for dataset in ("a1", "tonsil", "d1", "p22"):
            part = success[success.dataset == dataset]
            for metric in ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement"):
                item[f"{dataset}_mean_{metric}"] = float(part[metric].mean())
                item[f"{dataset}_mean_delta_{metric}"] = float(part[f"delta_{metric}"].mean())
            item[f"{dataset}_q_wins"] = int((part.delta_q > 0).sum())
        item["Q_HLN"] = .5 * (item["a1_mean_q"] + item["d1_mean_q"])
        item["delta_Q_HLN"] = .5 * (item["a1_mean_delta_q"] + item["d1_mean_delta_q"])
        item["priority_macro_Q"] = .45 * item["Q_HLN"] + .45 * item["p22_mean_q"] + .10 * item["tonsil_mean_q"]
        item["priority_macro_delta_Q"] = .45 * item["delta_Q_HLN"] + .45 * item["p22_mean_delta_q"] + .10 * item["tonsil_mean_delta_q"]
        item["important_worst_delta_Q"] = min(item["delta_Q_HLN"], item["p22_mean_delta_q"])
        item["paired_q_wins"] = int((success.delta_q > 0).sum())
        item["spatial_protection_pass"], item["spatial_failure_datasets"] = spatial_pass(success)
        item["complete"] = len(success) == len(rows)
        output.append(item)
    return pd.DataFrame(output).sort_values("config_id").reset_index(drop=True)


def rank_rows(frame: pd.DataFrame, primary: str) -> list[str]:
    columns = []
    for name in (primary, "priority_macro_delta_Q", "important_worst_delta_Q", "paired_q_wins", "complexity", "config_id"):
        if name not in columns:
            columns.append(name)
    ascending = [name in {"complexity", "config_id"} for name in columns]
    return frame.sort_values(columns, ascending=ascending).config_id.tolist()


def shortlist(summary: pd.DataFrame) -> tuple[list[str], dict]:
    eligible = summary[(summary.complete) & (summary.spatial_protection_pass)]
    unified = eligible[(eligible.delta_Q_HLN >= 0) & (eligible.p22_mean_delta_q >= 0) &
                       (eligible.tonsil_mean_delta_q >= -.01)]
    hln = eligible[eligible.p22_mean_delta_q >= -.01]
    p22 = eligible[eligible.delta_Q_HLN >= -.01]
    slots = {
        "unified_balanced": rank_rows(unified, "priority_macro_delta_Q")[0] if len(unified) else None,
        "human_lymph_frontier": rank_rows(hln, "Q_HLN")[0] if len(hln) else None,
        "P22_frontier": rank_rows(p22, "p22_mean_q")[0] if len(p22) else None,
    }
    ids = []
    for value in slots.values():
        if value is not None and value not in ids:
            ids.append(value)
    return ids[:3], slots


def bootstrap(values: np.ndarray, seed: int = 20260820, reps: int = 100000) -> dict:
    values = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(seed)
    chunk = 10000; samples = []
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start); idx = rng.integers(0, len(values), size=(count, len(values)))
        samples.append(values[idx].mean(1))
    result = np.concatenate(samples)
    return {"mean": float(values.mean()), "ci_lower": float(np.quantile(result, .025)),
            "ci_upper": float(np.quantile(result, .975)), "repetitions": reps, "seed": seed}


def holm(raw: dict[str, float]) -> dict[str, float]:
    order = sorted(raw, key=raw.get); n = len(order); adjusted = {}; running = 0.0
    for i, key in enumerate(order):
        running = max(running, (n - i) * raw[key]); adjusted[key] = min(1.0, running)
    return adjusted


def statistics(frame: pd.DataFrame) -> dict:
    raw_p = {}; detail = {}
    for (config_id, dataset), rows in frame.groupby(["config_id", "dataset"]):
        values = rows.delta_q.to_numpy(dtype=float); nz = values[values != 0]
        sign = 1.0 if len(nz) == 0 else float(binomtest(int((nz > 0).sum()), len(nz), .5, alternative="two-sided").pvalue)
        try:
            wp = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            wp = 1.0
        key = f"{config_id}|{dataset}"; raw_p[key] = sign
        detail[key] = {"n": len(values), "wins": int((values > 0).sum()), "losses": int((values < 0).sum()),
                       "zeros": int((values == 0).sum()), "exact_sign_p": sign, "wilcoxon_p": wp,
                       "bootstrap_Q": bootstrap(values)}
    adjusted = holm(raw_p)
    for key in detail: detail[key]["holm_exact_sign_p"] = adjusted[key]
    return {"zero_rule": "exact zeros excluded from sign-test n; retained as zero in bootstrap and Wilcoxon",
            "holm_family": "all finalist-by-dataset primary Q sign tests", "tests": detail}


def pareto(summary: pd.DataFrame) -> list[str]:
    result = []
    values = summary[["config_id", "delta_Q_HLN", "p22_mean_delta_q", "tonsil_mean_delta_q", "priority_macro_delta_Q"]].to_dict("records")
    for row in values:
        dominated = any(other["config_id"] != row["config_id"] and
                        other["delta_Q_HLN"] >= row["delta_Q_HLN"] and
                        other["p22_mean_delta_q"] >= row["p22_mean_delta_q"] and
                        other["tonsil_mean_delta_q"] >= row["tonsil_mean_delta_q"] and
                        (other["delta_Q_HLN"] > row["delta_Q_HLN"] or other["p22_mean_delta_q"] > row["p22_mean_delta_q"] or
                         other["tonsil_mean_delta_q"] > row["tonsil_mean_delta_q"])
                        for other in values)
        if not dominated: result.append(row["config_id"])
    return sorted(result)


def window1(registry: dict) -> None:
    r1m = verify_stage("R1"); r2m = verify_stage("R2")
    labels, label_audit = load_labels(); sources = source_map(); reference = historical_references()
    r1 = add_deltas(evaluate_stage("R1", r1m, labels, sources), reference)
    r2 = add_deltas(evaluate_stage("R2", r2m, labels, sources), reference)
    baseline = r2[r2.config_id == "B00_FAMILY_REFERENCE"]
    parity_error = float(np.nanmax(np.abs(baseline[["delta_ari", "delta_nmi", "delta_q"]].to_numpy(dtype=float))))
    if len(baseline) != 12 or parity_error > 1e-12:
        raise RuntimeError(f"BLOCKED_HISTORICAL_RECOMPUTE: B00 family reference parity {parity_error}")
    r1.to_csv(OUT / "r1_per_seed_metrics.csv", index=False); r2.to_csv(OUT / "r2_per_seed_metrics.csv", index=False)
    summary = summaries(r2, registry); summary.to_csv(OUT / "r2_candidate_summary.csv", index=False)
    ids, slots = shortlist(summary)
    metrics_sha = file_sha(OUT / "r2_per_seed_metrics.csv")
    shortlist_payload = {"status": "LOCKED", "finalist_ids": ids, "source_metrics_sha256": metrics_sha,
                         "label_window_closed": True}
    atomic_json(OUT / "r2_shortlist_ids.json", shortlist_payload)
    atomic_json(OUT / "dev_window_1_audit.json", {
        "status": "CLOSED", "authorized_role": "single_evaluator", "labels_opened_after_R1_R2_total_lock": True,
        "label_audit": label_audit, "slots": slots, "finalist_ids": ids,
        "B00_historical_reference_parity_max_abs_error": parity_error,
        "configuration_or_code_changes_after_labels": False, "r3_trainer_receives_only_shortlist_ids": True,
        "r1_metrics_sha256": file_sha(OUT / "r1_per_seed_metrics.csv"),
        "r2_metrics_sha256": metrics_sha, "shortlist_sha256": file_sha(OUT / "r2_shortlist_ids.json")})
    print(json.dumps({"event": "DEV_WINDOW_1_CLOSED", "finalist_ids": ids, "slots": slots}, sort_keys=True))


def window2(registry: dict) -> None:
    r2m = verify_stage("R2"); r3m = verify_stage("R3")
    labels, label_audit = load_labels(); sources = source_map(); reference = historical_references()
    r2 = add_deltas(evaluate_stage("R2", r2m, labels, sources), reference)
    r3 = add_deltas(evaluate_stage("R3", r3m, labels, sources), reference)
    finalists = json.loads((OUT / "r2_shortlist_ids.json").read_text())["finalist_ids"]
    full = pd.concat((r2[r2.config_id.isin(finalists)], r3), ignore_index=True)
    if full.duplicated(["config_id", "dataset", "seed"]).any():
        raise RuntimeError("duplicate final metric key")
    full.to_csv(OUT / "final_per_seed_metrics.csv", index=False)
    summary = summaries(full, registry); summary.to_csv(OUT / "final_candidate_summary.csv", index=False)
    stats = statistics(full); atomic_json(OUT / "final_statistics.json", stats)
    decisions = []
    for row in summary.to_dict("records"):
        balanced = bool(row["priority_macro_delta_Q"] >= .010 and row["delta_Q_HLN"] >= 0 and
                        row["p22_mean_delta_q"] >= 0 and row["spatial_protection_pass"])
        decisions.append({"config_id": row["config_id"], "balanced_material_gate": balanced,
                          "human_lymph_frontier": bool(row["p22_mean_delta_q"] >= -.01 and row["spatial_protection_pass"]),
                          "P22_frontier": bool(row["delta_Q_HLN"] >= -.01 and row["spatial_protection_pass"]),
                          "priority_macro_delta_Q": row["priority_macro_delta_Q"], "delta_Q_HLN": row["delta_Q_HLN"],
                          "delta_Q_P22": row["p22_mean_delta_q"], "delta_Q_tonsil": row["tonsil_mean_delta_q"]})
    balanced = [x["config_id"] for x in decisions if x["balanced_material_gate"]]
    decision = {"status": "NIGHT8A_DEVELOPMENT_COMPLETE", "balanced_material_winners": balanced,
                "unified_winner": None if not balanced else rank_rows(summary[summary.config_id.isin(balanced)], "priority_macro_delta_Q")[0],
                "decisions": decisions, "pareto_frontier": pareto(summary),
                "external_confirmation_required": True, "R02_remains_development_reference": True,
                "labels_closed_after_window": True}
    atomic_json(OUT / "night8a_decision.json", decision)
    atomic_json(OUT / "pareto_frontier.json", {"status": "LOCKED", "candidate_ids": decision["pareto_frontier"]})
    atomic_json(OUT / "dev_window_2_audit.json", {
        "status": "CLOSED", "authorized_role": "single_evaluator", "labels_opened_after_R3_total_lock": True,
        "label_audit": label_audit, "configuration_or_code_changes_after_labels": False,
        "final_metrics_sha256": file_sha(OUT / "final_per_seed_metrics.csv"),
        "statistics_sha256": file_sha(OUT / "final_statistics.json"),
        "decision_sha256": file_sha(OUT / "night8a_decision.json")})
    print(json.dumps({"event": "DEV_WINDOW_2_CLOSED", "decision": decision}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--window", choices=("1", "2"), required=True)
    registry = json.loads(REGISTRY.read_text())
    global CONFIGS; CONFIGS = {x["id"]: x for x in registry["R2_configs"]}
    if ap.parse_args().window == "1": window1(registry)
    else: window2(registry)


if __name__ == "__main__":
    main()
