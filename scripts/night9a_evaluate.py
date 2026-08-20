#!/usr/bin/env python3
"""Stage-locked P22 evaluator and label-free MISAR fidelity selector for Night-9A.

This is the only Night-9A file allowed to read the already locked P22 label
snapshot.  It has no MISAR annotation path and never opens a MISAR HDF5 file.
"""
from __future__ import annotations

import argparse
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

from SpaLORA.night1_evaluation import _mean_cluster_moran
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency
from SpaLORA.night6c_pipeline import array_sha
from SpaLORA.night7a_consensus import canonical_partition
from SpaLORA.night9a_efficient import sha256_file


RAW = Path("/root/autodl-fs/night9a_efficient_topology_transfer_20260820")
OUT = REPO / "outputs/night9a"
REGISTRY = REPO / "protocols/night9a/SpaLORA_Night9A_Efficient_Topology_Transfer_Registry_2026-08-20.json"
LABEL = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots/p22_labels_locked.npz")
LABEL_AUTHORITY = REPO / "outputs/night7a_handoff/label_window_audit.json"
SPATIAL_THRESHOLDS = {"neighbor_min": -0.01, "moran_min": -0.02,
                      "geary_max": 0.02, "boundary_max": 0.01}


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prelabel_push_gate(stage: str) -> dict:
    lock = OUT / f"locked_{stage}_manifest.json"
    if not lock.is_file():
        raise RuntimeError("stage lock missing")
    if git("status", "--porcelain", "--", str(lock.relative_to(REPO))):
        raise RuntimeError("stage lock is not committed")
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{u}")
    if head != upstream:
        raise RuntimeError("stage lock has not been pushed before P22 label access")
    payload = {"status": "PASS", "stage": stage, "head": head, "upstream": upstream,
               "locked_manifest_sha256": sha256_file(lock),
               "p22_label_access_before": 0, "misar_Y_access": 0}
    atomic_json(OUT / f"{stage}_prelabel_push_gate.json", payload)
    return payload


def load_p22_labels() -> tuple[np.ndarray, np.ndarray, dict]:
    authority = json.loads(LABEL_AUTHORITY.read_text())["datasets"]["p22"]
    digest = sha256_file(LABEL)
    if digest != authority["snapshot_sha256"]:
        raise RuntimeError("P22 locked label snapshot SHA mismatch")
    with np.load(LABEL, allow_pickle=False) as value:
        if set(value.files) != {"observation_id", "label"}:
            raise RuntimeError("P22 locked label snapshot schema mismatch")
        ids = np.asarray(value["observation_id"]).astype(str)
        labels = np.asarray(value["label"]).astype(str)
    if len(ids) != authority["aligned_rows"] or len(np.unique(labels)) != authority["known_k"]:
        raise RuntimeError("P22 locked label snapshot cardinality mismatch")
    if array_sha(labels) != authority["ordered_label_vector_sha256"]:
        raise RuntimeError("P22 ordered label vector SHA mismatch")
    return ids, labels, {"snapshot_sha256": digest, "rows": len(ids),
                         "reference_K": len(np.unique(labels)),
                         "authorized_role": "night9a_stage_evaluator"}


def load_clusters(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path)
    ids = table["observation_id"].astype(str).to_numpy()
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError(f"cluster observation order mismatch: {path}")
    return canonical_partition(table["cluster"].to_numpy())


def independent_ari_nmi(true, pred) -> tuple[float, float]:
    _, ti = np.unique(np.asarray(true), return_inverse=True)
    _, pi = np.unique(np.asarray(pred), return_inverse=True)
    table = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(table, (ti, pi), 1)
    n = int(table.sum())
    comb = lambda x: x * (x - 1.0) / 2.0
    sum_cells = float(comb(table).sum())
    sum_rows = float(comb(table.sum(axis=1)).sum())
    sum_cols = float(comb(table.sum(axis=0)).sum())
    total = float(comb(n))
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    ari = (sum_cells - expected) / (maximum - expected)
    pxy = table / n; px = pxy.sum(axis=1); py = pxy.sum(axis=0)
    rows, cols = np.nonzero(table)
    mi = float(sum(pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
                   for i, j in zip(rows, cols)))
    hx = float(-sum(value * math.log(value) for value in px if value > 0))
    hy = float(-sum(value * math.log(value) for value in py if value > 0))
    nmi = mi / ((hx + hy) / 2.0)
    return float(ari), float(nmi)


def metric(true, pred, graph) -> tuple[dict, dict]:
    rows, cols = graph.nonzero()
    neighbor = float(np.mean(pred[rows] == pred[cols]))
    moran = float(_mean_cluster_moran(pred, graph))
    geary, _ = mean_one_vs_rest_geary(pred, graph)
    ari = float(adjusted_rand_score(true, pred))
    nmi = float(normalized_mutual_info_score(true, pred))
    independent_ari, independent_nmi = independent_ari_nmi(true, pred)
    audit = {"ari_abs_error": abs(ari - independent_ari),
             "nmi_abs_error": abs(nmi - independent_nmi)}
    if max(audit.values()) > 1e-12:
        raise RuntimeError(f"independent contingency evaluator mismatch: {audit}")
    return {"ari": ari, "nmi": nmi, "q": (ari + nmi) / 2.0,
            "neighbor_agreement": neighbor, "moran_i": moran,
            "geary_c": float(geary), "boundary_disagreement": 1.0 - neighbor}, audit


def stage_seeds(stage: str) -> list[int]:
    return {"R1": [0, 1, 2], "R2": [3, 4], "R3": [5, 6, 7, 8, 9]}[stage]


def candidate_ids(stage: str) -> list[str]:
    registry = json.loads(REGISTRY.read_text())
    if stage == "R1": return [row["id"] for row in registry["candidates"]]
    if stage == "R2": return json.loads((OUT / "R1_shortlist.json").read_text())["shortlist_candidate_ids"]
    return [json.loads((OUT / "night9a_final_candidate_lock.json").read_text())["final_candidate_id"]]


def stage_rows(stage: str, ids: np.ndarray, true: np.ndarray, graph) -> tuple[list[dict], list[dict]]:
    rows = []; independent = []
    lock = json.loads((OUT / f"locked_{stage}_manifest.json").read_text())
    by_key = {(row.get("candidate_id"), row.get("dataset"), int(row.get("seed", -1))): row
              for row in lock["rows"]}
    for seed in stage_seeds(stage):
        for reference in ("U00", "FULL_F00"):
            path = RAW / f"p0/references/p22/seed_{seed}/{reference}/clusters.csv"
            pred = load_clusters(path, ids)
            values, audit = metric(true, pred, graph)
            rows.append({"stage": stage, "candidate_id": reference, "dataset": "p22",
                         "seed": seed, "status": "success", **values})
            independent.append({"stage": stage, "candidate_id": reference,
                                "seed": seed, **audit})
        for candidate in candidate_ids(stage):
            manifest = by_key.get((candidate, "p22", seed), {})
            status = manifest.get("status", "MISSING")
            row = {"stage": stage, "candidate_id": candidate, "dataset": "p22",
                   "seed": seed, "status": status}
            if status == "success":
                pred = load_clusters(Path(manifest["artifacts"]["clusters"]["path"]), ids)
                values, audit = metric(true, pred, graph)
                row.update(values); independent.append({"stage": stage,
                                                         "candidate_id": candidate,
                                                         "seed": seed, **audit})
            else:
                row.update({key: np.nan for key in ("ari", "nmi", "q", "neighbor_agreement",
                                                    "moran_i", "geary_c", "boundary_disagreement")})
            rows.append(row)
    return rows, independent


def aggregate_candidate(candidate: str, stages: list[str]) -> dict:
    metrics = pd.concat([pd.read_csv(OUT / f"{stage}_p22_per_seed_metrics.csv")
                         for stage in stages], ignore_index=True)
    candidate_rows = metrics[metrics.candidate_id == candidate].sort_values("seed")
    full = metrics[metrics.candidate_id == "FULL_F00"].set_index("seed")
    u00 = metrics[metrics.candidate_id == "U00"].set_index("seed")
    expected_seeds = sorted(set(sum((stage_seeds(stage) for stage in stages), [])))
    complete = (len(candidate_rows) == len(expected_seeds) and
                candidate_rows.status.eq("success").all() and
                candidate_rows.seed.tolist() == expected_seeds)
    lock_rows = []
    for stage in stages:
        lock = json.loads((OUT / f"locked_{stage}_manifest.json").read_text())
        lock_rows.extend(row for row in lock["rows"] if row.get("candidate_id") == candidate)
    expected_cells = 2 * len(expected_seeds)
    complete = complete and len(lock_rows) == expected_cells and all(row.get("status") == "success" for row in lock_rows)
    result = {"candidate_id": candidate, "stages": stages, "seeds": expected_seeds,
              "complete": bool(complete), "expected_cells": expected_cells,
              "observed_cells": len(lock_rows)}
    if not complete:
        result.update({"all_hard_gates_pass": False, "gate_failures": ["incomplete_candidate"]})
        return result
    p22 = candidate_rows.set_index("seed")
    for name in ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement"):
        result[f"p22_mean_{name}"] = float(p22[name].mean())
        result[f"p22_mean_delta_{name}_vs_full_f00"] = float((p22[name] - full.loc[expected_seeds, name]).mean())
        result[f"p22_mean_delta_{name}_vs_u00"] = float((p22[name] - u00.loc[expected_seeds, name]).mean())
        result[f"p22_wins_{name}_vs_u00"] = int(np.sum(p22[name] > u00.loc[expected_seeds, name]))
    result["p22_full_f00_mean_q"] = float(full.loc[expected_seeds, "q"].mean())
    # Label-free fidelity/resource rows are independently loaded from locked chain manifests.
    misar = [row for row in lock_rows if row["dataset"] == "misar"]
    fidelities = [row["fidelity_vs_full_f00"] for row in misar]
    result.update({
        "misar_mean_partition_ari": float(np.mean([x["partition_ari"] for x in fidelities])),
        "misar_min_partition_ari": float(np.min([x["partition_ari"] for x in fidelities])),
        "misar_mean_partition_nmi": float(np.mean([x["partition_nmi"] for x in fidelities])),
        "misar_exact_partition_count": int(sum(x["partition_exact_up_to_permutation"] for x in fidelities)),
    })
    cand_e2e = np.asarray([row["resource"]["candidate_end_to_end_seconds"] for row in lock_rows])
    u00_e2e = np.asarray([row["resource"]["u00_end_to_end_seconds"] for row in lock_rows])
    cand_gpu = np.asarray([row["resource"]["candidate_peak_gpu_mib"] for row in lock_rows])
    u00_gpu = np.asarray([row["resource"]["u00_peak_gpu_mib"] for row in lock_rows])
    result.update({
        "mean_candidate_end_to_end_seconds": float(cand_e2e.mean()),
        "mean_u00_end_to_end_seconds": float(u00_e2e.mean()),
        "runtime_ratio_vs_u00": float(cand_e2e.mean() / u00_e2e.mean()),
        "mean_candidate_peak_gpu_mib": float(cand_gpu.mean()),
        "mean_u00_peak_gpu_mib": float(u00_gpu.mean()),
        "peak_gpu_ratio_vs_u00": float(cand_gpu.mean() / max(u00_gpu.mean(), 1e-12)),
    })
    spatial = {
        "neighbor": result["p22_mean_delta_neighbor_agreement_vs_full_f00"] >= SPATIAL_THRESHOLDS["neighbor_min"],
        "moran": result["p22_mean_delta_moran_i_vs_full_f00"] >= SPATIAL_THRESHOLDS["moran_min"],
        "geary": result["p22_mean_delta_geary_c_vs_full_f00"] <= SPATIAL_THRESHOLDS["geary_max"],
        "boundary": result["p22_mean_delta_boundary_disagreement_vs_full_f00"] <= SPATIAL_THRESHOLDS["boundary_max"],
    }
    result["spatial_protection"] = {"pass": all(spatial.values()), "components": spatial,
                                    "thresholds": SPATIAL_THRESHOLDS}
    registry = json.loads(REGISTRY.read_text())
    gates = registry["r1_r2_selection_gates"]
    checks = {
        "runtime": result["runtime_ratio_vs_u00"] <= gates["runtime_ratio_vs_u00_max"],
        "peak_gpu": result["peak_gpu_ratio_vs_u00"] <= gates["peak_gpu_ratio_vs_u00_max"],
        "p22_q_vs_full": result["p22_mean_delta_q_vs_full_f00"] >= gates["p22_mean_q_delta_vs_full_f00_min"],
        "p22_q_vs_u00": result["p22_mean_delta_q_vs_u00"] >= gates["p22_mean_q_delta_vs_u00_min"],
        "p22_ari_vs_full": result["p22_mean_delta_ari_vs_full_f00"] >= gates["p22_mean_ari_delta_vs_full_f00_min"],
        "p22_nmi_vs_full": result["p22_mean_delta_nmi_vs_full_f00"] >= gates["p22_mean_nmi_delta_vs_full_f00_min"],
        "p22_spatial": result["spatial_protection"]["pass"],
        "misar_mean_ari": result["misar_mean_partition_ari"] >= gates["misar_mean_partition_ari_vs_full_f00_min"],
        "misar_min_ari": result["misar_min_partition_ari"] >= gates["misar_min_partition_ari_vs_full_f00_min"],
        "misar_mean_nmi": result["misar_mean_partition_nmi"] >= gates["misar_mean_partition_nmi_vs_full_f00_min"],
    }
    result["hard_gates"] = checks
    result["all_hard_gates_pass"] = all(checks.values())
    result["gate_failures"] = [name for name, value in checks.items() if not value]
    p22_preservation = 1.0 + result["p22_mean_delta_q_vs_full_f00"] / max(
        abs(result["p22_full_f00_mean_q"]), 1e-12)
    result["worst_dataset_normalized_preservation"] = min(p22_preservation,
                                                            result["misar_mean_partition_ari"])
    return result


def paired_bootstrap(values: np.ndarray, repetitions: int = 100000) -> list[float]:
    rng = np.random.default_rng(20260820)
    n = len(values); chunk = 10000; estimates = []
    for start in range(0, repetitions, chunk):
        count = min(chunk, repetitions - start)
        indices = rng.integers(0, n, size=(count, n))
        estimates.append(values[indices].mean(axis=1))
    sample = np.concatenate(estimates)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def select(stage: str) -> dict:
    registry = json.loads(REGISTRY.read_text())
    candidates = candidate_ids(stage)
    stages = ["R1"] if stage == "R1" else (["R1", "R2"] if stage == "R2" else ["R1", "R2", "R3"])
    summary = [aggregate_candidate(candidate, stages) for candidate in candidates]
    order = {row["id"]: index for index, row in enumerate(registry["candidates"])}
    eligible = [row for row in summary if row.get("all_hard_gates_pass")]
    eligible.sort(key=lambda row: (-row["worst_dataset_normalized_preservation"],
                                  -row["misar_mean_partition_ari"],
                                  -row["p22_mean_q"], row["runtime_ratio_vs_u00"],
                                  order[row["candidate_id"]]))
    pd.DataFrame(summary).to_json(OUT / f"{stage}_candidate_summary.json", orient="records", indent=2)
    if stage == "R1":
        selected = [row["candidate_id"] for row in eligible[:3]]
        payload = {"status": "PASS" if selected else "NO_ELIGIBLE_CANDIDATE",
                   "shortlist_candidate_ids": selected, "selection_slots_max": 3,
                   "ranking_rule": registry["r1_r2_selection_gates"]["candidate_ranking_after_hard_gates"],
                   "summary": summary, "p22_label_access_this_stage": 1,
                   "misar_Y_access": 0, "lineage_raw_Y_read_count": 2}
        atomic_json(OUT / "R1_shortlist.json", payload)
        return payload
    if stage == "R2":
        selected = eligible[0]["candidate_id"] if eligible else None
        payload = {"status": "PASS" if selected else "NO_ELIGIBLE_CANDIDATE",
                   "final_candidate_id": selected, "selection_slots": 1,
                   "candidate_config": next((x for x in registry["candidates"] if x["id"] == selected), None),
                   "candidate_config_sha256": canonical_sha(next((x for x in registry["candidates"] if x["id"] == selected), {})),
                   "seeds_0_4_summary": summary, "future_R3_seeds": [5, 6, 7, 8, 9],
                   "p22_label_access_this_stage": 1, "misar_Y_access": 0,
                   "lineage_raw_Y_read_count": 2}
        atomic_json(OUT / "night9a_final_candidate_lock.json", payload)
        return payload
    final = summary[0]
    metrics = pd.concat([pd.read_csv(OUT / f"{value}_p22_per_seed_metrics.csv")
                         for value in ("R1", "R2", "R3")], ignore_index=True)
    candidate = final["candidate_id"]
    r3 = metrics[(metrics.candidate_id == candidate) & metrics.seed.isin([5,6,7,8,9])].set_index("seed")
    u00 = metrics[metrics.candidate_id == "U00"].set_index("seed")
    q_delta = r3["q"] - u00.loc[[5,6,7,8,9], "q"]
    confirmation = registry["r3_confirmation_gates"]
    r3_checks = {
        "runtime": final["runtime_ratio_vs_u00"] <= confirmation["runtime_ratio_vs_u00_max"],
        "peak_gpu": final["peak_gpu_ratio_vs_u00"] <= confirmation["peak_gpu_ratio_vs_u00_max"],
        "p22_q_vs_full": final["p22_mean_delta_q_vs_full_f00"] >= confirmation["p22_mean_q_delta_vs_full_f00_min"],
        "p22_q_vs_u00": final["p22_mean_delta_q_vs_u00"] >= confirmation["p22_mean_q_delta_vs_u00_min"],
        "p22_q_wins_vs_u00": int(np.sum(q_delta > 0)) >= confirmation["p22_q_wins_vs_u00_min"],
        "spatial": final["spatial_protection"]["pass"],
    }
    exact = final["misar_exact_partition_count"] == 10
    high = (final["misar_mean_partition_ari"] >= confirmation["misar_high_fidelity_mean_ari_min"] and
            final["misar_min_partition_ari"] >= confirmation["misar_high_fidelity_min_ari_min"])
    if all(r3_checks.values()) and exact:
        terminal = "NIGHT9A_BALANCED_EFFICIENT_R02_CONFIRMED_WITH_MISAR_EXACT_INHERITANCE"
    elif all(r3_checks.values()) and high:
        terminal = "NIGHT9A_EFFICIENT_R02_P22_CONFIRMED_MISAR_HIGH_FIDELITY"
    elif all(r3_checks.values()):
        terminal = "NIGHT9A_EFFICIENT_R02_LOCKED_FOR_NEW_EXTERNAL_VALIDATION"
    else:
        terminal = "NIGHT9A_PARTIAL_OR_MIXED_EVIDENCE"
    payload = {"terminal_status": terminal, "final_candidate_id": candidate,
               "full_10_seed_summary": final, "R3_confirmation_checks": r3_checks,
               "R3_q_delta_vs_u00": q_delta.tolist(),
               "R3_paired_bootstrap_delta_q_95CI": paired_bootstrap(q_delta.to_numpy()),
               "misar_exact_metric_inheritance_authorized": exact,
               "misar_high_fidelity": high, "misar_Y_access": 0,
               "lineage_raw_Y_read_count": 2, "claim_pristine_holdout": False,
               "claim_sota": False, "third_party_benchmark": 0}
    atomic_json(OUT / "night9a_decision.json", payload)
    return payload


def canonical_sha(value) -> str:
    import hashlib
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def evaluate(stage: str) -> None:
    gate = prelabel_push_gate(stage)
    ids, labels, access = load_p22_labels()
    coords = np.load(Path("/root/autodl-fs/night7b_score_rnd_20260818/source/u020/coordinates.npy"),
                     allow_pickle=False)
    source_ids = np.asarray([x.strip() for x in Path(
        "/root/autodl-fs/night7b_score_rnd_20260818/source/u020/observation_ids.txt"
    ).read_text().splitlines() if x.strip()])
    if not np.array_equal(ids, source_ids) or len(coords) != len(ids):
        raise RuntimeError("P22 coordinate/label order mismatch")
    graph = symmetric_knn_adjacency(coords, 18)
    rows, independent = stage_rows(stage, ids, labels, graph)
    pd.DataFrame(rows).to_csv(OUT / f"{stage}_p22_per_seed_metrics.csv", index=False)
    max_error = max(max(row["ari_abs_error"], row["nmi_abs_error"]) for row in independent)
    atomic_json(OUT / f"{stage}_independent_metric_recompute.json", {
        "status": "PASS" if max_error <= 1e-12 else "FAIL", "max_abs_error": max_error,
        "tolerance": 1e-12, "rows": independent,
    })
    payload = select(stage)
    atomic_json(OUT / f"{stage}_label_window_audit.json", {
        "status": "PASS", "stage": stage, "stage_lock_sha256": gate["locked_manifest_sha256"],
        "label_opened_only_after_stage_lock_and_push": True, "p22_label_access_this_stage": 1,
        "label_authority": access, "return_to_completed_stage_candidates": False,
        "misar_Y_access": 0, "lineage_raw_Y_read_count": 2,
        "selection_output_sha256": sha256_file(
            OUT / ("R1_shortlist.json" if stage == "R1" else
                   "night9a_final_candidate_lock.json" if stage == "R2" else "night9a_decision.json")),
    })
    print(json.dumps({"event": "stage_evaluated", "stage": stage,
                      "status": payload.get("status", payload.get("terminal_status"))}))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("R1", "R2", "R3"))
    evaluate(parser.parse_args().stage)


if __name__ == "__main__":
    main()
