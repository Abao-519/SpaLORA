#!/usr/bin/env python3
"""One-way post-lock evaluation for the Night-8B uniform-head recovery."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
    fowlkes_mallows_score, homogeneity_score, normalized_mutual_info_score,
    v_measure_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran
from SpaLORA.night3af_cache import load_cache, sha256_file
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency

ORIGINAL = Path("/root/autodl-fs/night8b_raw_runs_20260820")
RECOVERY = Path("/root/autodl-fs/night8b_head_recovery_20260820")
OUT = REPO / "outputs/night8b_head_recovery"
ORIGINAL_OUT = REPO / "outputs/night8b_handoff"
BRANCH = "revision/q2-night8b-uniform-head-recovery-20260820"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def metric(truth: np.ndarray, prediction: np.ndarray, graph: sp.csr_matrix) -> dict:
    row, col = graph.nonzero()
    ari = float(adjusted_rand_score(truth, prediction))
    nmi = float(normalized_mutual_info_score(truth, prediction))
    neighbor = float(np.mean(prediction[row] == prediction[col]))
    geary, _ = mean_one_vs_rest_geary(prediction, graph)
    return {
        "ari": ari, "nmi": nmi, "q": (ari + nmi) / 2.0,
        "ami": float(adjusted_mutual_info_score(truth, prediction)),
        "fmi": float(fowlkes_mallows_score(truth, prediction)),
        "homogeneity": float(homogeneity_score(truth, prediction)),
        "completeness": float(completeness_score(truth, prediction)),
        "v_measure": float(v_measure_score(truth, prediction)),
        "neighbor_agreement": neighbor,
        "moran_i": float(_mean_cluster_moran(prediction, graph)),
        "geary_c": float(geary), "boundary_disagreement": 1.0 - neighbor,
    }


def signflip(values: np.ndarray) -> dict:
    observed = float(values.mean())
    means = np.asarray([np.mean(values * np.asarray(signs, dtype=np.float64))
                        for signs in itertools.product((-1.0, 1.0), repeat=10)])
    tail = int(np.sum(means >= observed - 1e-15))
    return {"observed_mean": observed, "enumerations": 1024,
            "tail_count": tail, "p_one_sided": tail / 1024.0}


def bootstrap(values: np.ndarray) -> dict:
    rng = np.random.default_rng(20260820)
    indices = rng.integers(0, 10, size=(100000, 10))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {"replicates": 100000, "seed": 20260820,
            "mean": float(values.mean()), "ci_lower": float(lower),
            "ci_upper": float(upper)}


def read_partition(method: str, seed: int) -> tuple[np.ndarray, list[str]]:
    path = RECOVERY / f"partitions/{method}/seed_{seed}/clusters.csv"
    table = pd.read_csv(path)
    return table["cluster"].to_numpy(np.int64), table["observation_id"].astype(str).tolist()


def resource_components(seed: int, method: str, head_seconds: float) -> dict:
    base_path = ORIGINAL / f"formal/base/seed_{seed}/attempt_001/base_unit_manifest.json"
    base = json.loads(base_path.read_text())
    by_graph = {row["graph_id"]: row for row in base["submodels"]}
    g00 = by_graph["G00_SP18_F20_CORR_UNION"]
    g04 = by_graph["G04_SP10_F10_EUC_UNION"]
    adapter_path = ORIGINAL / f"formal/adapter/formal/seed_{seed}/attempt_001/adapter_unit_manifest.json"
    adapter = json.loads(adapter_path.read_text())["worker"]
    if method == "HR_U00":
        base_seconds = float(g04["runtime_seconds"])
        adapter_seconds = 0.0
        peak = float(g04["peak_gpu_mib"])
    else:
        base_seconds = float(g00["runtime_seconds"]) + float(g04["runtime_seconds"])
        adapter_seconds = float(adapter["runtime_seconds"])
        peak = max(float(g00["peak_gpu_mib"]), float(g04["peak_gpu_mib"]),
                   float(adapter["peak_gpu_mib"]))
    return {"base_training_seconds": base_seconds,
            "adapter_increment_seconds": adapter_seconds,
            "recovery_head_seconds": float(head_seconds),
            "end_to_end_seconds": base_seconds + adapter_seconds + float(head_seconds),
            "peak_gpu_mib": peak}


def prelabel_gate() -> tuple[dict, dict]:
    lock_path = OUT / "locked_recovery_partition_manifest.json"
    push_path = OUT / "prelabel_push_audit.json"
    lock = json.loads(lock_path.read_text()); push = json.loads(push_path.read_text())
    if (lock.get("status") != "TOTAL_LOCKED_BEFORE_LABEL_ACCESS"
            or lock.get("row_count") != 20 or lock.get("exact_K") != "20/20"
            or lock.get("deterministic_exact") != "20/20"):
        raise RuntimeError("recovery partition total-lock gate failed")
    if push.get("status") != "PASS" or push.get("label_access") is not False:
        raise RuntimeError("prelabel ordinary-push gate failed")
    commit = push["commit"]
    remote = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", BRANCH], cwd=REPO,
        text=True).split()[0]
    if remote != commit:
        raise RuntimeError(f"prelabel push commit mismatch {remote} != {commit}")
    for row in lock["rows"]:
        clusters = Path(row["clusters_path"])
        manifest = Path(row["transform_manifest_path"])
        if sha256_file(clusters) != row["clusters_file_sha256"] or sha256_file(manifest) != row["transform_manifest_sha256"]:
            raise RuntimeError("post-lock partition SHA drift")
    return lock, push


def main() -> None:
    lock, push = prelabel_gate()
    guard = RECOVERY / "evaluation/label_window_guard.json"
    if guard.exists():
        raise RuntimeError("one-way label window already attempted; rerun forbidden")
    atomic_json(guard, {"status": "STARTED_ONE_WAY", "return_to_head_forbidden": True,
                        "training": 0, "adapter": 0, "affinity_rebuild": 0})
    mapping = pd.read_csv(ORIGINAL_OUT / "prelabel_observation_mapping.csv")
    expected_ids = mapping["observation_id"].astype(str).tolist()
    carrier = ORIGINAL / "annotation_carrier/MISAR_seq_mouse_E15_brain_ATAC_data.h5"
    # Sole direct read of Y in this recovery.  Every partition is already locked and pushed.
    with h5py.File(carrier, "r") as handle:
        raw_y = np.asarray(handle["Y"][:])
    decoded = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in raw_y])
    truth = decoded[mapping["carrier_row"].to_numpy(np.int64)]
    if len(truth) != 1949 or np.unique(truth).size != 12:
        raise RuntimeError("locked MISAR annotation contract K=12 failed")
    cache_manifest = ORIGINAL / "cache/base/manifest.json"
    prepared = load_cache(ORIGINAL / "cache/base", sha256_file(cache_manifest))
    if prepared.obs_names.astype(str).tolist() != expected_ids:
        raise RuntimeError("mapping and cache observation order mismatch")
    graph = symmetric_knn_adjacency(prepared.coordinates, 18).tocsr()
    predictions: Dict[str, np.ndarray] = {}
    rows = []
    lock_rows = {(row["method"], int(row["seed"])): row for row in lock["rows"]}
    for method in ("HR_U00", "HR_F00"):
        for seed in range(10):
            prediction, ids = read_partition(method, seed)
            if ids != expected_ids:
                raise RuntimeError("locked partition observation order mismatch")
            predictions[f"prediction_{method}_{seed}"] = prediction
            resources = resource_components(seed, method,
                lock_rows[(method, seed)]["recovery_head_effective_seconds"])
            rows.append({"method": method, "seed": seed,
                         **metric(truth, prediction, graph), **resources,
                         "canonical_partition_sha256": lock_rows[(method, seed)]["canonical_partition_sha256"]})
    frame = pd.DataFrame(rows).sort_values(["method", "seed"])
    frame.to_csv(OUT / "recovery_20row_metrics.csv", index=False)
    paired = []
    metric_keys = ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
                   "geary_c", "boundary_disagreement", "end_to_end_seconds",
                   "peak_gpu_mib")
    for seed in range(10):
        u = frame[(frame.method == "HR_U00") & (frame.seed == seed)].iloc[0]
        f = frame[(frame.method == "HR_F00") & (frame.seed == seed)].iloc[0]
        paired.append({"seed": seed, **{f"delta_{key}": float(f[key] - u[key])
                                        for key in metric_keys}})
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(OUT / "recovery_paired_deltas.csv", index=False)
    delta_q = paired_frame["delta_q"].to_numpy(np.float64)
    statistics = {
        "mean_delta_ari": float(paired_frame.delta_ari.mean()),
        "std_delta_ari": float(paired_frame.delta_ari.std(ddof=1)),
        "mean_delta_nmi": float(paired_frame.delta_nmi.mean()),
        "std_delta_nmi": float(paired_frame.delta_nmi.std(ddof=1)),
        "mean_delta_q": float(delta_q.mean()),
        "std_delta_q": float(delta_q.std(ddof=1)),
        "q_wins": int((delta_q > 0).sum()),
        "exact_sign_flip": signflip(delta_q),
        "bootstrap_delta_q": bootstrap(delta_q),
        "inference_note": "10 seeds measure algorithmic stability, not independent biological replication",
        "per_seed": paired,
    }
    atomic_json(OUT / "recovery_paired_statistics.json", statistics)
    spatial = {
        "mean_delta_neighbor": float(paired_frame.delta_neighbor_agreement.mean()),
        "mean_delta_moran": float(paired_frame.delta_moran_i.mean()),
        "mean_delta_geary": float(paired_frame.delta_geary_c.mean()),
        "mean_delta_boundary": float(paired_frame.delta_boundary_disagreement.mean()),
        "thresholds": {"neighbor_min": -0.01, "moran_min": -0.02,
                       "geary_max": 0.02, "boundary_max": 0.01},
    }
    spatial["pass"] = bool(spatial["mean_delta_neighbor"] >= -0.01
                           and spatial["mean_delta_moran"] >= -0.02
                           and spatial["mean_delta_geary"] <= 0.02
                           and spatial["mean_delta_boundary"] <= 0.01)
    atomic_json(OUT / "recovery_spatial_protection.json", spatial)
    urows = frame[frame.method == "HR_U00"]; frows = frame[frame.method == "HR_F00"]
    resource_audit = {
        "effective_end_to_end_definition": "locked original base training + locked F00 adapter where applicable + first recovery head call; duplicate determinism audit excluded",
        "HR_U00_mean_seconds": float(urows.end_to_end_seconds.mean()),
        "HR_F00_mean_seconds": float(frows.end_to_end_seconds.mean()),
        "runtime_ratio": float(frows.end_to_end_seconds.mean() / urows.end_to_end_seconds.mean()),
        "HR_U00_mean_peak_gpu_mib": float(urows.peak_gpu_mib.mean()),
        "HR_F00_mean_peak_gpu_mib": float(frows.peak_gpu_mib.mean()),
        "peak_gpu_ratio": float(frows.peak_gpu_mib.mean() / urows.peak_gpu_mib.mean()),
        "runtime_ratio_max": 1.5, "peak_gpu_ratio_max": 1.25,
        "recovery_gpu_used": False,
    }
    resource_audit["pass"] = bool(resource_audit["runtime_ratio"] <= 1.5
                                  and resource_audit["peak_gpu_ratio"] <= 1.25)
    atomic_json(OUT / "recovery_resource_audit.json", resource_audit)
    old_rows = []
    for seed in (0, 1, 2, 3, 4, 5, 7, 8, 9):
        row = {"seed": seed, "terminal_decision_input": False,
               "description_only": True}
        for method in ("U00", "F00"):
            table = pd.read_csv(ORIGINAL / f"formal/transforms/{method}/seed_{seed}/clusters.csv")
            pred = table["cluster"].to_numpy(np.int64)
            values = metric(truth, pred, graph)
            for key, value in values.items():
                row[f"{method}_{key}"] = value
        for key in ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
                    "geary_c", "boundary_disagreement"):
            row[f"delta_{key}"] = row[f"F00_{key}"] - row[f"U00_{key}"]
        old_rows.append(row)
    pd.DataFrame(old_rows).to_csv(OUT / "original_spectral_9pair_sensitivity.csv", index=False)
    snapshot = RECOVERY / "evaluation/authorized_metric_snapshot.npz"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(snapshot, truth=truth, graph_data=graph.data,
                        graph_indices=graph.indices, graph_indptr=graph.indptr,
                        graph_shape=np.asarray(graph.shape, dtype=np.int64),
                        **predictions)
    independent_path = OUT / "recovery_independent_recalculation.json"
    subprocess.run([sys.executable, str(REPO / "scripts/night8b_head_recovery_independent.py"),
                    "--snapshot", str(snapshot), "--output", str(independent_path)],
                   check=True, cwd=REPO)
    independent = json.loads(independent_path.read_text())
    errors = []
    for row in rows:
        other = next(x for x in independent["rows"] if x["method"] == row["method"] and x["seed"] == row["seed"])
        for key in ("ari", "nmi", "q", "ami", "fmi", "homogeneity",
                    "completeness", "v_measure", "neighbor_agreement",
                    "moran_i", "geary_c", "boundary_disagreement"):
            errors.append(abs(float(row[key]) - float(other[key])))
    for row in paired:
        other = next(x for x in independent["paired"] if x["seed"] == row["seed"])
        for key in ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
                    "geary_c", "boundary_disagreement"):
            errors.append(abs(float(row[f"delta_{key}"]) - float(other[f"delta_{key}"])))
    for key in ("mean_delta_ari", "mean_delta_nmi", "mean_delta_q", "std_delta_q"):
        errors.append(abs(float(statistics[key]) - float(independent["statistics"][key])))
    errors.extend([
        abs(float(statistics["exact_sign_flip"]["p_one_sided"]) - float(independent["statistics"]["exact_sign_flip"]["p_one_sided"])),
        abs(float(statistics["bootstrap_delta_q"]["ci_lower"]) - float(independent["statistics"]["bootstrap_delta_q"]["ci_lower"])),
        abs(float(statistics["bootstrap_delta_q"]["ci_upper"]) - float(independent["statistics"]["bootstrap_delta_q"]["ci_upper"])),
    ])
    for key in ("mean_delta_neighbor", "mean_delta_moran", "mean_delta_geary", "mean_delta_boundary"):
        errors.append(abs(float(spatial[key]) - float(independent["spatial"][key])))
    maximum_error = float(max(errors))
    independent.update({"maximum_absolute_error_vs_primary": maximum_error,
                        "key_coverage": "20/20 primary, 10/10 paired, all statistics and spatial",
                        "status": "PASS" if maximum_error <= 1e-12 else "FAIL"})
    atomic_json(independent_path, independent)
    if maximum_error > 1e-12:
        terminal = "RECOVERY_SEMANTICS_INVALID"
        science_pass = False
    else:
        science_pass = bool(statistics["mean_delta_q"] >= 0.010
                            and statistics["mean_delta_ari"] >= 0.0
                            and statistics["mean_delta_nmi"] >= 0.0
                            and statistics["q_wins"] >= 8
                            and statistics["exact_sign_flip"]["p_one_sided"] < 0.05
                            and statistics["bootstrap_delta_q"]["ci_lower"] > 0.0
                            and spatial["pass"])
        if science_pass and resource_audit["pass"]:
            terminal = "NIGHT8B_HEAD_RECOVERY_BALANCED_CONFIRMED"
        elif science_pass:
            terminal = "NIGHT8B_HEAD_RECOVERY_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST"
        elif statistics["mean_delta_q"] > 0.0:
            terminal = "NIGHT8B_HEAD_RECOVERY_PARTIAL_OR_MIXED_EVIDENCE"
        else:
            terminal = "NIGHT8B_HEAD_RECOVERY_FAMILY_POLICY_NOT_GENERALIZED"
    label_audit = {
        "schema_version": 1, "status": "PASS", "one_authorized_window": True,
        "Y_values_read_once": True, "Y_rows": 1949, "K": 12,
        "total_lock_commit": push["commit"],
        "return_to_training_adapter_affinity_or_head": False,
        "training": 0, "adapter": 0, "affinity_rebuild": 0,
        "primary_partition_source": "20/20 recovery head only",
        "old_spectral_role": "descriptive_sensitivity_only",
    }
    atomic_json(OUT / "label_window_audit.json", label_audit)
    atomic_json(guard, {**label_audit, "status": "COMPLETED_ONE_WAY"})
    decision = {
        "schema_version": 1, "terminal_status": terminal,
        "original_night8b_terminal_status": "INFRASTRUCTURE_BLOCKED",
        "science_gate_pass": science_pass,
        "resource_gate_pass": resource_audit["pass"],
        "mean_delta_q": statistics["mean_delta_q"], "q_wins": statistics["q_wins"],
        "uniform_head_external_confirmation": True,
        "original_H05_endpoint_complete_confirmation": False,
        "old_spectral_9pair_terminal_decision_input": False,
        "candidate_search": False, "third_party_benchmark_run": False,
        "claim_sota": False, "label_opened_post_total_lock_and_push": True,
    }
    atomic_json(OUT / "night8b_head_recovery_decision.json", decision)
    print(json.dumps({"terminal_status": terminal,
                      "mean_delta_q": statistics["mean_delta_q"],
                      "q_wins": statistics["q_wins"],
                      "independent_max_error": maximum_error}, sort_keys=True))


if __name__ == "__main__":
    main()
