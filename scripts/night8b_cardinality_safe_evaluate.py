#!/usr/bin/env python3
"""Single-use, same-process Night-8B cardinality-safe evaluator.

The script preloads every non-label input, performs the second and final raw-Y
read for the Night-8B lineage, atomically persists the reference-label contract,
then evaluates frozen partitions.  It has no training or transform interface.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night8b_cardinality_safe_eval import (
    atomic_json_fsync,
    canonicalize_labels,
    independent_decision_metrics,
    paired_rows,
    paired_statistics,
    primary_metrics,
    sha256_file,
    spatial_summary,
    symmetric_knn_adjacency,
)

ORIGINAL = Path("/root/autodl-fs/night8b_raw_runs_20260820")
RECOVERY = Path("/root/autodl-fs/night8b_head_recovery_20260820")
RAW_OUT = Path("/root/autodl-fs/night8b_cardinality_safe_eval_20260820")
OUT = REPO / "outputs/night8b_cardinality_safe_eval"
PARENT_OUT = REPO / "outputs/night8b_head_recovery"
ORIGINAL_OUT = REPO / "outputs/night8b_handoff"
LOCK_PATH = RECOVERY / "manifests/locked_recovery_partition_manifest.json"
LOCK_SHA = "8a696ec456b9abe45c9fd65c3b646f2654300f6c51e47d766fb0bfcef0686e6c"
EXPECTED_MAPPING_SHA = "322e7bf0f459998c882a0305e8aea129deee29570412982b3697976ac64b8ae5"
EXPECTED_ORDER_SHA = "9f0514cee55d307a0ff81d44ffffc2da742dbe2d02b849576b7ef5903743dd1b"
EXPECTED_BASE_CACHE_MANIFEST_SHA = "c5a8b3c3e6ffd0de378ce60f890aabcce9eccd57014c42b482e9dee180a5c544"
BRANCH = "revision/q2-night8b-cardinality-safe-eval-20260820"
EXPECTED_N = 1949
DECISION_KEYS = (
    "ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c",
    "boundary_disagreement",
)


def atomic_csv_fsync(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_both(name: str, payload: dict) -> None:
    atomic_json_fsync(RAW_OUT / "evaluation" / name, payload)
    atomic_json_fsync(OUT / name, payload)


def write_csv_both(name: str, frame: pd.DataFrame) -> None:
    atomic_csv_fsync(RAW_OUT / "evaluation" / name, frame)
    atomic_csv_fsync(OUT / name, frame)


def ordered_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def verify_expected_artifact(path: Path, expected: Dict[str, dict]) -> None:
    row = expected.get(str(path))
    if row is None:
        raise RuntimeError("input absent from immutable 297-row authority: %s" % path)
    wanted = row.get("expected_sha256") or row.get("actual_sha256")
    wanted_size = row.get("expected_size_bytes") or row.get("actual_size_bytes")
    if not path.is_file() or path.stat().st_size != wanted_size or sha256_file(path) != wanted:
        raise RuntimeError("immutable input drift: %s" % path)


def read_locked_partition(path: Path, expected_sha: str,
                          expected_ids: Sequence[str]) -> np.ndarray:
    if sha256_file(path) != expected_sha:
        raise RuntimeError("locked partition SHA mismatch: %s" % path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["observation_id", "cluster"]:
        raise RuntimeError("locked partition schema mismatch: %s" % path)
    ids = frame["observation_id"].astype(str).tolist()
    prediction = frame["cluster"].to_numpy(np.int64)
    if ids != list(expected_ids) or prediction.size != EXPECTED_N:
        raise RuntimeError("locked partition observation alignment mismatch")
    if np.unique(prediction).size != 12:
        raise RuntimeError("locked predicted K changed")
    return prediction


def resource_components(seed: int, method: str, head_seconds: float,
                        expected: Dict[str, dict]) -> dict:
    base_path = ORIGINAL / ("formal/base/seed_%d/attempt_001/base_unit_manifest.json" % seed)
    adapter_path = ORIGINAL / ("formal/adapter/formal/seed_%d/attempt_001/adapter_unit_manifest.json" % seed)
    verify_expected_artifact(base_path, expected)
    verify_expected_artifact(adapter_path, expected)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    by_graph = {row["graph_id"]: row for row in base["submodels"]}
    g00 = by_graph["G00_SP18_F20_CORR_UNION"]
    g04 = by_graph["G04_SP10_F10_EUC_UNION"]
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))["worker"]
    if method == "HR_U00":
        base_seconds = float(g04["runtime_seconds"])
        adapter_seconds = 0.0
        peak = float(g04["peak_gpu_mib"])
    elif method == "HR_F00":
        base_seconds = float(g00["runtime_seconds"]) + float(g04["runtime_seconds"])
        adapter_seconds = float(adapter["runtime_seconds"])
        peak = max(float(g00["peak_gpu_mib"]), float(g04["peak_gpu_mib"]),
                   float(adapter["peak_gpu_mib"]))
    else:
        raise RuntimeError("unknown fixed method")
    return {
        "base_training_seconds": base_seconds,
        "adapter_increment_seconds": adapter_seconds,
        "recovery_head_seconds": float(head_seconds),
        "end_to_end_seconds": base_seconds + adapter_seconds + float(head_seconds),
        "peak_gpu_mib": peak,
    }


def preload_nonlabel_inputs() -> dict:
    p0 = json.loads((OUT / "p0_eval_recovery_authority.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "evaluation_contract_lock.json").read_text(encoding="utf-8"))
    push = json.loads((RAW_OUT / "manifests/prelabel_push_audit.json").read_text(encoding="utf-8"))
    if p0.get("status") != "PASS" or contract.get("status") != "LOCKED_BEFORE_FINAL_Y_READ":
        raise RuntimeError("P0 or evaluation contract not locked")
    if push.get("status") != "PASS" or push.get("label_access_count_before") != 1:
        raise RuntimeError("pre-label ordinary push audit failed")
    remote_line = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin", BRANCH], cwd=str(REPO), text=True
    ).strip().split()
    if len(remote_line) != 2 or remote_line[0] != push["remote_branch_sha"]:
        raise RuntimeError("pre-label remote branch SHA drift")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", push["prelabel_semantics_commit"], remote_line[0]],
        cwd=str(REPO), check=False,
    ).returncode != 0:
        raise RuntimeError("pre-label semantics commit is not an ancestor of remote")
    if sha256_file(LOCK_PATH) != LOCK_SHA:
        raise RuntimeError("locked manifest SHA drift")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("row_count") != 20 or lock.get("exact_K") != "20/20" or lock.get("deterministic_exact") != "20/20":
        raise RuntimeError("20-partition lock semantic gate failed")

    mapping_path = ORIGINAL_OUT / "prelabel_observation_mapping.csv"
    if sha256_file(mapping_path) != EXPECTED_MAPPING_SHA:
        raise RuntimeError("mapping SHA drift")
    mapping = pd.read_csv(mapping_path)
    expected_ids = mapping["observation_id"].astype(str).tolist()
    carrier_rows = mapping["carrier_row"].to_numpy(np.int64)
    if len(expected_ids) != EXPECTED_N or ordered_sha(expected_ids) != EXPECTED_ORDER_SHA:
        raise RuntimeError("mapping observation order drift")
    if sorted(carrier_rows.tolist()) != list(range(EXPECTED_N)):
        raise RuntimeError("mapping carrier rows are not a permutation")

    original_authority = json.loads((PARENT_OUT / "original_artifact_manifest_before.json").read_text())
    expected = {str(Path(row["path"])): row for row in original_authority["rows"]}
    coordinates_path = ORIGINAL / "cache/base/coordinates.npy"
    observation_path = ORIGINAL / "cache/base/observation_ids.tsv"
    cache_manifest_path = ORIGINAL / "cache/base/manifest.json"
    if sha256_file(cache_manifest_path) != EXPECTED_BASE_CACHE_MANIFEST_SHA:
        raise RuntimeError("base cache manifest SHA drift")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    for path in (coordinates_path, observation_path):
        row = cache_manifest["files"].get(path.name)
        if (row is None or not path.is_file() or path.stat().st_size != row["size_bytes"]
                or sha256_file(path) != row["sha256"]):
            raise RuntimeError("base cache file drift: %s" % path)
    coordinates = np.load(coordinates_path, allow_pickle=False)
    cache_id_table = pd.read_csv(observation_path, sep="\t", header=0)
    if list(cache_id_table.columns) != ["observation_id"]:
        raise RuntimeError("cache observation-id schema mismatch")
    cache_ids = cache_id_table["observation_id"].astype(str).tolist()
    if cache_ids != expected_ids or coordinates.shape[0] != EXPECTED_N:
        raise RuntimeError("spatial/cache observation alignment mismatch")
    graph = symmetric_knn_adjacency(coordinates, 18)

    lock_rows = {(row["method"], int(row["seed"])): row for row in lock["rows"]}
    primary_predictions: Dict[Tuple[str, int], np.ndarray] = {}
    resources: Dict[Tuple[str, int], dict] = {}
    for method in ("HR_U00", "HR_F00"):
        for seed in range(10):
            row = lock_rows[(method, seed)]
            cluster_path = Path(row["clusters_path"])
            manifest_path = Path(row["transform_manifest_path"])
            if sha256_file(manifest_path) != row["transform_manifest_sha256"]:
                raise RuntimeError("locked transform manifest drift")
            primary_predictions[(method, seed)] = read_locked_partition(
                cluster_path, row["clusters_file_sha256"], expected_ids
            )
            resources[(method, seed)] = resource_components(
                seed, method, float(row["recovery_head_effective_seconds"]), expected
            )

    sensitivity_predictions: Dict[Tuple[str, int], np.ndarray] = {}
    for method in ("U00", "F00"):
        for seed in (0, 1, 2, 3, 4, 5, 7, 8, 9):
            cluster_path = ORIGINAL / ("formal/transforms/%s/seed_%d/clusters.csv" % (method, seed))
            transform_path = ORIGINAL / ("formal/transforms/%s/seed_%d/transform_manifest.json" % (method, seed))
            verify_expected_artifact(cluster_path, expected)
            verify_expected_artifact(transform_path, expected)
            frame = pd.read_csv(cluster_path)
            ids = frame["observation_id"].astype(str).tolist()
            if ids != expected_ids or len(frame) != EXPECTED_N:
                raise RuntimeError("original spectral alignment mismatch")
            sensitivity_predictions[(method, seed)] = frame["cluster"].to_numpy(np.int64)

    carrier = ORIGINAL / "annotation_carrier/MISAR_seq_mouse_E15_brain_ATAC_data.h5"
    carrier_sha = sha256_file(carrier)
    locked_input = push["locked_input_sha256"]
    if locked_input.get("carrier") != carrier_sha:
        raise RuntimeError("pre-label carrier SHA lock mismatch")
    if locked_input.get("base_cache_manifest") != EXPECTED_BASE_CACHE_MANIFEST_SHA:
        raise RuntimeError("pre-label cache manifest lock mismatch")
    return {
        "p0": p0, "contract": contract, "push": push, "lock": lock,
        "expected_ids": expected_ids, "carrier_rows": carrier_rows,
        "coordinates": coordinates, "graph": graph,
        "primary_predictions": primary_predictions,
        "sensitivity_predictions": sensitivity_predictions,
        "resources": resources, "carrier": carrier,
        "carrier_sha256": carrier_sha,
        "input_sha256": {
            "locked_partition_manifest": LOCK_SHA,
            "annotation_mapping": EXPECTED_MAPPING_SHA,
            "coordinates": sha256_file(coordinates_path),
            "observation_ids": sha256_file(observation_path),
            "cache_manifest": sha256_file(cache_manifest_path),
            "carrier": carrier_sha,
        },
    }


def independent_paired(rows: Sequence[dict], bootstrap_indices: np.ndarray) -> Tuple[List[dict], dict, dict]:
    paired = paired_rows(rows)
    output: Dict[str, object] = {
        "paired_units": len(paired),
        "inference_note": "seeds measure algorithmic stability, not independent biological replication",
    }
    for name in ("ari", "nmi", "q"):
        values = np.asarray([item["delta_" + name] for item in paired], dtype=np.float64)
        bootstrap_means = np.asarray([
            float(np.sum(values[index_row]) / len(index_row))
            for index_row in np.asarray(bootstrap_indices, dtype=np.int64)
        ], dtype=np.float64)
        output["mean_delta_" + name] = float(np.sum(values) / values.size)
        output["std_delta_" + name] = float(np.sqrt(np.sum((values - values.mean()) ** 2) / (values.size - 1)))
        output[name + "_wins"] = int(np.count_nonzero(values > 0.0))
        output["bootstrap_delta_" + name] = {
            "replicates": 100000, "seed": 20260820,
            "mean": float(np.sum(values) / values.size),
            "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
            "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
        }
    delta_q = np.asarray([item["delta_q"] for item in paired], dtype=np.float64)
    observed = float(np.sum(delta_q) / delta_q.size)
    enumerated = []
    for bitmask in range(1 << delta_q.size):
        signs = np.asarray([1.0 if (bitmask >> index) & 1 else -1.0
                            for index in range(delta_q.size)], dtype=np.float64)
        enumerated.append(float(np.sum(delta_q * signs) / delta_q.size))
    tail = sum(value >= observed - 1e-15 for value in enumerated)
    output["exact_sign_flip_delta_q"] = {
        "observed_mean": observed, "enumerations": 1024,
        "tail_count": int(tail), "p_one_sided": float(tail / 1024.0),
    }
    spatial = {
        "mean_delta_neighbor": float(np.sum([x["delta_neighbor_agreement"] for x in paired]) / 10.0),
        "mean_delta_moran": float(np.sum([x["delta_moran_i"] for x in paired]) / 10.0),
        "mean_delta_geary": float(np.sum([x["delta_geary_c"] for x in paired]) / 10.0),
        "mean_delta_boundary": float(np.sum([x["delta_boundary_disagreement"] for x in paired]) / 10.0),
    }
    return paired, output, spatial


def maximum_error(primary_rows: Sequence[dict], independent_rows: Sequence[dict],
                  primary_paired: Sequence[dict], independent_paired_rows: Sequence[dict],
                  primary_stats: dict, independent_stats: dict,
                  primary_spatial: dict, independent_spatial: dict) -> Tuple[float, list]:
    errors = []
    independent_by_key = {(row["method"], row["seed"]): row for row in independent_rows}
    for row in primary_rows:
        other = independent_by_key[(row["method"], row["seed"])]
        for key in DECISION_KEYS:
            errors.append({"key": "%s/%d/%s" % (row["method"], row["seed"], key),
                           "absolute_error": abs(float(row[key]) - float(other[key]))})
    independent_pair_by_seed = {row["seed"]: row for row in independent_paired_rows}
    for row in primary_paired:
        other = independent_pair_by_seed[row["seed"]]
        for key in DECISION_KEYS:
            errors.append({"key": "paired/%d/%s" % (row["seed"], key),
                           "absolute_error": abs(float(row["delta_" + key]) - float(other["delta_" + key]))})
    for key in ("mean_delta_ari", "std_delta_ari", "ari_wins",
                "mean_delta_nmi", "std_delta_nmi", "nmi_wins",
                "mean_delta_q", "std_delta_q", "q_wins"):
        errors.append({"key": "stats/" + key,
                       "absolute_error": abs(float(primary_stats[key]) - float(independent_stats[key]))})
    for name in ("ari", "nmi", "q"):
        for key in ("mean", "ci_lower", "ci_upper"):
            errors.append({"key": "bootstrap/%s/%s" % (name, key),
                           "absolute_error": abs(float(primary_stats["bootstrap_delta_" + name][key]) - float(independent_stats["bootstrap_delta_" + name][key]))})
    for key in ("observed_mean", "enumerations", "tail_count", "p_one_sided"):
        errors.append({"key": "signflip/" + key,
                       "absolute_error": abs(float(primary_stats["exact_sign_flip_delta_q"][key]) - float(independent_stats["exact_sign_flip_delta_q"][key]))})
    for key in ("mean_delta_neighbor", "mean_delta_moran", "mean_delta_geary", "mean_delta_boundary"):
        errors.append({"key": "spatial/" + key,
                       "absolute_error": abs(float(primary_spatial[key]) - float(independent_spatial[key]))})
    return float(max(row["absolute_error"] for row in errors)), errors


def evaluate_after_contract(truth: np.ndarray, preloaded: dict,
                            reference_contract_sha: str) -> dict:
    graph = preloaded["graph"]
    rng = np.random.default_rng(20260820)
    bootstrap_indices = rng.integers(0, 10, size=(100000, 10))
    primary_rows_list: List[dict] = []
    independent_rows_list: List[dict] = []
    for method in ("HR_U00", "HR_F00"):
        for seed in range(10):
            prediction = preloaded["primary_predictions"][(method, seed)]
            resource = preloaded["resources"][(method, seed)]
            primary_rows_list.append({"method": method, "seed": seed,
                                      **primary_metrics(truth, prediction, graph), **resource})
            independent_rows_list.append({
                "method": method, "seed": seed,
                **independent_decision_metrics(truth, prediction, graph), **resource,
            })
    primary_pairs = paired_rows(primary_rows_list)
    primary_stats = paired_statistics(primary_pairs, bootstrap_indices)
    primary_spatial = spatial_summary(primary_pairs)
    independent_pairs, independent_stats, independent_spatial = independent_paired(
        independent_rows_list, bootstrap_indices
    )
    error, error_rows = maximum_error(
        primary_rows_list, independent_rows_list, primary_pairs, independent_pairs,
        primary_stats, independent_stats, primary_spatial, independent_spatial,
    )

    secondary_failures = []
    for row in primary_rows_list:
        for key in ("ari", "ami"):
            if not np.isfinite(row[key]) or not (-1.0 - 1e-12 <= row[key] <= 1.0 + 1e-12):
                secondary_failures.append("%s/%s/%s" % (row["method"], row["seed"], key))
        for key in ("nmi", "fmi", "homogeneity", "completeness", "v_measure"):
            if not np.isfinite(row[key]) or not (-1e-12 <= row[key] <= 1.0 + 1e-12):
                secondary_failures.append("%s/%s/%s" % (row["method"], row["seed"], key))

    sensitivity = []
    for seed in (0, 1, 2, 3, 4, 5, 7, 8, 9):
        u = primary_metrics(truth, preloaded["sensitivity_predictions"][("U00", seed)], graph)
        f = primary_metrics(truth, preloaded["sensitivity_predictions"][("F00", seed)], graph)
        sensitivity.append({
            "seed": seed,
            **{"U00_" + key: float(u[key]) for key in u},
            **{"F00_" + key: float(f[key]) for key in f},
            **{"delta_" + key: float(f[key]) - float(u[key]) for key in u},
        })
    sensitivity_mean_delta_q = float(np.mean([row["delta_q"] for row in sensitivity]))
    uniform_direction = int(np.sign(primary_stats["mean_delta_q"]))
    sensitivity_direction = int(np.sign(sensitivity_mean_delta_q))

    urows = [row for row in primary_rows_list if row["method"] == "HR_U00"]
    frows = [row for row in primary_rows_list if row["method"] == "HR_F00"]
    u_runtime = float(np.mean([row["end_to_end_seconds"] for row in urows]))
    f_runtime = float(np.mean([row["end_to_end_seconds"] for row in frows]))
    u_peak = float(np.mean([row["peak_gpu_mib"] for row in urows]))
    f_peak = float(np.mean([row["peak_gpu_mib"] for row in frows]))
    resource = {
        "definition": "original training plus fixed adapter when applicable plus locked recovery head cost",
        "HR_U00_mean_end_to_end_seconds": u_runtime,
        "HR_F00_mean_end_to_end_seconds": f_runtime,
        "runtime_ratio": f_runtime / u_runtime,
        "runtime_ratio_max": 1.5,
        "HR_U00_mean_peak_gpu_mib": u_peak,
        "HR_F00_mean_peak_gpu_mib": f_peak,
        "peak_gpu_ratio": f_peak / u_peak,
        "peak_gpu_ratio_max": 1.25,
        "evaluation_gpu_used": False,
    }
    resource["pass"] = bool(resource["runtime_ratio"] <= 1.5 and resource["peak_gpu_ratio"] <= 1.25)

    independent_pass = error <= 1e-12 and not secondary_failures
    science_core_pass = bool(
        primary_stats["mean_delta_q"] >= 0.010
        and primary_stats["mean_delta_ari"] >= 0.0
        and primary_stats["mean_delta_nmi"] >= 0.0
        and primary_stats["q_wins"] >= 8
        and primary_stats["exact_sign_flip_delta_q"]["p_one_sided"] < 0.05
        and primary_stats["bootstrap_delta_q"]["ci_lower"] > 0.0
    )
    if not independent_pass:
        terminal = "EVAL_RECOVERY_SEMANTICS_INVALID"
    elif science_core_pass and primary_spatial["pass"] and resource["pass"]:
        terminal = "NIGHT8B_CARDINALITY_SAFE_FAMILY_POLICY_CONFIRMED"
    elif science_core_pass and primary_spatial["pass"]:
        terminal = "NIGHT8B_CARDINALITY_SAFE_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST"
    elif primary_stats["mean_delta_q"] > 0.0:
        terminal = "NIGHT8B_CARDINALITY_SAFE_PARTIAL_OR_MIXED_EVIDENCE"
    else:
        terminal = "NIGHT8B_CARDINALITY_SAFE_FAMILY_POLICY_NOT_GENERALIZED"

    primary_frame = pd.DataFrame(primary_rows_list).sort_values(["method", "seed"])
    paired_frame = pd.DataFrame(primary_pairs).sort_values("seed")
    sensitivity_frame = pd.DataFrame(sensitivity).sort_values("seed")
    write_csv_both("cardinality_safe_20row_metrics.csv", primary_frame)
    write_csv_both("cardinality_safe_paired_deltas.csv", paired_frame)
    write_csv_both("original_spectral_9pair_sensitivity.csv", sensitivity_frame)
    write_json_both("cardinality_safe_paired_statistics.json", primary_stats)
    write_json_both("cardinality_safe_spatial_protection.json", primary_spatial)
    write_json_both("cardinality_safe_resource_audit.json", resource)
    independent_payload = {
        "schema_version": 1,
        "same_process_same_in_memory_Y": True,
        "reference_label_contract_sha256": reference_contract_sha,
        "primary_key_coverage": "20/20",
        "paired_key_coverage": "10/10",
        "maximum_absolute_error_vs_primary": error,
        "threshold": 1e-12,
        "secondary_metric_failures": secondary_failures,
        "status": "PASS" if independent_pass else "FAIL",
        "rows": independent_rows_list,
        "paired": independent_pairs,
        "statistics": independent_stats,
        "spatial": independent_spatial,
        "error_rows": error_rows,
    }
    write_json_both("cardinality_safe_independent_recompute.json", independent_payload)
    sensitivity_payload = {
        "complete_pairs": 9, "missing_seed": 6,
        "terminal_decision_input": False,
        "same_in_memory_Y": True,
        "mean_delta_q": sensitivity_mean_delta_q,
        "direction_same_as_uniform_head": uniform_direction == sensitivity_direction,
    }
    write_json_both("original_spectral_sensitivity_audit.json", sensitivity_payload)
    decision = {
        "schema_version": 1,
        "terminal_status": terminal,
        "evidence_label": "POST_LOCK_EXTERNAL_EVALUATION_RECOVERY_NOT_PRISTINE_HOLDOUT",
        "reference_K": None,
        "predicted_K": 12,
        "predicted_reference_K_equality_required": False,
        "science_core_gate_pass": science_core_pass,
        "spatial_gate_pass": primary_spatial["pass"],
        "resource_gate_pass": resource["pass"],
        "independent_recompute_pass": independent_pass,
        "mean_delta_ari": primary_stats["mean_delta_ari"],
        "mean_delta_nmi": primary_stats["mean_delta_nmi"],
        "mean_delta_q": primary_stats["mean_delta_q"],
        "q_wins": primary_stats["q_wins"],
        "exact_sign_flip_p": primary_stats["exact_sign_flip_delta_q"]["p_one_sided"],
        "bootstrap_delta_q_95CI": [primary_stats["bootstrap_delta_q"]["ci_lower"], primary_stats["bootstrap_delta_q"]["ci_upper"]],
        "original_spectral_terminal_decision_input": False,
        "candidate_search": False, "third_party_benchmark": False,
        "claim_pristine_holdout": False, "claim_sota": False,
        "scope_counts": {"training": 0, "checkpoint_load": 0, "forward": 0,
                         "adapter": 0, "affinity_rebuild": 0, "head_transform": 0},
    }
    return {
        "decision": decision, "primary_rows": primary_rows_list,
        "paired": primary_pairs, "statistics": primary_stats,
        "spatial": primary_spatial, "resource": resource,
        "independent": independent_payload, "sensitivity": sensitivity_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Night-8B K=12 partitions only")
    parser.add_argument("--execute-final-authorized-Y-read", action="store_true", required=True)
    args = parser.parse_args()
    del args
    signal.alarm(30 * 60)
    started = time.monotonic()
    preloaded = preload_nonlabel_inputs()
    guard = RAW_OUT / "evaluation/final_Y_read_guard.json"
    if guard.exists():
        raise RuntimeError("Night-8B final raw-Y read already attempted; third read forbidden")
    atomic_json_fsync(guard, {
        "schema_version": 1, "status": "STARTED_SECOND_AND_FINAL_RAW_Y_READ",
        "lineage_read_count_before": 1, "this_task_authorized_reads": 1,
        "lineage_read_count_after": 2, "third_read_forbidden": True,
        "all_nonlabel_inputs_preloaded_and_verified": True,
        "input_sha256": preloaded["input_sha256"],
        "started_unix_time": time.time(),
    })

    carrier = preloaded["carrier"]
    with h5py.File(str(carrier), "r") as handle:
        if "Y" not in handle:
            raise RuntimeError("authorized carrier has no Y dataset")
        dataset = handle["Y"]
        y_shape = list(dataset.shape)
        y_dtype = str(dataset.dtype)
        # Exactly one raw-Y deserialization in this task and the final one in the lineage.
        raw_y = np.asarray(dataset[()])
        canonical_carrier, canonical_contract = canonicalize_labels(raw_y, EXPECTED_N)
        truth = canonical_carrier[preloaded["carrier_rows"]]
        if truth.size != EXPECTED_N:
            raise RuntimeError("mapped truth length mismatch")
        reference_contract = {
            "schema_version": 1,
            "status": "PERSISTED_BEFORE_ANY_METRIC",
            "raw_Y_file_sha256": preloaded["carrier_sha256"],
            "Y_dataset_shape": y_shape,
            "Y_dataset_dtype": y_dtype,
            **canonical_contract,
            "predicted_K": 12,
            "predicted_K_may_differ_from_reference_K": True,
            "observation_alignment_checks": {
                "expected_spots": EXPECTED_N, "mapped_spots": int(truth.size),
                "carrier_rows_unique": int(np.unique(preloaded["carrier_rows"]).size),
                "carrier_rows_exact_permutation": sorted(preloaded["carrier_rows"].tolist()) == list(range(EXPECTED_N)),
                "observation_order_sha256": EXPECTED_ORDER_SHA,
                "annotation_mapping_sha256": EXPECTED_MAPPING_SHA,
                "pass": True,
            },
            "lineage_raw_Y_read_count_before": 1,
            "this_task_raw_Y_read_count": 1,
            "lineage_raw_Y_read_count_after": 2,
            "this_is_final_raw_Y_read_for_night8b_lineage": True,
            "raw_label_vector_persisted": False,
            "metrics_started_before_contract_fsync": False,
        }
        raw_contract_path = RAW_OUT / "evaluation/reference_label_contract.json"
        repo_contract_path = OUT / "reference_label_contract.json"
        atomic_json_fsync(raw_contract_path, reference_contract)
        atomic_json_fsync(repo_contract_path, reference_contract)
        reference_sha = sha256_file(raw_contract_path)
        if sha256_file(repo_contract_path) != reference_sha:
            raise RuntimeError("reference contract dual-copy mismatch")
        result = evaluate_after_contract(truth, preloaded, reference_sha)
        result["decision"]["reference_K"] = int(reference_contract["reference_K"])
        write_json_both("night8b_cardinality_safe_eval_decision.json", result["decision"])

    label_audit = {
        "schema_version": 1, "status": "PASS",
        "one_evaluator_process": True, "one_raw_Y_deserialization_this_task": True,
        "lineage_raw_Y_read_count_before": 1, "this_task": 1, "after": 2,
        "third_read_forbidden": True,
        "reference_K_persisted_before_metrics": True,
        "same_process_same_in_memory_Y_primary_and_independent": True,
        "carrier_closed_after_evaluation": True,
        "return_to_training_checkpoint_forward_adapter_affinity_head_mapping_code_or_K": False,
        "scope_counts": {"training": 0, "checkpoint_load": 0, "forward": 0,
                         "adapter": 0, "affinity_rebuild": 0, "head_transform": 0},
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_both("label_access_and_evaluation_order_audit.json", label_audit)
    atomic_json_fsync(guard, {**label_audit, "status": "COMPLETED_SECOND_AND_FINAL_RAW_Y_READ"})
    print(json.dumps({
        "terminal_status": result["decision"]["terminal_status"],
        "reference_K": result["decision"]["reference_K"],
        "predicted_K": 12,
        "mean_delta_q": result["decision"]["mean_delta_q"],
        "q_wins": result["decision"]["q_wins"],
        "independent_max_error": result["independent"]["maximum_absolute_error_vs_primary"],
        "elapsed_seconds": label_audit["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
