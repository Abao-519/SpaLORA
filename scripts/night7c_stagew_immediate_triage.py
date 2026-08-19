#!/usr/bin/env python3
"""Label-free Stage-W semantic triage and immediate-stop bounded continuation.

This script is intentionally split into two phases:

* ``triage`` verifies all 48 immutable training cells and materializes only the
  registered pre-clustering affinity diagnostics.  It never calls H01.
* ``coordinator`` is enabled only by a PASS triage report and runs the unchanged
  H01 transform worker for W01--W05 under the amended resource bounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import sklearn
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from SpaLORA.night6c_pipeline import array_sha, sparse_sha, spectral  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, atomic_sparse, sha256_file  # noqa: E402
from SpaLORA.night7b_adaptive import row_sparse_strict, self_tuning_affinity, sym_zero  # noqa: E402
from SpaLORA.night7c_conflict import weighted_mnn_weights  # noqa: E402
from scripts.night7b_adapter_stage import endpoint_affinity  # noqa: E402
from scripts.night7b_train import state_sha  # noqa: E402
from scripts.night7c_p1 import training_map  # noqa: E402
from scripts.night7c_replay_recovery_stage_w import CANDIDATES, OUT, RAW, pilot_units, transform_cell  # noqa: E402
from scripts.night7c_replay_recovery_stage_w_resume import load_verified_training  # noqa: E402
from scripts import night7c_stagew_resource_bounded as bounded  # noqa: E402

RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
HANDOFF7B = RAW7B / "official_compact/handoff"
FORMAL = RAW / "stage_w/formal"
INFRA = RAW / "infrastructure"
TRIAGE_ROOT = RAW / "stage_w_semantic_triage"
AFFINITY_ROOT = TRIAGE_ROOT / "preclustering_affinity"
AMENDMENT = REPO / "protocols/night7c_replay_recovery/SpaLORA_Night7C_StageW_Immediate_Stop_and_Semantic_Triage_Amendment_2026-08-19.md"
AMENDMENT_SHA = "24dba16c3aae027e392062b6c81f93732585e119f07e3b87e962e22fb3770a75"
INCIDENT = RAW / "incidents/w00_user_stop_20260819T151604Z"
INCIDENT_STATUS = "RESOURCE_CENSORED_USER_STOP_AFTER_EXTREME_LONGTAIL"
W00 = "W00_FILTER75"
REMAINING = CANDIDATES[1:]
WORKERS = 4
UNIT_LIMIT_SECONDS = 60 * 60.0
TOTAL_LIMIT_SECONDS = 12 * 60 * 60.0
TRIAGE_LIMIT_SECONDS = 30 * 60
COORD_LOG = INFRA / "stagew_immediate_bounded_coordinator.jsonl"
WORKER_ROOT = INFRA / "stagew_immediate_bounded_workers"
PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(name: str, **values: object) -> None:
    row = {"timestamp_utc": utc_now(), "event": name, **values}
    COORD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COORD_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def canonical_json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_inventory(root: Path) -> list[dict]:
    return [
        {"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def no_label_outputs() -> dict:
    names = (
        "total_prelabel_lock.json",
        "label_window_audit.json",
        "gate_audit.json",
        "routing_per_seed_metrics.csv",
        "weighted_mnn_per_seed_metrics.csv",
    )
    state = {name: (OUT / name).exists() for name in names}
    require(not any(state.values()), f"label/evaluation output exists before triage: {state}")
    return state


def source_rows() -> list[dict]:
    rows = list(csv.DictReader((HANDOFF7B / "source_unit_index.csv").open(newline="", encoding="utf-8")))
    return rows


def c00_map() -> dict[str, Path]:
    result: dict[str, Path] = {}
    with (HANDOFF7B / "historical_reference_partition_index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["reference"] == "C00":
                result[row["unit_id"]] = Path(row["clusters_path"]).parent / "affinity.npz"
    require(len(result) == 30, "C00 affinity map cardinality mismatch")
    return result


def graph_stats(matrix: sp.spmatrix, *, k: int) -> dict:
    value = matrix.tocsr().astype(np.float64)
    value.sort_indices()
    diff = (value - value.T).tocsr()
    degree = np.asarray(value.sum(axis=1)).ravel()
    components, labels = connected_components(value, directed=False)
    sizes = np.bincount(labels, minlength=components).astype(np.int64)
    median = float(np.median(degree))
    ratios = degree / max(median, np.finfo(np.float64).tiny)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nnz": int(value.nnz),
        "density": float(value.nnz / float(value.shape[0] * value.shape[1])),
        "finite": bool(np.isfinite(value.data).all()),
        "nonnegative": bool(np.all(value.data >= 0)),
        "negative_value_count": int(np.sum(value.data < 0)),
        "symmetry_max_error": float(np.max(np.abs(diff.data))) if diff.nnz else 0.0,
        "diagonal_max_abs": float(np.max(np.abs(value.diagonal()))) if value.shape[0] else 0.0,
        "degree_min": float(degree.min()),
        "degree_median": median,
        "degree_max": float(degree.max()),
        "degree_quantiles": [float(x) for x in np.quantile(degree, [0, .01, .05, .25, .5, .75, .95, .99, 1])],
        "zero_degree_count": int(np.sum(degree <= 0)),
        "near_isolated_below_1e_6_median": int(np.sum(ratios < 1e-6)),
        "near_isolated_below_1e_3_median": int(np.sum(ratios < 1e-3)),
        "near_isolated_below_1e_2_median": int(np.sum(ratios < 1e-2)),
        "connected_component_count": int(components),
        "component_sizes_desc": sorted(map(int, sizes), reverse=True),
        "largest_component_fraction": float(sizes.max() / value.shape[0]),
        "component_count_ge_k": bool(components >= int(k)),
        "canonical_affinity_sha256": sparse_sha(value),
    }


def code_contract() -> dict:
    transform_source = inspect.getsource(transform_cell)
    endpoint_source = inspect.getsource(endpoint_affinity)
    spectral_source = inspect.getsource(spectral)
    require('endpoint_affinity("E1_ADAPTER_C06_MEAN", embedding, c06, ids)' in transform_source, "transform endpoint call changed")
    require('run_partition("H01", affinity, int(unit["K"]), [])' in transform_source, "H01 call changed")
    require('affinity="precomputed"' in spectral_source, "spectral affinity contract changed")
    require('assign_labels="discretize"' in spectral_source, "spectral assign_labels changed")
    require('n_init=20' in spectral_source and 'random_state=2020' in spectral_source, "spectral fixed parameters changed")
    require("toarray(" not in transform_source + endpoint_source and ".A" not in endpoint_source, "unexpected explicit affinity densification in call chain")
    from sklearn.cluster import SpectralClustering
    signature = inspect.signature(SpectralClustering)
    return {
        "status": "PASS",
        "label_access": False,
        "call_chain": "transform_cell -> endpoint_affinity(E1_ADAPTER_C06_MEAN) -> run_partition(H01) -> spectral -> sklearn.cluster.SpectralClustering",
        "transform_module": inspect.getsourcefile(transform_cell),
        "transform_module_sha256": sha256_file(Path(inspect.getsourcefile(transform_cell))),
        "transform_function_sha256": hashlib.sha256(transform_source.encode()).hexdigest(),
        "endpoint_function_sha256": hashlib.sha256(endpoint_source.encode()).hexdigest(),
        "spectral_function_sha256": hashlib.sha256(spectral_source.encode()).hexdigest(),
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
        "H01": {
            "affinity": "precomputed",
            "assign_labels": "discretize",
            "n_init": 20,
            "random_state": 2020,
            "eigen_solver": signature.parameters["eigen_solver"].default,
            "eigen_tol": signature.parameters["eigen_tol"].default,
            "maxiter": None,
            "fallback": False,
        },
        "explicit_dense_square_conversion": False,
    }


def verify_incident() -> dict:
    require(INCIDENT.is_dir(), f"missing immutable incident: {INCIDENT}")
    inventory = json.loads((INCIDENT / "incident_inventory.json").read_text(encoding="utf-8"))
    require(inventory["status"] == INCIDENT_STATUS and inventory["label_access"] is False, "incident status/firewall mismatch")
    for row in inventory["files"]:
        path = INCIDENT / row["path"]
        require(path.is_file() and path.stat().st_size == int(row["size"]) and sha256_file(path) == row["sha256"], f"incident artifact mismatch: {path}")
    final = json.loads((INCIDENT / "process_control_final.json").read_text(encoding="utf-8"))
    require(final["manifest_after_sigstop"] is False, "W00 had a natural manifest at the stop boundary")
    require(final["driver_termination"]["sigkill_needed"] is False, "unexpected W00 SIGKILL")
    for pid in (929, 67932, 101248):
        require(not Path(f"/proc/{pid}").exists(), f"stopped actor still exists: PID {pid}")
    return {"status": INCIDENT_STATUS, "path": str(INCIDENT), "inventory_sha256": sha256_file(INCIDENT / "incident_inventory.json"), "process_evidence_sha256": sha256_file(INCIDENT / "process_control_final.json")}


def verify_training_cell(candidate: str, unit: dict, row: dict, prior: dict) -> tuple[dict, np.ndarray, list[str], sp.csr_matrix]:
    unit_id = unit["unit_id"]
    attempt = FORMAL / candidate / unit_id / "attempt_001"
    worker = attempt / "worker"
    tm_path = worker / "training_manifest.json"
    reload_path = worker / "reload_forward_audit.json"
    checkpoint_path = worker / "model_final.pt"
    embedding_path = worker / "embedding.npy"
    gate_path = worker / "gate_weights.npy"
    raw_weight_path = worker / "mnn_raw_weights.npy"
    norm_weight_path = worker / "mnn_normalized_weights.npy"
    curve_path = worker / "loss_curve.csv"
    tm = json.loads(tm_path.read_text(encoding="utf-8"))
    reload = json.loads(reload_path.read_text(encoding="utf-8"))
    stage, authority = prior
    config_path = Path(authority["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter_input = RAW7B / "adapter_inputs" / unit_id / "worker_input.json"
    feature_path = RAW / "p1b_features" / unit_id / "features.npz"
    ids_path = RAW7B / "source" / unit_id / "observation_ids.txt"
    ids = [value.strip() for value in ids_path.read_text(encoding="utf-8").splitlines() if value.strip()]
    n = int(unit["observation_count"])
    k = int(unit["K"])
    errors: list[str] = []
    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)
    check(row.get("status") == "success" and row.get("candidate_id") == candidate and row.get("unit_id") == unit_id, "cell primary key/status mismatch")
    check(row.get("label_access") is False and row.get("retry") is False and row.get("fallback") is False, "cell firewall/retry/fallback mismatch")
    check(sha256_file(tm_path) == row.get("training_manifest_sha256"), "training manifest SHA mismatch")
    check(sha256_file(reload_path) == row.get("reload_audit_sha256"), "reload audit SHA mismatch")
    check(reload.get("status") == "PASS" and reload.get("fresh_process") is True and reload.get("embedding_exact") is True and reload.get("gate_exact") is True, "reload audit mismatch")
    check(tm.get("status") == "success" and tm.get("candidate_id") == candidate and tm.get("unit_id") == unit_id, "training manifest key/status mismatch")
    check(tm.get("label_access") is False and tm.get("retry") is False and tm.get("fallback") is False, "training firewall/retry/fallback mismatch")
    check(tm.get("scientific_training") is True and tm.get("cuda_tensor_verified") is True and tm.get("gpu_model") == "NVIDIA GeForce RTX 4080 SUPER", "CUDA training provenance mismatch")
    check(int(tm.get("seed", -1)) == int(unit["seed"]) == int(config["seed"]), "seed mismatch")
    check(int(tm.get("epochs", -1)) == int(config["epochs"]) == 160, "epoch mismatch")
    check(config.get("recipe_id") == "R02" and config.get("fusion") == "equal" and tuple(config.get("losses", [])) == ("RECON", "MNN"), "canonical R02 config mismatch")
    check(sha256_file(config_path) == tm.get("base_config_sha256"), "base config SHA mismatch")
    check(sha256_file(adapter_input) == tm.get("worker_input_sha256"), "adapter worker input SHA mismatch")
    check(sha256_file(feature_path) == tm.get("feature_file_sha256"), "feature SHA mismatch")
    check(sha256_file(checkpoint_path) == tm.get("checkpoint_sha256"), "checkpoint file SHA mismatch")
    check(sha256_file(curve_path) == tm.get("loss_curve_sha256"), "loss curve SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    check(checkpoint.get("candidate_id") == candidate, "checkpoint candidate mismatch")
    check(checkpoint.get("base_config") == config, "checkpoint config mismatch")
    check(checkpoint.get("unit_contract", {}).get("unit_id") == unit_id and int(checkpoint.get("unit_contract", {}).get("K", -1)) == k, "checkpoint unit/K mismatch")
    checkpoint_state_sha = state_sha(checkpoint["model_state"])
    check(checkpoint_state_sha == tm.get("state_tensor_sha256") == reload.get("state_tensor_sha256"), "checkpoint tensor-state SHA mismatch")
    embedding = np.load(embedding_path, allow_pickle=False)
    gates = np.load(gate_path, allow_pickle=False)
    raw_weights = np.load(raw_weight_path, allow_pickle=False)
    norm_weights = np.load(norm_weight_path, allow_pickle=False)
    check(embedding.shape == (n, 64) and embedding.dtype == np.float32, "embedding shape/dtype mismatch")
    check(bool(np.isfinite(embedding).all()), "embedding NaN/Inf")
    constant_dims = int(np.sum(np.ptp(embedding, axis=0) == 0)) if embedding.ndim == 2 else -1
    check(constant_dims == 0, "embedding has all-constant dimensions")
    check(array_sha(embedding) == tm.get("embedding_sha256") == checkpoint.get("embedding_sha256"), "embedding canonical SHA mismatch")
    check(gates.shape[0] == n and np.isfinite(gates).all() and array_sha(gates) == tm.get("gate_sha256"), "gate artifact mismatch")
    check(len(ids) == n and len(set(ids)) == n, "observation ID cardinality/order uniqueness mismatch")
    with np.load(feature_path, allow_pickle=False) as feature:
        rank_c, quality, support = feature["rank_c"], feature["quality"], feature["support"]
        positive, negative = feature["positive"], feature["negative"]
    check(all(value.shape == (n,) for value in (rank_c, quality, support, positive, negative)), "feature shape mismatch")
    check(all(np.isfinite(value).all() for value in (rank_c, quality, support)), "feature NaN/Inf")
    check(np.all((positive >= 0) & (positive < n)) and np.all((negative >= 0) & (negative < n)), "MNN index bounds mismatch")
    expected_raw, expected_norm = weighted_mnn_weights(candidate, rank_c, quality, support, ids)
    check(np.array_equal(raw_weights, expected_raw) and np.array_equal(norm_weights, expected_norm), "candidate weight semantic mismatch")
    check(array_sha(raw_weights) == tm.get("raw_weight_sha256") and array_sha(norm_weights) == tm.get("normalized_weight_sha256"), "weight SHA mismatch")
    check(int(tm.get("nonzero_weight_count", -1)) == int(np.count_nonzero(raw_weights)), "nonzero weight count mismatch")
    c06 = sp.load_npz(RAW7B / "source" / unit_id / "c06_affinity.npz").tocsr()
    check(c06.shape == (n, n) and np.isfinite(c06.data).all(), "C06 shape/finite mismatch")
    return ({
        "candidate_id": candidate,
        "unit_id": unit_id,
        "dataset": unit["dataset"],
        "seed": int(unit["seed"]),
        "K": k,
        "observation_count": n,
        "night7b_stage": stage,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "label_access": False,
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_state_sha256": checkpoint_state_sha,
        "training_manifest_sha256": sha256_file(tm_path),
        "embedding_file_sha256": sha256_file(embedding_path),
        "embedding_canonical_sha256": array_sha(embedding),
        "embedding_shape": list(embedding.shape),
        "embedding_dtype": str(embedding.dtype),
        "embedding_finite": bool(np.isfinite(embedding).all()),
        "constant_dimension_count": constant_dims,
        "observation_ids_file_sha256": sha256_file(ids_path),
        "ordered_observation_sha256_authority": unit["ordered_observation_sha256"],
        "base_config_sha256": sha256_file(config_path),
        "worker_input_sha256": sha256_file(adapter_input),
        "feature_file_sha256": sha256_file(feature_path),
        "raw_weight_sha256": array_sha(raw_weights),
        "normalized_weight_sha256": array_sha(norm_weights),
        "cuda_tensor_verified": tm.get("cuda_tensor_verified"),
        "gpu_model": tm.get("gpu_model"),
        "epochs": tm.get("epochs"),
    }, embedding, ids, c06)


def triage() -> dict:
    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"label-free semantic triage exceeded {TRIAGE_LIMIT_SECONDS} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TRIAGE_LIMIT_SECONDS)
    require(sha256_file(AMENDMENT) == AMENDMENT_SHA, "immediate-stop amendment SHA mismatch")
    require(not TRIAGE_ROOT.exists(), f"triage root already exists: {TRIAGE_ROOT}")
    firewall = no_label_outputs()
    incident = verify_incident()
    code = code_contract()
    units, training_rows = load_verified_training()
    require(len(units) == 8 and len(training_rows) == 48, "48-cell training authority mismatch")
    expected_keys = [(candidate, unit["unit_id"]) for candidate in CANDIDATES for unit in units]
    lookup = {(row["candidate_id"], row["unit_id"]): row for row in training_rows}
    require(set(lookup) == set(expected_keys), "training candidate/unit mapping mismatch")
    prior = training_map()
    require(set(unit["unit_id"] for unit in units).issubset(prior), "Night-7B R02 authority mapping incomplete")
    baselines = c00_map()
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    try:
        for candidate, unit_id in expected_keys:
            unit = next(item for item in units if item["unit_id"] == unit_id)
            started = time.perf_counter()
            training, embedding, ids, c06 = verify_training_cell(candidate, unit, lookup[(candidate, unit_id)], prior[unit_id])
            k = int(unit["K"])
            az = self_tuning_affinity(embedding, 10, ids)
            affinity = endpoint_affinity("E1_ADAPTER_C06_MEAN", embedding, c06, ids)
        # Nonnegative operands cannot cancel; this support is the exact upper bound
        # for the symmetric endpoint and detects any accidental densification.
            directed_support = ((az != 0).astype(np.int8) + (c06 != 0).astype(np.int8)).tocsr()
            symmetric_support = ((directed_support != 0).astype(np.int8) + (directed_support.T != 0).astype(np.int8)).tocsr()
            symmetric_support.setdiag(0)
            symmetric_support.eliminate_zeros()
            affinity_path = AFFINITY_ROOT / candidate / unit_id / "affinity.npz"
            atomic_sparse(affinity_path, affinity)
            actual = graph_stats(affinity, k=k)
            baseline_path = baselines[unit_id]
            baseline = sp.load_npz(baseline_path)
            baseline_stats = graph_stats(baseline, k=k)
            contract_errors = list(training["errors"])
            ordered_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
            if ordered_sha != unit["ordered_observation_sha256"]: contract_errors.append("ordered observation SHA mismatch")
            if actual["shape"] != [int(unit["observation_count"]), int(unit["observation_count"])]: contract_errors.append("affinity shape mismatch")
            if not actual["finite"] or not actual["nonnegative"]: contract_errors.append("affinity finite/nonnegative mismatch")
            if actual["symmetry_max_error"] != 0.0 or actual["diagonal_max_abs"] != 0.0: contract_errors.append("affinity symmetry/diagonal mismatch")
            if actual["zero_degree_count"] != 0: contract_errors.append("affinity zero-degree rows")
            if actual["nnz"] > int(symmetric_support.nnz): contract_errors.append("affinity support exceeds sparse construction bound")
            if sha256_file(affinity_path) == "": contract_errors.append("affinity file hash missing")
            structural_pathology = bool(
                actual["component_count_ge_k"]
                or actual["largest_component_fraction"] < .95
                or actual["near_isolated_below_1e_6_median"] > 0
            )
            rows.append({
            **training,
            "status": "PASS" if not contract_errors else "FAIL",
            "errors": contract_errors,
            "affinity_file": str(affinity_path),
            "affinity_file_sha256": sha256_file(affinity_path),
            "affinity": actual,
            "self_tuning_nnz": int(az.nnz),
            "c06_nnz": int(c06.nnz),
            "symmetric_sparse_support_bound_nnz": int(symmetric_support.nnz),
            "unexpected_densification": bool(actual["nnz"] > int(symmetric_support.nnz)),
            "baseline_c00_path": str(baseline_path),
            "baseline_c00_file_sha256": sha256_file(baseline_path),
            "baseline_c00": baseline_stats,
            "affinity_vs_c00": {
                "density_ratio": float(actual["density"] / baseline_stats["density"]),
                "degree_median_ratio": float(actual["degree_median"] / baseline_stats["degree_median"]),
                "component_count_delta": int(actual["connected_component_count"] - baseline_stats["connected_component_count"]),
                "largest_component_fraction_delta": float(actual["largest_component_fraction"] - baseline_stats["largest_component_fraction"]),
            },
            "structural_pathology": structural_pathology,
                "ordered_observation_sha256_actual": ordered_sha,
                "runtime_seconds": time.perf_counter() - started,
            })
    finally:
        signal.alarm(0)
    require(len(rows) == 48, "triage cell cardinality mismatch")
    mapping_duplicates = len(expected_keys) - len(set(expected_keys))
    contract_failures = [row for row in rows if row["status"] != "PASS"]
    remaining_pathologies = [row for row in rows if row["candidate_id"] in REMAINING and row["structural_pathology"]]
    by_candidate = []
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate_id"] == candidate]
        by_candidate.append({
            "candidate_id": candidate,
            "cells": len(selected),
            "contract_pass": sum(row["status"] == "PASS" for row in selected),
            "structural_pathology_count": sum(row["structural_pathology"] for row in selected),
            "component_counts": [row["affinity"]["connected_component_count"] for row in selected],
            "density_range": [min(row["affinity"]["density"] for row in selected), max(row["affinity"]["density"] for row in selected)],
        })
    if contract_failures or mapping_duplicates:
        decision = "IMPLEMENTATION_SEMANTICS_INVALID"
    elif len(remaining_pathologies) >= 2:
        decision = "BLOCKED_NUMERICAL_ENDPOINT_SCALABILITY"
    else:
        decision = "PASS_W01_W05_BOUNDED_CONTINUATION_AUTHORIZED"
    report = {
        "schema_version": 1,
        "status": decision,
        "label_access": False,
        "authority_sha256": sha256_file(AMENDMENT),
        "incident": incident,
        "firewall_outputs_present": firewall,
        "code_contract": code,
        "training_cells": 48,
        "triage_total_wall_limit_seconds": TRIAGE_LIMIT_SECONDS,
        "checkpoint_roundtrip_authority_pass": sum(row["status"] == "PASS" for row in rows),
        "candidate_unit_mapping_duplicates": mapping_duplicates,
        "contract_failure_count": len(contract_failures),
        "remaining_structural_pathology_count": len(remaining_pathologies),
        "pathology_rule": "component_count>=K OR largest_component_fraction<0.95 OR degree<1e-6*median",
        "candidate_summary": by_candidate,
        "W00_observed_resource_status": INCIDENT_STATUS,
        "W00_scientific_interpretation": "resource-censored numerical endpoint longtail; no scientific clustering result",
        "W01_W05_formal_queue_authorized": decision == "PASS_W01_W05_BOUNDED_CONTINUATION_AUTHORIZED",
        "cells": rows,
    }
    atomic_json(OUT / "stagew_immediate_semantic_triage.json", report)
    manifest = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "triage_status": decision,
        "affinity_files": [
            {"candidate_id": row["candidate_id"], "unit_id": row["unit_id"], "path": row["affinity_file"], "file_sha256": row["affinity_file_sha256"], "canonical_sha256": row["affinity"]["canonical_affinity_sha256"]}
            for row in rows
        ],
    }
    atomic_json(OUT / "stagew_immediate_triage_affinity_manifest.json", manifest)
    with (OUT / "stagew_immediate_semantic_triage_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["candidate_id", "unit_id", "dataset", "seed", "K", "observation_count", "status", "structural_pathology", "affinity_file_sha256", "canonical_affinity_sha256", "nnz", "density", "degree_min", "degree_median", "degree_max", "zero_degree_count", "connected_component_count", "largest_component_fraction", "runtime_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            graph = row["affinity"]
            writer.writerow({
                "candidate_id": row["candidate_id"], "unit_id": row["unit_id"], "dataset": row["dataset"], "seed": row["seed"], "K": row["K"], "observation_count": row["observation_count"],
                "status": row["status"], "structural_pathology": row["structural_pathology"], "affinity_file_sha256": row["affinity_file_sha256"], "canonical_affinity_sha256": graph["canonical_affinity_sha256"],
                "nnz": graph["nnz"], "density": graph["density"], "degree_min": graph["degree_min"], "degree_median": graph["degree_median"], "degree_max": graph["degree_max"], "zero_degree_count": graph["zero_degree_count"],
                "connected_component_count": graph["connected_component_count"], "largest_component_fraction": graph["largest_component_fraction"], "runtime_seconds": row["runtime_seconds"],
            })
    completion = {
        "schema_version": 1,
        "status": decision,
        "label_access": False,
        "triage_report_sha256": sha256_file(OUT / "stagew_immediate_semantic_triage.json"),
        "affinity_manifest_sha256": sha256_file(OUT / "stagew_immediate_triage_affinity_manifest.json"),
        "cell_table_sha256": sha256_file(OUT / "stagew_immediate_semantic_triage_cells.csv"),
        "triage_raw_inventory": file_inventory(TRIAGE_ROOT),
    }
    atomic_json(INFRA / "stagew_immediate_semantic_triage_completion.json", completion)
    print(json.dumps({k: completion[k] for k in ("status", "label_access", "triage_report_sha256", "affinity_manifest_sha256", "cell_table_sha256")}, sort_keys=True))
    return report


def exact_worker_payload(candidate: str, unit_id: str, training: dict, unit: dict) -> Path:
    payload = WORKER_ROOT / candidate / unit_id / "payload.json"
    require(not payload.exists(), f"worker payload exists: {payload}")
    payload.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, {"candidate": candidate, "unit_id": unit_id, "training": training, "unit": unit})
    return payload


def start_worker(candidate: str, unit: dict, training: dict) -> dict:
    unit_id = unit["unit_id"]
    target = FORMAL / candidate / unit_id / "attempt_001/transform"
    require(not target.exists(), f"transform target already exists: {target}")
    payload = exact_worker_payload(candidate, unit_id, training, unit)
    log = payload.parent / "worker.log"
    handle = log.open("xb")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1"})
    process = subprocess.Popen(
        [str(PYTHON), str(REPO / "scripts/night7c_stagew_resource_bounded.py"), "worker", "--payload", str(payload)],
        cwd=REPO,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    time.sleep(.05)
    identity = bounded.proc_identity(process.pid)
    require(identity["pgid"] == process.pid and identity["sid"] == process.pid, "worker not isolated")
    row = {
        "candidate_id": candidate,
        "unit_id": unit_id,
        "process": process,
        "pid": process.pid,
        "pgid": identity["pgid"],
        "start_ticks": identity["start_ticks"],
        "started_monotonic": time.monotonic(),
        "deadline_monotonic": time.monotonic() + UNIT_LIMIT_SECONDS,
        "payload_path": payload,
        "payload_sha256": sha256_file(payload),
        "log_path": log,
    }
    event("transform_worker_started", candidate_id=candidate, unit_id=unit_id, pid=process.pid, pgid=identity["pgid"], start_ticks=identity["start_ticks"], payload_sha256=row["payload_sha256"])
    return row


def atomic_outcome(candidate: str, unit_id: str, status: str, **extra: object) -> dict:
    return bounded.atomic_outcome(candidate, unit_id, status, **extra)


def censor_worker(task: dict, status: str, classification: str, limit_seconds: float) -> dict:
    # Reuse the already synthetic-tested exact process-group implementation.
    original_log, original_root = bounded.COORD_LOG, bounded.WORKER_ROOT
    bounded.COORD_LOG, bounded.WORKER_ROOT = COORD_LOG, WORKER_ROOT
    try:
        return bounded.censor_worker(task, status, classification, limit_seconds)
    finally:
        bounded.COORD_LOG, bounded.WORKER_ROOT = original_log, original_root


def write_locked_manifests(training_rows: list[dict], units: list[dict], outcomes: dict[tuple[str, str], dict], launched: int, continuation_seconds: float) -> dict:
    all_keys = [(candidate, unit["unit_id"]) for candidate in CANDIDATES for unit in units]
    require(set(outcomes) == set(all_keys), f"outcome cardinality mismatch: {len(outcomes)}/48")
    ordered = [outcomes[key] for key in all_keys]
    status_counts = Counter(row["status"] for row in ordered)
    candidate_audit = []
    for candidate in CANDIDATES:
        statuses = [outcomes[(candidate, unit["unit_id"])]["status"] for unit in units]
        if all(value == "success" for value in statuses):
            status = "ELIGIBLE_COMPLETE_SUCCESS"
        elif any(value.startswith("RESOURCE_CENSORED") or value.startswith("SKIPPED_") for value in statuses):
            status = "INELIGIBLE_RESOURCE_CENSORED"
        else:
            status = "INELIGIBLE_NATURAL_FAILURE"
        candidate_audit.append({"candidate_id": candidate, "status": status, "statuses": statuses})
    eligible = [row["candidate_id"] for row in candidate_audit if row["status"] == "ELIGIBLE_COMPLETE_SUCCESS"]
    plan = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "w00_mode": "user_immediate_resource_stop_after_extreme_longtail",
        "all_cells": [{"registry_index": index, "candidate_id": key[0], "unit_id": key[1], "outcome_status": ordered[index]["status"]} for index, key in enumerate(all_keys)],
        "candidate_eligibility": candidate_audit,
        "eligible_weighted_mnn_candidates": eligible,
        "workers": WORKERS,
        "unit_wall_limit_seconds": UNIT_LIMIT_SECONDS,
        "continuation_wall_limit_seconds": TOTAL_LIMIT_SECONDS,
        "scientific_retry": 0,
        "fallback": 0,
        "triage_report_sha256": sha256_file(OUT / "stagew_immediate_semantic_triage.json"),
    }
    atomic_json(OUT / "stagew_resource_bounded_plan_and_eligibility.json", plan)
    require(not (OUT / "weighted_mnn_training_manifest.json").exists(), "training aggregate already exists")
    require(not (OUT / "weighted_mnn_transform_manifest.json").exists(), "transform aggregate already exists")
    training_manifest = {
        "schema_version": 1, "status": "LOCKED_PRE_LABEL", "label_access": False,
        "planned_training": 48, "training_attempts": 48, "successful_training": 48, "failed_training": 0,
        "scientific_retry": 0, "fallback_count": 0, "reconstructed_from_immutable_completed_cells": True,
        "new_training_during_resume": 0, "training_cells": training_rows,
    }
    transform_manifest = {
        "schema_version": 1, "status": "LOCKED_PRE_LABEL", "label_access": False,
        "planned_transforms": 48, "transform_attempts": 48, "transform_outcomes": 48,
        "physical_transform_invocations": 2 + launched, "prior_infrastructure_aborted_invocations": 1,
        "successful_transforms": status_counts.get("success", 0), "failed_transforms": 48 - status_counts.get("success", 0),
        "failed_natural_transforms": status_counts.get("scientific_numerical_failure", 0),
        "resource_censored_transforms": sum(value for key, value in status_counts.items() if key.startswith("RESOURCE_CENSORED")),
        "skipped_transforms": sum(value for key, value in status_counts.items() if key.startswith("SKIPPED_")),
        "status_counts": dict(status_counts), "scientific_retry": 0, "fallback_count": 0,
        "w00_mode": "user_immediate_resource_stop_after_extreme_longtail",
        "runtime_seconds": continuation_seconds,
        "resource_bounded_continuation_seconds": continuation_seconds,
        "resource_cutoff_contract": {
            "authority_sha256": AMENDMENT_SHA, "workers": WORKERS,
            "unit_wall_limit_seconds": UNIT_LIMIT_SECONDS, "continuation_wall_limit_seconds": TOTAL_LIMIT_SECONDS,
            "scientific_retry": 0, "fallback": 0, "algorithm_unchanged": True,
            "worker_call_chain": "immediate coordinator -> resource-bounded worker -> exact imported transform_cell(training, unit)",
        },
        "eligible_weighted_mnn_candidates": eligible, "candidate_eligibility": candidate_audit, "transforms": ordered,
    }
    atomic_json(OUT / "weighted_mnn_training_manifest.json", training_manifest)
    atomic_json(OUT / "weighted_mnn_transform_manifest.json", transform_manifest)
    summary = {
        "schema_version": 1, "status": "LOCKED_PRE_LABEL", "label_access": False,
        "status_counts": dict(status_counts), "eligible_weighted_mnn_candidates": eligible,
        "training_manifest_sha256": sha256_file(OUT / "weighted_mnn_training_manifest.json"),
        "transform_manifest_sha256": sha256_file(OUT / "weighted_mnn_transform_manifest.json"),
        "plan_sha256": sha256_file(OUT / "stagew_resource_bounded_plan_and_eligibility.json"),
    }
    atomic_json(INFRA / "stagew_immediate_bounded_completion.json", summary)
    return summary


def coordinator() -> dict:
    require(sha256_file(AMENDMENT) == AMENDMENT_SHA, "immediate amendment SHA mismatch")
    require(not COORD_LOG.exists(), "immediate coordinator log already exists")
    require(no_label_outputs() == {name: False for name in no_label_outputs()}, "firewall output mismatch")
    triage_report = json.loads((OUT / "stagew_immediate_semantic_triage.json").read_text(encoding="utf-8"))
    require(triage_report["status"] == "PASS_W01_W05_BOUNDED_CONTINUATION_AUTHORIZED" and triage_report["label_access"] is False, "triage did not authorize queue")
    completion = json.loads((INFRA / "stagew_immediate_semantic_triage_completion.json").read_text(encoding="utf-8"))
    require(completion["triage_report_sha256"] == sha256_file(OUT / "stagew_immediate_semantic_triage.json"), "triage report SHA changed")
    units, training_rows = load_verified_training()
    unit_lookup = {unit["unit_id"]: unit for unit in units}
    training_lookup = {(row["candidate_id"], row["unit_id"]): row for row in training_rows}
    outcomes: dict[tuple[str, str], dict] = {}
    # Mechanically lock the user-stopped W00 family before starting any new process.
    outcomes[(W00, units[0]["unit_id"])] = atomic_outcome(
        W00, units[0]["unit_id"], INCIDENT_STATUS,
        failure_type="resource_censored_user_stop_after_extreme_longtail",
        candidate_eligibility="INELIGIBLE_RESOURCE_CENSORED",
        natural_scientific_terminal=False,
        incident_path=str(INCIDENT),
        incident_inventory_sha256=sha256_file(INCIDENT / "incident_inventory.json"),
    )
    for unit in units[1:]:
        outcomes[(W00, unit["unit_id"])] = atomic_outcome(
            W00, unit["unit_id"], "SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER",
            failure_type="candidate_resource_circuit_breaker",
            triggering_unit_id=units[0]["unit_id"], triggering_status=INCIDENT_STATUS,
            natural_scientific_terminal=False,
        )
    continuation_start = time.monotonic()
    continuation_deadline = continuation_start + TOTAL_LIMIT_SECONDS
    launched = 0
    total_cutoff = False
    event("bounded_queue_started", candidates=REMAINING, units=[unit["unit_id"] for unit in units], workers=WORKERS, label_access=False)
    for candidate in REMAINING:
        keys = [(candidate, unit["unit_id"]) for unit in units]
        pending = list(keys)
        circuit = False
        while pending and not circuit and not total_cutoff:
            if time.monotonic() >= continuation_deadline:
                total_cutoff = True
                break
            batch = pending[:WORKERS]
            running = {key: start_worker(key[0], unit_lookup[key[1]], training_lookup[key]) for key in batch}
            launched += len(running)
            while running:
                for key, task in list(running.items()):
                    process = task["process"]
                    manifest = FORMAL / key[0] / key[1] / "attempt_001/transform/transform_manifest.json"
                    if process.poll() is not None:
                        require(process.returncode == 0, f"worker exited without terminal manifest: {key}, rc={process.returncode}")
                        outcomes[key] = bounded.validate_natural(key[0], unit_lookup[key[1]], training_lookup[key])
                        event("transform_natural_terminal", candidate_id=key[0], unit_id=key[1], status=outcomes[key]["status"], manifest_sha256=sha256_file(manifest))
                        del running[key]
                        continue
                    if manifest.is_file():
                        process.wait(timeout=60)
                        require(process.returncode == 0, f"worker nonzero after manifest: {key}")
                        outcomes[key] = bounded.validate_natural(key[0], unit_lookup[key[1]], training_lookup[key])
                        event("transform_natural_terminal", candidate_id=key[0], unit_id=key[1], status=outcomes[key]["status"], manifest_sha256=sha256_file(manifest))
                        del running[key]
                        continue
                    if time.monotonic() >= continuation_deadline:
                        outcomes[key] = censor_worker(task, "RESOURCE_CENSORED_TOTAL_WALLTIME_12H", "stagew_immediate_total_resource_censored_12h_20260820", TOTAL_LIMIT_SECONDS)
                        del running[key]
                        total_cutoff = True
                        continue
                    if time.monotonic() >= task["deadline_monotonic"]:
                        outcomes[key] = censor_worker(task, "RESOURCE_CENSORED_WALLTIME_60M", "stagew_immediate_unit_resource_censored_60m_20260820", UNIT_LIMIT_SECONDS)
                        del running[key]
                        circuit = True
                time.sleep(.25)
            pending = [key for key in keys if key not in outcomes]
        if total_cutoff:
            for later_candidate in REMAINING[REMAINING.index(candidate):]:
                for unit in units:
                    key = (later_candidate, unit["unit_id"])
                    if key not in outcomes:
                        outcomes[key] = atomic_outcome(key[0], key[1], "SKIPPED_TOTAL_RESOURCE_CUTOFF_12H", failure_type="total_resource_cutoff", triggering_status="RESOURCE_CENSORED_TOTAL_WALLTIME_12H", natural_scientific_terminal=False)
            break
        if circuit:
            trigger = next(key for key in keys if outcomes.get(key, {}).get("status") == "RESOURCE_CENSORED_WALLTIME_60M")
            for key in pending:
                outcomes[key] = atomic_outcome(key[0], key[1], "SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER", failure_type="candidate_resource_circuit_breaker", triggering_unit_id=trigger[1], triggering_status=outcomes[trigger]["status"], natural_scientific_terminal=False)
    summary = write_locked_manifests(training_rows, units, outcomes, launched, time.monotonic() - continuation_start)
    event("bounded_queue_complete", summary=summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("triage", "coordinator"))
    args = parser.parse_args()
    if args.mode == "triage":
        triage()
    else:
        coordinator()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
