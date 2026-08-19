#!/usr/bin/env python3
"""Isolated-process driver for the label-free Night-7C Stage-W triage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from SpaLORA.night7a_consensus import atomic_json, atomic_sparse, sha256_file  # noqa: E402
from SpaLORA.night7b_adaptive import self_tuning_affinity  # noqa: E402
from scripts.night7b_adapter_stage import endpoint_affinity  # noqa: E402
from scripts.night7c_p1 import training_map  # noqa: E402
from scripts.night7c_replay_recovery_stage_w import CANDIDATES, OUT, RAW, pilot_units  # noqa: E402
from scripts.night7c_replay_recovery_stage_w_resume import load_verified_training  # noqa: E402
from scripts import night7c_stagew_immediate_triage as base  # noqa: E402
from scripts import night7c_stagew_resource_bounded as bounded  # noqa: E402

PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")
FORMAL = RAW / "stage_w/formal"
TRIAGE_ROOT = RAW / "stage_w_semantic_triage"
CELL_ROOT = TRIAGE_ROOT / "cells"
AFFINITY_ROOT = TRIAGE_ROOT / "preclustering_affinity"
INFRA = RAW / "infrastructure"
WORKERS = 4
CELL_LIMIT_SECONDS = 5 * 60.0
TOTAL_LIMIT_SECONDS = 30 * 60.0
REMAINING = CANDIDATES[1:]
CRASH_ARCHIVE = RAW / "invalid_attempts/stagew_immediate_triage_native_crash_20260819T153708Z"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cell_result_path(candidate: str, unit_id: str) -> Path:
    return CELL_ROOT / candidate / unit_id / "cell.json"


def build_cell(candidate: str, unit_id: str) -> dict:
    require(candidate in CANDIDATES, "candidate is not registered")
    unit = next(row for row in pilot_units() if row["unit_id"] == unit_id)
    row = json.loads((FORMAL / candidate / unit_id / "attempt_001/cell_manifest.json").read_text(encoding="utf-8"))
    prior = training_map()
    training, embedding, ids, c06 = base.verify_training_cell(candidate, unit, row, prior[unit_id])
    k = int(unit["K"])
    az = self_tuning_affinity(embedding, 10, ids)
    affinity = endpoint_affinity("E1_ADAPTER_C06_MEAN", embedding, c06, ids)
    directed_support = ((az != 0).astype(np.int8) + (c06 != 0).astype(np.int8)).tocsr()
    symmetric_support = ((directed_support != 0).astype(np.int8) + (directed_support.T != 0).astype(np.int8)).tocsr()
    symmetric_support.setdiag(0)
    symmetric_support.eliminate_zeros()
    affinity_path = AFFINITY_ROOT / candidate / unit_id / "affinity.npz"
    atomic_sparse(affinity_path, affinity)
    actual = base.graph_stats(affinity, k=k)
    baseline_path = base.c00_map()[unit_id]
    baseline = base.graph_stats(sp.load_npz(baseline_path), k=k)
    errors = list(training["errors"])
    ordered_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if ordered_sha != unit["ordered_observation_sha256"]:
        errors.append("ordered observation SHA mismatch")
    if actual["shape"] != [int(unit["observation_count"]), int(unit["observation_count"])]:
        errors.append("affinity shape mismatch")
    if not actual["finite"] or not actual["nonnegative"]:
        errors.append("affinity finite/nonnegative mismatch")
    if actual["symmetry_max_error"] != 0.0 or actual["diagonal_max_abs"] != 0.0:
        errors.append("affinity symmetry/diagonal mismatch")
    if actual["zero_degree_count"] != 0:
        errors.append("affinity zero-degree rows")
    if actual["nnz"] > int(symmetric_support.nnz):
        errors.append("affinity support exceeds sparse construction bound")
    pathology = bool(
        actual["component_count_ge_k"]
        or actual["largest_component_fraction"] < .95
        or actual["near_isolated_below_1e_6_median"] > 0
    )
    result = {
        **training,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "ordered_observation_sha256_actual": ordered_sha,
        "affinity_file": str(affinity_path),
        "affinity_file_sha256": sha256_file(affinity_path),
        "affinity": actual,
        "self_tuning_nnz": int(az.nnz),
        "c06_nnz": int(c06.nnz),
        "symmetric_sparse_support_bound_nnz": int(symmetric_support.nnz),
        "unexpected_densification": bool(actual["nnz"] > int(symmetric_support.nnz)),
        "baseline_c00_path": str(baseline_path),
        "baseline_c00_file_sha256": sha256_file(baseline_path),
        "baseline_c00": baseline,
        "affinity_vs_c00": {
            "density_ratio": float(actual["density"] / baseline["density"]),
            "degree_median_ratio": float(actual["degree_median"] / baseline["degree_median"]),
            "component_count_delta": int(actual["connected_component_count"] - baseline["connected_component_count"]),
            "largest_component_fraction_delta": float(actual["largest_component_fraction"] - baseline["largest_component_fraction"]),
        },
        "structural_pathology": pathology,
        "label_access": False,
    }
    target = cell_result_path(candidate, unit_id)
    require(not target.exists(), f"cell result exists: {target}")
    atomic_json(target, result)
    print(json.dumps({"candidate_id": candidate, "unit_id": unit_id, "status": result["status"], "structural_pathology": pathology}, sort_keys=True))
    return result


def process_identity(pid: int) -> dict:
    return bounded.proc_identity(pid)


def stop_group(task: dict, reason: str) -> dict:
    identity = process_identity(task["pid"])
    require(identity["start_ticks"] == task["start_ticks"] and identity["pgid"] == task["pgid"] == task["pid"], "triage worker identity changed")
    os.killpg(task["pgid"], signal.SIGSTOP)
    for _ in range(500):
        current = process_identity(task["pid"])
        if current["state"] in {"T", "t"}:
            break
        time.sleep(.01)
    termination = bounded.stop_exact_group(task["pid"], task["start_ticks"], task["pgid"], already_stopped=True, grace_seconds=10.0)
    return {"reason": reason, "identity": identity, "termination": termination}


def start_cell(candidate: str, unit_id: str) -> dict:
    log = CELL_ROOT / candidate / unit_id / "cell.log"
    require(not log.exists(), f"cell log exists: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("xb")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__)), "cell", "--candidate", candidate, "--unit-id", unit_id],
        cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    time.sleep(.05)
    identity = process_identity(process.pid)
    require(identity["pgid"] == process.pid and identity["sid"] == process.pid, "triage cell process is not isolated")
    return {
        "candidate_id": candidate, "unit_id": unit_id, "process": process,
        "pid": process.pid, "pgid": identity["pgid"], "start_ticks": identity["start_ticks"],
        "started_monotonic": time.monotonic(), "deadline_monotonic": time.monotonic() + CELL_LIMIT_SECONDS,
        "log": log,
    }


def aggregate(rows: list[dict], code: dict, incident: dict, firewall: dict) -> dict:
    expected = [(candidate, unit["unit_id"]) for candidate in CANDIDATES for unit in pilot_units()]
    require([(row["candidate_id"], row["unit_id"]) for row in rows] == expected, "triage aggregation order mismatch")
    failures = [row for row in rows if row["status"] != "PASS"]
    pathologies = [row for row in rows if row["candidate_id"] in REMAINING and row["structural_pathology"]]
    summaries = []
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate_id"] == candidate]
        summaries.append({
            "candidate_id": candidate, "cells": len(selected),
            "contract_pass": sum(row["status"] == "PASS" for row in selected),
            "structural_pathology_count": sum(row["structural_pathology"] for row in selected),
            "component_counts": [row["affinity"]["connected_component_count"] for row in selected],
            "density_range": [min(row["affinity"]["density"] for row in selected), max(row["affinity"]["density"] for row in selected)],
        })
    if failures:
        decision = "IMPLEMENTATION_SEMANTICS_INVALID"
    elif len(pathologies) >= 2:
        decision = "BLOCKED_NUMERICAL_ENDPOINT_SCALABILITY"
    else:
        decision = "PASS_W01_W05_BOUNDED_CONTINUATION_AUTHORIZED"
    report = {
        "schema_version": 1, "status": decision, "label_access": False,
        "authority_sha256": sha256_file(base.AMENDMENT), "incident": incident,
        "prior_infrastructure_only_triage_crash": {
            "path": str(CRASH_ARCHIVE), "manifest_sha256": sha256_file(CRASH_ARCHIVE / "incident_manifest.json"),
            "scientific_retry": False,
        },
        "firewall_outputs_present": firewall, "code_contract": code,
        "training_cells": 48, "checkpoint_roundtrip_authority_pass": sum(row["status"] == "PASS" for row in rows),
        "candidate_unit_mapping_duplicates": 0, "contract_failure_count": len(failures),
        "remaining_structural_pathology_count": len(pathologies),
        "pathology_rule": "component_count>=K OR largest_component_fraction<0.95 OR degree<1e-6*median",
        "candidate_summary": summaries,
        "W00_observed_resource_status": base.INCIDENT_STATUS,
        "W00_scientific_interpretation": "resource-censored numerical endpoint longtail; no scientific clustering result",
        "W01_W05_formal_queue_authorized": decision == "PASS_W01_W05_BOUNDED_CONTINUATION_AUTHORIZED",
        "triage_execution": {"isolated_process_per_cell": True, "workers": WORKERS, "cell_wall_limit_seconds": CELL_LIMIT_SECONDS, "total_wall_limit_seconds": TOTAL_LIMIT_SECONDS, "native_crash_retry": 0},
        "cells": rows,
    }
    atomic_json(OUT / "stagew_immediate_semantic_triage.json", report)
    manifest = {
        "schema_version": 1, "status": "LOCKED_PRE_LABEL", "label_access": False, "triage_status": decision,
        "affinity_files": [{"candidate_id": row["candidate_id"], "unit_id": row["unit_id"], "path": row["affinity_file"], "file_sha256": row["affinity_file_sha256"], "canonical_sha256": row["affinity"]["canonical_affinity_sha256"]} for row in rows],
    }
    atomic_json(OUT / "stagew_immediate_triage_affinity_manifest.json", manifest)
    with (OUT / "stagew_immediate_semantic_triage_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["candidate_id", "unit_id", "dataset", "seed", "K", "observation_count", "status", "structural_pathology", "affinity_file_sha256", "canonical_affinity_sha256", "nnz", "density", "degree_min", "degree_median", "degree_max", "zero_degree_count", "connected_component_count", "largest_component_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            graph = row["affinity"]
            writer.writerow({"candidate_id": row["candidate_id"], "unit_id": row["unit_id"], "dataset": row["dataset"], "seed": row["seed"], "K": row["K"], "observation_count": row["observation_count"], "status": row["status"], "structural_pathology": row["structural_pathology"], "affinity_file_sha256": row["affinity_file_sha256"], "canonical_affinity_sha256": graph["canonical_affinity_sha256"], "nnz": graph["nnz"], "density": graph["density"], "degree_min": graph["degree_min"], "degree_median": graph["degree_median"], "degree_max": graph["degree_max"], "zero_degree_count": graph["zero_degree_count"], "connected_component_count": graph["connected_component_count"], "largest_component_fraction": graph["largest_component_fraction"]})
    completion = {
        "schema_version": 1, "status": decision, "label_access": False,
        "triage_report_sha256": sha256_file(OUT / "stagew_immediate_semantic_triage.json"),
        "affinity_manifest_sha256": sha256_file(OUT / "stagew_immediate_triage_affinity_manifest.json"),
        "cell_table_sha256": sha256_file(OUT / "stagew_immediate_semantic_triage_cells.csv"),
        "triage_raw_inventory": base.file_inventory(TRIAGE_ROOT),
    }
    atomic_json(INFRA / "stagew_immediate_semantic_triage_completion.json", completion)
    print(json.dumps({key: completion[key] for key in ("status", "label_access", "triage_report_sha256", "affinity_manifest_sha256", "cell_table_sha256")}, sort_keys=True))
    return report


def driver() -> dict:
    require(sha256_file(base.AMENDMENT) == base.AMENDMENT_SHA, "immediate amendment SHA mismatch")
    require(not TRIAGE_ROOT.exists(), "triage root already exists")
    require(CRASH_ARCHIVE.is_dir(), "prior infrastructure crash archive missing")
    firewall = base.no_label_outputs()
    incident = base.verify_incident()
    code = base.code_contract()
    units, training_rows = load_verified_training()
    require(len(units) == 8 and len(training_rows) == 48, "training authority is not 48/48")
    keys = [(candidate, unit["unit_id"]) for candidate in CANDIDATES for unit in units]
    require(len(keys) == len(set(keys)) == 48, "candidate/unit key mismatch")
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=False)
    pending = list(keys)
    running: dict[tuple[str, str], dict] = {}
    complete: dict[tuple[str, str], dict] = {}
    failures: list[dict] = []
    started = time.monotonic(); total_deadline = started + TOTAL_LIMIT_SECONDS
    while pending or running:
        while pending and len(running) < WORKERS and not failures:
            key = pending.pop(0)
            running[key] = start_cell(*key)
        for key, task in list(running.items()):
            process = task["process"]
            if process.poll() is not None:
                result_path = cell_result_path(*key)
                if process.returncode == 0 and result_path.is_file():
                    complete[key] = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    failures.append({"candidate_id": key[0], "unit_id": key[1], "status": "NATIVE_PROCESS_CRASH", "returncode": process.returncode, "log": str(task["log"]), "log_sha256": sha256_file(task["log"])})
                del running[key]
                continue
            if time.monotonic() >= task["deadline_monotonic"]:
                termination = stop_group(task, "TRIAGE_CELL_WALLTIME_5M")
                failures.append({"candidate_id": key[0], "unit_id": key[1], "status": "TRIAGE_CELL_TIMEOUT", "termination": termination, "log": str(task["log"]), "log_sha256": sha256_file(task["log"])})
                del running[key]
        if time.monotonic() >= total_deadline and (pending or running):
            for key, task in list(running.items()):
                failures.append({"candidate_id": key[0], "unit_id": key[1], "status": "TRIAGE_TOTAL_TIMEOUT", "termination": stop_group(task, "TRIAGE_TOTAL_WALLTIME_30M")})
                del running[key]
            pending.clear()
        if failures:
            for key, task in list(running.items()):
                failures.append({"candidate_id": key[0], "unit_id": key[1], "status": "CANCELLED_AFTER_PEER_INFRA_FAILURE", "termination": stop_group(task, "PEER_INFRA_FAILURE")})
                del running[key]
            break
        time.sleep(.1)
    if failures:
        blocked = {"schema_version": 1, "status": "PRECHECK_INFRASTRUCTURE_INVALID", "label_access": False, "scientific_retry": 0, "completed_cells": len(complete), "failures": failures, "pending_cells": [{"candidate_id": key[0], "unit_id": key[1]} for key in pending], "raw_inventory": base.file_inventory(TRIAGE_ROOT)}
        atomic_json(INFRA / "stagew_immediate_semantic_triage_infrastructure_block.json", blocked)
        print(json.dumps({"status": blocked["status"], "completed_cells": len(complete), "failure_count": len(failures)}, sort_keys=True))
        return blocked
    rows = [complete[key] for key in keys]
    return aggregate(rows, code, incident, firewall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("driver", "cell"))
    parser.add_argument("--candidate")
    parser.add_argument("--unit-id")
    args = parser.parse_args()
    if args.mode == "cell":
        require(args.candidate and args.unit_id, "cell requires candidate and unit")
        build_cell(args.candidate, args.unit_id)
    else:
        driver()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
