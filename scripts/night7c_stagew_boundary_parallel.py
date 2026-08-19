#!/usr/bin/env python3
"""Preserve the running W00 boundary, then run the locked remaining 47 transforms."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import (  # noqa: E402
    atomic_json,
    canonical_partition,
    sha256_file,
)
from scripts.night7c_replay_recovery_stage_w import (  # noqa: E402
    CANDIDATES,
    OUT,
    RAW,
    pilot_units,
    sampler,
    transform_cell,
)
from scripts.night7c_replay_recovery_stage_w_resume import load_verified_training  # noqa: E402

RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
FORMAL = RAW / "stage_w/formal"
INFRA = RAW / "infrastructure"
GUARD_LOG = INFRA / "w00_boundary_guard.jsonl"
COORD_LOG = INFRA / "stagew_boundary_parallel_coordinator.jsonl"
DRIVER_PID = 929
DRIVER_START_TICKS = 489569013
DRIVER_PGID = 929
DRIVER_CMDLINE = (
    "/root/miniconda3/envs/SpaLORA/bin/python "
    "scripts/night7c_replay_recovery_stage_w_resume.py"
)
W00_KEY = ("W00_FILTER75", "u000")
W00_MANIFEST = FORMAL / W00_KEY[0] / W00_KEY[1] / "attempt_001/transform/transform_manifest.json"
INCIDENT = RAW / "invalid_attempts/stagew_post_w00_boundary_spill_20260819"
P2 = OUT / "p2_runtime_contract.json"
OLD_AMENDMENT = REPO / "protocols/night7c_replay_recovery/SpaLORA_Night7C_StageW_LongTail_Parallel_Recovery_Amendment_2026-08-19.md"
NEW_AMENDMENT = REPO / "protocols/night7c_replay_recovery/SpaLORA_Night7C_StageW_W00_Boundary_Preserve_Parallel_Amendment_2026-08-19.md"
OLD_AMENDMENT_SHA = "fb0bc530fb7d12d9f57907498510e7b32a73c8cd89600ec4bcbdd49bbd990d31"
NEW_AMENDMENT_SHA = "98387915dec4d5a1d5a6fb5963889fbcb90e7088dd2385fdc9f4f5557c109653"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(name: str, **values: object) -> None:
    INFRA.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": now(), "event": name, **values}
    with COORD_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def proc_identity(pid: int) -> dict:
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat_text.rfind(")")
    fields = stat_text[close + 2 :].split()
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
        "cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
    }


def same_driver() -> dict:
    value = proc_identity(DRIVER_PID)
    require(value["start_ticks"] == DRIVER_START_TICKS, "driver PID start-time mismatch")
    require(value["pgid"] == DRIVER_PGID, "driver PGID mismatch")
    require(value["cmdline"] == DRIVER_CMDLINE, "driver command line mismatch")
    return value


def guard_events() -> list[dict]:
    require(GUARD_LOG.is_file(), "guard log missing")
    rows = [json.loads(line) for line in GUARD_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(sum(x["event"] == "guard_armed" for x in rows) == 1, "guard arm cardinality mismatch")
    return rows


def wait_for_guard() -> list[dict]:
    event("coordinator_waiting_for_guard", label_access=False)
    while True:
        rows = guard_events()
        if any(x["event"] == "driver_stop_confirmed" for x in rows):
            require(rows[-1]["event"] == "driver_stop_confirmed", "guard terminal event is not stop confirmation")
            return rows
        bad = [x for x in rows if x["event"] not in {"guard_armed", "sigstop_sent"}]
        require(not bad, f"guard entered non-success terminal state: {bad[-1] if bad else None}")
        time.sleep(1.0)


def load_clusters(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows and set(rows[0]) == {"observation_id", "cluster"}, "cluster CSV schema mismatch")
    return [x["observation_id"] for x in rows], np.asarray([int(x["cluster"]) for x in rows], dtype=np.int64)


def validate_w00(training: dict, unit: dict) -> tuple[dict, dict]:
    require(W00_MANIFEST.is_file(), "W00 terminal manifest missing after guard stop")
    row = json.loads(W00_MANIFEST.read_text(encoding="utf-8"))
    required = {
        "candidate_id": W00_KEY[0],
        "unit_id": W00_KEY[1],
        "label_access": False,
        "formal_transform": True,
        "retry": False,
        "fallback": False,
    }
    for key, expected in required.items():
        require(row.get(key) == expected, f"W00 manifest mismatch for {key}: {row.get(key)!r}")
    require(row.get("status") in {"success", "scientific_numerical_failure"}, "W00 terminal status invalid")
    manifest_dir = W00_MANIFEST.parent
    audit = {
        "manifest": str(W00_MANIFEST),
        "manifest_sha256": sha256_file(W00_MANIFEST),
        "manifest_size": W00_MANIFEST.stat().st_size,
        "status": row["status"],
        "label_access": False,
        "artifacts": [],
    }
    if row["status"] == "success":
        affinity_path = manifest_dir / "affinity.npz"
        clusters_path = manifest_dir / "clusters.csv"
        require(affinity_path.is_file() and clusters_path.is_file(), "W00 success artifacts missing")
        affinity = sp.load_npz(affinity_path)
        ids, labels = load_clusters(clusters_path)
        expected_ids = [x.strip() for x in (RAW7B / "source" / unit["unit_id"] / "observation_ids.txt").read_text().splitlines() if x.strip()]
        require(ids == expected_ids, "W00 cluster observation order mismatch")
        require(affinity.shape == (len(ids), len(ids)), "W00 affinity shape mismatch")
        require(sha256_file(affinity_path) == row.get("affinity_file_sha256"), "W00 affinity file SHA mismatch")
        require(sparse_sha(affinity) == row.get("canonical_affinity_sha256"), "W00 canonical affinity SHA mismatch")
        require(sha256_file(clusters_path) == row.get("clusters_file_sha256"), "W00 clusters file SHA mismatch")
        require(array_sha(canonical_partition(labels)) == row.get("canonical_partition_sha256"), "W00 partition SHA mismatch")
        require(int(row.get("cluster_count")) == int(unit["K"]), "W00 cluster count differs from locked K")
        require(len(np.unique(labels)) == int(unit["K"]), "W00 cluster file unique count differs from locked K")
        require(row.get("training_manifest_sha256") == training.get("training_manifest_sha256"), "W00 training manifest link mismatch")
        embedding_path = W00_MANIFEST.parent.parent / "worker/embedding.npy"
        embedding = np.load(embedding_path, allow_pickle=False)
        require(array_sha(embedding) == row.get("embedding_sha256"), "W00 embedding SHA mismatch")
        audit["artifacts"] = [
            {"path": str(affinity_path), "sha256": sha256_file(affinity_path), "size": affinity_path.stat().st_size},
            {"path": str(clusters_path), "sha256": sha256_file(clusters_path), "size": clusters_path.stat().st_size},
        ]
        audit["cluster_count"] = int(row["cluster_count"])
    else:
        require(bool(row.get("failure_type")), "W00 failure manifest lacks failure_type")
        require("failure_message" in row, "W00 failure manifest lacks failure_message")
        audit["failure_type"] = row["failure_type"]
        audit["failure_message"] = row["failure_message"]
    atomic_json(INFRA / "w00_boundary_validation.json", audit)
    event("w00_terminal_validated", audit=audit)
    return row, audit


def terminate_old_driver() -> dict:
    before = same_driver()
    require(before["state"] in {"T", "t"}, f"driver is not stopped: {before['state']}")
    os.kill(DRIVER_PID, signal.SIGTERM)
    event("driver_sigterm_sent", driver=before)
    os.kill(DRIVER_PID, signal.SIGCONT)
    event("driver_sigcont_sent_for_term_delivery", driver_pid=DRIVER_PID)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            current = proc_identity(DRIVER_PID)
        except FileNotFoundError:
            result = {"terminated_by": "SIGTERM", "driver": before}
            event("driver_terminated", **result)
            return result
        require(current["start_ticks"] == DRIVER_START_TICKS, "PID reused while terminating driver")
        if current["state"] == "Z":
            result = {"terminated_by": "SIGTERM", "driver": before, "observed_state": "Z"}
            event("driver_terminated", **result)
            return result
        time.sleep(0.1)
    current = same_driver()
    os.kill(DRIVER_PID, signal.SIGKILL)
    result = {"terminated_by": "SIGKILL_AFTER_60S", "driver": current}
    event("driver_sigkill_sent", **result)
    return result


def file_inventory(path: Path) -> list[dict]:
    rows = []
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rows.append({"relative_path": str(item.relative_to(path)), "sha256": sha256_file(item), "size": item.stat().st_size})
    return rows


def isolate_boundary_spill() -> list[dict]:
    expected = W00_MANIFEST.parent
    paths = sorted(FORMAL.glob("*/*/attempt_001/transform"))
    spills = [x for x in paths if x != expected]
    rows = []
    for source in spills:
        require(not (source / "transform_manifest.json").exists(), f"unexpected complete post-boundary transform: {source}")
        inventory = file_inventory(source)
        relative = source.relative_to(RAW)
        target = INCIDENT / relative
        require(not target.exists(), f"spill archive target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        rows.append({"source": str(source), "archive": str(target), "files": inventory})
    audit = {"schema_version": 1, "classification": "post_w00_boundary_spill", "label_access": False, "spill_count": len(rows), "spills": rows}
    atomic_json(INCIDENT / "incident_manifest.json", audit)
    event("boundary_spill_isolated", audit=audit)
    return rows


def _parallel_transform(payload: tuple[dict, dict]) -> dict:
    training, unit = payload
    return transform_cell(training, unit)


def verify_authority() -> dict:
    require(sha256_file(OLD_AMENDMENT) == OLD_AMENDMENT_SHA, "old amendment SHA mismatch")
    require(sha256_file(NEW_AMENDMENT) == NEW_AMENDMENT_SHA, "boundary amendment SHA mismatch")
    p2 = json.loads(P2.read_text(encoding="utf-8"))
    require(p2.get("status") == "PASS", "P2 status not PASS")
    require(p2.get("canonical_exact") is True, "P2 scientific artifacts were not exact")
    require(int(p2.get("parity_cells")) == 4, "P2 parity cardinality mismatch")
    require(p2.get("label_access") is False, "P2 label firewall mismatch")
    require(abs(float(p2.get("speedup")) - 1.1749768413491453) < 1e-12, "P2 speedup mismatch")
    transform_source = inspect.getsource(transform_cell).encode("utf-8")
    authority = {
        "schema_version": 1,
        "status": "PASS_BY_AUTHORIZED_BOUNDARY_PARALLEL_AMENDMENT",
        "label_access": False,
        "p2_report": str(P2),
        "p2_report_sha256": sha256_file(P2),
        "p2": p2,
        "old_amendment_sha256": sha256_file(OLD_AMENDMENT),
        "boundary_amendment_sha256": sha256_file(NEW_AMENDMENT),
        "transform_module": inspect.getsourcefile(transform_cell),
        "transform_module_sha256": sha256_file(Path(inspect.getsourcefile(transform_cell))),
        "transform_function_source_sha256": hashlib.sha256(transform_source).hexdigest(),
        "workers": 4,
        "worker_environment": {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"},
        "algorithm_unchanged": True,
        "gpu_backend": False,
        "scientific_retry": 0,
    }
    atomic_json(OUT / "stagew_parallel_amendment_contract.json", authority)
    return authority


def run_remaining(training_rows: list[dict], units: list[dict], w00_row: dict, authority: dict) -> dict:
    all_cells = [(candidate, unit["unit_id"]) for candidate in CANDIDATES for unit in units]
    require(len(all_cells) == 48 and len(set(all_cells)) == 48, "locked Stage-W matrix is not 48 unique cells")
    require(all_cells[0] == W00_KEY, f"W00 is not first registry cell: {all_cells[0]}")
    remaining = [key for key in all_cells if key != W00_KEY]
    require(len(remaining) == 47, f"mechanical remaining cardinality is {len(remaining)}, not 47")
    training_lookup = {(x["candidate_id"], x["unit_id"]): x for x in training_rows}
    unit_lookup = {x["unit_id"]: x for x in units}
    require(set(training_lookup) == set(all_cells), "training manifest keys differ from locked matrix")
    for candidate, unit_id in remaining:
        target = FORMAL / candidate / unit_id / "attempt_001/transform"
        require(not target.exists(), f"remaining transform target already exists: {target}")

    plan = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "all_cells": [{"registry_index": i, "candidate_id": c, "unit_id": u} for i, (c, u) in enumerate(all_cells)],
        "preserved_w00": {"candidate_id": W00_KEY[0], "unit_id": W00_KEY[1], "manifest_sha256": sha256_file(W00_MANIFEST)},
        "remaining_count": len(remaining),
        "remaining": [{"registry_index": all_cells.index(key), "candidate_id": key[0], "unit_id": key[1]} for key in remaining],
        "workers": 4,
        "submission_order": "registry_order",
        "completion_order": "may_differ",
        "aggregation_order": "registry_order",
        "scientific_retry": 0,
        "authority_contract_sha256": sha256_file(OUT / "stagew_parallel_amendment_contract.json"),
    }
    atomic_json(OUT / "stagew_remaining47_plan.json", plan)
    event("parallel47_plan_locked", remaining_count=47, plan_sha256=sha256_file(OUT / "stagew_remaining47_plan.json"))

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "8"
    payloads = [(training_lookup[key], unit_lookup[key[1]]) for key in remaining]
    stop = threading.Event()
    resource_path = RAW / "stage_w_parallel_resource.csv"
    resource_thread = threading.Thread(target=sampler, args=(stop, resource_path, os.getpid()), daemon=True)
    resource_thread.start()
    started = time.perf_counter()
    results: dict[tuple[str, str], dict] = {}
    completion: list[dict] = []
    fatal: list[dict] = []
    try:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
            future_map = {}
            for index, (key, payload) in enumerate(zip(remaining, payloads)):
                future = pool.submit(_parallel_transform, payload)
                future_map[future] = (index, key)
            for future in concurrent.futures.as_completed(future_map):
                index, key = future_map[future]
                try:
                    row = future.result()
                    require((row.get("candidate_id"), row.get("unit_id")) == key, f"worker returned wrong key for {key}")
                    results[key] = row
                    completion.append({"completion_index": len(completion), "registry_index": all_cells.index(key), "candidate_id": key[0], "unit_id": key[1], "status": row.get("status"), "manifest_sha256": sha256_file(FORMAL / key[0] / key[1] / "attempt_001/transform/transform_manifest.json"), "timestamp_utc": now()})
                except Exception as exc:
                    fatal.append({"registry_index": all_cells.index(key), "candidate_id": key[0], "unit_id": key[1], "error_type": type(exc).__name__, "error": repr(exc), "timestamp_utc": now()})
    finally:
        stop.set()
        resource_thread.join(timeout=3)
    parallel_seconds = time.perf_counter() - started
    atomic_json(INFRA / "stagew_parallel_completion_order.json", {"schema_version": 1, "label_access": False, "completion": completion, "fatal": fatal})
    require(not fatal, f"parallel scheduler fatal failures: {fatal}")
    require(len(results) == 47, f"parallel result cardinality is {len(results)}, not 47")

    ordered = []
    for key in all_cells:
        row = w00_row if key == W00_KEY else results[key]
        manifest_path = FORMAL / key[0] / key[1] / "attempt_001/transform/transform_manifest.json"
        require(manifest_path.is_file(), f"final transform manifest missing: {key}")
        require(json.loads(manifest_path.read_text(encoding="utf-8")) == row, f"in-memory/disk manifest mismatch: {key}")
        require(row.get("label_access") is False, f"label access detected: {key}")
        require(row.get("retry") is False and row.get("fallback") is False, f"retry/fallback detected: {key}")
        ordered.append(row)

    require(not (OUT / "weighted_mnn_training_manifest.json").exists(), "overall training manifest already exists")
    require(not (OUT / "weighted_mnn_transform_manifest.json").exists(), "overall transform manifest already exists")
    training_manifest = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "planned_training": 48,
        "training_attempts": 48,
        "successful_training": 48,
        "failed_training": 0,
        "scientific_retry": 0,
        "fallback_count": 0,
        "reconstructed_from_immutable_completed_cells": True,
        "new_training_during_resume": 0,
        "training_cells": training_rows,
    }
    transform_manifest = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "planned_transforms": 48,
        "transform_attempts": 48,
        "physical_transform_invocations": 49,
        "infrastructure_aborted_invocations": 1,
        "infrastructure_corrections": 2,
        "successful_transforms": sum(x["status"] == "success" for x in ordered),
        "failed_transforms": sum(x["status"] != "success" for x in ordered),
        "scientific_retry": 0,
        "fallback_count": 0,
        "runtime_seconds": float(w00_row.get("runtime_seconds", 0.0)) + parallel_seconds,
        "preserved_w00_runtime_seconds": float(w00_row.get("runtime_seconds", 0.0)),
        "remaining47_parallel_wall_seconds": parallel_seconds,
        "parallel_workers": 4,
        "parallel_authority": authority,
        "transforms": ordered,
    }
    atomic_json(OUT / "weighted_mnn_training_manifest.json", training_manifest)
    atomic_json(OUT / "weighted_mnn_transform_manifest.json", transform_manifest)
    summary = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "preserved_w00": True,
        "remaining_parallel_completed": 47,
        "successful_transforms": transform_manifest["successful_transforms"],
        "failed_transforms": transform_manifest["failed_transforms"],
        "parallel_wall_seconds": parallel_seconds,
        "training_manifest_sha256": sha256_file(OUT / "weighted_mnn_training_manifest.json"),
        "transform_manifest_sha256": sha256_file(OUT / "weighted_mnn_transform_manifest.json"),
    }
    atomic_json(INFRA / "stagew_boundary_parallel_completion.json", summary)
    event("stagew_parallel47_complete", summary=summary)
    return summary


def main() -> None:
    require(os.getpgrp() != DRIVER_PGID, "coordinator shares driver process group")
    require(not COORD_LOG.exists(), "coordinator log already exists")
    authority = verify_authority()
    guard = wait_for_guard()
    driver = same_driver()
    require(driver["state"] in {"T", "t"}, f"guard did not leave driver stopped: {driver['state']}")
    units, training_rows = load_verified_training()
    training_lookup = {(x["candidate_id"], x["unit_id"]): x for x in training_rows}
    unit_lookup = {x["unit_id"]: x for x in units}
    w00_row, validation = validate_w00(training_lookup[W00_KEY], unit_lookup[W00_KEY[1]])
    termination = terminate_old_driver()
    spill = isolate_boundary_spill()
    event("boundary_ready_for_parallel47", guard_terminal=guard[-1], validation=validation, termination=termination, spill=spill)
    summary = run_remaining(training_rows, units, w00_row, authority)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
