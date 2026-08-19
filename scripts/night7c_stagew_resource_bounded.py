#!/usr/bin/env python3
"""Night-7C Stage-W resource-cutoff coordinator and exact transform worker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import signal
import subprocess
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
from SpaLORA.night7a_consensus import atomic_json, canonical_partition, sha256_file  # noqa: E402
from scripts.night7c_replay_recovery_stage_w import (  # noqa: E402
    CANDIDATES,
    OUT,
    RAW,
    pilot_units,
    sampler,
    transform_cell,
)
from scripts.night7c_replay_recovery_stage_w_resume import load_verified_training  # noqa: E402

PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")
RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
FORMAL = RAW / "stage_w/formal"
INFRA = RAW / "infrastructure"
INVALID = RAW / "invalid_attempts"
DRIVER_PID = 929
DRIVER_START_TICKS = 489569013
DRIVER_PGID = 929
DRIVER_CMDLINE = f"{PYTHON} scripts/night7c_replay_recovery_stage_w_resume.py"
W00_KEY = ("W00_FILTER75", "u000")
W00_MANIFEST = FORMAL / W00_KEY[0] / W00_KEY[1] / "attempt_001/transform/transform_manifest.json"
GUARD_PID = 67932
GUARD_LOG = INFRA / "w00_boundary_guard.jsonl"
OLD_COORD_PID = 68756
OLD_COORD_START_TICKS = 492636241
OLD_COORD_CODE = REPO / "scripts/night7c_stagew_boundary_parallel.py"
OLD_COORD_LOG = INFRA / "stagew_boundary_parallel_coordinator.jsonl"
OLD_COORD_NOHUP = INFRA / "stagew_boundary_parallel.nohup.log"
OLD_COORD_CODE_SHA = "6df51d00967a9e223a88736f1cf941487d6b9bbe41edc4c38166a3f0327278e4"
OLD_COORD_LOG_SHA = "4609d9408f94f461ee1a37a7b2c0a948759dde11680d98739acb52c0f0f98399"
OLD_COORD_NOHUP_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
P2 = OUT / "p2_runtime_contract.json"
AMENDMENT = REPO / "protocols/night7c_replay_recovery/SpaLORA_Night7C_StageW_Resource_Cutoff_Amendment_2026-08-19.md"
AMENDMENT_SHA = "060858d84e57947de870e4fe92a5e04ce95954035ff4c6fec4f2d0f708db7536"
BOUNDARY_AMENDMENT = REPO / "protocols/night7c_replay_recovery/SpaLORA_Night7C_StageW_W00_Boundary_Preserve_Parallel_Amendment_2026-08-19.md"
BOUNDARY_AMENDMENT_SHA = "98387915dec4d5a1d5a6fb5963889fbcb90e7088dd2385fdc9f4f5557c109653"
COORD_LOG = INFRA / "stagew_resource_bounded_coordinator.jsonl"
WORKER_ROOT = INFRA / "stagew_resource_bounded_workers"
W00_LIMIT_SECONDS = 16 * 3600.0
UNIT_LIMIT_SECONDS = 60 * 60.0
CONTINUATION_LIMIT_SECONDS = 12 * 3600.0
WORKERS = 4


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(name: str, **values: object) -> None:
    INFRA.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": utc_now(), "event": name, **values}
    with COORD_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def proc_identity(pid: int) -> dict:
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat_text.rfind(")")
    require(close > 0, "malformed /proc stat")
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


def process_exists(pid: int) -> bool:
    try:
        proc_identity(pid)
        return True
    except FileNotFoundError:
        return False


def same_driver() -> dict:
    value = proc_identity(DRIVER_PID)
    require(value["start_ticks"] == DRIVER_START_TICKS, "W00 driver start-time mismatch")
    require(value["pgid"] == DRIVER_PGID, "W00 driver PGID mismatch")
    require(value["cmdline"] == DRIVER_CMDLINE, "W00 driver command mismatch")
    return value


def boot_seconds() -> float:
    return float(Path("/proc/uptime").read_text().split()[0])


def w00_deadline_boot_seconds() -> float:
    ticks = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    return DRIVER_START_TICKS / ticks + W00_LIMIT_SECONDS


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def inventory(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        {
            "relative_path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
            "size_bytes": item.stat().st_size,
        }
        for item in sorted(x for x in path.rglob("*") if x.is_file())
    ]


def process_snapshot(pid: int) -> dict:
    identity = proc_identity(pid)
    status = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in {"VmRSS", "Threads", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"}:
                status[key] = value.strip()
    children = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) == pid:
            continue
        try:
            child = proc_identity(int(item.name))
        except (FileNotFoundError, PermissionError, RuntimeError):
            continue
        if child["ppid"] == pid or child["pgid"] == identity["pgid"]:
            children.append(child)
    return {"identity": identity, "status": status, "same_group_processes": children}


def stop_exact_process(pid: int, start_ticks: int, pgid: int, *, already_stopped: bool, grace_seconds: float = 60.0) -> dict:
    before = proc_identity(pid)
    require(before["start_ticks"] == start_ticks, f"PID {pid} start-time mismatch")
    require(before["pgid"] == pgid, f"PID {pid} PGID mismatch")
    os.kill(pid, signal.SIGTERM)
    if already_stopped:
        os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            current = proc_identity(pid)
        except FileNotFoundError:
            return {"pid": pid, "start_ticks": start_ticks, "pgid": pgid, "terminated_by": "SIGTERM"}
        require(current["start_ticks"] == start_ticks, "PID reused during exact termination")
        if current["state"] == "Z":
            return {"pid": pid, "start_ticks": start_ticks, "pgid": pgid, "terminated_by": "SIGTERM", "observed_state": "Z"}
        time.sleep(0.1)
    current = proc_identity(pid)
    require(current["start_ticks"] == start_ticks and current["pgid"] == pgid, "identity changed before SIGKILL")
    os.kill(pid, signal.SIGKILL)
    return {"pid": pid, "start_ticks": start_ticks, "pgid": pgid, "terminated_by": "SIGKILL_AFTER_GRACE"}


def stop_exact_group(pid: int, start_ticks: int, pgid: int, *, already_stopped: bool, grace_seconds: float = 60.0) -> dict:
    before = proc_identity(pid)
    require(before["start_ticks"] == start_ticks, f"group leader {pid} start-time mismatch")
    require(before["pgid"] == pgid == pid, f"group leader {pid} is not an isolated PGID")
    os.killpg(pgid, signal.SIGTERM)
    if already_stopped:
        os.killpg(pgid, signal.SIGCONT)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            current = proc_identity(pid)
        except FileNotFoundError:
            return {"pid": pid, "start_ticks": start_ticks, "pgid": pgid, "terminated_by": "SIGTERM"}
        require(current["start_ticks"] == start_ticks, "group leader PID reused during termination")
        if current["state"] == "Z":
            return {"pid": pid, "start_ticks": start_ticks, "pgid": pgid, "terminated_by": "SIGTERM", "observed_state": "Z"}
        time.sleep(0.1)
    current = proc_identity(pid)
    require(current["start_ticks"] == start_ticks and current["pgid"] == pgid, "group identity changed before SIGKILL")
    os.killpg(pgid, signal.SIGKILL)
    return {"pid": pid, "start_ticks": start_ticks, "pgid": pgid, "terminated_by": "SIGKILL_AFTER_GRACE"}


def atomic_outcome(candidate: str, unit_id: str, status: str, **extra: object) -> dict:
    target = FORMAL / candidate / unit_id / "attempt_001/transform"
    manifest = target / "transform_manifest.json"
    require(not manifest.exists(), f"refusing to overwrite complete manifest: {manifest}")
    target.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1,
        "candidate_id": candidate,
        "unit_id": unit_id,
        "status": status,
        "label_access": False,
        "formal_transform": True,
        "retry": False,
        "fallback": False,
        **extra,
    }
    atomic_json(manifest, row)
    return row


def isolate_partial(candidate: str, unit_id: str, classification: str, *, allow_complete: bool = False) -> dict:
    source = FORMAL / candidate / unit_id / "attempt_001/transform"
    require(source.exists(), f"partial source missing: {source}")
    contained_manifest = (source / "transform_manifest.json").is_file()
    require(allow_complete or not contained_manifest, f"partial unexpectedly has complete manifest: {source}")
    record = {"source": str(source), "files": inventory(source), "classification": classification, "contained_terminal_manifest": contained_manifest}
    target = INVALID / classification / source.relative_to(RAW)
    require(not target.exists(), f"partial archive exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    record["archive"] = str(target)
    return record


def load_clusters(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows and set(rows[0]) == {"observation_id", "cluster"}, "cluster CSV schema mismatch")
    return [x["observation_id"] for x in rows], np.asarray([int(x["cluster"]) for x in rows], dtype=np.int64)


def validate_natural(candidate: str, unit: dict, training: dict) -> dict:
    manifest = FORMAL / candidate / unit["unit_id"] / "attempt_001/transform/transform_manifest.json"
    require(manifest.is_file(), f"natural terminal manifest missing: {candidate}/{unit['unit_id']}")
    row = json.loads(manifest.read_text(encoding="utf-8"))
    for key, expected in {
        "candidate_id": candidate,
        "unit_id": unit["unit_id"],
        "label_access": False,
        "formal_transform": True,
        "retry": False,
        "fallback": False,
    }.items():
        require(row.get(key) == expected, f"natural manifest mismatch {candidate}/{unit['unit_id']} {key}")
    require(row.get("status") in {"success", "scientific_numerical_failure"}, "natural terminal status invalid")
    audit = {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "status": row["status"]}
    if row["status"] == "success":
        directory = manifest.parent
        affinity_path, clusters_path = directory / "affinity.npz", directory / "clusters.csv"
        require(affinity_path.is_file() and clusters_path.is_file(), "success artifacts missing")
        affinity = sp.load_npz(affinity_path)
        ids, labels = load_clusters(clusters_path)
        expected_ids = [x.strip() for x in (RAW7B / "source" / unit["unit_id"] / "observation_ids.txt").read_text().splitlines() if x.strip()]
        require(ids == expected_ids, "cluster observation order mismatch")
        require(sha256_file(affinity_path) == row.get("affinity_file_sha256"), "affinity file SHA mismatch")
        require(sparse_sha(affinity) == row.get("canonical_affinity_sha256"), "canonical affinity SHA mismatch")
        require(sha256_file(clusters_path) == row.get("clusters_file_sha256"), "cluster file SHA mismatch")
        require(array_sha(canonical_partition(labels)) == row.get("canonical_partition_sha256"), "partition SHA mismatch")
        require(len(np.unique(labels)) == int(unit["K"]) == int(row.get("cluster_count")), "cluster count mismatch")
        require(row.get("training_manifest_sha256") == training.get("training_manifest_sha256"), "training manifest link mismatch")
        embedding = np.load(manifest.parent.parent / "worker/embedding.npy", allow_pickle=False)
        require(array_sha(embedding) == row.get("embedding_sha256"), "embedding SHA mismatch")
        audit["artifacts"] = [
            {"path": str(affinity_path), "sha256": sha256_file(affinity_path), "size_bytes": affinity_path.stat().st_size},
            {"path": str(clusters_path), "sha256": sha256_file(clusters_path), "size_bytes": clusters_path.stat().st_size},
        ]
    else:
        require(bool(row.get("failure_type")) and "failure_message" in row, "natural failure evidence incomplete")
    return row


def stop_guard_exact() -> dict:
    if not process_exists(GUARD_PID):
        return {"guard_pid": GUARD_PID, "already_exited": True}
    identity = proc_identity(GUARD_PID)
    require(identity["pgid"] == GUARD_PID and identity["sid"] == GUARD_PID, "guard isolation mismatch")
    return stop_exact_process(GUARD_PID, identity["start_ticks"], identity["pgid"], already_stopped=False, grace_seconds=10.0)


def stop_driver_after_natural() -> dict:
    driver = same_driver()
    require(driver["state"] in {"T", "t"}, f"natural guard did not stop driver: {driver['state']}")
    return stop_exact_process(DRIVER_PID, DRIVER_START_TICKS, DRIVER_PGID, already_stopped=True, grace_seconds=60.0)


def isolate_post_boundary_spill() -> list[dict]:
    expected = W00_MANIFEST.parent
    rows = []
    for source in sorted(FORMAL.glob("*/*/attempt_001/transform")):
        if source == expected:
            continue
        require(not (source / "transform_manifest.json").exists(), f"unexpected complete post-boundary unit: {source}")
        candidate, unit_id = source.parts[-4], source.parts[-3]
        rows.append(isolate_partial(candidate, unit_id, "stagew_post_w00_boundary_spill_20260819_resource_cutoff"))
    return rows


def capture_w00_resource_cutoff() -> tuple[dict, dict]:
    before = process_snapshot(DRIVER_PID)
    require(before["identity"]["start_ticks"] == DRIVER_START_TICKS, "W00 identity changed at cutoff")
    require(not W00_MANIFEST.exists(), "W00 manifest appeared before resource stop")
    os.kill(DRIVER_PID, signal.SIGSTOP)
    for _ in range(500):
        current = same_driver()
        if current["state"] in {"T", "t"}:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("W00 SIGSTOP not confirmed")
    guard_stop = stop_guard_exact()
    late_manifest = W00_MANIFEST.is_file()
    partial = isolate_partial(
        W00_KEY[0],
        W00_KEY[1],
        "stagew_w00_resource_censored_16h_20260820",
        allow_complete=late_manifest,
    )
    termination = stop_exact_process(DRIVER_PID, DRIVER_START_TICKS, DRIVER_PGID, already_stopped=True, grace_seconds=60.0)
    row = atomic_outcome(
        W00_KEY[0],
        W00_KEY[1],
        "RESOURCE_CENSORED_WALLTIME_16H",
        failure_type="resource_censored_walltime",
        resource_limit_seconds=W00_LIMIT_SECONDS,
        continuous_wall_seconds=max(0.0, boot_seconds() - DRIVER_START_TICKS / float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))),
        candidate_eligibility="INELIGIBLE_RESOURCE_CENSORED",
        natural_scientific_terminal=False,
        post_deadline_terminal_archived=late_manifest,
        partial_archive=partial,
        process_snapshot=before,
        guard_termination=guard_stop,
        driver_termination=termination,
    )
    evidence = {"schema_version": 1, "status": row["status"], "label_access": False, "manifest_sha256": sha256_file(W00_MANIFEST), "row": row}
    atomic_json(INFRA / "w00_resource_cutoff_16h.json", evidence)
    event("w00_resource_censored_16h", evidence=evidence)
    return row, before


def wait_for_w00(training: dict, unit: dict) -> tuple[dict, str]:
    deadline = w00_deadline_boot_seconds()
    event(
        "w00_resource_deadline_armed",
        driver=same_driver(),
        deadline_boot_seconds=deadline,
        remaining_seconds=max(0.0, deadline - boot_seconds()),
        wall_limit_seconds=W00_LIMIT_SECONDS,
        label_access=False,
    )
    while True:
        guard_rows = read_jsonl(GUARD_LOG)
        if W00_MANIFEST.is_file() or any(x.get("event") == "driver_stop_confirmed" for x in guard_rows):
            wait_deadline = time.monotonic() + 10.0
            while time.monotonic() < wait_deadline:
                guard_rows = read_jsonl(GUARD_LOG)
                if any(x.get("event") == "driver_stop_confirmed" for x in guard_rows):
                    break
                time.sleep(0.05)
            require(W00_MANIFEST.is_file(), "guard event without W00 manifest")
            row = validate_natural(W00_KEY[0], unit, training)
            termination = stop_driver_after_natural()
            guard_cleanup = stop_guard_exact()
            spill = isolate_post_boundary_spill()
            event("w00_natural_terminal_preserved", status=row["status"], manifest_sha256=sha256_file(W00_MANIFEST), driver_termination=termination, guard_cleanup=guard_cleanup, spill=spill)
            return row, "natural"
        if boot_seconds() >= deadline:
            if W00_MANIFEST.is_file():
                continue
            row, snapshot = capture_w00_resource_cutoff()
            return row, "resource_censored_16h"
        time.sleep(0.25)


def worker_payload_path(candidate: str, unit_id: str) -> Path:
    return WORKER_ROOT / candidate / unit_id / "payload.json"


def worker_main(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    row = transform_cell(payload["training"], payload["unit"])
    print(json.dumps({"candidate_id": row["candidate_id"], "unit_id": row["unit_id"], "status": row["status"]}, sort_keys=True))
    return 0


def start_worker(candidate: str, unit: dict, training: dict) -> dict:
    unit_id = unit["unit_id"]
    target = FORMAL / candidate / unit_id / "attempt_001/transform"
    require(not target.exists(), f"worker target already exists: {target}")
    payload_path = worker_payload_path(candidate, unit_id)
    require(not payload_path.exists(), f"worker payload exists: {payload_path}")
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload_path, {"candidate": candidate, "unit_id": unit_id, "training": training, "unit": unit})
    log_path = payload_path.parent / "worker.log"
    handle = log_path.open("wb")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1"})
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__)), "worker", "--payload", str(payload_path)],
        cwd=REPO,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    time.sleep(0.05)
    identity = proc_identity(process.pid)
    require(identity["pgid"] == process.pid and identity["sid"] == process.pid, "worker is not in isolated process group/session")
    row = {
        "candidate_id": candidate,
        "unit_id": unit_id,
        "process": process,
        "pid": process.pid,
        "pgid": identity["pgid"],
        "start_ticks": identity["start_ticks"],
        "started_monotonic": time.monotonic(),
        "deadline_monotonic": time.monotonic() + UNIT_LIMIT_SECONDS,
        "payload_path": payload_path,
        "payload_sha256": sha256_file(payload_path),
        "log_path": log_path,
    }
    event("transform_worker_started", candidate_id=candidate, unit_id=unit_id, pid=process.pid, pgid=identity["pgid"], start_ticks=identity["start_ticks"], payload_sha256=row["payload_sha256"])
    return row


def censor_worker(task: dict, status: str, classification: str, limit_seconds: float) -> dict:
    candidate, unit_id = task["candidate_id"], task["unit_id"]
    snapshot = process_snapshot(task["pid"])
    require(snapshot["identity"]["start_ticks"] == task["start_ticks"], "worker identity changed at cutoff")
    os.killpg(task["pgid"], signal.SIGSTOP)
    for _ in range(500):
        current = proc_identity(task["pid"])
        if current["state"] in {"T", "t"}:
            break
        time.sleep(0.01)
    manifest = FORMAL / candidate / unit_id / "attempt_001/transform/transform_manifest.json"
    late_manifest = manifest.is_file()
    partial = isolate_partial(candidate, unit_id, classification, allow_complete=late_manifest)
    termination = stop_exact_group(task["pid"], task["start_ticks"], task["pgid"], already_stopped=True, grace_seconds=60.0)
    row = atomic_outcome(
        candidate,
        unit_id,
        status,
        failure_type="resource_censored_walltime",
        resource_limit_seconds=limit_seconds,
        observed_wall_seconds=time.monotonic() - task["started_monotonic"],
        natural_scientific_terminal=False,
        post_deadline_terminal_archived=late_manifest,
        partial_archive=partial,
        process_snapshot=snapshot,
        worker_termination=termination,
        worker_payload_sha256=task["payload_sha256"],
        worker_log_sha256=sha256_file(task["log_path"]),
    )
    event("transform_resource_censored", candidate_id=candidate, unit_id=unit_id, status=status, manifest_sha256=sha256_file(manifest))
    return row


def verify_authority() -> dict:
    require(sha256_file(AMENDMENT) == AMENDMENT_SHA, "resource-cutoff amendment SHA mismatch")
    require(sha256_file(BOUNDARY_AMENDMENT) == BOUNDARY_AMENDMENT_SHA, "boundary amendment SHA mismatch")
    require(sha256_file(OLD_COORD_CODE) == OLD_COORD_CODE_SHA, "old coordinator code SHA changed")
    require(sha256_file(OLD_COORD_LOG) == OLD_COORD_LOG_SHA, "old coordinator log SHA changed")
    require(sha256_file(OLD_COORD_NOHUP) == OLD_COORD_NOHUP_SHA, "old coordinator nohup SHA changed")
    require(not process_exists(OLD_COORD_PID), "old unbounded coordinator is still running")
    p2 = json.loads(P2.read_text(encoding="utf-8"))
    require(p2.get("status") == "PASS" and p2.get("canonical_exact") is True and int(p2.get("parity_cells")) == 4, "P2 exact parity authority failed")
    require(p2.get("label_access") is False, "P2 label access mismatch")
    source = inspect.getsource(transform_cell).encode("utf-8")
    contract = {
        "schema_version": 1,
        "status": "PASS_BY_RESOURCE_CUTOFF_AMENDMENT",
        "label_access": False,
        "old_unbounded_coordinator": {
            "pid": OLD_COORD_PID,
            "start_ticks": OLD_COORD_START_TICKS,
            "running": False,
            "code_sha256": sha256_file(OLD_COORD_CODE),
            "log_sha256": sha256_file(OLD_COORD_LOG),
            "nohup_log_sha256": sha256_file(OLD_COORD_NOHUP),
            "preserved": True,
        },
        "resource_amendment_sha256": sha256_file(AMENDMENT),
        "boundary_amendment_sha256": sha256_file(BOUNDARY_AMENDMENT),
        "p2_report_sha256": sha256_file(P2),
        "p2": p2,
        "transform_module": inspect.getsourcefile(transform_cell),
        "transform_module_sha256": sha256_file(Path(inspect.getsourcefile(transform_cell))),
        "transform_function_source_sha256": hashlib.sha256(source).hexdigest(),
        "worker_call_chain": "resource_bounded_worker -> exact imported transform_cell(training, unit)",
        "workers": WORKERS,
        "worker_environment": {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"},
        "w00_wall_limit_seconds": W00_LIMIT_SECONDS,
        "unit_wall_limit_seconds": UNIT_LIMIT_SECONDS,
        "continuation_wall_limit_seconds": CONTINUATION_LIMIT_SECONDS,
        "scientific_retry": 0,
        "fallback": 0,
        "algorithm_unchanged": True,
        "gpu_backend": False,
    }
    atomic_json(OUT / "stagew_resource_cutoff_contract.json", contract)
    return contract


def run_bounded(training_rows: list[dict], units: list[dict], w00_row: dict, w00_mode: str, contract: dict) -> dict:
    all_keys = [(candidate, unit["unit_id"]) for candidate in CANDIDATES for unit in units]
    require(len(all_keys) == len(set(all_keys)) == 48 and all_keys[0] == W00_KEY, "locked 48-cell order mismatch")
    training_lookup = {(x["candidate_id"], x["unit_id"]): x for x in training_rows}
    unit_lookup = {x["unit_id"]: x for x in units}
    outcomes: dict[tuple[str, str], dict] = {W00_KEY: w00_row}
    launched = 0
    continuation_start = time.monotonic()
    continuation_deadline = continuation_start + CONTINUATION_LIMIT_SECONDS
    candidate_audit = []
    total_cutoff = False
    resource_states = {"RESOURCE_CENSORED_WALLTIME_16H", "RESOURCE_CENSORED_WALLTIME_60M", "RESOURCE_CENSORED_TOTAL_WALLTIME_12H"}

    for candidate in CANDIDATES:
        keys = [(candidate, unit["unit_id"]) for unit in units]
        circuit = candidate == W00_KEY[0] and w00_row["status"] == "RESOURCE_CENSORED_WALLTIME_16H"
        pending = [key for key in keys if key not in outcomes]
        if circuit:
            for key in pending:
                outcomes[key] = atomic_outcome(
                    key[0], key[1], "SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER",
                    failure_type="candidate_resource_circuit_breaker",
                    triggering_unit_id=W00_KEY[1],
                    triggering_status=w00_row["status"],
                    natural_scientific_terminal=False,
                )
            candidate_audit.append({"candidate_id": candidate, "status": "INELIGIBLE_RESOURCE_CENSORED", "trigger": W00_KEY[1], "started_units": 1, "skipped_units": 7})
            continue

        while pending and not circuit and not total_cutoff:
            if time.monotonic() >= continuation_deadline:
                total_cutoff = True
                break
            batch_keys = pending[:WORKERS]
            running = {key: start_worker(key[0], unit_lookup[key[1]], training_lookup[key]) for key in batch_keys}
            launched += len(running)
            while running:
                for key, task in list(running.items()):
                    process = task["process"]
                    manifest = FORMAL / key[0] / key[1] / "attempt_001/transform/transform_manifest.json"
                    if process.poll() is not None:
                        require(process.returncode == 0, f"worker process failed without allowed scientific terminal: {key}, rc={process.returncode}")
                        outcomes[key] = validate_natural(key[0], unit_lookup[key[1]], training_lookup[key])
                        event("transform_natural_terminal", candidate_id=key[0], unit_id=key[1], status=outcomes[key]["status"], manifest_sha256=sha256_file(manifest))
                        del running[key]
                        continue
                    if manifest.is_file():
                        process.wait(timeout=60)
                        require(process.returncode == 0, f"worker exited nonzero after manifest: {key}")
                        outcomes[key] = validate_natural(key[0], unit_lookup[key[1]], training_lookup[key])
                        event("transform_natural_terminal", candidate_id=key[0], unit_id=key[1], status=outcomes[key]["status"], manifest_sha256=sha256_file(manifest))
                        del running[key]
                        continue
                    if time.monotonic() >= continuation_deadline:
                        outcomes[key] = censor_worker(task, "RESOURCE_CENSORED_TOTAL_WALLTIME_12H", "stagew_total_resource_censored_12h_20260820", CONTINUATION_LIMIT_SECONDS)
                        del running[key]
                        total_cutoff = True
                        continue
                    if time.monotonic() >= task["deadline_monotonic"]:
                        outcomes[key] = censor_worker(task, "RESOURCE_CENSORED_WALLTIME_60M", "stagew_unit_resource_censored_60m_20260820", UNIT_LIMIT_SECONDS)
                        circuit = True
                        del running[key]
                time.sleep(0.25)
            pending = [key for key in keys if key not in outcomes]

        if total_cutoff:
            for key in [x for x in all_keys if x not in outcomes]:
                outcomes[key] = atomic_outcome(
                    key[0], key[1], "SKIPPED_TOTAL_RESOURCE_CUTOFF_12H",
                    failure_type="total_resource_cutoff",
                    triggering_status="RESOURCE_CENSORED_TOTAL_WALLTIME_12H",
                    natural_scientific_terminal=False,
                )
            break
        if circuit:
            trigger = next(key for key in keys if outcomes[key]["status"] == "RESOURCE_CENSORED_WALLTIME_60M")
            for key in [x for x in keys if x not in outcomes]:
                outcomes[key] = atomic_outcome(
                    key[0], key[1], "SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER",
                    failure_type="candidate_resource_circuit_breaker",
                    triggering_unit_id=trigger[1],
                    triggering_status=outcomes[trigger]["status"],
                    natural_scientific_terminal=False,
                )
            candidate_audit.append({"candidate_id": candidate, "status": "INELIGIBLE_RESOURCE_CENSORED", "trigger": trigger[1]})
        else:
            rows = [outcomes[key] for key in keys]
            eligible = all(x["status"] == "success" for x in rows)
            candidate_audit.append({"candidate_id": candidate, "status": "ELIGIBLE_COMPLETE_SUCCESS" if eligible else "INELIGIBLE_NATURAL_FAILURE", "statuses": [x["status"] for x in rows]})

    require(set(outcomes) == set(all_keys), f"Stage-W outcome keys incomplete: {len(outcomes)}/48")
    ordered = [outcomes[key] for key in all_keys]
    for key, row in zip(all_keys, ordered):
        manifest = FORMAL / key[0] / key[1] / "attempt_001/transform/transform_manifest.json"
        require(manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) == row, f"outcome manifest mismatch: {key}")
        require(row.get("label_access") is False and row.get("retry") is False and row.get("fallback") is False, f"firewall/retry mismatch: {key}")
    candidate_audit = []
    for candidate in CANDIDATES:
        rows = [outcomes[(candidate, unit["unit_id"])] for unit in units]
        statuses = [x["status"] for x in rows]
        if all(status == "success" for status in statuses):
            candidate_status = "ELIGIBLE_COMPLETE_SUCCESS"
        elif any(status in resource_states or status.startswith("SKIPPED_") for status in statuses):
            candidate_status = "INELIGIBLE_RESOURCE_CENSORED"
        else:
            candidate_status = "INELIGIBLE_NATURAL_FAILURE"
        candidate_audit.append({"candidate_id": candidate, "status": candidate_status, "statuses": statuses})
    require(len(candidate_audit) == 6, "candidate eligibility cardinality mismatch")
    eligible = [x["candidate_id"] for x in candidate_audit if x["status"] == "ELIGIBLE_COMPLETE_SUCCESS"]
    plan = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "w00_mode": w00_mode,
        "all_cells": [{"registry_index": i, "candidate_id": key[0], "unit_id": key[1], "outcome_status": ordered[i]["status"]} for i, key in enumerate(all_keys)],
        "candidate_eligibility": candidate_audit,
        "eligible_weighted_mnn_candidates": eligible,
        "workers": WORKERS,
        "unit_wall_limit_seconds": UNIT_LIMIT_SECONDS,
        "continuation_wall_limit_seconds": CONTINUATION_LIMIT_SECONDS,
        "scientific_retry": 0,
        "fallback": 0,
    }
    atomic_json(OUT / "stagew_resource_bounded_plan_and_eligibility.json", plan)
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
    status_counts = {}
    for row in ordered:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    transform_manifest = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "planned_transforms": 48,
        "transform_attempts": 48,
        "transform_outcomes": 48,
        "physical_transform_invocations": 2 + launched,
        "prior_infrastructure_aborted_invocations": 1,
        "successful_transforms": status_counts.get("success", 0),
        "failed_transforms": 48 - status_counts.get("success", 0),
        "failed_natural_transforms": status_counts.get("scientific_numerical_failure", 0),
        "resource_censored_transforms": sum(status_counts.get(x, 0) for x in resource_states),
        "skipped_transforms": sum(v for k, v in status_counts.items() if k.startswith("SKIPPED_")),
        "status_counts": status_counts,
        "scientific_retry": 0,
        "fallback_count": 0,
        "w00_mode": w00_mode,
        "runtime_seconds": time.monotonic() - continuation_start + float(w00_row.get("runtime_seconds", w00_row.get("continuous_wall_seconds", 0.0))),
        "resource_bounded_continuation_seconds": time.monotonic() - continuation_start,
        "resource_cutoff_contract": contract,
        "eligible_weighted_mnn_candidates": eligible,
        "candidate_eligibility": candidate_audit,
        "transforms": ordered,
    }
    atomic_json(OUT / "weighted_mnn_training_manifest.json", training_manifest)
    atomic_json(OUT / "weighted_mnn_transform_manifest.json", transform_manifest)
    summary = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "w00_mode": w00_mode,
        "status_counts": status_counts,
        "eligible_weighted_mnn_candidates": eligible,
        "training_manifest_sha256": sha256_file(OUT / "weighted_mnn_training_manifest.json"),
        "transform_manifest_sha256": sha256_file(OUT / "weighted_mnn_transform_manifest.json"),
        "plan_sha256": sha256_file(OUT / "stagew_resource_bounded_plan_and_eligibility.json"),
    }
    atomic_json(INFRA / "stagew_resource_bounded_completion.json", summary)
    event("stagew_resource_bounded_complete", summary=summary)
    return summary


def coordinator_main() -> int:
    require(os.getpgrp() not in {DRIVER_PGID, GUARD_PID}, "coordinator process group is not isolated")
    require(not COORD_LOG.exists(), "resource-bounded coordinator log already exists")
    contract = verify_authority()
    units, training_rows = load_verified_training()
    training_lookup = {(x["candidate_id"], x["unit_id"]): x for x in training_rows}
    unit_lookup = {x["unit_id"]: x for x in units}
    w00_row, w00_mode = wait_for_w00(training_lookup[W00_KEY], unit_lookup[W00_KEY[1]])
    summary = run_bounded(training_rows, units, w00_row, w00_mode, contract)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("coordinator")
    worker = sub.add_parser("worker")
    worker.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "worker":
        return worker_main(args.payload)
    return coordinator_main()


if __name__ == "__main__":
    raise SystemExit(main())
