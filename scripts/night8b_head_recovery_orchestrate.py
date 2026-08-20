#!/usr/bin/env python3
"""Bounded four-worker coordinator and pre-label partition lock."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
RAW = Path("/root/autodl-fs/night8b_head_recovery_20260820")
ORIGINAL = Path("/root/autodl-fs/night8b_raw_runs_20260820")
OUT = REPO / "outputs/night8b_head_recovery"
RUNNER = REPO / "scripts/night8b_head_recovery_transform.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def start_cell(method: str, seed: int):
    log = RAW / "logs" / f"{method}_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("wb")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    command = [sys.executable, str(RUNNER), "--method", method, "--seed", str(seed)]
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return {"method": method, "seed": seed, "process": process,
            "handle": handle, "log": str(log), "started": time.time(),
            "command": command}


def run_all() -> list:
    jobs = [(method, seed) for method in ("HR_U00", "HR_F00") for seed in range(10)]
    if (RAW / "coordinator_state.json").exists():
        raise RuntimeError("no-retry coordinator refuses an existing state")
    started = time.time(); pending = list(jobs); active = []; results = []
    atomic_json(RAW / "coordinator_state.json", {
        "status": "running", "started_unix": started, "jobs": jobs,
        "worker_limit": 4, "cell_timeout_seconds": 600,
        "wall_limit_seconds": 7200, "scientific_retry": 0, "fallback": 0,
        "label_access": False,
    })
    while pending or active:
        if time.time() - started > 7200:
            raise RuntimeError("RECOVERY_BLOCKED_HEAD_NUMERICS: total wall-time limit")
        while pending and len(active) < 4:
            method, seed = pending.pop(0)
            active.append(start_cell(method, seed))
        for job in list(active):
            process = job["process"]
            elapsed = time.time() - job["started"]
            rc = process.poll()
            if rc is None and elapsed > 600:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
                rc = process.returncode
                status = "resource_censored_10m"
            elif rc is None:
                continue
            else:
                status = "success" if rc == 0 else "failed"
            job["handle"].close()
            row = {key: job[key] for key in ("method", "seed", "log")}
            row.update({"status": status, "returncode": rc,
                        "wall_seconds": elapsed, "scientific_retry": False,
                        "fallback": False})
            results.append(row); active.remove(job)
            state = json.loads((RAW / "coordinator_state.json").read_text())
            state["results"] = results
            atomic_json(RAW / "coordinator_state.json", state)
        time.sleep(0.25)
    if len(results) != 20 or any(row["status"] != "success" for row in results):
        state = json.loads((RAW / "coordinator_state.json").read_text())
        state.update({"status": "RECOVERY_BLOCKED_HEAD_NUMERICS",
                      "elapsed_seconds": time.time() - started})
        atomic_json(RAW / "coordinator_state.json", state)
        raise RuntimeError("RECOVERY_BLOCKED_HEAD_NUMERICS: transform failure")
    state = json.loads((RAW / "coordinator_state.json").read_text())
    state.update({"status": "complete", "elapsed_seconds": time.time() - started,
                  "results": sorted(results, key=lambda x: (x["method"], x["seed"]))})
    atomic_json(RAW / "coordinator_state.json", state)
    return results


def lock_partitions() -> dict:
    input_lock = json.loads((OUT / "recovery_input_view_manifest.json").read_text())
    expected_config = input_lock["head_config_sha256"]
    rows = []
    for method in ("HR_U00", "HR_F00"):
        for seed in range(10):
            directory = RAW / "partitions" / method / f"seed_{seed}"
            manifest_path = directory / "transform_manifest.json"
            clusters_path = directory / "clusters.csv"
            manifest = json.loads(manifest_path.read_text())
            table = pd.read_csv(clusters_path)
            if (manifest.get("status") != "success" or manifest.get("K") != 12
                    or manifest.get("head_config_sha256") != expected_config
                    or manifest.get("head_function_arguments") != ["affinity"]
                    or not manifest["determinism_audit"].get("deterministic_exact")
                    or table["cluster"].nunique() != 12
                    or len(table) != 1949
                    or sha256_file(clusters_path) != manifest["clusters_file_sha256"]):
                raise RuntimeError(f"RECOVERY_BLOCKED_HEAD_NUMERICS lock {method}/{seed}")
            rows.append({
                "method": method, "seed": seed, "K": 12,
                "status": "success", "head_id": manifest["head_id"],
                "head_config_sha256": manifest["head_config_sha256"],
                "canonical_partition_sha256": manifest["canonical_partition_sha256"],
                "clusters_path": str(clusters_path),
                "clusters_file_sha256": manifest["clusters_file_sha256"],
                "transform_manifest_path": str(manifest_path),
                "transform_manifest_sha256": sha256_file(manifest_path),
                "deterministic_exact": True,
                "input_file_sha256": manifest["input_file_sha256"],
                "input_canonical_sparse_sha256": manifest["input_canonical_sparse_sha256"],
                "recovery_head_effective_seconds": manifest["recovery_head_effective_seconds"],
                "determinism_audit_seconds": manifest["determinism_audit"]["second_determinism_seconds"],
                "peak_rss_mib": manifest["peak_rss_mib"],
            })
    payload = {
        "schema_version": 1, "status": "TOTAL_LOCKED_BEFORE_LABEL_ACCESS",
        "row_count": len(rows), "exact_K": "20/20",
        "deterministic_exact": "20/20", "sha_complete": "20/20",
        "same_head_config": len({row["head_config_sha256"] for row in rows}) == 1,
        "training": 0, "adapter": 0, "affinity_rebuild": 0,
        "label_access": False, "rows": rows,
    }
    atomic_json(OUT / "locked_recovery_partition_manifest.json", payload)
    atomic_json(RAW / "manifests/locked_recovery_partition_manifest.json", payload)
    return payload


def prelabel_concordance() -> None:
    rows = []
    for method, old_method in (("HR_U00", "U00"), ("HR_F00", "F00")):
        for seed in range(10):
            old = ORIGINAL / f"formal/transforms/{old_method}/seed_{seed}/clusters.csv"
            if not old.is_file():
                continue
            new = RAW / f"partitions/{method}/seed_{seed}/clusters.csv"
            old_table = pd.read_csv(old); new_table = pd.read_csv(new)
            if old_table.iloc[:, 0].astype(str).tolist() != new_table.iloc[:, 0].astype(str).tolist():
                raise RuntimeError("prelabel concordance observation order mismatch")
            a = old_table["cluster"].to_numpy(); b = new_table["cluster"].to_numpy()
            rows.append({"method": method, "seed": seed,
                         "old_partition_sha256": sha256_file(old),
                         "recovery_partition_sha256": sha256_file(new),
                         "assignment_ari": adjusted_rand_score(a, b),
                         "assignment_nmi": normalized_mutual_info_score(a, b),
                         "label_access": False, "terminal_decision_input": False})
    if len(rows) != 19:
        raise RuntimeError(f"expected 19 prelabel old partitions, found {len(rows)}")
    path = OUT / "prelabel_head_partition_concordance.csv"
    pd.DataFrame(rows).sort_values(["method", "seed"]).to_csv(path, index=False)


def main() -> None:
    run_all()
    lock = lock_partitions()
    prelabel_concordance()
    print(json.dumps({"status": lock["status"], "partitions": lock["row_count"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
