#!/usr/bin/env python3
"""P2 parity probe and Stage-T conflict-gated affinity transforms."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "8")

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import (  # noqa: E402
    atomic_json, atomic_sparse, canonical_partition, sha256_file,
)
from SpaLORA.night7b_adaptive import run_partition  # noqa: E402
from SpaLORA.night7c_conflict import mix_affinities, routing_weights  # noqa: E402
from scripts.night7b_adapter_stage import save_clusters  # noqa: E402

RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
HANDOFF7B = RAW7B / "official_compact/handoff"
OUT = REPO / "outputs/night7c_replay_recovery_handoff"
CANDIDATES = [
    "T02_GLOBAL_WIDE", "T03_GLOBAL_CONSERVATIVE", "T04_HARD_CONFLICT_030",
    "T05_LOCAL_CONFLICT", "T06_LOCAL_CONFLICT_SQUARED",
    "T07_LOCAL_QUALITY_CONFLICT", "T08_LOCAL_SHARED_SUPPORT",
    "T09_THREE_SPECIALIST_EXPLORATORY",
]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def source_rows() -> list[dict]:
    rows = list(csv.DictReader((HANDOFF7B / "source_unit_index.csv").open(newline="")))
    require(len(rows) == 30, "source unit count mismatch")
    return rows


def specialist_map(recipe: str) -> dict[str, dict]:
    out = {}
    for stage in ("R1", "R2"):
        locked = json.loads((HANDOFF7B / f"locked_{stage}_manifest.json").read_text())
        for row in locked["transforms"]:
            if (row["recipe_id"] == recipe and row["endpoint"] == "E1_ADAPTER_C06_MEAN"
                    and row["head_id"] == "H01"):
                path = RAW7B / "adapter_stage" / stage / "formal" / recipe / row["unit_id"] / "attempt_001" / "transforms" / row["endpoint"] / row["head_id"] / "affinity.npz"
                out[row["unit_id"]] = {"path": path, "row": row, "stage": stage}
    require(len(out) == 30, f"{recipe} specialist map is not 30")
    return out


def c00_map() -> dict[str, Path]:
    out = {}
    for row in csv.DictReader((HANDOFF7B / "historical_reference_partition_index.csv").open(newline="")):
        if row["reference"] == "C00":
            out[row["unit_id"]] = Path(row["clusters_path"]).parent / "affinity.npz"
    require(len(out) == 30, "C00 affinity map is not 30")
    return out


def load_cell_inputs(unit_id: str):
    feature_path = RAW / "p1b_features" / unit_id / "features.npz"
    with np.load(feature_path, allow_pickle=False) as f:
        m = float(f["m_initial"]); rank_c = f["rank_c"]; quality = f["quality"]; support = f["support"]
    s0_path = c00_map()[unit_id]; s2_entry = specialist_map("R02")[unit_id]
    s8_entry = specialist_map("R08")[unit_id]
    s0 = sp.load_npz(s0_path); s2 = sp.load_npz(s2_entry["path"]); s8 = sp.load_npz(s8_entry["path"])
    return feature_path, m, rank_c, quality, support, s0, s2, s8, s0_path, s2_entry, s8_entry


def cell(candidate: str, unit_id: str, root: Path) -> dict:
    target = root / candidate / unit_id
    require(not target.exists(), f"transform target exists: {target}")
    target.mkdir(parents=True)
    start = time.perf_counter()
    base = {"schema_version": 1, "candidate_id": candidate, "unit_id": unit_id,
            "status": "scientific_numerical_failure", "label_access": False,
            "formal_training": False, "formal_transform": root.name == "formal",
            "retry": False, "fallback": False}
    try:
        feature_path, m, rank_c, quality, support, s0, s2, s8, s0p, s2e, s8e = load_cell_inputs(unit_id)
        weights = routing_weights(candidate, m, rank_c, quality, support)
        affinity = mix_affinities(weights, s0, s2, s8 if candidate.startswith("T09_") else None)
        source = next(x for x in source_rows() if x["unit_id"] == unit_id)
        ids = [x.strip() for x in (RAW7B / "source" / unit_id / "observation_ids.txt").read_text().splitlines() if x.strip()]
        labels, partition = run_partition("H01", affinity, int(source["K"]), [])
        atomic_sparse(target / "affinity.npz", affinity)
        save_clusters(target / "clusters.csv", ids, labels)
        np.save(target / "weights.npy", weights, allow_pickle=False)
        base.update({
            "status": "success", "partition": partition,
            "m_initial": m, "m_initial_source": "night7b_sha_locked_loss_curve_first_row_MNN",
            "feature_file_sha256": sha256_file(feature_path),
            "s0_path": str(s0p), "s0_canonical_sha256": sparse_sha(s0),
            "s2_path": str(s2e["path"]), "s2_canonical_sha256": sparse_sha(s2),
            "s8_path": str(s8e["path"]), "s8_canonical_sha256": sparse_sha(s8) if candidate.startswith("T09_") else None,
            "weights_file_sha256": sha256_file(target / "weights.npy"),
            "weights_column_summary": [[float(x) for x in (weights[:, j].min(), weights[:, j].mean(), weights[:, j].max())] for j in range(weights.shape[1])],
            "affinity_file_sha256": sha256_file(target / "affinity.npz"),
            "canonical_affinity_sha256": sparse_sha(affinity),
            "clusters_file_sha256": sha256_file(target / "clusters.csv"),
            "canonical_partition_sha256": array_sha(canonical_partition(labels)),
            "cluster_count": int(len(np.unique(labels))),
        })
    except Exception as exc:
        base["failure_type"] = type(exc).__name__; base["failure_message"] = str(exc)
    base["runtime_seconds"] = time.perf_counter() - start
    atomic_json(target / "transform_manifest.json", base)
    return base


def _subprocess_cell(args: tuple[str, str, str]) -> dict:
    candidate, unit_id, root = args
    cmd = [sys.executable, str(Path(__file__).resolve()), "cell", "--candidate", candidate,
           "--unit-id", unit_id, "--root", root]
    done = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    target = Path(root) / candidate / unit_id
    (target / "cell.log").write_text(done.stdout + done.stderr)
    require(done.returncode == 0, f"transform process failed before manifest: {candidate}/{unit_id}")
    return json.loads((target / "transform_manifest.json").read_text())


def sample_resources(stop: threading.Event, target: Path, phase: str, parent_pid: int) -> None:
    fields = ["timestamp", "phase", "cpu_percent", "rss_mib", "process_count", "gpu_util_percent", "gpu_memory_mib", "gpu_power_w", "gpu_sm_clock_mhz"]
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        while not stop.is_set():
            try:
                parent = psutil.Process(parent_pid); procs = [parent] + parent.children(recursive=True)
                rss = sum(p.memory_info().rss for p in procs if p.is_running()) / 1048576.0
                cpu = sum(p.cpu_percent(None) for p in procs if p.is_running())
            except Exception:
                rss = cpu = 0.0; procs = []
            gpu = ["", "", "", ""]
            try:
                gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm", "--format=csv,noheader,nounits"], text=True, timeout=3).strip().split(", ")
            except Exception:
                pass
            writer.writerow(dict(zip(fields, [time.time(), phase, cpu, rss, len(procs), *gpu]))); handle.flush(); stop.wait(1.0)


def run_jobs(jobs: list[tuple[str, str, str]], workers: int, phase: str, log_name: str) -> tuple[list[dict], float]:
    stop = threading.Event(); sampler = threading.Thread(target=sample_resources, args=(stop, RAW / log_name, phase, os.getpid()), daemon=True); sampler.start()
    start = time.perf_counter()
    try:
        if workers == 1:
            rows = [_subprocess_cell(job) for job in jobs]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(_subprocess_cell, jobs))
    finally:
        elapsed = time.perf_counter() - start; stop.set(); sampler.join(timeout=3)
    return rows, elapsed


def p2() -> None:
    serial_root = RAW / "p2_runtime/serial"; parallel_root = RAW / "p2_runtime/parallel"
    require(not serial_root.exists() and not parallel_root.exists(), "P2 roots already exist")
    units = [x["unit_id"] for x in source_rows() if x["unit_id"] in {"u020", "u021", "u022", "u023"}]
    jobs_s = [("T02_GLOBAL_WIDE", u, str(serial_root)) for u in units]
    jobs_p = [("T02_GLOBAL_WIDE", u, str(parallel_root)) for u in units]
    serial, serial_s = run_jobs(jobs_s, 1, "p2_serial", "p2_serial_resource.csv")
    parallel, parallel_s = run_jobs(jobs_p, 4, "p2_parallel", "p2_parallel_resource.csv")
    exact = all(a["status"] == b["status"] == "success" and
                a["canonical_affinity_sha256"] == b["canonical_affinity_sha256"] and
                a["canonical_partition_sha256"] == b["canonical_partition_sha256"]
                for a, b in zip(serial, parallel))
    speedup = serial_s / parallel_s
    accepted = exact and speedup >= 1.5
    result = {"schema_version": 1, "status": "PASS", "parity_cells": 4,
              "canonical_exact": exact, "serial_seconds": serial_s,
              "parallel_seconds": parallel_s, "speedup": speedup,
              "parallel_backend_accepted": accepted,
              "formal_backend": "parallel_4x8" if accepted else "serial_reference",
              "gpu_zero_during_cpu_transform_allowed": True, "label_access": False}
    atomic_json(OUT / "p2_runtime_contract.json", result); print(json.dumps(result, sort_keys=True))


def formal() -> None:
    contract = json.loads((OUT / "p2_runtime_contract.json").read_text())
    workers = 4 if contract["parallel_backend_accepted"] else 1
    root = RAW / "stage_t/formal"; require(not root.exists(), "formal Stage T root exists")
    jobs = [(c, x["unit_id"], str(root)) for c in CANDIDATES for x in source_rows()]
    rows, elapsed = run_jobs(jobs, workers, "stage_t_transform", "stage_t_resource.csv")
    require(len(rows) == 240, "Stage T transform cardinality mismatch")
    manifest = {"schema_version": 1, "status": "LOCKED_PRE_LABEL", "label_access": False,
                "planned_transforms": 240, "transform_attempts": 240,
                "successful_transforms": sum(x["status"] == "success" for x in rows),
                "failed_transforms": sum(x["status"] != "success" for x in rows),
                "scientific_retry": 0, "fallback_count": 0, "backend": contract["formal_backend"],
                "runtime_seconds": elapsed, "transforms": rows}
    atomic_json(OUT / "routing_transform_manifest.json", manifest)
    with (OUT / "routing_weights_manifest.csv").open("w", newline="") as h:
        fields = ["candidate_id", "unit_id", "status", "m_initial", "weights_file_sha256", "canonical_affinity_sha256", "canonical_partition_sha256", "runtime_seconds"]
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader();
        for row in rows: w.writerow({k: row.get(k, "") for k in fields})
    print(json.dumps({"status": "LOCKED_PRE_LABEL", "attempts": 240,
                      "success": manifest["successful_transforms"], "failure": manifest["failed_transforms"],
                      "backend": manifest["backend"], "runtime_seconds": elapsed}, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("cell", "p2", "formal")); p.add_argument("--candidate"); p.add_argument("--unit-id"); p.add_argument("--root")
    a = p.parse_args()
    if a.mode == "cell":
        require(a.candidate in CANDIDATES and a.unit_id and a.root, "cell arguments invalid")
        value = cell(a.candidate, a.unit_id, Path(a.root)); print(json.dumps(value, sort_keys=True))
    elif a.mode == "p2": p2()
    else: formal()


if __name__ == "__main__": main()
