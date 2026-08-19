#!/usr/bin/env python3
"""Run the fixed 48-cell weighted-MNN pilot and 48 H01 transforms."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, atomic_sparse, canonical_partition, sha256_file  # noqa: E402
from SpaLORA.night7b_adaptive import run_partition  # noqa: E402
from scripts.night7b_adapter_stage import endpoint_affinity, save_clusters  # noqa: E402
from scripts.night7c_p1 import training_map  # noqa: E402

PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")
RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
HANDOFF7B = RAW7B / "official_compact/handoff"
OUT = REPO / "outputs/night7c_replay_recovery_handoff"
TRAINER = REPO / "scripts/night7c_weighted_train.py"
CANDIDATES = ["W00_FILTER75", "W01_QUALITY_SOFT", "W02_CONFLICT_RANK",
              "W03_QUALITY_CONFLICT", "W04_QUALITY_SHARED", "W05_QUALITY_CONFLICT_SHARED"]


def require(ok: bool, message: str) -> None:
    if not ok: raise RuntimeError(message)


def pilot_units() -> list[dict]:
    rows = list(csv.DictReader((HANDOFF7B / "source_unit_index.csv").open(newline="")))
    selected = []
    seen = {}
    for row in rows:
        key = row["dataset"]; seen[key] = seen.get(key, 0)
        if seen[key] < 2: selected.append(row)
        seen[key] += 1
    require(len(selected) == 8 and len({x["dataset"] for x in selected}) == 4,
            "pilot unit matrix is not four datasets x first two seeds")
    return selected


def run_logged(cmd: list[str], path: Path) -> int:
    start = time.perf_counter(); done = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(done.stdout + done.stderr)
    return done.returncode


def train_cell(candidate: str, unit: dict) -> dict:
    unit_id = unit["unit_id"]; target = RAW / "stage_w/formal" / candidate / unit_id / "attempt_001"
    require(not target.exists(), f"Stage W target exists: {candidate}/{unit_id}")
    target.mkdir(parents=True); worker = target / "worker"
    _, train = training_map()[unit_id]
    config = Path(train["config_path"]); unit_dir = RAW7B / "adapter_inputs" / unit_id
    feature = RAW / "p1b_features" / unit_id / "features.npz"
    start = time.perf_counter()
    train_rc = run_logged([str(PYTHON), str(TRAINER), "train", "--unit-dir", str(unit_dir),
                           "--config", str(config), "--feature", str(feature), "--candidate", candidate,
                           "--output", str(worker)], target / "train.log")
    row = {"schema_version": 1, "candidate_id": candidate, "unit_id": unit_id,
           "status": "scientific_numerical_failure", "scientific_training": True,
           "label_access": False, "retry": False, "fallback": False,
           "train_returncode": train_rc, "runtime_seconds": time.perf_counter() - start}
    if train_rc == 0 and (worker / "training_manifest.json").exists():
        reload_rc = run_logged([str(PYTHON), str(TRAINER), "reload", "--unit-dir", str(unit_dir),
                                "--config", str(config), "--output", str(worker)], target / "reload.log")
        row["reload_returncode"] = reload_rc
        if reload_rc == 0:
            row.update({"status": "success", "training_manifest": json.loads((worker / "training_manifest.json").read_text()),
                        "training_manifest_sha256": sha256_file(worker / "training_manifest.json"),
                        "reload_audit": json.loads((worker / "reload_forward_audit.json").read_text()),
                        "reload_audit_sha256": sha256_file(worker / "reload_forward_audit.json")})
        else: row["failure_type"] = "checkpoint_roundtrip_failure"
    else: row["failure_type"] = "training_failure"
    atomic_json(target / "cell_manifest.json", row); return row


def transform_cell(training: dict, unit: dict) -> dict:
    candidate, unit_id = training["candidate_id"], unit["unit_id"]
    target = RAW / "stage_w/formal" / candidate / unit_id / "attempt_001/transform"
    require(not target.exists(), f"Stage W transform exists: {candidate}/{unit_id}")
    target.mkdir(parents=True)
    base = {"schema_version": 1, "candidate_id": candidate, "unit_id": unit_id,
            "status": "scientific_numerical_failure", "label_access": False,
            "formal_transform": True, "retry": False, "fallback": False}
    start = time.perf_counter()
    try:
        require(training["status"] == "success", "training cell unsuccessful")
        worker = target.parent / "worker"
        embedding = np.load(worker / "embedding.npy", allow_pickle=False)
        ids = [x.strip() for x in (RAW7B / "source" / unit_id / "observation_ids.txt").read_text().splitlines() if x.strip()]
        c06 = sp.load_npz(RAW7B / "source" / unit_id / "c06_affinity.npz")
        affinity = endpoint_affinity("E1_ADAPTER_C06_MEAN", embedding, c06, ids)
        labels, partition = run_partition("H01", affinity, int(unit["K"]), [])
        atomic_sparse(target / "affinity.npz", affinity); save_clusters(target / "clusters.csv", ids, labels)
        base.update({"status": "success", "partition": partition,
                     "training_manifest_sha256": training["training_manifest_sha256"],
                     "embedding_sha256": array_sha(embedding),
                     "affinity_file_sha256": sha256_file(target / "affinity.npz"),
                     "canonical_affinity_sha256": sparse_sha(affinity),
                     "clusters_file_sha256": sha256_file(target / "clusters.csv"),
                     "canonical_partition_sha256": array_sha(canonical_partition(labels)),
                     "cluster_count": int(len(np.unique(labels)))})
    except Exception as exc:
        base["failure_type"] = type(exc).__name__; base["failure_message"] = str(exc)
    base["runtime_seconds"] = time.perf_counter() - start
    atomic_json(target / "transform_manifest.json", base); return base


def sampler(stop: threading.Event, path: Path, pid: int) -> None:
    fields = ["timestamp", "phase", "cpu_percent", "rss_mib", "process_count", "gpu_util_percent", "gpu_memory_mib", "gpu_power_w", "gpu_sm_clock_mhz"]
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        while not stop.is_set():
            try:
                p = psutil.Process(pid); ps = [p] + p.children(recursive=True)
                rss = sum(x.memory_info().rss for x in ps if x.is_running()) / 1048576.; cpu = sum(x.cpu_percent(None) for x in ps if x.is_running())
            except Exception: ps=[]; rss=cpu=0
            gpu=["", "", "", ""]
            try: gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm", "--format=csv,noheader,nounits"], text=True, timeout=3).strip().split(", ")
            except Exception: pass
            w.writerow(dict(zip(fields,[time.time(),"stage_w_training_and_transform",cpu,rss,len(ps),*gpu]))); h.flush(); stop.wait(1)


def driver() -> None:
    root = RAW / "stage_w/formal"; require(not root.exists(), "Stage W formal root exists")
    root.mkdir(parents=True)
    stop=threading.Event(); thread=threading.Thread(target=sampler,args=(stop,RAW/"stage_w_resource.csv",os.getpid()),daemon=True); thread.start()
    training=[]; transforms=[]; start=time.perf_counter()
    try:
        units=pilot_units()
        for candidate in CANDIDATES:
            for unit in units: training.append(train_cell(candidate, unit))
        require(len(training)==48,"training cardinality mismatch")
        lookup={(x["candidate_id"],x["unit_id"]):x for x in training}
        for candidate in CANDIDATES:
            for unit in units: transforms.append(transform_cell(lookup[(candidate,unit["unit_id"])],unit))
    finally: stop.set(); thread.join(timeout=3)
    train_manifest={"schema_version":1,"status":"LOCKED_PRE_LABEL","label_access":False,"planned_training":48,"training_attempts":48,
                    "successful_training":sum(x["status"]=="success" for x in training),"failed_training":sum(x["status"]!="success" for x in training),
                    "scientific_retry":0,"fallback_count":0,"training_cells":training}
    transform_manifest={"schema_version":1,"status":"LOCKED_PRE_LABEL","label_access":False,"planned_transforms":48,"transform_attempts":48,
                        "successful_transforms":sum(x["status"]=="success" for x in transforms),"failed_transforms":sum(x["status"]!="success" for x in transforms),
                        "scientific_retry":0,"fallback_count":0,"runtime_seconds":time.perf_counter()-start,"transforms":transforms}
    atomic_json(OUT/"weighted_mnn_training_manifest.json",train_manifest); atomic_json(OUT/"weighted_mnn_transform_manifest.json",transform_manifest)
    print(json.dumps({"status":"LOCKED_PRE_LABEL","training_attempts":48,"training_success":train_manifest["successful_training"],
                      "transform_attempts":48,"transform_success":transform_manifest["successful_transforms"],"runtime_seconds":transform_manifest["runtime_seconds"]},sort_keys=True))


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("driver",)); a=p.parse_args(); driver()


if __name__=="__main__": main()
