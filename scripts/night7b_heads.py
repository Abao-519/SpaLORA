#!/usr/bin/env python3
"""Execute and lock the 540 registered Night-7B head transforms."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("R_HOME", "/root/miniconda3/envs/SpaLORA/lib/R")
os.environ["LD_LIBRARY_PATH"] = "/root/miniconda3/envs/SpaLORA/lib/R/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, atomic_sparse, canonical_partition, sha256_file  # noqa: E402
from SpaLORA.night7b_adaptive import (  # noqa: E402
    HEAD_ORDER, VIEWS, affinity_audit, build_head_affinity, run_partition,
)

OUT = REPO / "outputs/night7b_handoff"
RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
SOURCE = RAW / "source"
HEAD_RAW = RAW / "head_stage"
REG = REPO / "protocols/night7b/SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json"


def save_clusters(path: Path, ids, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame({"observation_id": ids, "cluster": np.asarray(labels, dtype=np.int64)}).to_csv(tmp, index=False)
    os.replace(tmp, path)


def file_row(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def canonical_json_sha(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    registry = json.loads(REG.read_text())
    if tuple(registry["head_candidate_order"]) != HEAD_ORDER:
        raise RuntimeError("head order mismatch")
    candidates = {x["id"]: x for x in registry["head_candidates"]}
    resolution = registry["partition_heads"]["LEIDEN_EXACT_K"]["resolution_grid"]
    units = list(csv.DictReader((OUT / "source_unit_index.csv").open(newline="")))
    if len(units) != 30: raise RuntimeError("source unit cardinality mismatch")
    rows = []; ordinal = 0
    for unit in units:
        unit_id = unit["unit_id"]; unit_dir = SOURCE / unit_id
        ids = [x.strip() for x in (unit_dir / "observation_ids.txt").read_text().splitlines() if x.strip()]
        with np.load(unit_dir / "g00_views.npz", allow_pickle=False) as x:
            v00 = {key: np.asarray(x[key]) for key in VIEWS}
        with np.load(unit_dir / "g04_views.npz", allow_pickle=False) as x:
            v04 = {key: np.asarray(x[key]) for key in VIEWS}
        s00, s04 = sp.load_npz(unit_dir / "s00.npz"), sp.load_npz(unit_dir / "s04.npz")
        representatives = ("H00", "H01", "H02", "H03", "H04", "H05", "H06", "H07")
        cache = {hid: build_head_affinity(hid, s00, s04, v00, v04, ids) for hid in representatives}
        affinity_source = {"H08":"H00", "H09":"H01", "H10":"H03", "H11":"H06",
                           "H12":"H00", "H13":"H01", "H14":"H00", "H15":"H01",
                           "H16":"H00", "H17":"H01"}
        for head_id in HEAD_ORDER:
            ordinal += 1; start = time.perf_counter(); before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            target = HEAD_RAW / "formal" / unit_id / head_id / "attempt_001"
            if target.exists(): raise RuntimeError("formal head cell already exists: %s" % target)
            target.mkdir(parents=True)
            source_id = affinity_source.get(head_id, head_id)
            affinity, diagnostics = cache[source_id]
            affinity_path = target / "affinity.npz"; atomic_sparse(affinity_path, affinity)
            row = {"ordinal": ordinal, "unit_id": unit_id, "dataset": unit["dataset"],
                   "seed": int(unit["seed"]), "head_id": head_id,
                   "head_contract": candidates[head_id], "attempt": 1,
                   "config_sha256": canonical_json_sha(candidates[head_id]),
                   "input_worker_sha256": unit["worker_input_sha256"],
                   "obs_order_sha256": unit["ordered_observation_sha256"],
                   "canonical_affinity_sha256": sparse_sha(affinity),
                   "affinity_audit": affinity_audit(affinity),
                   "affinity_diagnostics": diagnostics,
                   "affinity_artifact": file_row(affinity_path),
                   "fallback": False, "retry": False, "label_access": False,
                   "gpu_allocation_mib": 0.0}
            try:
                labels, part = run_partition(head_id, affinity, int(unit["K"]), resolution)
                clusters = target / "clusters.csv"; save_clusters(clusters, ids, labels)
                row.update({"status":"success", "partition":part,
                            "canonical_partition_sha256": array_sha(canonical_partition(labels)),
                            "cluster_count": int(len(np.unique(labels))),
                            "clusters_artifact": file_row(clusters), "failure_type": None})
            except Exception as exc:
                trace = target / "failure_trace.txt"; trace.write_text(traceback.format_exc(), encoding="utf-8")
                row.update({"status":"scientific_numerical_failure", "failure_type":type(exc).__name__,
                            "failure_message":str(exc), "failure_trace":file_row(trace),
                            "canonical_partition_sha256":None, "cluster_count":None})
            row["runtime_seconds"] = time.perf_counter() - start
            row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            manifest_path = target / "transform_manifest.json"; atomic_json(manifest_path, row)
            row["manifest_path"] = str(manifest_path); row["manifest_sha256"] = sha256_file(manifest_path)
            rows.append(row)
            print("H_CELL", ordinal, unit_id, head_id, row["status"], flush=True)
    if len(rows) != 540: raise RuntimeError("H matrix incomplete")
    locked = {"schema_version":1, "stage":"H", "planned":540, "attempts":540,
              "success":sum(x["status"] == "success" for x in rows),
              "scientific_numerical_failures":sum(x["status"] != "success" for x in rows),
              "fallback_count":0, "retry_count":0, "label_access":False,
              "locked_before_evaluation":True, "registry_sha256":sha256_file(REG),
              "fixed_order":"dataset_then_seed_then_H00_to_H17", "transforms":rows}
    atomic_json(OUT / "locked_head_transform_manifest.json", locked)


if __name__ == "__main__": main()
