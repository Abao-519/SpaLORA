#!/usr/bin/env python3
"""Run the locked 80-cell Night-6D head transform matrix."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3af_cache import load_cache, sha256_file
from SpaLORA.night6d_firewall import reject_transform_payload
from SpaLORA.night6d_pipeline import (
    DATASET_CFG, GRAPHS, HEADS, NumericalHeadFailure, atomic_json,
    canonical_json_sha, file_row, load_views, run_head, runtime_resources,
)

OUT = REPO / "outputs/night6d_handoff"
BASE = {
    "d1": Path("/root/autodl-fs/night6d_cache_20260817/base/d1"),
    "p22": Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22"),
}


def main() -> None:
    reject_transform_payload()
    training_path = OUT / "locked_training_manifest.json"
    training = json.loads(training_path.read_text())
    if training["status"] != "LOCKED" or training["success_count"] != 40:
        raise RuntimeError("40-unit training manifest not locked")
    if any(not row["checkpoint_round_trip_pass"] for row in training["runs"]):
        raise RuntimeError("checkpoint round-trip incomplete")
    by_key = {(x["dataset"], x["graph_id"], int(x["seed"])): x for x in training["runs"]}
    if len(by_key) != 40:
        raise RuntimeError("training primary-key duplication")
    prepared = {}
    for dataset, path in BASE.items():
        prepared[dataset] = load_cache(path, sha256_file(path / "manifest.json"))
    rows = []
    ordinal = 0
    for dataset in ("d1", "p22"):
        ids = prepared[dataset].obs_names.astype(str).to_numpy()
        for graph_id in GRAPHS:
            for head_id in HEADS:
                for seed in range(10):
                    ordinal += 1
                    training_run = by_key[(dataset, graph_id, seed)]
                    run_dir = Path(training_run["run_dir"])
                    locked_ids = pd.read_csv(run_dir / "observation_ids.csv")["observation_id"].astype(str).to_numpy()
                    if not np.array_equal(ids, locked_ids):
                        raise RuntimeError("transform observation order mismatch")
                    views = load_views(run_dir / "views.npz")
                    root = run_dir / "heads" / head_id
                    completed = []
                    for path in sorted(root.glob("attempt_*/transform_manifest.json")):
                        row = json.loads(path.read_text())
                        if row.get("status") == "success":
                            if sha256_file(Path(row["head_dir"]) / "clusters.csv") != row["cluster_file_sha256"]:
                                raise RuntimeError("existing transform output SHA mismatch")
                        if row.get("status") in {"success", "scientific_numerical_failure_no_retry"}:
                            completed.append(row)
                    if len(completed) > 1:
                        raise RuntimeError("multiple terminal transform attempts")
                    if completed:
                        rows.append(completed[0])
                        continue
                    attempts = sorted(root.glob("attempt_*")) if root.exists() else []
                    attempt = len(attempts) + 1
                    target = root / f"attempt_{attempt:03d}"
                    target.mkdir(parents=True)
                    started = time.perf_counter()
                    try:
                        labels, audit = run_head(
                            HEADS[head_id], views, DATASET_CFG[dataset]["n_clusters"],
                            prepared[dataset].coordinates, ids, target,
                        )
                        if len(np.unique(labels)) != DATASET_CFG[dataset]["n_clusters"]:
                            raise NumericalHeadFailure("fixed head returned invalid K")
                        cluster_path = target / "clusters.csv"
                        pd.DataFrame({"observation_id": ids, "cluster": labels}).to_csv(cluster_path, index=False)
                        atomic_json(target / "head_audit.json", audit)
                        resource = runtime_resources(started)
                        artifacts = {p.name: file_row(p) for p in sorted(target.iterdir()) if p.is_file()}
                        row = {
                            "ordinal": ordinal,
                            "dataset": dataset,
                            "graph_id": graph_id,
                            "seed": seed,
                            "head_id": head_id,
                            "status": "success",
                            "fallback": False,
                            "attempt": attempt,
                            "run_dir": str(run_dir),
                            "head_dir": str(target),
                            "cluster_file_sha256": sha256_file(cluster_path),
                            "head_config_sha256": canonical_json_sha(HEADS[head_id]),
                            "artifacts": artifacts,
                            "label_values_deserialized": False,
                            "label_values_used": False,
                            **resource,
                        }
                        atomic_json(target / "transform_manifest.json", row)
                        rows.append(row)
                    except NumericalHeadFailure as exc:
                        resource = runtime_resources(started)
                        row = {
                            "ordinal": ordinal, "dataset": dataset, "graph_id": graph_id,
                            "seed": seed, "head_id": head_id, "attempt": attempt,
                            "status": "scientific_numerical_failure_no_retry", "fallback": False,
                            "run_dir": str(run_dir), "head_dir": str(target),
                            "cluster_file_sha256": None,
                            "head_config_sha256": canonical_json_sha(HEADS[head_id]),
                            "exception_type": type(exc).__name__, "message": str(exc),
                            "label_values_deserialized": False, "label_values_used": False,
                            **resource,
                        }
                        atomic_json(target / "failure.json", row)
                        atomic_json(target / "transform_manifest.json", row)
                        rows.append(row)
                    except Exception as exc:
                        atomic_json(target / "failure.json", {
                            "ordinal": ordinal, "dataset": dataset, "graph_id": graph_id,
                            "seed": seed, "head_id": head_id, "attempt": attempt,
                            "status": "implementation_or_infrastructure_failure",
                            "exception_type": type(exc).__name__, "message": str(exc),
                            "label_values_deserialized": False,
                        })
                        raise
                    print(json.dumps({"event": "transform_terminal", "ordinal": ordinal,
                                      "planned": 80, "dataset": dataset,
                                      "graph_id": graph_id, "head_id": head_id,
                                      "seed": seed, "status": rows[-1]["status"]}, sort_keys=True), flush=True)
    corrections = 0
    for run in training["runs"]:
        for head_id in HEADS:
            for failure in (Path(run["run_dir"]) / "heads" / head_id).glob("attempt_*/failure.json"):
                terminal = failure.parent / "transform_manifest.json"
                if not terminal.exists() or json.loads(terminal.read_text()).get("status") != "scientific_numerical_failure_no_retry":
                    corrections += 1
    if corrections > 8:
        raise RuntimeError("transform correction budget exceeded")
    if len(rows) != 80:
        raise RuntimeError("fixed transform coverage incomplete")
    success = sum(x["status"] == "success" for x in rows)
    numerical = sum(x["status"] == "scientific_numerical_failure_no_retry" for x in rows)
    atomic_json(OUT / "locked_transform_manifest.json", {
        "schema_version": 1,
        "status": "LOCKED",
        "locked_before_label_access": True,
        "training_manifest_sha256": sha256_file(training_path),
        "planned_transforms": 80,
        "attempted_transforms": len(rows),
        "success_count": success,
        "scientific_numerical_failure_count": numerical,
        "transform_corrections": corrections,
        "fixed_order": "dataset_graph_head_seed",
        "transforms": rows,
    })
    print(json.dumps({"event": "transform_total_lock", "attempted": len(rows),
                      "success": success, "scientific_failures": numerical,
                      "corrections": corrections}, sort_keys=True))


if __name__ == "__main__":
    main()
