#!/usr/bin/env python3
"""Run preregistered Night-6C label-free cluster heads in fixed order."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3af_cache import load_cache, sha256_file
from SpaLORA.night6c_pipeline import (
    atomic_json, canonical_json_sha, file_row, load_views, parse_registry, run_head, runtime_resources,
)

OUT = REPO / "outputs/night6c_handoff"
REG_PATH = REPO / "protocols/night6c/SpaLORA_Night6B_Candidate_Registry_2026-08-17.json"
CACHE = Path("/root/autodl-fs/night6c_cache_20260817/base")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("R1", "R2"), required=True)
    args = ap.parse_args()
    registry = json.loads(REG_PATH.read_text(encoding="utf-8"))
    _, heads = parse_registry(registry)
    training = json.loads((OUT / f"{args.stage.lower()}_training_manifest.json").read_text())
    if training["status"] != "LOCKED" or not training["locked_before_label_access"]:
        raise RuntimeError("training manifest not locked")
    if any(not r["checkpoint_round_trip_pass"] for r in training["runs"]):
        raise RuntimeError("run without checkpoint round-trip entered transforms")
    if args.stage == "R1":
        selected_heads = list(heads); planned = len(training["runs"]) * 12
    else:
        decision = json.loads((OUT / "r1_decision.json").read_text())
        selected_heads = [list(heads)[0]] + list(decision["advanced_heads"])
        if len(selected_heads) > 4 or len(set(selected_heads)) != len(selected_heads):
            raise RuntimeError("R2 head selection contract invalid")
        planned = len(training["runs"]) * len(selected_heads)
    rows = []; ordinal = 0
    for training_run in training["runs"]:
        run_dir = Path(training_run["run_dir"])
        dataset = training_run["dataset"]
        ids = pd.read_csv(run_dir / "observation_ids.csv")["observation_id"].astype(str).to_numpy()
        base_dir = CACHE / dataset
        prepared = load_cache(base_dir, sha256_file(base_dir / "manifest.json"))
        if not np.array_equal(ids, prepared.obs_names.astype(str).to_numpy()):
            raise RuntimeError("transform observation order mismatch")
        views = load_views(run_dir / "views.npz")
        for head_id in selected_heads:
            ordinal += 1
            target = run_dir / "heads" / head_id
            if target.exists():
                raise RuntimeError(f"refusing to overwrite head output: {target}")
            target.mkdir(parents=True)
            started = time.perf_counter()
            try:
                labels, aux = run_head(heads[head_id], views,
                                       int(registry["development_datasets"][dataset].get("known_k", 4)
                                           if dataset == "a1" else 4),
                                       prepared.coordinates, ids, target)
                # A1 registry known_k is numeric 10; tonsil is authority-resolved to 4.
                expected_k = 10 if dataset == "a1" else 4
                if len(np.unique(labels)) != expected_k:
                    raise RuntimeError("head K mismatch")
                cluster_path = target / "clusters.csv"
                pd.DataFrame({"observation_id": ids, "cluster": labels}).to_csv(cluster_path, index=False)
                atomic_json(target / "head_audit.json", aux)
                resource = runtime_resources(started)
                artifacts = {p.name: file_row(p) for p in sorted(target.iterdir()) if p.is_file()}
                row = {
                    "stage": args.stage, "ordinal": ordinal,
                    "dataset": dataset, "graph_id": training_run["graph_id"],
                    "seed": int(training_run["seed"]), "head_id": head_id,
                    "status": "success", "fallback": False,
                    "run_dir": str(run_dir), "head_dir": str(target),
                    "cluster_file_sha256": sha256_file(cluster_path),
                    "head_config_sha256": canonical_json_sha(heads[head_id]),
                    "artifacts": artifacts, "label_values_deserialized": False,
                    "label_values_used": False, **resource,
                }
                rows.append(row)
            except Exception as exc:
                atomic_json(target / "failure.json", {
                    "stage": args.stage, "ordinal": ordinal, "dataset": dataset,
                    "graph_id": training_run["graph_id"], "seed": int(training_run["seed"]),
                    "head_id": head_id, "status": "implementation_or_infrastructure_failure",
                    "exception_type": type(exc).__name__, "message": str(exc),
                    "label_values_deserialized": False,
                })
                raise
            print(json.dumps({"event": "transform_complete", "stage": args.stage,
                              "ordinal": ordinal, "planned": planned,
                              "dataset": dataset, "graph_id": training_run["graph_id"],
                              "seed": training_run["seed"], "head_id": head_id},
                             sort_keys=True), flush=True)
    aggregate = {
        "schema_version": 1, "stage": args.stage, "status": "LOCKED",
        "locked_before_label_access": True, "training_manifest_sha256":
            sha256_file(OUT / f"{args.stage.lower()}_training_manifest.json"),
        "planned_transforms": planned, "attempted_transforms": len(rows),
        "success_count": len(rows), "failure_count": 0,
        "formal_head_transforms": len(rows), "transform_corrections": 0,
        "selected_heads": selected_heads, "transforms": rows,
    }
    atomic_json(OUT / f"{args.stage.lower()}_transform_manifest.json", aggregate)
    print(json.dumps({"event": "transform_stage_locked", "stage": args.stage,
                      "success": len(rows), "planned": planned}, sort_keys=True))


if __name__ == "__main__":
    main()
