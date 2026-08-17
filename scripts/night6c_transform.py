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
    NumericalHeadFailure, atomic_json, canonical_json_sha, file_row, load_views,
    parse_registry, run_head, runtime_resources,
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
                raise RuntimeError("multiple terminal transform attempts for one cell")
            if completed:
                rows.append(completed[0]); continue
            attempts = sorted(root.glob("attempt_*")) if root.exists() else []
            attempt = len(attempts) + 1
            target = root / f"attempt_{attempt:03d}"
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
                    "status": "success", "fallback": False, "attempt": attempt,
                    "run_dir": str(run_dir), "head_dir": str(target),
                    "cluster_file_sha256": sha256_file(cluster_path),
                    "head_config_sha256": canonical_json_sha(heads[head_id]),
                    "artifacts": artifacts, "label_values_deserialized": False,
                    "label_values_used": False, **resource,
                }
                atomic_json(target / "transform_manifest.json", row)
                rows.append(row)
            except NumericalHeadFailure as exc:
                resource = runtime_resources(started)
                row = {"stage": args.stage, "ordinal": ordinal, "dataset": dataset,
                       "graph_id": training_run["graph_id"], "seed": int(training_run["seed"]),
                       "head_id": head_id, "attempt": attempt,
                       "status": "scientific_numerical_failure_no_retry",
                       "fallback": False, "run_dir": str(run_dir), "head_dir": str(target),
                       "cluster_file_sha256": None,
                       "head_config_sha256": canonical_json_sha(heads[head_id]),
                       "exception_type": type(exc).__name__, "message": str(exc),
                       "label_values_deserialized": False, "label_values_used": False,
                       **resource}
                atomic_json(target / "failure.json", row)
                atomic_json(target / "transform_manifest.json", row)
                rows.append(row)
                print(json.dumps({"event": "transform_scientific_failure_no_retry",
                                  "stage": args.stage, "ordinal": ordinal,
                                  "dataset": dataset, "graph_id": training_run["graph_id"],
                                  "seed": training_run["seed"], "head_id": head_id,
                                  "message": str(exc)}, sort_keys=True), flush=True)
                continue
            except Exception as exc:
                atomic_json(target / "failure.json", {
                    "stage": args.stage, "ordinal": ordinal, "dataset": dataset,
                    "graph_id": training_run["graph_id"], "seed": int(training_run["seed"]),
                    "head_id": head_id, "attempt": attempt,
                    "status": "implementation_or_infrastructure_failure",
                    "exception_type": type(exc).__name__, "message": str(exc),
                    "label_values_deserialized": False,
                })
                raise
            print(json.dumps({"event": "transform_complete", "stage": args.stage,
                              "ordinal": ordinal, "planned": planned,
                              "dataset": dataset, "graph_id": training_run["graph_id"],
                              "seed": training_run["seed"], "head_id": head_id},
                             sort_keys=True), flush=True)
    implementation_corrections = 0
    for training_run in training["runs"]:
        for head_id in selected_heads:
            for failure in (Path(training_run["run_dir"]) / "heads" / head_id).glob("attempt_*/failure.json"):
                terminal = failure.parent / "transform_manifest.json"
                if not terminal.exists() or json.loads(terminal.read_text()).get("status") != "scientific_numerical_failure_no_retry":
                    implementation_corrections += 1
    if implementation_corrections > 48:
        raise RuntimeError("head transform correction budget exceeded")
    success_count = sum(x["status"] == "success" for x in rows)
    scientific_failures = sum(x["status"] == "scientific_numerical_failure_no_retry" for x in rows)
    if len(rows) != planned:
        raise RuntimeError("fixed transform coverage incomplete")
    aggregate = {
        "schema_version": 1, "stage": args.stage, "status": "LOCKED",
        "locked_before_label_access": True, "training_manifest_sha256":
            sha256_file(OUT / f"{args.stage.lower()}_training_manifest.json"),
        "planned_transforms": planned, "attempted_transforms": len(rows),
        "success_count": success_count,
        "scientific_numerical_failure_count": scientific_failures,
        "failure_count": scientific_failures + implementation_corrections,
        "formal_head_transforms": len(rows),
        "transform_corrections": implementation_corrections,
        "selected_heads": selected_heads, "transforms": rows,
    }
    atomic_json(OUT / f"{args.stage.lower()}_transform_manifest.json", aggregate)
    print(json.dumps({"event": "transform_stage_locked", "stage": args.stage,
                      "success": success_count, "scientific_failures": scientific_failures,
                      "planned": planned}, sort_keys=True))


if __name__ == "__main__":
    main()
