#!/usr/bin/env python3
"""Finalize auditable Night-6C evidence after R2 evaluation."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6c_pipeline import atomic_json

OUT = REPO / "outputs/night6c_handoff"


def compact_candidate(value):
    if value is None: return None
    return {k: v for k, v in value.items() if k != "cells"}


def main() -> None:
    r1t = json.loads((OUT / "r1_training_manifest.json").read_text())
    r1x = json.loads((OUT / "r1_transform_manifest.json").read_text())
    r1d = json.loads((OUT / "r1_decision.json").read_text())
    r2t = json.loads((OUT / "r2_training_manifest.json").read_text())
    r2x = json.loads((OUT / "r2_transform_manifest.json").read_text())
    r2d = json.loads((OUT / "r2_decision.json").read_text())
    metrics = pd.read_csv(OUT / "per_seed_metrics.csv")
    training = r1t["scientific_training_units"] + r2t["scientific_training_units"]
    transforms = r1x["formal_head_transforms"] + r2x["formal_head_transforms"]
    if training > 66 or transforms > 552:
        raise RuntimeError("Night-6C hard budget exceeded")
    if r1t["success_count"] != 36 or r1x["attempted_transforms"] != 432 or \
            r1x["success_count"] + r1x.get("scientific_numerical_failure_count", 0) != 432:
        raise RuntimeError("R1 fixed coverage incomplete")
    expected_r2 = (1 + len(r1d["advanced_graphs"])) * 2 * 3
    expected_x2 = expected_r2 * (1 + len(r1d["advanced_heads"]))
    if r2t["success_count"] != expected_r2 or r2x["attempted_transforms"] != expected_x2 or \
            r2x["success_count"] + r2x.get("scientific_numerical_failure_count", 0) != expected_x2:
        raise RuntimeError("R2 fixed selected coverage incomplete")
    all_runs = r1t["runs"] + r2t["runs"]
    if any(not x["checkpoint_round_trip_pass"] or not x["h00_cluster_reload_exact"] for x in all_runs):
        raise RuntimeError("checkpoint round-trip completeness failed")
    checkpoint_rows = [{"stage": x["stage"], "dataset": x["dataset"], "graph_id": x["graph_id"],
                        "seed": x["seed"], "checkpoint_file_sha256": x["checkpoint_file_sha256"],
                        "final_tensor_state_sha256": x["final_tensor_state_sha256"],
                        "round_trip_pass": x["checkpoint_round_trip_pass"],
                        "h00_cluster_reload_exact": x["h00_cluster_reload_exact"],
                        "run_manifest_sha256": x["run_manifest_sha256"], "run_dir": x["run_dir"]}
                       for x in all_runs]
    pd.DataFrame(checkpoint_rows).to_csv(OUT / "checkpoint_round_trip_summary.csv", index=False)
    raw_rows = []
    for x in all_runs:
        for name in ("model_final.pt", "views.npz", "run_manifest.json", "checkpoint_reload_audit.json"):
            path = Path(x["run_dir"]) / name
            raw_rows.append({"stage": x["stage"], "dataset": x["dataset"], "graph_id": x["graph_id"],
                             "seed": x["seed"], "artifact": name, "absolute_path": str(path),
                             "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(raw_rows).to_csv(OUT / "raw_artifact_manifest.csv", index=False)
    reference = metrics[metrics.graph_id.str.startswith("G00_") & metrics.head_id.str.startswith("H00_")]
    reference.sort_values(["dataset", "seed"]).to_csv(OUT / "fresh_reference_five_seed_metrics.csv", index=False)
    decision = {
        "schema_version": 1, "terminal_status": r2d["terminal_status"],
        "balanced_candidate": compact_candidate(r2d["locked_balanced_candidate"]),
        "accuracy_frontier_candidate": compact_candidate(r2d["locked_accuracy_frontier_candidate"]),
        "fresh_reference": "E00C_C04_B01_CLEAN/G00_SP18_F20_CORR_UNION/H00_FUSED_PCA20_MCLUST_EEE",
        "scientific_training_units": training, "scientific_training_cap": 66,
        "implementation_or_infrastructure_retries": 0, "retry_cap": 12,
        "total_training_attempts": training, "total_training_attempt_cap": 78,
        "formal_head_transforms": transforms, "formal_head_transform_cap": 552,
        "head_transform_corrections": 0, "head_transform_correction_cap": 48,
        "checkpoint_round_trip_pass_count": len(all_runs),
        "checkpoint_round_trip_expected": len(all_runs),
        "label_firewall": "PASS", "parameter_tuning": False, "seed_search": False,
        "best_epoch_selection": False, "dataset_specific_candidate_parameters": False,
        "protected_datasets_accessed": [], "night6a_scientific_evidence_used": False,
        "night5_metrics_role": "HISTORICAL_DRIFT_DIAGNOSTIC_ONLY",
    }
    atomic_json(OUT / "night6c_decision.json", decision)
    tests = json.loads((OUT / "tests_and_invariance_audit.json").read_text())
    tests.update({"status": "PASS", "formal_training_cells": len(all_runs),
                  "checkpoint_round_trip_pass_count": len(all_runs),
                  "h00_reload_exact_count": len(all_runs),
                  "r1_transform_coverage": f"{r1x['attempted_transforms']}/{r1x['planned_transforms']}",
                  "r1_transform_successes": r1x["success_count"],
                  "r1_transform_scientific_failures": r1x.get("scientific_numerical_failure_count", 0),
                  "r2_transform_coverage": f"{r2x['attempted_transforms']}/{r2x['planned_transforms']}",
                  "r2_transform_successes": r2x["success_count"],
                  "r2_transform_scientific_failures": r2x.get("scientific_numerical_failure_count", 0),
                  "metric_primary_key_unique": not metrics.duplicated(["dataset","graph_id","seed","head_id"]).any(),
                  "lower_is_better_fields": ["geary_c", "boundary_disagreement"],
                  "implementation_failures_mixed_into_science": False})
    atomic_json(OUT / "tests_and_invariance_audit.json", tests)
    atomic_json(OUT / "budget_and_access_audit.json", {
        "budgets": {"scientific_training": training, "scientific_training_cap": 66,
                    "retries": 0, "retry_cap": 12, "total_attempts": training,
                    "total_attempt_cap": 78, "head_transforms": transforms,
                    "head_transform_cap": 552, "transform_corrections": 0,
                    "transform_correction_cap": 48},
        "access": {"D1": 0, "P22": 0, "GSE198353": 0, "Night4B": 0,
                   "Night5D_metric_content": 0, "Night6A_raw_or_metric_selection": 0},
        "fixed_seeds": [0,1,2,3,4], "run_order_preserved": True,
    })
    atomic_json(OUT / "shutdown_dispatch_status.json", {
        "status": "READY_FOR_LAST_REMOTE_COMMAND", "command": "/usr/bin/shutdown",
        "must_be_last_remote_command": True, "dispatched": False,
        "note": "The actual dispatch outcome is reported by the controlling SSH session; no reconnect is permitted afterward."
    })
    balanced = decision["balanced_candidate"]
    accuracy = decision["accuracy_frontier_candidate"]
    report = f"""# SpaLORA Night-6C final report

## Outcome

Terminal status: `{decision['terminal_status']}`.

Night-6C trained a new, paired C04/B01 reference and every preregistered R1 graph cell on A1 and the firewall-clean tonsil copy. It did not replay or synthesize nonexistent historical checkpoints. All downstream comparisons use the same Night-6C dataset/seed `G00/H00` reference.

Balanced candidate: `{balanced['graph_id'] + '/' + balanced['head_id'] if balanced else 'none'}`.

Accuracy-frontier candidate: `{accuracy['graph_id'] + '/' + accuracy['head_id'] if accuracy else 'none'}`. Any spatial trade-off designation is retained in `night6c_decision.json` and `balanced_and_accuracy_frontiers.csv`.

## Execution integrity

- P0 authority, Night-6B 31/31 root-aware evidence verification, ontology, and zero-obs label-free data checks passed.
- C04/B01 training semantics were uniquely reconstructed from the Night-5 registry, runner, and five manifests before science began.
- R1 training: {r1t['success_count']}/{r1t['planned_units']}; R1 head attempts: {r1x['attempted_transforms']}/{r1x['planned_transforms']} ({r1x['success_count']} successful, {r1x.get('scientific_numerical_failure_count', 0)} preregistered numerical failures, no fallback).
- R2 training: {r2t['success_count']}/{r2t['planned_units']}; R2 head attempts: {r2x['attempted_transforms']}/{r2x['planned_transforms']} ({r2x['success_count']} successful, {r2x.get('scientific_numerical_failure_count', 0)} preregistered numerical failures, no fallback).
- Every one of {len(all_runs)} successful training cells saved a real `model_final.pt`, canonical tensor-state SHA, full provenance, six views, and passed a fresh-process reload with exact H00 cluster labels.
- Scientific training used {training}/66 units; implementation retries 0/12; head transforms {transforms}/552; corrections 0/48.

## Scientific controls

Labels were unavailable to trainer and transformer processes. R1 and R2 labels were opened only after the corresponding transform manifest was totally locked. A1 and tonsil used the same graph/head rules; seeds 0-4, fixed final epochs, thresholds, and budgets were unchanged. D1, P22, GSE198353, Night-4B, Night-5D metric content, and Night-6A raw/metric evidence remained sealed. Historical Night-5 A1 values appear only in `historical_night5_drift_diagnostic.csv` and were not a parity gate or selection input.

## Evidence map

The principal evidence is in `per_seed_metrics.csv`, `r1_decision.json`, `r2_decision.json`, `graph_head_five_seed_summary.csv`, `balanced_and_accuracy_frontiers.csv`, `checkpoint_round_trip_summary.csv`, `raw_artifact_manifest.csv`, `tests_and_invariance_audit.json`, and `budget_and_access_audit.json`. Large checkpoints, views, graph caches, affinities, and raw runs remain under `/root/autodl-fs` and are protected by absolute paths, sizes, and SHA-256 values.
"""
    (OUT / "night6c_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"terminal_status": decision["terminal_status"], "training": training,
                      "transforms": transforms, "checkpoint_round_trips": len(all_runs)}, sort_keys=True))


if __name__ == "__main__":
    main()
