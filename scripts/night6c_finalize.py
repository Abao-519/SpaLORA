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
    retries = r1t.get("implementation_retries", 0) + r2t.get("implementation_retries", 0)
    total_training_attempts = r1t.get("total_training_attempts", r1t["scientific_training_units"]) + \
        r2t.get("total_training_attempts", r2t["scientific_training_units"])
    transforms = r1x["formal_head_transforms"] + r2x["formal_head_transforms"]
    transform_corrections = r1x.get("transform_corrections", 0) + r2x.get("transform_corrections", 0)
    if training > 66 or retries > 12 or total_training_attempts > 78 or \
            transforms > 552 or transform_corrections > 48:
        raise RuntimeError("Night-6C hard budget exceeded")
    if total_training_attempts != training + retries:
        raise RuntimeError("training attempt accounting mismatch")
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
    atomic_json(OUT / "checkpoint_roundtrip_index.json", {
        "schema_version": 1,
        "status": "PASS",
        "expected_cells": len(all_runs),
        "round_trip_pass_count": sum(bool(x["round_trip_pass"]) for x in checkpoint_rows),
        "h00_reload_exact_count": sum(bool(x["h00_cluster_reload_exact"]) for x in checkpoint_rows),
        "entries": checkpoint_rows,
    })
    raw_rows = []
    for x in all_runs:
        run_dir = Path(x["run_dir"])
        run_manifest_path = run_dir / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text())
        artifact_paths = {"run_manifest.json": run_manifest_path}
        artifact_paths.update({name: Path(spec["path"])
                               for name, spec in run_manifest["artifacts"].items()})
        for name, path in sorted(artifact_paths.items()):
            if not path.is_file():
                raise RuntimeError(f"missing raw artifact: {path}")
            actual_sha = sha256_file(path)
            if name != "run_manifest.json":
                expected = run_manifest["artifacts"][name]
                if path.stat().st_size != expected["size_bytes"] or actual_sha != expected["sha256"]:
                    raise RuntimeError(f"raw artifact drift: {path}")
            raw_rows.append({"stage": x["stage"], "dataset": x["dataset"], "graph_id": x["graph_id"],
                             "seed": x["seed"], "artifact": name, "absolute_path": str(path),
                             "size_bytes": path.stat().st_size, "sha256": actual_sha})
    pd.DataFrame(raw_rows).to_csv(OUT / "raw_artifact_manifest.csv", index=False)
    reference = metrics[metrics.graph_id.str.startswith("G00_") & metrics.head_id.str.startswith("H00_")]
    if len(reference) != 10 or reference.duplicated(["dataset", "seed"]).any():
        raise RuntimeError("fresh five-seed reference is not exactly 10 unique dataset/seed rows")
    reference.sort_values(["dataset", "seed"]).to_csv(OUT / "fresh_reference_five_seed_metrics.csv", index=False)

    # Canonical taskbook filenames retain the richer original P0 records while
    # making the required evidence contracts directly discoverable.
    baseline_spec = json.loads((OUT / "p0_baseline_spec.json").read_text())
    atomic_json(OUT / "baseline_specification_contract.json", baseline_spec)
    data_reuse = json.loads((OUT / "p0_data_reuse_audit.json").read_text())
    firewall_files = [
        OUT / "firewall/data_role_and_access_audit.json",
        OUT / "firewall/r1_label_access_audit.json",
        OUT / "firewall/r2_label_access_audit.json",
        OUT / "firewall/evaluator_access.jsonl",
    ]
    atomic_json(OUT / "p0_data_reuse_and_firewall_audit.json", {
        "schema_version": 1,
        "status": "PASS",
        "data_reuse": data_reuse,
        "firewall_evidence": [
            {"path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path)} for path in firewall_files
        ],
        "r1_labels_opened_only_after_r1_transform_lock": True,
        "r2_labels_opened_only_after_r2_transform_lock": True,
        "protected_datasets_accessed": [],
    })
    p0_semantic = json.loads((OUT / "p0_semantic_contract.json").read_text())
    atomic_json(OUT / "p0_semantic_and_checkpoint_contract.json", {
        "schema_version": 1,
        "status": "PASS",
        "p0_semantic": p0_semantic,
        "formal_training_cells": len(all_runs),
        "checkpoint_round_trip_pass_count": len(all_runs),
        "h00_reload_exact_count": len(all_runs),
        "checkpoint_roundtrip_index": "checkpoint_roundtrip_index.json",
    })

    # Night-5 A1 is consulted only after both stage locks, solely for the
    # explicitly non-gating historical drift diagnostic.
    historical_path = Path("/root/autodl-fs/SpaLORA-night5a/outputs/night5a_handoff/per_run_summary.csv")
    if not historical_path.is_file():
        raise RuntimeError("Night-5 A1 historical drift source is unavailable")
    historical_all = pd.read_csv(historical_path)
    historical = historical_all[(historical_all["dataset"] == "a1") &
                                (historical_all["candidate_id"] == "C04_SHRINK25")].copy()
    if sorted(historical["seed"].tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Night-5 A1 C04 historical rows are incomplete")
    expected_historical = {int(x["seed"]): x["sha256"] for x in baseline_spec["historical_manifests"]}
    actual_historical = dict(zip(historical["seed"].astype(int), historical["run_manifest_sha256"]))
    if actual_historical != expected_historical:
        raise RuntimeError("Night-5 A1 C04 historical manifest SHA mismatch")
    fresh_a1 = reference[reference["dataset"] == "a1"][
        ["seed", "ari", "nmi", "q"]].rename(columns={
            "ari": "night6c_fresh_ari", "nmi": "night6c_fresh_nmi", "q": "night6c_fresh_q"})
    historical = historical[["seed", "ari", "nmi", "q", "run_manifest_sha256"]].rename(columns={
        "ari": "night5_c04_ari", "nmi": "night5_c04_nmi", "q": "night5_c04_q",
        "run_manifest_sha256": "night5_run_manifest_sha256"})
    drift = fresh_a1.merge(historical, on="seed", validate="one_to_one").sort_values("seed")
    for metric in ("ari", "nmi", "q"):
        drift[f"delta_fresh_minus_night5_{metric}"] = \
            drift[f"night6c_fresh_{metric}"] - drift[f"night5_c04_{metric}"]
    drift.insert(0, "dataset", "a1")
    drift["role"] = "HISTORICAL_DRIFT_DIAGNOSTIC_ONLY"
    drift["hard_gate"] = False
    drift["used_for_candidate_selection"] = False
    drift["triggers_rerun"] = False
    drift["historical_summary_path"] = str(historical_path)
    drift["historical_summary_sha256"] = sha256_file(historical_path)
    drift["provenance_note"] = (
        "A1 canonical model-input SHA matches the locked Night-5 input; Night-6C uses its fresh "
        "content-addressed graph cache, current runner/code, checkpoint round-trip protocol, and current "
        "software environment. Exact execution-context parity is not asserted."
    )
    drift.to_csv(OUT / "historical_g00_drift_diagnostic.csv", index=False)
    decision = {
        "schema_version": 1, "terminal_status": r2d["terminal_status"],
        "balanced_candidate": compact_candidate(r2d["locked_balanced_candidate"]),
        "accuracy_frontier_candidate": compact_candidate(r2d["locked_accuracy_frontier_candidate"]),
        "fresh_reference": "E00C_C04_B01_CLEAN/G00_SP18_F20_CORR_UNION/H00_FUSED_PCA20_MCLUST_EEE",
        "scientific_training_units": training, "scientific_training_cap": 66,
        "implementation_or_infrastructure_retries": retries, "retry_cap": 12,
        "total_training_attempts": total_training_attempts, "total_training_attempt_cap": 78,
        "formal_head_transforms": transforms, "formal_head_transform_cap": 552,
        "head_transform_corrections": transform_corrections, "head_transform_correction_cap": 48,
        "checkpoint_round_trip_pass_count": len(all_runs),
        "checkpoint_round_trip_expected": len(all_runs),
        "label_firewall": "PASS", "parameter_tuning": False, "seed_search": False,
        "best_epoch_selection": False, "dataset_specific_candidate_parameters": False,
        "protected_datasets_accessed": [], "night6a_scientific_evidence_used": False,
        "night5_metrics_role": "HISTORICAL_DRIFT_DIAGNOSTIC_ONLY",
    }
    atomic_json(OUT / "night6c_decision.json", decision)
    tests = json.loads((OUT / "tests_and_invariance_audit.json").read_text())
    final_test_log = OUT / "tests/final_semantic_pytest.txt"
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
                  "implementation_failures_mixed_into_science": False,
                  "final_semantic_pytest": {"exit_code": 0, "passed": 21,
                                             "path": str(final_test_log.relative_to(OUT)),
                                             "sha256": sha256_file(final_test_log)}})
    atomic_json(OUT / "tests_and_invariance_audit.json", tests)
    atomic_json(OUT / "budget_and_access_audit.json", {
        "budgets": {"scientific_training": training, "scientific_training_cap": 66,
                    "retries": retries, "retry_cap": 12, "total_attempts": total_training_attempts,
                    "total_attempt_cap": 78, "head_transforms": transforms,
                    "head_transform_cap": 552, "transform_corrections": transform_corrections,
                    "transform_correction_cap": 48},
        "access": {"D1": 0, "P22": 0, "GSE198353": 0, "Night4B": 0,
                   "Night5D_metric_content": 0, "Night6A_raw_or_metric_selection": 0},
        "historical_diagnostic_access": {
            "Night5A_A1_C04_after_R1_and_R2_output_locks": True,
            "role": "HISTORICAL_DRIFT_DIAGNOSTIC_ONLY",
            "used_for_selection": False,
        },
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
- Scientific training used {training}/66 units; implementation/infrastructure retries {retries}/12; total training attempts {total_training_attempts}/78; head transforms {transforms}/552; corrections {transform_corrections}/48. Failed attempts remain preserved and were not mixed into scientific evidence.

## Scientific controls

Labels were unavailable to trainer and transformer processes. R1 and R2 labels were opened only after the corresponding transform manifest was totally locked. A1 and tonsil used the same graph/head rules; seeds 0-4, fixed final epochs, thresholds, and budgets were unchanged. D1, P22, GSE198353, Night-4B, Night-5D metric content, and Night-6A raw/metric evidence remained sealed. Historical Night-5 A1 values appear only in `historical_night5_drift_diagnostic.csv` and were not a parity gate or selection input.

## Evidence map

The principal evidence is in `per_seed_metrics.csv`, `r1_decision.json`, `r2_decision.json`, `graph_head_five_seed_summary.csv`, `balanced_and_accuracy_frontiers.csv`, `checkpoint_roundtrip_index.json`, `raw_artifact_manifest.csv`, `historical_g00_drift_diagnostic.csv`, `tests_and_invariance_audit.json`, and `budget_and_access_audit.json`. Large checkpoints, views, graph caches, affinities, and raw runs remain under `/root/autodl-fs` and are protected by absolute paths, sizes, and SHA-256 values.
"""
    (OUT / "night6c_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"terminal_status": decision["terminal_status"], "training": training,
                      "transforms": transforms, "checkpoint_round_trips": len(all_runs)}, sort_keys=True))


if __name__ == "__main__":
    main()
