#!/usr/bin/env python3
"""Read-only end-to-end verification after Night-7A evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import (  # noqa: E402
    CANDIDATE_ORDER, DATASETS, atomic_json, sha256_file,
)

OUT = REPO / "outputs/night7a_handoff"
ALLOWED_METHOD = {
    "READY_COMMON_PROTOCOL", "READY_WITH_FIXED_ENDPOINT_ADAPTER",
    "SOURCE_ONLY_LICENSE_BLOCKED", "BLOCKED_PRIVATE_ASSET_OR_THIRD_MODALITY",
    "BLOCKED_NO_JOINT_CLUSTER_ENDPOINT", "BLOCKED_ENVIRONMENT",
}
ALLOWED_DATA = {
    "READY_FRESH_ANNOTATED_CONFIRMATION", "READY_LABEL_FREE_REPLICATION",
    "NEEDS_MANUAL_PROVENANCE_REVIEW", "NOT_SUITABLE",
}


def main() -> None:
    views = pd.read_csv(OUT / "source_views_index.csv")
    predictions = pd.read_csv(OUT / "source_prediction_index.csv")
    if len(views) != 60 or len(predictions) != 120:
        raise RuntimeError(f"source cardinality mismatch: {len(views)}/{len(predictions)}")
    if views.duplicated(["dataset", "seed", "graph_id"]).any():
        raise RuntimeError("duplicate source-view key")
    semantic = json.loads((OUT / "p0_semantic_contract.json").read_text())
    if (semantic["status"] != "PASS" or len(semantic["cells"]) != 60 or
            not all(row["exact_partition_parity"] and row["repeat_deterministic"]
                    for row in semantic["cells"]) or
            semantic["c02_six_view_identity_max_error"] > 1e-12 or
            not semantic.get("consensus_repeat_sparse_sha_all_equal")):
        raise RuntimeError("P0 semantic evidence is incomplete")
    transform_path = OUT / "locked_consensus_transform_manifest.json"
    transform = json.loads(transform_path.read_text())
    expected_keys = [
        (dataset, seed, candidate)
        for dataset in DATASETS
        for seed in (range(5) if dataset in {"a1", "tonsil"} else range(10))
        for candidate in CANDIDATE_ORDER
    ]
    observed_keys = [
        (row["dataset"], int(row["seed"]), row["candidate_id"])
        for row in transform["transforms"]
    ]
    if observed_keys != expected_keys or len(set(observed_keys)) != 360:
        raise RuntimeError("formal transform order/cardinality mismatch")
    for row in transform["transforms"]:
        audit = Path(row["candidate_audit_path"])
        if sha256_file(audit) != row["candidate_audit_sha256"]:
            raise RuntimeError(f"candidate audit SHA mismatch: {audit}")
        for artifact in row.get("artifacts", {}).values():
            path = Path(artifact["path"])
            if (path.stat().st_size != artifact["size_bytes"] or
                    sha256_file(path) != artifact["sha256"]):
                raise RuntimeError(f"candidate artifact SHA mismatch: {path}")
        weights = row.get("local_reliability_weights")
        if weights:
            path = Path(weights["path"])
            if (path.stat().st_size != weights["size_bytes"] or
                    sha256_file(path) != weights["sha256"]):
                raise RuntimeError(f"reliability weight SHA mismatch: {path}")
    if not (transform["formal_transform_attempts"] == 360 and
            transform["implementation_corrections"] <= 12 and
            transform["total_transform_attempts"] <= 372 and
            transform["scientific_training"] == 0 and
            transform["checkpoint_forward"] == 0 and
            transform["diffusion"] == 0 and
            transform["gpu_allocation_mib"] == 0):
        raise RuntimeError("transform budget audit failed")
    readiness = json.loads((OUT / "external_method_readiness.json").read_text())
    if (readiness["methods_audited"] != 13 or readiness["formal_benchmark_runs"] != 0 or
            any(row["status"] not in ALLOWED_METHOD for row in readiness["audits"])):
        raise RuntimeError("external source audit cardinality/status mismatch")
    fresh = json.loads((OUT / "fresh_dataset_preflight.json").read_text())
    if (fresh["fresh_external_per_spot_label_reads"] != 0 or
            fresh["large_archives_downloaded"] or len(fresh["contracts"]) != 4 or
            any(row["status"] not in ALLOWED_DATA for row in fresh["contracts"])):
        raise RuntimeError("fresh-data preflight contract mismatch")
    label = json.loads((OUT / "label_window_audit.json").read_text())
    if (label["status"] != "PASS" or label["fresh_external_label_reads"] != 0 or
            set(label["datasets"]) != set(DATASETS) or
            not label["single_authorized_evaluator_process"]):
        raise RuntimeError("label-window audit mismatch")
    replay = json.loads((OUT / "historical_metric_replay.json").read_text())
    if replay["status"] != "PASS" or replay["rows"] != "90/90" or replay["maximum_absolute_error"] > 1e-12:
        raise RuntimeError("historical metric replay mismatch")
    metrics = pd.read_csv(OUT / "per_seed_metrics.csv")
    if metrics.duplicated(["dataset", "seed", "candidate_id"]).any():
        raise RuntimeError("duplicate metric key")
    gates = pd.read_csv(OUT / "candidate_gate_table.csv")
    if gates.candidate_id.tolist() != list(CANDIDATE_ORDER):
        raise RuntimeError("gate table order mismatch")
    decision = json.loads((OUT / "night7a_decision.json").read_text())
    atomic_json(OUT / "end_to_end_verification.json", {
        "status": "PASS", "source_views": "60/60",
        "source_predictions": "120/120", "p0_h05_parity": "30/30 x 2",
        "formal_transform_keys": "360/360 fixed order unique",
        "successful_transform_artifacts_reverified": transform["success_count"],
        "scientific_numerical_failures_preserved": transform["scientific_numerical_failure_count"],
        "historical_metric_replay": "90/90 within 1e-12",
        "metric_primary_keys": len(metrics), "candidate_gate_rows": len(gates),
        "external_methods": "13/13", "fresh_data_contracts": "4/4",
        "fresh_external_label_reads": 0, "formal_benchmark_runs": 0,
        "scientific_training": 0, "checkpoint_forward": 0, "diffusion": 0,
        "gpu_use": 0, "terminal_status": decision["terminal_status"],
        "selected_structure": decision["selected_structure"],
        "transform_manifest_sha256": sha256_file(transform_path),
    })
    print(json.dumps({"status": "PASS", "terminal": decision["terminal_status"],
                      "selected": decision["selected_structure"]}, sort_keys=True))


if __name__ == "__main__":
    main()
