#!/usr/bin/env python3
"""P0 authority and immutable-input audit for Night-8B evaluation recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night8b_cardinality_safe_eval import atomic_json_fsync, sha256_file

BASE_COMMIT = "4b4c10c32220c9210a366f8e0b3691b2b46cf4fe"
BASE_TAG = "night8b-head-recovery-final-20260820"
BASE_REPO = Path("/root/autodl-fs/SpaLORA-night8b-head-recovery")
ORIGINAL = Path("/root/autodl-fs/night8b_raw_runs_20260820")
RECOVERY = Path("/root/autodl-fs/night8b_head_recovery_20260820")
NEW_RAW = Path("/root/autodl-fs/night8b_cardinality_safe_eval_20260820")
PARENT_OUT = REPO / "outputs/night8b_head_recovery"
ORIGINAL_OUT = REPO / "outputs/night8b_handoff"
PROTOCOL = REPO / "protocols/night8b_cardinality_safe_eval"
EXPECTED_ORDER_SHA = "9f0514cee55d307a0ff81d44ffffc2da742dbe2d02b849576b7ef5903743dd1b"
EXPECTED_MAPPING_SHA = "322e7bf0f459998c882a0305e8aea129deee29570412982b3697976ac64b8ae5"
LOCK_SHA = "8a696ec456b9abe45c9fd65c3b646f2654300f6c51e47d766fb0bfcef0686e6c"


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=str(BASE_REPO), text=True).strip()


def snapshot_tree(root: Path) -> list:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(stat.st_size),
            "sha256": sha256_file(path),
        })
    return rows


def verify_index(index_path: Path, root: Path) -> dict:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = payload["files"] if "files" in payload else payload["rows"]
    failures = []
    for row in rows:
        relative = row.get("path") or row.get("relative_path")
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        expected = row.get("sha256") or row.get("file_sha256")
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    return {"expected_rows": len(rows), "verified_rows": len(rows) - len(failures),
            "failures": failures, "pass": not failures}


def verify_original_297() -> dict:
    authority = json.loads((PARENT_OUT / "original_artifact_manifest_before.json").read_text())
    rows = authority["rows"]
    failures = []
    for row in rows:
        path = Path(row["path"])
        actual_sha = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        expected_sha = row.get("expected_sha256") or row.get("actual_sha256")
        expected_size = row.get("expected_size_bytes") or row.get("actual_size_bytes")
        if actual_sha != expected_sha or actual_size != expected_size:
            failures.append({
                "path": str(path), "expected_sha256": expected_sha,
                "actual_sha256": actual_sha, "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
            })
    return {"expected_rows": 297, "verified_rows": len(rows) - len(failures),
            "source_row_count": len(rows), "failures": failures,
            "pass": len(rows) == 297 and not failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-attestation", required=True)
    args = parser.parse_args()

    failures = []
    current = git("rev-parse", "HEAD")
    base_tag = git("rev-parse", BASE_TAG + "^{commit}")
    clean = git("status", "--porcelain") == ""
    if current != BASE_COMMIT or base_tag != BASE_COMMIT or not clean:
        failures.append("git_base_or_clean_gate")

    windows = json.loads(Path(args.windows_attestation).read_text(encoding="utf-8"))
    if windows.get("status") != "PASS" or windows.get("authority_count") != 12:
        failures.append("windows_authority_12_of_12")

    tracked = verify_index(PARENT_OUT / "tracked_delivery_index.json", REPO)
    if tracked["expected_rows"] != 45 or not tracked["pass"]:
        failures.append("tracked_index_45_of_45")

    declared_lock = RECOVERY / "outputs/night8b_head_recovery/locked_recovery_partition_manifest.json"
    raw_lock = RECOVERY / "manifests/locked_recovery_partition_manifest.json"
    tracked_lock = PARENT_OUT / "locked_recovery_partition_manifest.json"
    lock_candidates = [p for p in (declared_lock, raw_lock, tracked_lock)
                       if p.is_file() and sha256_file(p) == LOCK_SHA]
    if len(lock_candidates) != 2 or raw_lock not in lock_candidates or tracked_lock not in lock_candidates:
        failures.append("unique_content_addressed_lock_resolution")
        lock = {"rows": []}
    else:
        lock = json.loads(raw_lock.read_text(encoding="utf-8"))

    partition_failures = []
    keys = []
    expected_ids = None
    for row in lock.get("rows", []):
        cluster_path = Path(row["clusters_path"])
        manifest_path = Path(row["transform_manifest_path"])
        key = (row.get("method"), int(row.get("seed", -1)))
        keys.append(key)
        if not cluster_path.is_file() or sha256_file(cluster_path) != row["clusters_file_sha256"]:
            partition_failures.append({"key": key, "field": "clusters_sha"})
            continue
        if not manifest_path.is_file() or sha256_file(manifest_path) != row["transform_manifest_sha256"]:
            partition_failures.append({"key": key, "field": "manifest_sha"})
            continue
        table = pd.read_csv(cluster_path)
        ids = table["observation_id"].astype(str).tolist()
        actual_k = int(table["cluster"].nunique())
        if len(table) != 1949 or actual_k != 12 or row.get("K") != 12:
            partition_failures.append({"key": key, "field": "shape_or_K"})
        if row.get("status") != "success" or row.get("deterministic_exact") is not True:
            partition_failures.append({"key": key, "field": "status_or_determinism"})
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            partition_failures.append({"key": key, "field": "observation_order"})
    expected_keys = [(method, seed) for method in ("HR_U00", "HR_F00") for seed in range(10)]
    if sorted(keys) != sorted(expected_keys) or len(lock.get("rows", [])) != 20:
        partition_failures.append({"field": "key_coverage_20"})
    if partition_failures:
        failures.append("partition_manifest_or_bytes")

    order_sha = hashlib.sha256("\n".join(expected_ids or []).encode("utf-8")).hexdigest()
    if len(expected_ids or []) != 1949 or order_sha != EXPECTED_ORDER_SHA:
        failures.append("observation_order_1949_sha")

    mapping_path = ORIGINAL_OUT / "prelabel_observation_mapping.csv"
    mapping_sha = sha256_file(mapping_path)
    mapping = pd.read_csv(mapping_path)
    mapping_alignment = (
        mapping_sha == EXPECTED_MAPPING_SHA
        and mapping["observation_id"].astype(str).tolist() == (expected_ids or [])
        and sorted(mapping["carrier_row"].astype(int).tolist()) == list(range(1949))
    )
    if not mapping_alignment:
        failures.append("annotation_mapping")

    prior_label = json.loads((PARENT_OUT / "label_window_audit.json").read_text())
    prior_failure = json.loads((PARENT_OUT / "evaluation_failure_audit.json").read_text())
    lineage = {
        "raw_Y_read_count": int(prior_failure.get("Y_authorized_read_count", -1)),
        "metrics_computed": int(prior_failure.get("metrics_computed", -1)),
        "actual_reference_K_persisted": not bool(prior_failure.get("actual_K_not_persisted_before_process_exit", False)),
        "label_firewall_breach": bool(prior_label.get("label_firewall_breach", True)),
    }
    if lineage != {"raw_Y_read_count": 1, "metrics_computed": 0,
                   "actual_reference_K_persisted": False, "label_firewall_breach": False}:
        failures.append("lineage_read1_metrics0_actualK_absent")

    original = verify_original_297()
    if not original["pass"]:
        failures.append("original_raw_297")
    recovery_snapshot = snapshot_tree(RECOVERY)
    if not recovery_snapshot:
        failures.append("head_recovery_snapshot_empty")

    scope_zero = {
        "training": 0, "checkpoint_load": 0, "forward": 0,
        "adapter": 0, "affinity_rebuild": 0, "head_transform": 0,
        "gpu_utilization_expected": 0,
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if not failures else "EVAL_RECOVERY_BLOCKED_INPUT_INTEGRITY",
        "failures": failures,
        "git": {"head": current, "base_tag_commit": base_tag, "clean": clean},
        "windows_authority": windows,
        "tracked_index": tracked,
        "locked_manifest_resolution": {
            "registry_declared_path": str(declared_lock),
            "registry_declared_path_exists": declared_lock.exists(),
            "resolved_immutable_path": str(raw_lock),
            "resolved_tracked_path": str(tracked_lock),
            "content_sha256": LOCK_SHA,
            "resolution_rule": "taskbook locate-plus-SHA: use byte-identical immutable raw manifests copy; never write either old root",
            "matching_authoritative_copies": [str(p) for p in lock_candidates],
        },
        "partition_verification": {
            "rows": len(lock.get("rows", [])), "clusters_verified": 20 - len([x for x in partition_failures if x.get("field") == "clusters_sha"]),
            "transform_manifests_verified": 20 - len([x for x in partition_failures if x.get("field") == "manifest_sha"]),
            "failures": partition_failures,
        },
        "observation_order": {"rows": len(expected_ids or []), "sha256": order_sha,
                              "expected_sha256": EXPECTED_ORDER_SHA, "pass": order_sha == EXPECTED_ORDER_SHA},
        "annotation_mapping": {"path": str(mapping_path), "sha256": mapping_sha,
                               "expected_sha256": EXPECTED_MAPPING_SHA, "alignment_pass": mapping_alignment},
        "prior_lineage": lineage,
        "original_raw_297": original,
        "head_recovery_before_snapshot": {
            "root": str(RECOVERY), "row_count": len(recovery_snapshot), "rows": recovery_snapshot,
        },
        "new_scope_counts": scope_zero,
        "immutable_roots_read_only": [str(ORIGINAL), str(RECOVERY)],
    }
    repo_output = REPO / "outputs/night8b_cardinality_safe_eval/p0_eval_recovery_authority.json"
    raw_output = NEW_RAW / "manifests/p0_eval_recovery_authority.json"
    atomic_json_fsync(raw_output, payload)
    atomic_json_fsync(repo_output, payload)
    atomic_json_fsync(NEW_RAW / "manifests/head_recovery_before_snapshot.json", {
        "schema_version": 1, "root": str(RECOVERY),
        "row_count": len(recovery_snapshot), "rows": recovery_snapshot,
    })
    print(json.dumps({"status": payload["status"], "failures": failures,
                      "tracked": tracked["verified_rows"],
                      "partitions": len(lock.get("rows", [])),
                      "original": original["verified_rows"],
                      "head_recovery_snapshot": len(recovery_snapshot)}, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
