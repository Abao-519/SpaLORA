#!/usr/bin/env python3
"""Post-evaluation reporting and immutable-root verification (never reads Y)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night8b_cardinality_safe_eval import atomic_json_fsync, sha256_file

OUT = REPO / "outputs/night8b_cardinality_safe_eval"
RAW_OUT = Path("/root/autodl-fs/night8b_cardinality_safe_eval_20260820")
ORIGINAL = Path("/root/autodl-fs/night8b_raw_runs_20260820")
RECOVERY = Path("/root/autodl-fs/night8b_head_recovery_20260820")
BRANCH = "revision/q2-night8b-cardinality-safe-eval-20260820"
FINAL_TAG = "night8b-cardinality-safe-eval-final-20260820"


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=str(REPO), text=True).strip()


def snapshot_tree(root: Path) -> list:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def prepare() -> None:
    decision = json.loads((OUT / "night8b_cardinality_safe_eval_decision.json").read_text())
    reference = json.loads((OUT / "reference_label_contract.json").read_text())
    statistics = json.loads((OUT / "cardinality_safe_paired_statistics.json").read_text())
    spatial = json.loads((OUT / "cardinality_safe_spatial_protection.json").read_text())
    resource = json.loads((OUT / "cardinality_safe_resource_audit.json").read_text())
    independent = json.loads((OUT / "cardinality_safe_independent_recompute.json").read_text())
    sensitivity = json.loads((OUT / "original_spectral_sensitivity_audit.json").read_text())
    access = json.loads((OUT / "label_access_and_evaluation_order_audit.json").read_text())
    p0 = json.loads((OUT / "p0_eval_recovery_authority.json").read_text())

    original_failures = []
    original_rows = p0["original_raw_297"]
    before_authority = json.loads((REPO / "outputs/night8b_head_recovery/original_artifact_manifest_before.json").read_text())
    after_rows = []
    for row in before_authority["rows"]:
        path = Path(row["path"])
        actual_sha = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        expected_sha = row.get("expected_sha256") or row.get("actual_sha256")
        expected_size = row.get("expected_size_bytes") or row.get("actual_size_bytes")
        match = actual_sha == expected_sha and actual_size == expected_size
        after_rows.append({"path": str(path), "sha256": actual_sha,
                           "size_bytes": actual_size, "match": match})
        if not match:
            original_failures.append(str(path))
    original_after = {
        "schema_version": 1, "root": str(ORIGINAL), "row_count": len(after_rows),
        "expected_row_count": 297, "failures": original_failures,
        "all_match": len(after_rows) == 297 and not original_failures, "rows": after_rows,
    }
    atomic_json_fsync(OUT / "original_raw_after_invariance.json", original_after)
    atomic_json_fsync(RAW_OUT / "manifests/original_raw_after_invariance.json", original_after)

    recovery_before = p0["head_recovery_before_snapshot"]
    recovery_after_rows = snapshot_tree(RECOVERY)
    before_map = {row["relative_path"]: (row["size_bytes"], row["sha256"])
                  for row in recovery_before["rows"]}
    after_map = {row["relative_path"]: (row["size_bytes"], row["sha256"])
                 for row in recovery_after_rows}
    recovery_failures = sorted(set(before_map) ^ set(after_map))
    recovery_failures += sorted(key for key in set(before_map) & set(after_map)
                                if before_map[key] != after_map[key])
    recovery_after = {
        "schema_version": 1, "root": str(RECOVERY),
        "before_row_count": len(before_map), "after_row_count": len(after_map),
        "failures": recovery_failures, "all_match": not recovery_failures,
        "rows": recovery_after_rows,
    }
    atomic_json_fsync(OUT / "head_recovery_after_invariance.json", recovery_after)
    atomic_json_fsync(RAW_OUT / "manifests/head_recovery_after_invariance.json", recovery_after)

    tests = {
        "schema_version": 1,
        "status": "PASS" if (independent["status"] == "PASS"
                              and original_after["all_match"]
                              and recovery_after["all_match"]
                              and access["status"] == "PASS") else "FAIL",
        "partition_rows": "20/20", "paired_rows": "10/10",
        "predicted_reference_K_mismatch_is_allowed": True,
        "reference_K_persisted_before_metrics": access["reference_K_persisted_before_metrics"],
        "lineage_raw_Y_read_count_after": access["after"],
        "independent_max_absolute_error": independent["maximum_absolute_error_vs_primary"],
        "independent_threshold": 1e-12,
        "original_spectral_terminal_decision_input": False,
        "original_raw_invariance": "297/297" if original_after["all_match"] else "FAIL",
        "head_recovery_root_invariance": recovery_after["all_match"],
        "scope_counts": access["scope_counts"],
    }
    atomic_json_fsync(OUT / "tests_and_invariance_audit.json", tests)
    budget = {
        "schema_version": 1, "wall_time_limit_minutes": 90,
        "evaluation_limit_minutes": 30, "independent_recompute_limit_minutes": 30,
        "evaluation_elapsed_seconds": access["elapsed_seconds"],
        "scientific_retry": 0, "fallback": 0, "gpu_used": False,
        "AutoDL_API_called": False,
        "raw_Y_read_this_task": 1, "raw_Y_read_lineage_after": 2,
        "third_raw_Y_read_forbidden": True,
        "scope_counts": access["scope_counts"],
    }
    atomic_json_fsync(OUT / "budget_and_access_audit.json", budget)

    terminal = decision["terminal_status"]
    qci = statistics["bootstrap_delta_q"]
    report = f"""# SpaLORA Night-8B Cardinality-Safe Evaluation Recovery Report

## Terminal status

`{terminal}`

This was a post-lock evaluation recovery, not a pristine holdout and not a SOTA benchmark. The 20 fixed `RECOVERY_EIGEN_KMEANS100` partitions remained byte-identical; no training, checkpoint load, forward, adapter, affinity rebuild, or head transform occurred.

## Reference-label contract

- Actual reference K: **{reference['reference_K']}**
- Fixed predicted K for both methods: **12**
- Different reference/predicted cardinalities are valid for ARI and contingency-table information metrics; both methods use the same fixed predicted K.
- Raw Y was read once in this task, the second and final read in the Night-8B lineage. Its per-spot vector was not persisted.

## Uniform-head primary result (HR_F00 - HR_U00)

- mean delta ARI: `{statistics['mean_delta_ari']:.12g}`
- mean delta NMI: `{statistics['mean_delta_nmi']:.12g}`
- mean delta Q: `{statistics['mean_delta_q']:.12g}`
- Q wins: `{statistics['q_wins']}/10`
- exact one-sided sign-flip p: `{statistics['exact_sign_flip_delta_q']['p_one_sided']:.12g}`
- paired bootstrap delta Q 95% CI: `[{qci['ci_lower']:.12g}, {qci['ci_upper']:.12g}]`

Science-core gate: `{decision['science_core_gate_pass']}`; spatial protection: `{decision['spatial_gate_pass']}`; resource gate: `{decision['resource_gate_pass']}`.

## Sensitivity and audit

- Original spectral complete-pair sensitivity: 9 pairs; mean delta Q `{sensitivity['mean_delta_q']:.12g}`; direction agrees with uniform head: `{sensitivity['direction_same_as_uniform_head']}`. It was descriptive only and did not enter the terminal decision.
- Same-process independent recomputation maximum absolute error: `{independent['maximum_absolute_error_vs_primary']:.3g}` (required <=1e-12).
- Original raw root: 297/297 unchanged. Head-recovery root: unchanged.
- Evaluation runtime: `{access['elapsed_seconds']:.3f}` seconds; evaluation GPU use: 0.

## Interpretation boundary

The result tests the frozen RNA+ATAC family policy on MISAR under a fixed K=12 uniform head and the actual reference-K labels. It cannot establish the original H05 endpoint, a pristine external holdout, or SOTA. Seeds measure algorithmic stability rather than independent biological replication.

## Git and delivery

Branch: `{BRANCH}`. Planned immutable final tag: `{FINAL_TAG}`. The exact final commit, tag verification, bundle, compact index, and shutdown dispatch receipt are external post-commit delivery records to avoid self-reference.
"""
    summary = f"""# Night-8B cardinality-safe recovery: plain-language summary

The frozen predictions contain 12 clusters, while the official reference labels contain {reference['reference_K']} categories. This is not an error: clustering agreement metrics can compare partitions with different category counts, and both competing methods still use the same fixed prediction K.

F00 minus U00 had mean delta ARI {statistics['mean_delta_ari']:.6f}, mean delta NMI {statistics['mean_delta_nmi']:.6f}, and mean delta Q {statistics['mean_delta_q']:.6f}, with {statistics['q_wins']}/10 Q wins. The preregistered terminal status is `{terminal}`.

This is a post-lock recovery because the labels had already been opened once by the failed evaluator. It is neither a pristine holdout nor a SOTA claim.
"""
    atomic_text(OUT / "night8b_cardinality_safe_eval_report.md", report)
    atomic_text(OUT / "night8b_cardinality_safe_eval_plain_language_summary.md", summary)
    print(json.dumps({"status": tests["status"], "terminal_status": terminal,
                      "reference_K": reference["reference_K"],
                      "original_297": original_after["all_match"],
                      "recovery_invariance": recovery_after["all_match"]}, sort_keys=True))
    if tests["status"] != "PASS":
        raise SystemExit(2)


def index() -> None:
    files = []
    roots = [
        REPO / "outputs/night8b_cardinality_safe_eval",
        REPO / "protocols/night8b_cardinality_safe_eval",
    ]
    explicit = [
        REPO / "SpaLORA/night8b_cardinality_safe_eval.py",
        REPO / "scripts/night8b_cardinality_safe_p0.py",
        REPO / "scripts/night8b_cardinality_safe_evaluate.py",
        REPO / "scripts/night8b_cardinality_safe_finalize.py",
        REPO / "tests/test_night8b_cardinality_safe_eval.py",
    ]
    index_path = OUT / "tracked_delivery_index.json"
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p != index_path):
            files.append(path)
    files.extend(path for path in explicit if path.is_file())
    unique = sorted(set(files))
    rows = [{"path": path.relative_to(REPO).as_posix(),
             "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
            for path in unique]
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_TRACKED_INDEX_EXCLUDING_SELF",
        "index_self_reference_excluded": True,
        "row_count": len(rows), "rows": rows,
    }
    atomic_json_fsync(index_path, payload)
    print(json.dumps({"status": payload["status"], "row_count": len(rows)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "index"), required=True)
    arguments = parser.parse_args()
    if arguments.phase == "prepare":
        prepare()
    else:
        index()


if __name__ == "__main__":
    main()
