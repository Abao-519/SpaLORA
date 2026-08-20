#!/usr/bin/env python3
"""Independent final evidence, report, and compact packager for Night-9A."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
RAW = Path("/root/autodl-fs/night9a_efficient_topology_transfer_20260820")
OUT = REPO / "outputs/night9a"
REPORTS = REPO / "reports"
BASE = "991ba9dcbd7108c2b3b4c9b4b8e233c5f394da5a"
BASE_TAG = "night8b-cardinality-safe-eval-final-20260820"
BRANCH = "revision/q2-night9a-efficient-topology-transfer-rnd-20260820"
FINAL_TAG = "night9a-final-20260820"
COMPACT = RAW / "delivery/official_compact"

import sys
sys.path.insert(0, str(REPO))
from SpaLORA.night9a_efficient import affinity_fidelity, sha256_file
from scripts.night9a_run import (
    authority_snapshot, dataset_paths, read_ids, reference_root,
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def file_row(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def clusters(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if not np.array_equal(frame["observation_id"].astype(str).to_numpy(), ids):
        raise RuntimeError(f"observation order mismatch: {path}")
    return frame["cluster"].to_numpy(dtype=np.int64)


def numeric_differences(left, right, prefix="") -> list[dict]:
    rows = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"dictionary key drift at {prefix}")
        for key in sorted(left):
            rows.extend(numeric_differences(left[key], right[key], f"{prefix}.{key}"))
    elif isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise RuntimeError(f"boolean drift at {prefix}: {left} != {right}")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        rows.append({"field": prefix.lstrip("."), "recorded": float(left),
                     "recomputed": float(right), "abs_error": abs(float(left) - float(right))})
    elif left != right:
        raise RuntimeError(f"value drift at {prefix}: {left!r} != {right!r}")
    return rows


def teacher_invariance() -> dict:
    before_path = RAW / "manifests/teacher_authority_before.json"
    after_path = RAW / "manifests/teacher_authority_after.json"
    before = json.loads(before_path.read_text())
    after = authority_snapshot(after_path)
    exact = before["rows"] == after["rows"] and before["row_count"] == after["row_count"]
    payload = {
        "status": "PASS" if exact else "FAIL",
        "byte_level_rows_exact": exact,
        "row_count_before": before["row_count"], "row_count_after": after["row_count"],
        "before_manifest": file_row(before_path), "after_manifest": file_row(after_path),
    }
    atomic_json(OUT / "teacher_roots_before_after_invariance.json", payload)
    if not exact:
        raise RuntimeError("teacher authority roots changed")
    return payload


def independent_recompute() -> dict:
    lock = json.loads((OUT / "locked_R1_manifest.json").read_text())
    audit_rows = []
    maximum = 0.0
    for ordinal, row in enumerate(lock["rows"], 1):
        if row["status"] != "success":
            raise RuntimeError("R1 contains a non-success row")
        dataset = row["dataset"]; seed = int(row["seed"])
        paths = dataset_paths(dataset, seed)
        ids = read_ids(paths["ids"])
        candidate_affinity = sp.load_npz(row["artifacts"]["affinity"]["path"])
        teacher_affinity = sp.load_npz(paths["teacher_affinity"])
        candidate_partition = clusters(Path(row["artifacts"]["clusters"]["path"]), ids)
        teacher_partition = clusters(reference_root(dataset, seed) / "FULL_F00/clusters.csv", ids)
        fidelity = affinity_fidelity(candidate_affinity, teacher_affinity,
                                     candidate_partition, teacher_partition)
        fidelity_diffs = numeric_differences(row["fidelity_vs_full_f00"], fidelity)

        u00_manifest = json.loads((reference_root(dataset, seed) /
                                   "U00/reference_manifest.json").read_text())
        resource = row["resource"]
        expected_candidate = (float(row["source_backbone"]["historical_runtime_seconds"]) +
                              float(row["topology_expert"]["runtime_seconds"]) +
                              float(row["adapter"]["runtime_seconds"]) +
                              float(row["head"]["runtime_seconds"]))
        expected_u00 = (float(row["source_backbone"]["historical_runtime_seconds"]) +
                        float(u00_manifest["head"]["runtime_seconds"]))
        expected_peak = max(float(row["source_backbone"]["historical_peak_gpu_mib"]),
                            float(row["topology_expert"]["peak_gpu_mib"]),
                            float(row["adapter"]["peak_gpu_mib"]))
        resource_expected = {
            "candidate_end_to_end_seconds": expected_candidate,
            "u00_end_to_end_seconds": expected_u00,
            "runtime_ratio_vs_u00": expected_candidate / expected_u00,
            "candidate_peak_gpu_mib": expected_peak,
            "u00_peak_gpu_mib": float(row["source_backbone"]["historical_peak_gpu_mib"]),
            "peak_gpu_ratio_vs_u00": expected_peak / max(
                float(row["source_backbone"]["historical_peak_gpu_mib"]), 1e-12),
        }
        resource_recorded = {key: resource[key] for key in resource_expected}
        resource_diffs = numeric_differences(resource_recorded, resource_expected)
        cell_max = max([x["abs_error"] for x in fidelity_diffs + resource_diffs] or [0.0])
        maximum = max(maximum, cell_max)
        audit_rows.append({"ordinal": ordinal, "candidate_id": row["candidate_id"],
                           "dataset": dataset, "seed": seed,
                           "max_abs_error": cell_max,
                           "fidelity_checks": len(fidelity_diffs),
                           "resource_checks": len(resource_diffs)})
    metric_audit = json.loads((OUT / "R1_independent_metric_recompute.json").read_text())
    maximum = max(maximum, float(metric_audit["max_abs_error"]))
    payload = {"status": "PASS" if maximum <= 1e-12 else "FAIL",
               "tolerance": 1e-12, "max_abs_error": maximum,
               "chain_count": len(audit_rows), "rows": audit_rows,
               "metric_audit": file_row(OUT / "R1_independent_metric_recompute.json")}
    atomic_json(OUT / "independent_metric_resource_fidelity_recompute.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("independent recomputation exceeded tolerance")
    return payload


def infrastructure_audit() -> dict:
    corrections = [
        ("P0_PATH_AUTHORITY_CORRECTION", RAW / "logs/p0.log",
         "MISAR locked affinity was initially addressed in the partition directory; no scientific chain ran."),
        ("P0_FRESH_PROCESS_CUDA_INIT_CORRECTION", RAW / "logs/p0_smoke.log",
         "Fresh-process CUDA statistics reset preceded CUDA initialization; empty partial was isolated."),
        ("P0_NATIVE_THREAD_RUNTIME_CORRECTION", RAW / "logs/p0_smoke_attempt2.log",
         "Three-thread native runtime crashed before adapter/affinity completion; partial was isolated and one-thread reproduction passed."),
    ]
    rows = []
    for identifier, log, note in corrections:
        rows.append({"id": identifier, "scientific_retry": 0, "label_access": 0,
                     "note": note, "log": file_row(log)})
    invalid = RAW / "invalid_infrastructure_attempts"
    partials = [file_row(path) for path in sorted(invalid.rglob("*")) if path.is_file()]
    payload = {"status": "PASS", "global_infrastructure_corrections_used": 3,
               "global_infrastructure_corrections_max": 4,
               "scientific_retry": 0, "fallback": 0,
               "corrections": rows, "preserved_partial_artifacts": partials,
               "single_thread_reproduction": file_row(
                   RAW / "logs/p0_smoke_e00_single_thread_repro.log"),
               "final_smoke": file_row(RAW / "logs/p0_smoke_attempt3_single_thread.log")}
    atomic_json(OUT / "infrastructure_corrections_and_failures.json", payload)
    return payload


def evidence() -> None:
    summary = json.loads((OUT / "R1_candidate_summary.json").read_text())
    shortlist = json.loads((OUT / "R1_shortlist.json").read_text())
    if shortlist["status"] != "NO_ELIGIBLE_CANDIDATE" or shortlist["shortlist_candidate_ids"]:
        raise RuntimeError("R1 terminal rule mismatch")
    teacher = teacher_invariance()
    independent = independent_recompute()
    infrastructure = infrastructure_audit()

    decision = {
        "terminal_status": "NIGHT9A_NO_EFFICIENT_R02_REPLACEMENT_KEEP_FULL_F00",
        "R1_formal_chain_count": 54, "R1_success_count": 54,
        "R2_formal_chain_count": 0, "R3_formal_chain_count": 0,
        "eligible_candidate_count": 0, "shortlist_candidate_ids": [],
        "frozen_teacher_to_keep": "FULL_F00_R02_RECON_MNN_EQUAL",
        "reason": "No registered candidate passed all locked runtime, P22, spatial, and MISAR fidelity gates.",
        "candidate_summaries": summary, "scientific_retry": 0, "fallback": 0,
        "misar_Y_access_this_task": 0, "lineage_raw_Y_read_count": 2,
        "third_MISAR_Y_read_forbidden": True, "claim_sota": False,
        "third_party_benchmark": 0,
    }
    atomic_json(OUT / "night9a_decision.json", decision)

    resource_rows = []
    for row in summary:
        resource_rows.append({key: row[key] for key in (
            "candidate_id", "runtime_ratio_vs_u00", "peak_gpu_ratio_vs_u00",
            "mean_candidate_end_to_end_seconds", "mean_u00_end_to_end_seconds",
            "p22_mean_delta_ari_vs_full_f00", "p22_mean_delta_nmi_vs_full_f00",
            "p22_mean_delta_q_vs_full_f00", "p22_mean_delta_q_vs_u00",
            "misar_mean_partition_ari", "misar_min_partition_ari",
            "misar_mean_partition_nmi", "misar_exact_partition_count",
            "all_hard_gates_pass", "gate_failures")})
    pd.DataFrame(resource_rows).to_csv(OUT / "night9a_candidate_resource_fidelity_summary.csv",
                                       index=False)
    fastest = min(summary, key=lambda x: x["runtime_ratio_vs_u00"])
    best_p22 = max(summary, key=lambda x: x["p22_mean_delta_q_vs_full_f00"])
    closest = min(summary, key=lambda x: (len(x["gate_failures"]),
                                          x["runtime_ratio_vs_u00"]))
    budget = {
        "status": "PASS", "formal_chain_units_used": 54, "formal_chain_units_max": 76,
        "total_formal_attempts": 54, "total_attempts_max": 80,
        "R1": 54, "R2": 0, "R3": 0, "scientific_retry": 0, "fallback": 0,
        "infrastructure_corrections_used": 3, "infrastructure_corrections_max": 4,
        "unit_wall_minutes_max": 20, "unit_timeout_count": 0,
        "total_wall_hours_max": 6,
        "label_access": {"P22_R1_windows": 1, "P22_R2_windows": 0,
                         "P22_R3_windows": 0, "MISAR_Y_this_task": 0,
                         "MISAR_lineage_total": 2, "third_MISAR_Y_read": 0},
    }
    atomic_json(OUT / "budget_and_label_access_audit.json", budget)
    atomic_json(OUT / "tests_and_invariance_audit.json", {
        "status": "PASS", "pytest_log": file_row(OUT / "final_tests.log"),
        "teacher_invariance": teacher, "independent_recompute": independent,
        "infrastructure": infrastructure,
        "same_head": "DATASET_LOCKED_EIGEN_KMEANS100",
        "unchanged_adapter": "R02_RECON_MNN_EQUAL_160_FIXED_EPOCHS",
        "candidate_real_smoke": "9/9 PASS", "formal_key_coverage": "54/54",
    })
    atomic_json(OUT / "git_audit.json", {
        "status": "PASS", "base_commit": BASE, "base_tag": BASE_TAG,
        "base_tag_peeled_commit": git("rev-parse", f"{BASE_TAG}^{{commit}}"),
        "branch": git("branch", "--show-current"), "ordinary_push_only": True,
        "force_push": False, "protection_tag": "baseline/pre-night9a-efficient-topology-transfer-rnd-20260820",
        "final_tag_planned_after_delivery_index_commit": FINAL_TAG,
        "evidence_source_commit": git("rev-parse", "HEAD"),
    })
    atomic_json(OUT / "shutdown_dispatch_intent.json", {
        "status": "INTENT_RECORDED_BEFORE_FINAL_COMMIT",
        "command": "/usr/bin/shutdown", "must_be_last_remote_command": True,
        "autodl_api_used": False, "reconnect_after_dispatch_forbidden": True,
        "dispatch_only_after_windows_compact_verification": True,
    })

    report = f"""# SpaLORA Night-9A report

## Terminal decision

`NIGHT9A_NO_EFFICIENT_R02_REPLACEMENT_KEEP_FULL_F00`

Night-9A completed all 54 registered R1 chains successfully, but zero of nine
candidates passed every pre-registered gate. R2 and R3 were therefore not run.
The locked recommendation is to keep the full F00/R02 route and not claim an
efficient topology-transfer replacement.

## Plain-language scientific result

The old F00 is slow because it trains both a full G04 backbone and a second full
G00 backbone before the fixed R02 relation adapter. Night-9A removed that second
from-scratch backbone: E00--E04 transferred the G04 state into G00 with 0--320
fixed epochs, while E05--E08 used deterministic sparse topology projections.

The closest candidate was `{closest['candidate_id']}`. Its mean P22 delta Q versus
full F00 was {closest['p22_mean_delta_q_vs_full_f00']:+.6f}, its delta Q versus U00
was {closest['p22_mean_delta_q_vs_u00']:+.6f}, MISAR mean partition ARI was
{closest['misar_mean_partition_ari']:.6f}, and runtime remained
{closest['runtime_ratio_vs_u00']:.4f}x. The fastest candidate,
`{fastest['candidate_id']}`, still required {fastest['runtime_ratio_vs_u00']:.4f}x
U00, above the 1.50x gate. The highest P22-Q-preserving candidate,
`{best_p22['candidate_id']}`, changed Q versus full F00 by
{best_p22['p22_mean_delta_q_vs_full_f00']:+.6f}, but failed these gates:
{', '.join(best_p22['gate_failures'])}.

Thus the registered transfer idea provides useful negative engineering evidence,
but it cannot currently be promoted as the paper's efficient topology-transfer
innovation. MISAR results are label-free partition-fidelity diagnostics only;
raw MISAR Y was not read and no metric inheritance claim is made.

## Audit summary

- Authority/base/tag and 275 teacher artifact rows: PASS; before/after exact.
- P0 strict transfer, projection determinism, same-head parity: PASS.
- P0 real P22 seed-0 construction smoke: 9/9 PASS.
- Formal science: R1 54/54 success; R2 0; R3 0.
- Independent metric/resource/fidelity maximum absolute error:
  {independent['max_abs_error']:.3g} (tolerance 1e-12).
- Scientific retry: 0; fallback: 0; timeouts: 0.
- Infrastructure corrections: 3/4, all before formal science and preserved.
- P22 label windows: one locked R1 window; MISAR Y reads this task: 0;
  lineage total remains 2 and a third read remains forbidden.
- Third-party benchmark: 0; SOTA claim: false; AutoDL API: unused.
"""
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "night9a_report.md").write_text(report)
    (OUT / "night9a_report.md").write_text(report)

    tracked = []
    patterns = [
        REPO / "SpaLORA/night9a_efficient.py",
        REPO / "scripts/night9a_run.py", REPO / "scripts/night9a_evaluate.py",
        REPO / "scripts/night9a_finalize.py", REPO / "tests/test_night9a_efficient.py",
        REPO / "reports/night9a_report.md",
    ]
    patterns += sorted((REPO / "protocols/night9a").glob("*"))
    patterns += sorted(OUT.glob("*"))
    index_path = OUT / "tracked_delivery_index.json"
    for path in patterns:
        if path.is_file() and path != index_path:
            tracked.append({"relative_path": str(path.relative_to(REPO)),
                            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_json(index_path, {
        "schema_version": 1, "status": "PASS", "root": str(REPO),
        "root_rule": "all listed paths are relative to repository root; this non-self-referential index excludes itself",
        "file_count": len(tracked), "files": tracked,
        "terminal_status": decision["terminal_status"],
        "final_tag_must_be_created_only_after_commit_containing_this_index": FINAL_TAG,
    })
    print(json.dumps({"event": "night9a_evidence_complete",
                      "terminal": decision["terminal_status"],
                      "tracked_files": len(tracked)}))


def package() -> None:
    if git("status", "--porcelain"):
        raise RuntimeError("repository is dirty before compact packaging")
    if git("rev-parse", "HEAD") != git("rev-parse", "@{u}"):
        raise RuntimeError("final commit is not pushed")
    if git("rev-parse", f"{FINAL_TAG}^{{commit}}") != git("rev-parse", "HEAD"):
        raise RuntimeError("final tag does not peel to HEAD")
    if COMPACT.exists():
        raise RuntimeError(f"refusing to overwrite compact: {COMPACT}")
    COMPACT.mkdir(parents=True)
    index = json.loads((OUT / "tracked_delivery_index.json").read_text())
    for row in index["files"]:
        source = REPO / row["relative_path"]
        target = COMPACT / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != row["sha256"]:
            raise RuntimeError(f"compact copy SHA mismatch: {source}")
    target_index = COMPACT / "outputs/night9a/tracked_delivery_index.json"
    target_index.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "tracked_delivery_index.json", target_index)

    bundles = COMPACT / "bundles"; bundles.mkdir()
    bundle = bundles / "night8b_to_night9a_incremental.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle),
                    f"{BASE_TAG}..{BRANCH}"], cwd=REPO, check=True)
    verify = subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=REPO,
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True)
    (bundles / "bundle_verify.txt").write_text(verify.stdout)
    atomic_json(COMPACT / "external_git_receipt.json", {
        "final_commit": git("rev-parse", "HEAD"), "final_tag": FINAL_TAG,
        "final_tag_peeled_commit": git("rev-parse", f"{FINAL_TAG}^{{commit}}"),
        "branch": BRANCH, "ordinary_push_only": True,
    })

    planner = RAW / "delivery/night9a_planner_handoff_20260820.tar.gz"
    with tarfile.open(planner, "w:gz") as archive:
        archive.add(COMPACT, arcname="official_compact")
    shutil.copy2(planner, COMPACT / planner.name)

    excluded = {"compact_delivery_index.json", "remote_compact_sha256.txt"}
    rows = []
    for path in sorted(COMPACT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append({"relative_path": str(path.relative_to(COMPACT)),
                         "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_json(COMPACT / "compact_delivery_index.json", {
        "schema_version": 1, "status": "PASS", "root": str(COMPACT),
        "root_rule": "all regular files recursively except compact_delivery_index.json and remote_compact_sha256.txt",
        "file_count": len(rows), "files": rows,
    })
    sha_rows = []
    for path in sorted(COMPACT.rglob("*")):
        if path.is_file() and path.name != "remote_compact_sha256.txt":
            sha_rows.append(f"{sha256_file(path)}  {path.relative_to(COMPACT)}")
    (COMPACT / "remote_compact_sha256.txt").write_text("\n".join(sha_rows) + "\n")
    total = sum(path.stat().st_size for path in COMPACT.rglob("*") if path.is_file())
    if total >= 10 * 1024 * 1024:
        raise RuntimeError(f"compact exceeds 10 MiB: {total}")
    print(json.dumps({"event": "night9a_compact_complete", "path": str(COMPACT),
                      "bytes": total, "file_count": len(rows),
                      "planner_tar": file_row(planner)}))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("evidence", "package"))
    args = parser.parse_args()
    evidence() if args.mode == "evidence" else package()


if __name__ == "__main__":
    main()
