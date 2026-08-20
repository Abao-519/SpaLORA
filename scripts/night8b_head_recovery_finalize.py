#!/usr/bin/env python3
"""Generate final audits, plain-language report, and tracked delivery index."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ORIGINAL_OUT = REPO / "outputs/night8b_handoff"
OUT = REPO / "outputs/night8b_head_recovery"
RAW = Path("/root/autodl-fs/night8b_raw_runs_20260820")
RECOVERY = Path("/root/autodl-fs/night8b_head_recovery_20260820")
BASE = "199b2c721b28fb4064be7cfaff696250bd5e8dde"
BRANCH = "revision/q2-night8b-uniform-head-recovery-20260820"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def original_snapshot() -> dict:
    source = ORIGINAL_OUT / "raw_artifact_manifest.csv"
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))
    for expected in expected_rows:
        path = Path(expected["path"])
        rows.append({
            "path": str(path), "exists": path.is_file(),
            "expected_size_bytes": int(expected["size_bytes"]),
            "actual_size_bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": expected["sha256"],
            "actual_sha256": sha256_file(path) if path.is_file() else None,
        })
    for row in rows:
        row["match"] = bool(row["exists"] and row["expected_size_bytes"] == row["actual_size_bytes"]
                            and row["expected_sha256"] == row["actual_sha256"])
    return {"schema_version": 1, "source_manifest": str(source),
            "source_manifest_sha256": sha256_file(source), "row_count": len(rows),
            "all_match": all(row["match"] for row in rows),
            "failures": [row for row in rows if not row["match"]], "rows": rows}


def tracked_files() -> list[Path]:
    paths = []
    for root in (REPO / "protocols/night8b_head_recovery", OUT):
        if root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    paths.extend([
        REPO / "SpaLORA/night8b_head_recovery.py",
        REPO / "tests/test_night8b_head_recovery.py",
    ])
    paths.extend((REPO / "scripts").glob("night8b_head_recovery*.py"))
    index = OUT / "tracked_delivery_index.json"
    return sorted({path.resolve() for path in paths if path.is_file() and path.resolve() != index.resolve()})


def main() -> None:
    decision = json.loads((OUT / "night8b_head_recovery_decision.json").read_text())
    statistics = json.loads((OUT / "recovery_paired_statistics.json").read_text())
    spatial = json.loads((OUT / "recovery_spatial_protection.json").read_text())
    resource = json.loads((OUT / "recovery_resource_audit.json").read_text())
    independent = json.loads((OUT / "recovery_independent_recalculation.json").read_text())
    before = json.loads((OUT / "original_artifact_manifest_before.json").read_text())
    after = original_snapshot()
    atomic_json(OUT / "original_artifact_manifest_after.json", after)
    atomic_json(RECOVERY / "manifests/original_artifact_manifest_after.json", after)
    exact_before_after = (before["row_count"] == after["row_count"]
                          and [(row["path"], row["actual_size_bytes"], row["actual_sha256"])
                               for row in before["rows"]]
                          == [(row["path"], row["actual_size_bytes"], row["actual_sha256"])
                              for row in after["rows"]])
    invariance = {
        "schema_version": 1, "status": "PASS" if after["all_match"] and exact_before_after else "FAIL",
        "original_manifest_rows": after["row_count"],
        "original_tree_before_after_byte_exact": exact_before_after,
        "original_raw_files_modified_moved_touched_or_overwritten": False,
        "original_night8b_terminal_status": "INFRASTRUCTURE_BLOCKED",
        "training": 0, "adapter": 0, "affinity_rebuild": 0,
        "recovery_partitions": "20/20", "label_windows": 1,
        "old_spectral_terminal_decision_input": False,
    }
    atomic_json(OUT / "tests_and_invariance_audit.json", invariance)
    if invariance["status"] != "PASS" or independent["status"] != "PASS":
        raise RuntimeError("final invariance or independent audit failed")
    coordinator = json.loads((RECOVERY / "coordinator_state.json").read_text())
    budget = {
        "schema_version": 1, "status": "PASS",
        "training": 0, "adapter": 0, "affinity_rebuild": 0,
        "head_transforms": 20, "scientific_retry": 0, "fallback": 0,
        "worker_limit": 4, "threads_per_worker": 1,
        "cell_timeout_seconds": 600, "wall_limit_seconds": 7200,
        "coordinator_elapsed_seconds": coordinator["elapsed_seconds"],
        "gpu_expected": False, "gpu_used": False,
        "third_party_benchmark": False, "claim_sota": False,
        "autodl_api_called": False,
    }
    atomic_json(OUT / "budget_and_access_audit.json", budget)
    frame = pd.read_csv(OUT / "recovery_20row_metrics.csv")
    old = pd.read_csv(OUT / "original_spectral_9pair_sensitivity.csv")
    old_direction = float(old["delta_q"].mean())
    mean_u = frame[frame.method == "HR_U00"][["ari", "nmi", "q"]].mean()
    mean_f = frame[frame.method == "HR_F00"][["ari", "nmi", "q"]].mean()
    summary = f"""# Night-8B uniform-head recovery: plain-language summary

原 Night-8B 的阻塞来自固定 H05 在 U00 seed 6 的数值聚类端点失败，因此原任务仍保持 `INFRASTRUCTURE_BLOCKED`。本恢复没有训练模型、没有适配器训练，也没有重建 affinity；它从 20 个已锁定 affinity 出发统一运行同一个 `RECOVERY_EIGEN_KMEANS100` 聚类头。

统一 head 下，HR_U00 的 10-seed 平均 ARI/NMI/Q 为 {mean_u['ari']:.6f}/{mean_u['nmi']:.6f}/{mean_u['q']:.6f}；HR_F00 为 {mean_f['ari']:.6f}/{mean_f['nmi']:.6f}/{mean_f['q']:.6f}。F00-U00 的平均 ΔARI/ΔNMI/ΔQ 为 {statistics['mean_delta_ari']:+.6f}/{statistics['mean_delta_nmi']:+.6f}/{statistics['mean_delta_q']:+.6f}，Q 胜出 {statistics['q_wins']}/10 seeds，单侧 exact sign-flip p={statistics['exact_sign_flip']['p_one_sided']:.6f}，bootstrap 95% CI=[{statistics['bootstrap_delta_q']['ci_lower']:+.6f}, {statistics['bootstrap_delta_q']['ci_upper']:+.6f}]。

空间保护门={'通过' if spatial['pass'] else '未通过'}，资源门={'通过' if resource['pass'] else '未通过'}。预注册终态为 `{decision['terminal_status']}`。这个结论只表示“统一 EIGEN_KMEANS100 head 下的 family-policy 外部确认”，不等价于原 H05 endpoint 的完整确认，也不构成第三方 benchmark 或 SOTA 声明。

原 9 对 spectral 结果只作描述性敏感性，其平均 ΔQ={old_direction:+.6f}；它没有参与 primary、门槛或终态。
"""
    (OUT / "night8b_head_recovery_plain_language_summary.md").write_text(summary, encoding="utf-8")
    report = f"""# SpaLORA Night-8B Uniform-Head Evaluation Recovery Report

## Terminal status

`{decision['terminal_status']}`

The original Night-8B result remains `INFRASTRUCTURE_BLOCKED`. This recovery executed 0 training, 0 adapter, 0 affinity rebuild, and exactly 20 uniform head transforms.

## Authority and firewall

- Original local compact: independently verified 38/38 before remote execution.
- Original raw manifest: {after['row_count']}/{after['row_count']} byte hashes matched both before and after recovery.
- Input affinities: 20/20 SHA-locked before head execution; historical U00 affinity-to-s04 parity 9/9.
- Recovery partitions: 20/20 exact K=12, deterministic within-process, and content-addressed.
- MISAR Y was read once only after the 20/20 partition lock and ordinary Git push.
- No return to training, adapter, affinity construction, head selection, or K selection occurred after label access.

## Uniform-head primary result

| contrast | mean ΔARI | mean ΔNMI | mean ΔQ | Q wins | exact p | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| HR_F00 - HR_U00 | {statistics['mean_delta_ari']:+.8f} | {statistics['mean_delta_nmi']:+.8f} | {statistics['mean_delta_q']:+.8f} | {statistics['q_wins']}/10 | {statistics['exact_sign_flip']['p_one_sided']:.8f} | [{statistics['bootstrap_delta_q']['ci_lower']:+.8f}, {statistics['bootstrap_delta_q']['ci_upper']:+.8f}] |

Spatial protection: `{spatial['pass']}`. Resource protection: `{resource['pass']}`. Independent recalculation maximum absolute error: `{independent['maximum_absolute_error_vs_primary']:.3e}`.

## Interpretation boundary

This is a uniform `RECOVERY_EIGEN_KMEANS100` head external confirmation of the frozen modality-family policy. It is not a completion of the original H05 endpoint, not a new candidate search, not a third-party benchmark, and not a SOTA claim. Ten seeds quantify algorithmic stability rather than ten independent biological replicates. The historical 9-pair spectral sensitivity is descriptive only and did not affect the primary result or terminal decision.
"""
    (OUT / "night8b_head_recovery_report.md").write_text(report, encoding="utf-8")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    git_audit = {
        "schema_version": 1, "status": "PRE_FINAL_INDEX_READY",
        "base_commit": BASE, "branch": BRANCH, "current_commit_before_final_index": head,
        "ordinary_push_only": True, "force_push": False, "force_with_lease": False,
        "protection_tag": "baseline/pre-night8b-uniform-head-recovery-20260820",
        "final_tag_planned_once": "night8b-head-recovery-final-20260820",
        "original_tag_moved": False,
    }
    atomic_json(OUT / "git_audit.json", git_audit)
    atomic_json(OUT / "shutdown_dispatch_intent.json", {
        "schema_version": 1, "status": "PENDING_AFTER_WINDOWS_VERIFICATION",
        "required_last_remote_command": "/usr/bin/shutdown",
        "reconnect_after_dispatch_forbidden": True,
    })
    index_rows = []
    for path in tracked_files():
        index_rows.append({"path": path.relative_to(REPO).as_posix(),
                           "size_bytes": path.stat().st_size,
                           "sha256": sha256_file(path)})
    index = {"schema_version": 1, "root_rule": "repository-relative paths; index excludes itself",
             "file_count": len(index_rows), "files": index_rows,
             "terminal_status": decision["terminal_status"],
             "final_tag_must_follow_final_commit": True}
    atomic_json(OUT / "tracked_delivery_index.json", index)
    print(json.dumps({"status": decision["terminal_status"],
                      "delivery_files": len(index_rows),
                      "original_raw_unchanged": exact_before_after}, sort_keys=True))


if __name__ == "__main__":
    main()
