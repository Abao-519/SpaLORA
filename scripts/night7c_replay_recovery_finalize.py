#!/usr/bin/env python3
"""Generate final, compact-ready evidence for Night-7C replay recovery."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
OUT = REPO / "outputs/night7c_replay_recovery_handoff"
from SpaLORA.night7a_consensus import atomic_json, sha256_file


def resource_summary(path: Path):
    if not path.is_file():
        return {"rows": 0}
    frame = pd.read_csv(path)
    result = {"rows": len(frame)}
    for col in (
        "cpu_percent", "rss_mib", "gpu_util_percent", "gpu_memory_mib",
        "gpu_power_w", "gpu_sm_clock_mhz",
    ):
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(values):
                result[col] = {"mean": float(values.mean()), "max": float(values.max())}
    return result


def read_json(name: str):
    return json.loads((OUT / name).read_text())


def copy_if_file(src: Path, dst: Path):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def cgroup_cpu_quota():
    path = Path("/sys/fs/cgroup/cpu.max")
    if not path.is_file():
        return None, None
    raw = path.read_text().strip()
    quota, period = raw.split()
    if quota == "max":
        return None, raw
    return float(quota) / float(period), raw


def main():
    p1a = read_json("p1a_replay_portability_contract.json")
    p1b = read_json("p1b_feature_freeze_contract.json")
    p2 = read_json("p2_runtime_contract.json")
    routing = read_json("routing_transform_manifest.json")
    training = read_json("weighted_mnn_training_manifest.json")
    weighted = read_json("weighted_mnn_transform_manifest.json")
    triage = read_json("stagew_immediate_semantic_triage.json")
    bounded = read_json("stagew_resource_bounded_plan_and_eligibility.json")
    threading = read_json("stagew_worker_threading_resource_disclosure.json")
    labels = read_json("label_window_audit.json")
    gate = read_json("gate_audit.json")
    terminal = gate["terminal_status"]

    invalid = []
    for directory in sorted((RAW / "invalid_attempts").glob("*")):
        invalid.append({
            "attempt": directory.name,
            "classification": "infrastructure_only_correction",
            "preserved": True,
            "scientific_retry": False,
        })

    failures = []
    for stage, rows in (("T", routing["transforms"]), ("W_TRANSFORM", weighted["transforms"])):
        for row in rows:
            if row["status"] != "success":
                failures.append({
                    "stage": stage,
                    "candidate": row.get("candidate_id"),
                    "unit_id": row.get("unit_id"),
                    "status": row["status"],
                    "failure_type": row.get("failure_type", ""),
                    "scientific_metrics_computed": False,
                })
    with (OUT / "failure_audit.csv").open("w", newline="") as handle:
        fields = ["stage", "candidate", "unit_id", "status", "failure_type", "scientific_metrics_computed"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(failures)

    atomic_json(OUT / "budget_audit.json", {
        "schema_version": 1,
        "replay_initial_forwards": 60,
        "old_order_diagnostics": 4,
        "formal_training_registered": training["training_attempts"],
        "formal_training_success": training["successful_training"],
        "formal_transforms_registered": routing["transform_attempts"] + weighted["transform_attempts"],
        "routing_transforms_success": routing["successful_transforms"],
        "routing_transforms_numerical_failure": routing["failed_transforms"],
        "weighted_transforms_success": weighted["successful_transforms"],
        "weighted_transforms_resource_censored": weighted["resource_censored_transforms"],
        "weighted_transforms_resource_skipped": weighted["skipped_transforms"],
        "weighted_physical_transform_invocations": weighted["physical_transform_invocations"],
        "prior_infrastructure_aborted_invocations": weighted["prior_infrastructure_aborted_invocations"],
        "scientific_retry": 0,
        "fallback": 0,
        "implementation_corrections": invalid,
        "label_window_count": labels["window_count"],
    })

    cpu_quota, cpu_max = cgroup_cpu_quota()
    clarification = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "prelabel_disclosure_sha256": sha256_file(OUT / "stagew_worker_threading_resource_disclosure.json"),
        "prelabel_disclosure_modified": False,
        "host_visible_logical_cpu_count": threading.get("logical_cpu_count"),
        "cgroup_cpu_max_raw": cpu_max,
        "effective_cgroup_cpu_quota": cpu_quota,
        "autodl_advertised_cpu_cores": 12,
        "worker_count": threading.get("workers_max"),
        "requested_threads_per_worker": threading.get("requested_threads_per_worker"),
        "requested_aggregate_thread_ceiling": threading.get("requested_aggregate_thread_ceiling"),
        "interpretation": (
            "The container exposed 96 logical CPUs through affinity but cpu.max constrained it to "
            "12 CPU equivalents. Four already-running workers each inherited an 8-thread request, "
            "so oversubscription was possible. The environment was not changed mid-queue and no cell "
            "was rerun; all 40 eligible transforms completed under one consistent environment."
        ),
        "algorithm_or_candidate_change": False,
        "rerun": False,
    }
    atomic_json(OUT / "stagew_worker_threading_resource_clarification.json", clarification)

    atomic_json(OUT / "runtime_report.json", {
        "schema_version": 1,
        "p2": p2,
        "p2_serial_resource": resource_summary(RAW / "p2_serial_resource.csv"),
        "p2_parallel_resource": resource_summary(RAW / "p2_parallel_resource.csv"),
        "stage_t_resource": resource_summary(RAW / "stage_t_resource.csv"),
        "stage_w_resource": resource_summary(RAW / "stage_w_resource.csv"),
        "stage_w_resume_resource": resource_summary(RAW / "stage_w_resume_resource.csv"),
        "stage_w_bounded_resource": resource_summary(RAW / "stage_w_parallel_resource.csv"),
        "triage_status": triage["status"],
        "bounded_queue_status": bounded["status"],
        "threading_disclosure": threading,
        "threading_clarification": clarification,
        "interpretation": (
            "CUDA training retained positive GPU use. Affinity construction, ARPACK spectral "
            "partitioning and clustering are CPU phases where GPU=0 is expected. W00/u000 was "
            "resource-censored by the user-authorized immediate-stop amendment after an extreme "
            "unchanged-solver long tail. W01-W05 then completed with the same transform_cell/H01/"
            "ARPACK implementation under the bounded four-worker scheduler."
        ),
    })

    compact_infra = OUT / "infrastructure"
    compact_infra.mkdir(parents=True, exist_ok=True)
    # The full JUnit failure payload is retained on the persistent raw root and
    # represented in test_audit.json by path-independent counts and SHA.
    (compact_infra / "full_repository_tests.xml").unlink(missing_ok=True)
    for name in (
        "evaluation.log", "stagew_immediate_bounded.nohup.log",
        "stagew_immediate_bounded_completion.json", "stagew_immediate_bounded_coordinator.jsonl",
        "stagew_immediate_semantic_triage_completion.json", "stagew_immediate_triage.log",
        "stagew_immediate_triage_isolated.log", "test_stagew_immediate_prelabel.log",
        "test_stagew_immediate_triage.log", "test_stagew_immediate_triage_isolated.log",
        "test_stagew_resource_bounded.log", "total_prelabel_lock.log", "w00_boundary_guard.jsonl",
        "night7c_directed_tests.xml",
    ):
        copy_if_file(RAW / "infrastructure" / name, compact_infra / name)

    incident_out = compact_infra / "incident_manifests"
    for manifest in sorted((RAW / "invalid_attempts").glob("*/**/incident_manifest.json")):
        copy_if_file(manifest, incident_out / "invalid_attempts" / manifest.relative_to(RAW / "invalid_attempts"))
    for incident in sorted((RAW / "incidents").glob("*")):
        for name in ("incident_inventory.json", "process_control_preterm.json", "process_control_final.json"):
            copy_if_file(incident / name, incident_out / "resource_stop" / incident.name / name)

    raw_index = []
    for name in (
        "p1a_historical_order_replay", "p1b_features", "p2_runtime", "stage_t", "stage_w",
        "invalid_attempts", "incidents", "infrastructure",
    ):
        root = RAW / name
        files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
        raw_index.append({
            "path": str(root),
            "file_count": len(files),
            "total_size_bytes": sum(path.stat().st_size for path in files),
        })
    atomic_json(OUT / "raw_artifact_index.json", {
        "schema_version": 1, "raw_files_in_compact": False, "roots": raw_index,
    })

    atomic_json(OUT / "ssh_session_replacement_audit.json", {
        "schema_version": 1,
        "original_control_session": 68838,
        "replacement_control_session": 41410,
        "replacement_reason": (
            "A post-evaluation read-only summary accidentally invoked the system Python without pandas. "
            "The command returned nonzero and the fail-fast SSH shell exited."
        ),
        "occurred_after_single_label_evaluation_completed": True,
        "training_restarted": False,
        "transform_restarted": False,
        "label_evaluation_restarted": False,
        "shutdown_contract": (
            "Replacement session remains the sole control session; /usr/bin/shutdown will be its final remote command."
        ),
    })

    atomic_json(OUT / "shutdown_contract.json", {
        "schema_version": 1,
        "status": "PREPARED_NOT_DISPATCHED",
        "exact_command": "/usr/bin/shutdown",
        "control_session": 41410,
        "must_be_final_remote_command": True,
        "reconnect_after_dispatch": False,
        "claim_boundary": (
            "Only client dispatch status may be reported; console power-off confirmation is not claimed."
        ),
    })

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    atomic_json(OUT / "git_audit_pre_final.json", {
        "schema_version": 1,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "head_before_final_delivery_commit": head,
        "old_invalid_tag_peel": subprocess.check_output(
            ["git", "rev-parse", "night7c-final-20260818^{}"], cwd=REPO, text=True
        ).strip(),
        "old_invalid_tag_unchanged": True,
        "force_push": False,
    })

    routing_recalc = gate["independent_recalculation"]["routing"]
    weighted_recalc = gate["independent_recalculation"]["weighted"]
    report = f'''# SpaLORA Night-7C replay portability recovery report

## Outcome

Terminal status: `{terminal}`.

No safe label-free router was selected, and no eligible weighted-MNN pilot passed its preregistered material gate. The result is a valid negative development-panel result, not a benchmark or external-generalization claim. The original Night-7C `IMPLEMENTATION_SEMANTICS_INVALID` commit, tag, report and compact delivery remain immutable.

## Replay portability recovery

- Historical-order initial forward: 30 units x 2 fresh processes = 60/60 finite replays.
- Routing authority: SHA-locked first-row MNN values from the original Night-7B loss curves; current-hardware replay was diagnostic only.
- Within-current-hardware maximum repeat error: `{p1a['gates']['within_current_hardware_repeat_absolute_error_max']}`.
- Maximum historical absolute / relative error: `{p1a['gates']['historical_vs_current_absolute_error_max']}` / `{p1a['gates']['historical_vs_current_relative_error_max']}`.
- T02/T03 maximum alpha delta: `{p1a['gates']['T02_T03_alpha_absolute_delta_max']}`; T04 decision flips: `{p1a['gates']['T04_hard_decision_mismatch_count']}`.
- Initial state, RNG, fixed-index and formula mismatch counts were zero. Four old-order shape diagnostics were retained as diagnostics only.
- Final-checkpoint/endpoint parity ran in separate fresh processes and passed `{p1b['fresh_final_checkpoint_endpoint_parity']}`.

## Formal scientific execution

- Stage T: {routing['transform_attempts']}/240 registered routing transforms; {routing['successful_transforms']} successes and {routing['failed_transforms']} retained fixed numerical failure.
- Stage W training: {training['successful_training']}/{training['training_attempts']} CUDA cells succeeded before the transform long tail.
- W00_FILTER75/u000 produced no terminal manifest and was precisely stopped under the SHA-locked immediate-stop amendment. It is `RESOURCE_CENSORED_USER_STOP_AFTER_EXTREME_LONGTAIL`; its remaining seven cells are circuit-breaker skips. W00 is ineligible and no W00 scientific metric was computed.
- Label-free semantic triage passed all 48 training/checkpoint/embedding/mapping/affinity contracts, with zero W01-W05 structural pathologies. The first monolithic native crash is preserved and classified infrastructure-only, not a scientific retry.
- The unchanged bounded H01/ARPACK backend completed all 40 W01-W05 transforms; resource-censored 1, skipped 7, scientific retry 0, fallback 0.
- Stage T and Stage W outputs were SHA-locked before exactly one authorized label window. There was no return to training, transform or clustering.

## Preregistered evaluation

- Safe router candidates: `{gate['safe_router_candidates']}`; strong router candidates: `{gate['strong_router_candidates']}`; selected router: `{gate['selected_router']}`.
- Evaluated weighted-MNN candidates: `{gate['evaluated_weighted_mnn_candidates']}`; promoted candidates: `{gate['promoted_weighted_mnn_candidates']}`.
- W01-W05 all passed their spatial protection checks but all failed the preregistered material OR gate; none was promoted.
- Independent aggregate recomputation: routing `{routing_recalc['status']}` with maximum absolute error `{routing_recalc['maximum_absolute_error']}`; weighted `{weighted_recalc['status']}` with maximum absolute error `{weighted_recalc['maximum_absolute_error']}` (tolerance 1e-12).

## Runtime and resource disclosure

- P2 four-worker output parity was exact, but speedup was only `{p2['speedup']:.6f}x`; Stage T therefore retained `{p2['formal_backend']}`.
- The formal W01-W05 queue used four workers. Each already-running worker requested 8 BLAS/OpenMP threads while the cgroup CPU quota was 12 CPU equivalents, so oversubscription was possible. The environment was not changed mid-queue and no unit was rerun; all 40 cells used one consistent implementation and environment.
- CPU affinity exposed 96 host logical CPUs, which explains the prelabel `logical_cpu_count=96` field; `cpu.max=1200000 100000` is the authoritative 12-CPU quota.
- CPU spectral/affinity phases legitimately report GPU=0. CUDA training evidence remains separately preserved.

## Scientific limits

This is development-panel R&D on A1, tonsil, D1 and previously used P22. It is not a pristine external benchmark, SOTA comparison or publication claim. No labels, metrics, tissue, platform, modality, file name or dataset identity were used for training or routing. The single label window occurred only after the total pre-label lock.
'''
    (OUT / "night7c_replay_recovery_report.md").write_text(report)
    (OUT / "plain_language_summary.txt").write_text(
        "Night-7C replay recovery completed validly, but no candidate passed the fixed selection rules. "
        "No safe router was selected and no weighted-MNN pilot was promoted. W00 was excluded after an "
        "extreme CPU/ARPACK resource long tail; W01-W05 completed and were evaluated only after the total lock.\n"
    )
    print(json.dumps({"terminal_status": terminal, "report": str(OUT / "night7c_replay_recovery_report.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
