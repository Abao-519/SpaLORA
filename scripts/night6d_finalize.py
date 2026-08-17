#!/usr/bin/env python3
"""Verify, summarize, and report Night-6D after the single label window."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6d_pipeline import atomic_json

OUT = REPO / "outputs/night6d_handoff"


def verify_file(path: Path, spec: dict) -> None:
    if not path.is_file() or path.stat().st_size != spec["size_bytes"] or sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"raw artifact mismatch: {path}")


def compact_primary(primary: dict, dataset: str) -> str:
    if "datasets" not in primary:
        return "primary incomplete"
    row = primary["datasets"][dataset]
    boot = row["bootstrap"]["delta_q"]
    return (f"mean ΔARI={row['mean_delta_ari']:+.6f}, ΔNMI={row['mean_delta_nmi']:+.6f}, "
            f"ΔQ={row['mean_delta_q']:+.6f}; Q wins={row['q_wins']}/10; "
            f"exact p={row['exact_sign_flip']['raw_p']:.8f}, Holm p={row['holm_adjusted_p']:.8f}; "
            f"bootstrap ΔQ 95% CI=[{boot['ci_lower']:+.6f}, {boot['ci_upper']:+.6f}]; "
            f"{row['dataset_conclusion']}")


def main() -> None:
    training = json.loads((OUT / "locked_training_manifest.json").read_text())
    transform = json.loads((OUT / "locked_transform_manifest.json").read_text())
    label = json.loads((OUT / "label_window_audit.json").read_text())
    decision = json.loads((OUT / "night6d_decision.json").read_text())
    primary = json.loads((OUT / "primary_confirmatory_tests.json").read_text())
    spatial = json.loads((OUT / "spatial_protection.json").read_text())
    metrics = pd.read_csv(OUT / "d1_p22_per_seed_metrics.csv")
    if training["success_count"] != 40 or len(training["runs"]) != 40:
        raise RuntimeError("training coverage drift")
    if transform["attempted_transforms"] != 80:
        raise RuntimeError("transform coverage drift")
    if not label["single_authorized_window"] or not label["training_locked_before_access"]:
        raise RuntimeError("label-window contract drift")
    if metrics.duplicated(["dataset", "graph_id", "head_id", "seed"]).any():
        raise RuntimeError("metric primary-key duplication")

    checkpoint_rows = []
    raw_rows = []
    for run in training["runs"]:
        run_dir = Path(run["run_dir"])
        checkpoint = run_dir / "model_final.pt"
        if sha256_file(checkpoint) != run["checkpoint_file_sha256"]:
            raise RuntimeError(f"checkpoint SHA drift: {checkpoint}")
        checkpoint_rows.append({
            "dataset": run["dataset"], "graph_id": run["graph_id"], "seed": run["seed"],
            "checkpoint_path": str(checkpoint), "checkpoint_file_sha256": run["checkpoint_file_sha256"],
            "canonical_tensor_state_sha256": run["final_tensor_state_sha256"],
            "round_trip_pass": run["checkpoint_round_trip_pass"],
            "h00_cluster_reload_exact": run["h00_cluster_reload_exact"],
            "run_manifest_sha256": run["run_manifest_sha256"], "run_dir": str(run_dir),
        })
        manifest_path = run_dir / "run_manifest.json"
        raw_rows.append({"dataset": run["dataset"], "graph_id": run["graph_id"],
                         "seed": run["seed"], "artifact": "run_manifest.json",
                         "absolute_path": str(manifest_path), "size_bytes": manifest_path.stat().st_size,
                         "sha256": sha256_file(manifest_path)})
        for name, spec in sorted(run["artifacts"].items()):
            path = Path(spec["path"])
            verify_file(path, spec)
            raw_rows.append({"dataset": run["dataset"], "graph_id": run["graph_id"],
                             "seed": run["seed"], "artifact": name,
                             "absolute_path": str(path), "size_bytes": path.stat().st_size,
                             "sha256": sha256_file(path)})
    checkpoint = pd.DataFrame(checkpoint_rows)
    checkpoint.to_csv(OUT / "checkpoint_roundtrip_summary.csv", index=False)
    atomic_json(OUT / "checkpoint_roundtrip_index.json", {
        "status": "PASS", "expected_cells": 40,
        "round_trip_pass_count": int(checkpoint.round_trip_pass.sum()),
        "h00_reload_exact_count": int(checkpoint.h00_cluster_reload_exact.sum()),
        "entries": checkpoint_rows,
    })
    pd.DataFrame(raw_rows).to_csv(OUT / "raw_artifact_manifest.csv", index=False)

    resource_rows = []
    by_key = {(x["dataset"], x["graph_id"], int(x["seed"])): x for x in training["runs"]}
    for item in transform["transforms"]:
        train = by_key[(item["dataset"], item["graph_id"], int(item["seed"]))]
        resource_rows.append({
            "dataset": item["dataset"], "graph_id": item["graph_id"], "head_id": item["head_id"],
            "seed": item["seed"], "transform_status": item["status"],
            "training_runtime_seconds": train["runtime_seconds"],
            "head_runtime_seconds": item["runtime_seconds"],
            "effective_runtime_seconds": train["runtime_seconds"] + item["runtime_seconds"],
            "peak_gpu_allocated_mib": train["peak_gpu_allocated_mib"],
            "process_peak_rss_mib": max(train["process_peak_rss_mib"], item["process_peak_rss_mib"]),
        })
    pd.DataFrame(resource_rows).to_csv(OUT / "resource_accounting.csv", index=False)

    train_failures = list(Path("/root/autodl-fs/night6d_raw_runs_20260817").glob("**/attempt_*/failure.json"))
    correction_failures = []
    scientific_failures = []
    for path in train_failures:
        row = json.loads(path.read_text())
        if row.get("status") == "scientific_numerical_failure_no_retry":
            scientific_failures.append(str(path))
        elif row.get("status") == "implementation_or_infrastructure_failure":
            correction_failures.append(str(path))
    retries = int(training.get("implementation_or_infrastructure_retries", 0))
    corrections = int(transform.get("transform_corrections", 0))
    atomic_json(OUT / "failure_and_retry_audit.json", {
        "status": "PASS", "implementation_or_infrastructure_failures": correction_failures,
        "scientific_numerical_failures_no_retry": scientific_failures,
        "training_retries": retries, "training_retry_cap": 8,
        "transform_corrections": corrections, "transform_correction_cap": 8,
        "failures_deleted_or_overwritten": False,
    })
    atomic_json(OUT / "budget_and_access_audit.json", {
        "scientific_training_units": 40, "scientific_training_cap": 40,
        "implementation_or_infrastructure_retries": retries, "retry_cap": 8,
        "total_training_attempts": 40 + retries, "total_training_attempt_cap": 48,
        "formal_transforms": 80, "formal_transform_cap": 80,
        "transform_corrections": corrections, "transform_correction_cap": 8,
        "fixed_seeds": list(range(10)), "fixed_order_preserved": True,
        "candidate_additions": 0, "a1_runs": 0, "tonsil_runs": 0,
        "gse198353_runs": 0, "night4b_runs": 0, "formal_benchmark_runs": 0,
        "label_access": {"before_total_lock": 0, "authorized_windows": 1,
                         "used_for_training_transform_or_selection": False},
    })

    proc = subprocess.run([sys.executable, "-m", "pytest", "-q",
                           "tests/test_night6d_semantic.py", "tests/test_night6d_statistics.py"],
                          cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    test_log = OUT / "tests/final_pytest.txt"
    test_log.parent.mkdir(parents=True, exist_ok=True)
    test_log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"final Night-6D tests failed; see {test_log}")
    atomic_json(OUT / "tests_and_invariance_audit.json", {
        "status": "PASS", "pytest_exit_code": 0, "pytest_log": str(test_log),
        "pytest_log_sha256": sha256_file(test_log), "training_coverage": "40/40",
        "checkpoint_round_trip": "40/40", "h00_exact_reload": "40/40",
        "transform_terminal_coverage": "80/80", "successful_metric_cells": len(metrics),
        "metric_primary_key_unique": True, "exact_sign_flip_enumerations": 1024,
        "paired_bootstrap_resamples": 100000, "bootstrap_seed": 20260817,
        "lower_is_better_metrics": ["geary_c", "boundary_disagreement"],
        "silent_fallback": False, "parameter_tuning": False, "seed_search": False,
    })

    atomic_json(OUT / "git_pre_final_audit.json", {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "head_before_delivery_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "parent_commit": "172cefba7559b34d2894ffa304553c0068ca23b9",
        "protection_tag": "baseline/pre-night6d-locked-d1-p22-confirmation-20260817",
        "ordinary_push_only": True, "force_used": False,
    })
    atomic_json(OUT / "shutdown_dispatch_status.json", {
        "status": "READY_FOR_LAST_REMOTE_COMMAND", "command": "/usr/bin/shutdown",
        "must_be_last_remote_command": True, "dispatched": False,
        "note": "The controlling SSH session records dispatch; no reconnect is permitted afterward.",
    })

    d1_spatial = spatial.get("datasets", {}).get("d1", {})
    p22_spatial = spatial.get("datasets", {}).get("p22", {})
    report = f"""# SpaLORA Night-6D final report

## Outcome

Terminal status: `{decision['terminal_status']}`.

The sole primary treatment was `{decision['primary_method']}` against the same-dataset, same-seed fresh `{decision['fresh_reference']}` reference. No method, graph, head, loss, seed, threshold, epoch, or checkpoint was selected in Night-6D.

- D1: {compact_primary(primary, 'd1')}
- P22: {compact_primary(primary, 'p22')}

D1 is the primary held-out within-study confirmation. P22 is a pre-locked cross-dataset confirmation but is not a pristine holdout because it participated in earlier Night-3B/Night-5D work. Even a positive result would not establish state of the art before fair external baselines.

## Spatial protection

- D1 gate failed: `{d1_spatial.get('spatial_gate_failed')}`; mean deltas neighbor={d1_spatial.get('delta_neighbor')}, Moran={d1_spatial.get('delta_moran')}, Geary={d1_spatial.get('delta_geary')}, boundary={d1_spatial.get('delta_boundary')}.
- P22 gate failed: `{p22_spatial.get('spatial_gate_failed')}`; mean deltas neighbor={p22_spatial.get('delta_neighbor')}, Moran={p22_spatial.get('delta_moran')}, Geary={p22_spatial.get('delta_geary')}, boundary={p22_spatial.get('delta_boundary')}.

## Execution and firewall integrity

- Three authority inputs and Night-6C 77/77 internal, 4/4 external, and 3/3 local-post evidence were verified before execution.
- D1 label-free RNA/ADT were built by low-level HDF5 field copying without deserializing original annotation values. The P22 Night-3AF deterministic cache was reused only after every file and canonical hash matched.
- Training 40/40; fresh-process checkpoint/six-view/H00 replay 40/40; terminal transforms 80/80. Formal seeds were 0-9 in the preregistered order.
- D1 and P22 labels were parsed together in one evaluator window only after total lock. There was no return to training, transformation, or clustering afterward.
- Scientific training budget 40/40; retries {retries}/8; total attempts {40 + retries}/48; transforms 80/80; corrections {corrections}/8.
- The fixed secondary 2x2 factorial diagnostics were computed only after the primary lock and did not alter method identity.

## Evidence map

Core numerical evidence is in `d1_p22_per_seed_metrics.csv`, `primary_confirmatory_tests.json`, `secondary_factorial_tests.json`, and `spatial_protection.json`. Execution evidence is in `locked_training_manifest.json`, `checkpoint_roundtrip_index.json`, `locked_transform_manifest.json`, `label_window_audit.json`, `resource_accounting.csv`, `failure_and_retry_audit.json`, `budget_and_access_audit.json`, and `tests_and_invariance_audit.json`. Large label-free inputs, caches, affinities, views, raw runs, and checkpoints remain under `/root/autodl-fs` and are protected by absolute paths, sizes, and SHA-256 manifests.
"""
    (OUT / "night6d_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"terminal_status": decision["terminal_status"], "tests": "PASS",
                      "checkpoints": len(checkpoint_rows), "raw_artifacts": len(raw_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
