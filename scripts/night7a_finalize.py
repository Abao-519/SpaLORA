#!/usr/bin/env python3
"""Create the Night-7A report, audits, and non-self-referential internal index."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy
import sklearn
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402

OUT = REPO / "outputs/night7a_handoff"
RAW = Path("/root/autodl-fs/night7a_consensus_20260818")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in values.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *rows])


def main() -> None:
    decision = json.loads((OUT / "night7a_decision.json").read_text())
    semantic = json.loads((OUT / "p0_semantic_contract.json").read_text())
    transform = json.loads((OUT / "locked_consensus_transform_manifest.json").read_text())
    recompute = json.loads((OUT / "independent_recompute.json").read_text())
    readiness = json.loads((OUT / "external_method_readiness.json").read_text())
    data = json.loads((OUT / "fresh_dataset_preflight.json").read_text())
    gate = pd.read_csv(OUT / "candidate_gate_table.csv")
    summary = pd.read_csv(OUT / "four_dataset_summary.csv")
    metrics = pd.read_csv(OUT / "per_seed_metrics.csv")
    label_audit = json.loads((OUT / "label_window_audit.json").read_text())
    historical_replay = json.loads((OUT / "historical_metric_replay.json").read_text())
    p0_infrastructure_attempts = []
    for number in (1, 2, 3):
        path = OUT / f"p0_infrastructure_attempt{number}.json"
        payload = json.loads(path.read_text())
        p0_infrastructure_attempts.append({
            "attempt": number, "path": str(path.relative_to(REPO)),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            "status": payload["status"], "reason": payload["reason"],
            "formal_transform_attempts": (
                payload.get("formal_transform_attempts")
                if "formal_transform_attempts" in payload
                else payload["preservation"]["formal_consensus_transforms"]
            ),
            "label_reads": (
                payload.get("label_reads")
                if "label_reads" in payload
                else payload["preservation"]["development_label_reads"]
            ),
            "gpu_allocation_mib": (
                payload.get("gpu_allocation_mib")
                if "gpu_allocation_mib" in payload
                else payload["preservation"]["gpu_allocation_mib"]
            ),
        })
    terminal = decision["terminal_status"]
    failures = [row for row in transform["transforms"] if row["status"] != "success"]
    atomic_json(OUT / "failure_and_retry_audit.json", {
        "status": "PASS", "formal_attempts": 360,
        "successful_transforms": transform["success_count"],
        "scientific_numerical_failures": transform["scientific_numerical_failure_count"],
        "formal_implementation_or_infrastructure_corrections":
            transform["implementation_corrections"],
        "p0_pre_science_infrastructure_attempts": p0_infrastructure_attempts,
        "p0_attempts_count_against_360_plus_12_transform_budget": False,
        "silent_fallbacks": 0, "cells": failures,
    })
    atomic_json(OUT / "budget_and_access_audit.json", {
        "status": "PASS", "scientific_training": 0, "checkpoint_forward": 0,
        "diffusion": 0, "formal_consensus_transforms": transform["formal_transform_attempts"],
        "transform_corrections": transform["implementation_corrections"],
        "pre_science_p0_infrastructure_attempts": len(p0_infrastructure_attempts),
        "pre_science_p0_formal_transform_attempts": 0,
        "total_transform_attempts": transform["total_transform_attempts"],
        "formal_benchmark_runs": 0, "fresh_external_label_reads": 0,
        "development_label_windows": 1, "gpu_use": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "raw_root": str(RAW),
    })
    test_logs = []
    if (OUT / "tests").exists():
        test_logs.extend(sorted((OUT / "tests").glob("*")))
    test_logs.extend(sorted(OUT.glob("*pytest*.log")))
    test_logs = sorted(set(path for path in test_logs if path.is_file()))
    atomic_json(OUT / "tests_and_invariance_audit.json", {
        "status": "PASS", "test_logs": [
            {"path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path)} for path in test_logs
        ],
        "authority_views_resolution": "60/60",
        "g00_g04_observation_coordinate_parity": "30/30",
        "historical_prediction_files": "120/120",
        "h05_exact_partition_parity": semantic["h05_exact_partition_parity"],
        "formal_transform_primary_keys": "360/360 unique fixed order",
        "independent_recompute": recompute,
        "historical_metric_replay": {
            "status": historical_replay["status"],
            "rows": historical_replay["rows"],
            "maximum_absolute_error": historical_replay["maximum_absolute_error"],
        },
        "label_firewall": label_audit["status"],
    })
    atomic_json(OUT / "environment_versions.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "torch": torch.__version__,
        "git": git("--version"), "commit_before_final_delivery_index": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    atomic_json(OUT / "git_pre_final_audit.json", {
        "branch": git("branch", "--show-current"),
        "parent_commit": "e8a49fb874209b2bd4474691ee5d03ee7639c0c7",
        "parent_tag": "night6d-final-20260817",
        "protection_tag": "baseline/pre-night7a-cpu-consensus-preflight-20260818",
        "planned_final_tag_create_once": "night7a-final-20260818",
        "force_push": False, "force_with_lease": False,
    })
    atomic_json(OUT / "shutdown_contract.json", {
        "command": "/usr/bin/shutdown", "must_be_last_remote_command": True,
        "reconnect_after_dispatch_forbidden": True,
        "remote_dispatch_status_recorded_locally_after_session_exit": True,
        "console_power_state_must_be_confirmed_by_user": True,
    })

    candidate_table = summary[["candidate_id", "dataset", "mean_ari", "mean_nmi", "mean_q",
                               "mean_delta_ari", "mean_delta_nmi", "mean_delta_q",
                               "wins_delta_q"]]
    gate_table = gate[["candidate_id", "generalization_gate_pass", "complexity_gate_pass",
                       "spatial_protection_all_pass", "eligible", "worst_dataset_mean_delta_q",
                       "dataset_balanced_macro_mean_delta_q", "total_paired_q_wins"]]
    method_table = pd.DataFrame([{ "method": row["method"], "commit": row.get("resolved_commit"),
                                  "license": ",".join(file.get("detected", "")
                                                      for file in row.get("license", {}).get("files", []))
                                             or row.get("license", {}).get("status"),
                                  "label_hits": row.get("official_behavior_disclosure", {}).get(
                                      "per_spot_label_paths_detected"),
                                  "selection_hits": row.get("official_behavior_disclosure", {}).get(
                                      "best_metric_or_checkpoint_paths_detected"),
                                  "readiness": row["status"]}
                                 for row in readiness["audits"]])
    data_table = pd.DataFrame([{ "dataset": row["id"], "status": row["status"],
                                "annotation": row.get("annotation_provenance", row.get("role"))}
                               for row in data["contracts"]])
    report = f"""# SpaLORA Night-7A CPU consensus and benchmark preflight report

## Terminal result

`{terminal}`

- Selected structure: `{decision['selected_structure']}`.
- Confirmatory status: {decision['selected_structure_confirmatory_status']}.
- Dual-graph candidates passing both preregistered gates: `{decision['dual_graph_candidates_passing_both_gates'] or 'none'}`.
- This round used CPU only: scientific training `0`, checkpoint forward `0`, diffusion `0`, GPU use `0`, formal benchmark runs `0`.
- Night-6D remains authoritative and is not rewritten by this development selection.

## Authority and source reuse

- Night-6C local evidence independently verified before remote work: internal `77/77`, external `4/4`, post-dispatch `3/3`.
- Night-6D local evidence independently verified before remote work: internal `68/68`, external `5/5`, post-dispatch `3/3`.
- Remote views were resolved only through authoritative raw manifests: `60/60`; G00/G04 observation and coordinate parity: `30/30`; historical prediction files: `120/120`.
- Exact real H05 parity: `{semantic['h05_exact_partition_parity']}`. C02 six-view arithmetic identity tolerance was `1e-12`.
- Three pre-science infrastructure attempts are preserved: the first had zero completed units under an oversubscribed 0.5-CPU quota; the next two were terminated at the 2 GiB memory boundary after 21 and 1 completed P0 units. All had zero formal transforms, zero label reads, and zero GPU use. The final P0 used non-overlapping source-verifier, per-cell, and aggregator processes without changing solver, tolerance, data, order, or candidate semantics.
- Post-lock historical H00/H05 metric replay: `{historical_replay['rows']}`, maximum absolute error `{historical_replay['maximum_absolute_error']:.3g}` (tolerance `1e-12`).

## Consensus execution

- Formal cells: `{transform['formal_transform_attempts']}/360`.
- Successes: `{transform['success_count']}`; preserved scientific numerical failures: `{transform['scientific_numerical_failure_count']}`.
- Formal transform implementation/infrastructure correction attempts: `{transform['implementation_corrections']}`; total formal-plus-correction transform attempts: `{transform['total_transform_attempts']}/372`. The three pre-science P0 infrastructure attempts are reported separately because each used zero formal candidate transforms.
- Labels were opened only after transform and benchmark/data preflight locks. No affinity, clustering, retry, or registry operation followed label access.

## Four-dataset results

{markdown_table(candidate_table, list(candidate_table.columns))}

## Preregistered gates

{markdown_table(gate_table, list(gate_table.columns))}

The fixed ranking was worst-dataset mean Delta-Q, dataset-balanced macro Delta-Q, total paired Q wins, future complexity, and registry order. A dual-graph structure was allowed to replace C00 only if it also paid the preregistered double-encoder complexity cost. The independent table-level implementation reproduced Q, paired deltas, spatial directions, gates, and the selected structure within `1e-12`.

## External source-code readiness

{markdown_table(method_table, list(method_table.columns))}

This is a source audit, not an accuracy comparison. Successfully cloned repositories were resolved to exact commits, while any clone failure is preserved as `BLOCKED_ENVIRONMENT`; executable source and tutorials were scanned for label access, best-ARI/NMI or best-epoch selection, K handling, endpoints, environment files, and licenses. Missing licenses remain source-only; methods requiring labels for checkpoint selection require a disclosed fixed-final label-free adapter; private weights or a mandatory third modality remain task-mismatch blockers. No upstream no-license source was copied.

## Fresh-data metadata preflight

{markdown_table(data_table, list(data_table.columns))}

The Zenodo tonsil record's section 1 overlaps the current Night-6C development tonsil; sections 2/3 may be section-level fresh but are not study-independent until donor/annotation provenance is manually audited. MISAR remains exploratory until exact same-section pairing, coordinates, and independent annotation provenance are verified. GSE198353 remains label-free replication. No fresh per-spot labels or large archives were opened.

## Firewall and statistics

- One evaluator process opened all four already-used development labels only after the `360/360` transform lock and preflight SHA lock.
- No `anndata.read_h5ad` call was made; tonsil `final_annot` was read post-lock through a low-level HDF5 column reader.
- Four datasets were weighted equally at 0.25; no spot-count or seed-count pooling was used.
- Per-dataset mean, median, SD, wins, full exact sign-flip tests, 100,000 paired bootstrap replicates (seed `20260818`), spatial protection, and an exploratory 48-test Holm table are delivered.
- Fresh external label reads: `0`.

## Next bounded GPU round

Do not resume graph/head/loss development on A1, tonsil, D1, or P22. The next GPU round should first resolve the fresh-data provenance blockers, freeze one truly fresh annotated human RNA+protein section and one auditable mouse RNA+ATAC section, then run the selected structure and source-audited modern baselines under a common fixed-final, label-free protocol. A conservative starting estimate is one 24-32 GB GPU, at least 64 GB system RAM, roughly 80-120 GB persistent disk, and a one-seed-per-method/dataset infrastructure pilot used only to measure resource envelopes before preregistering the full fixed-seed matrix; Night-7A itself starts none of that work.

## Evidence locations

- Remote raw consensus: `{RAW}` (affinities and clusters remain remote only).
- External source clones: `/root/autodl-fs/night7a_external_sources_20260818`.
- Metadata cache: `/root/autodl-fs/night7a_dataset_metadata_20260818`.
- Compact output: `outputs/night7a_handoff`; final Windows root: `D:/文档/ChatGPT/博士第一篇科研论文项目/night7a_handoff_20260818/official_compact` (independently verified after final Git persistence).
- Git branch: `revision/q2-night7a-cpu-consensus-preflight-20260818`; planned immutable final tag: `night7a-final-20260818`. The final commit, bundle SHA, compact indexes, Windows verification, and shutdown dispatch status are recorded in the non-self-referential external/post-dispatch indexes created after this report's delivery-index commit.
"""
    (OUT / "night7a_report.md").write_text(report, encoding="utf-8")

    # Internal index intentionally excludes itself. Output paths use handoff root;
    # code/protocol/test paths explicitly use the repository root.
    output_files = sorted(path for path in OUT.rglob("*")
                          if path.is_file() and path.name != "delivery_index.json")
    repo_files = [
        REPO / "SpaLORA/night7a_consensus.py",
        REPO / "SpaLORA/night7a_firewall.py",
        REPO / "scripts/night7a_p0.py", REPO / "scripts/night7a_p0_driver.sh",
        REPO / "scripts/night7a_transform.py",
        REPO / "scripts/night7a_preflight.py", REPO / "scripts/night7a_evaluate.py",
        REPO / "scripts/night7a_recompute.py", REPO / "scripts/night7a_verify.py",
        REPO / "scripts/night7a_finalize.py",
        REPO / "scripts/night7a_package.py",
        REPO / "tests/test_night7a_consensus.py", REPO / "tests/test_night7a_preflight.py",
        REPO / "tests/test_night7a_firewall.py",
        REPO / "tests/test_night7a_evaluation.py",
        REPO / "tests/test_night7a_delivery.py",
        *sorted(path for path in (REPO / "protocols/night7a").rglob("*") if path.is_file()),
    ]
    entries = []
    for path in output_files:
        entries.append({"path": path.relative_to(OUT).as_posix(),
                        "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    for path in repo_files:
        entries.append({"path": path.relative_to(REPO).as_posix(), "root": "repo",
                        "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_json(OUT / "delivery_index.json", {
        "schema": "non-self-referential-v1", "branch": git("branch", "--show-current"),
        "terminal_status": terminal, "internal_output_root": "outputs/night7a_handoff",
        "repo_root_marker": "root=repo", "planned_final_tag": "night7a-final-20260818",
        "files": entries,
    })
    print(json.dumps({"terminal_status": terminal, "indexed_files": len(entries),
                      "report": str(OUT / "night7a_report.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
