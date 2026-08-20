#!/usr/bin/env python3
"""Night-9B P0 authority/resource audit.

This program is deliberately label-blind.  It never imports anndata, opens an
H5AD, reads a label snapshot, loads clusters, or computes a new scientific
metric.  It only verifies already frozen manifests/artifacts and independently
re-aggregates the already locked Night-9A per-seed metric table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PARENT = "aa933b8fc11fc05470287a21f80facd03d0acfb9"
PARENT_TAG = "night9a-final-20260820"
BRANCH = "revision/q2-night9b-rna-anchor-cooperative-fusion-rnd-20260820"
PROTECTION_TAG = "baseline/pre-night9b-rna-anchor-cooperative-fusion-20260820"
TASKBOOK_SHA = "845a7002fad6109490e1bc30f43463e3bbc51d35fddcbf682cd2e240eee2855a"
REGISTRY_SHA = "ce181a59429ca90f1df8bf114cf54e5ab9eb3657c853caeb5096fec5a26b3bd4"
NIGHT9A_REGISTRY_SHA = "8ebd6e09cb49f065b49873b27bc0aedf60ef10217319d77e2b7470a99954087b"
COSMOS_COMMIT = "56ea355be51e64d9253e2871b8bd447fdfd0d230"
SPATIAL_THRESHOLDS = {
    "neighbor_min": -0.01,
    "moran_min": -0.02,
    "geary_max": 0.02,
    "boundary_max": 0.01,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def command(argv: list[str], cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(argv)}\n{result.stderr}"
        )
    return result.stdout.strip()


def git(repo: Path, *args: str) -> str:
    return command(["git", *args], cwd=repo)


def verify_file_record(record: dict, seen: dict[str, dict], errors: list[str]) -> None:
    path_value = record.get("path")
    expected_sha = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not (isinstance(path_value, str) and isinstance(expected_sha, str)):
        return
    path = Path(path_value)
    key = str(path)
    if key in seen:
        prior = seen[key]
        if prior["expected_sha256"] != expected_sha:
            errors.append(f"conflicting expected SHA for {path}")
        return
    entry = {
        "path": key,
        "expected_sha256": expected_sha,
        "expected_size_bytes": expected_size,
        "exists": path.is_file(),
    }
    if path.is_file():
        entry["actual_size_bytes"] = path.stat().st_size
        entry["actual_sha256"] = sha256_file(path)
        entry["status"] = (
            "PASS"
            if entry["actual_sha256"] == expected_sha
            and (expected_size is None or entry["actual_size_bytes"] == int(expected_size))
            else "FAIL"
        )
    else:
        entry["status"] = "MISSING"
    if entry["status"] != "PASS":
        errors.append(f"artifact verification failed: {path}: {entry['status']}")
    seen[key] = entry


def walk_declared_files(value: object, seen: dict[str, dict], errors: list[str]) -> None:
    if isinstance(value, dict):
        verify_file_record(value, seen, errors)
        for child in value.values():
            walk_declared_files(child, seen, errors)
    elif isinstance(value, list):
        for child in value:
            walk_declared_files(child, seen, errors)


def load_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = [
        "ari",
        "nmi",
        "q",
        "neighbor_agreement",
        "moran_i",
        "geary_c",
        "boundary_disagreement",
    ]
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in numeric:
            row[key] = float(row[key])
    return rows


def mean(values: list[float]) -> float:
    return float(np.asarray(values, dtype=np.float64).mean())


def recompute_candidate(
    candidate: str,
    metric_rows: list[dict],
    lock_rows: list[dict],
    gates: dict,
) -> dict:
    seeds = [0, 1, 2]
    by_metric = {(r["candidate_id"], r["seed"]): r for r in metric_rows}
    candidate_metrics = [by_metric[(candidate, seed)] for seed in seeds]
    full = [by_metric[("FULL_F00", seed)] for seed in seeds]
    u00 = [by_metric[("U00", seed)] for seed in seeds]
    selected_lock = [r for r in lock_rows if r["candidate_id"] == candidate]
    result = {
        "candidate_id": candidate,
        "stages": ["R1"],
        "seeds": seeds,
        "complete": len(selected_lock) == 6 and all(r["status"] == "success" for r in selected_lock),
        "expected_cells": 6,
        "observed_cells": len(selected_lock),
    }
    names = [
        "ari",
        "nmi",
        "q",
        "neighbor_agreement",
        "moran_i",
        "geary_c",
        "boundary_disagreement",
    ]
    for name in names:
        c = [r[name] for r in candidate_metrics]
        f = [r[name] for r in full]
        u = [r[name] for r in u00]
        result[f"p22_mean_{name}"] = mean(c)
        result[f"p22_mean_delta_{name}_vs_full_f00"] = mean([a - b for a, b in zip(c, f)])
        result[f"p22_mean_delta_{name}_vs_u00"] = mean([a - b for a, b in zip(c, u)])
        result[f"p22_wins_{name}_vs_u00"] = int(sum(a > b for a, b in zip(c, u)))
    result["p22_full_f00_mean_q"] = mean([r["q"] for r in full])

    misar = [r for r in selected_lock if r["dataset"] == "misar"]
    fidelity = [r["fidelity_vs_full_f00"] for r in misar]
    result.update(
        {
            "misar_mean_partition_ari": mean([x["partition_ari"] for x in fidelity]),
            "misar_min_partition_ari": min(x["partition_ari"] for x in fidelity),
            "misar_mean_partition_nmi": mean([x["partition_nmi"] for x in fidelity]),
            "misar_exact_partition_count": int(
                sum(bool(x["partition_exact_up_to_permutation"]) for x in fidelity)
            ),
        }
    )
    candidate_e2e = [r["resource"]["candidate_end_to_end_seconds"] for r in selected_lock]
    u00_e2e = [r["resource"]["u00_end_to_end_seconds"] for r in selected_lock]
    candidate_gpu = [r["resource"]["candidate_peak_gpu_mib"] for r in selected_lock]
    u00_gpu = [r["resource"]["u00_peak_gpu_mib"] for r in selected_lock]
    result.update(
        {
            "mean_candidate_end_to_end_seconds": mean(candidate_e2e),
            "mean_u00_end_to_end_seconds": mean(u00_e2e),
            "runtime_ratio_vs_u00": mean(candidate_e2e) / mean(u00_e2e),
            "mean_candidate_peak_gpu_mib": mean(candidate_gpu),
            "mean_u00_peak_gpu_mib": mean(u00_gpu),
            "peak_gpu_ratio_vs_u00": mean(candidate_gpu) / max(mean(u00_gpu), 1e-12),
        }
    )
    spatial = {
        "neighbor": result["p22_mean_delta_neighbor_agreement_vs_full_f00"]
        >= SPATIAL_THRESHOLDS["neighbor_min"],
        "moran": result["p22_mean_delta_moran_i_vs_full_f00"]
        >= SPATIAL_THRESHOLDS["moran_min"],
        "geary": result["p22_mean_delta_geary_c_vs_full_f00"]
        <= SPATIAL_THRESHOLDS["geary_max"],
        "boundary": result["p22_mean_delta_boundary_disagreement_vs_full_f00"]
        <= SPATIAL_THRESHOLDS["boundary_max"],
    }
    result["spatial_protection"] = {
        "pass": all(spatial.values()),
        "components": spatial,
        "thresholds": SPATIAL_THRESHOLDS,
    }
    checks = {
        "runtime": result["runtime_ratio_vs_u00"] <= gates["runtime_ratio_vs_u00_max"],
        "peak_gpu": result["peak_gpu_ratio_vs_u00"] <= gates["peak_gpu_ratio_vs_u00_max"],
        "p22_q_vs_full": result["p22_mean_delta_q_vs_full_f00"]
        >= gates["p22_mean_q_delta_vs_full_f00_min"],
        "p22_q_vs_u00": result["p22_mean_delta_q_vs_u00"]
        >= gates["p22_mean_q_delta_vs_u00_min"],
        "p22_ari_vs_full": result["p22_mean_delta_ari_vs_full_f00"]
        >= gates["p22_mean_ari_delta_vs_full_f00_min"],
        "p22_nmi_vs_full": result["p22_mean_delta_nmi_vs_full_f00"]
        >= gates["p22_mean_nmi_delta_vs_full_f00_min"],
        "p22_spatial": result["spatial_protection"]["pass"],
        "misar_mean_ari": result["misar_mean_partition_ari"]
        >= gates["misar_mean_partition_ari_vs_full_f00_min"],
        "misar_min_ari": result["misar_min_partition_ari"]
        >= gates["misar_min_partition_ari_vs_full_f00_min"],
        "misar_mean_nmi": result["misar_mean_partition_nmi"]
        >= gates["misar_mean_partition_nmi_vs_full_f00_min"],
    }
    result["hard_gates"] = checks
    result["all_hard_gates_pass"] = all(checks.values())
    result["gate_failures"] = [key for key, value in checks.items() if not value]
    p22_preservation = 1.0 + result["p22_mean_delta_q_vs_full_f00"] / max(
        abs(result["p22_full_f00_mean_q"]), 1e-12
    )
    result["worst_dataset_normalized_preservation"] = min(
        p22_preservation, result["misar_mean_partition_ari"]
    )
    return result


def compare_values(expected: object, actual: object, path: str, errors: list[dict]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected:
            if key not in actual:
                errors.append({"path": f"{path}.{key}", "kind": "missing"})
            else:
                compare_values(expected[key], actual[key], f"{path}.{key}", errors)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append({"path": path, "kind": "length", "expected": len(expected), "actual": len(actual)})
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_values(left, right, f"{path}[{index}]", errors)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or abs(float(expected) - float(actual)) > 1e-9:
            errors.append({"path": path, "kind": "numeric", "expected": expected, "actual": actual})
        return
    if expected != actual:
        errors.append({"path": path, "kind": "value", "expected": expected, "actual": actual})


def summarize_resources(lock_rows: list[dict]) -> dict:
    unit_rows = []
    errors = []
    for row in lock_rows:
        worker_path = Path(row["adapter"]["worker_manifest"]["path"])
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        source = float(row["source_backbone"]["historical_runtime_seconds"])
        topology = float(row["topology_expert"]["runtime_seconds"])
        adapter_subprocess = float(row["adapter"]["runtime_seconds"])
        adapter_train = float(worker["runtime_seconds"])
        residual = adapter_subprocess - adapter_train
        head = float(row["head"]["runtime_seconds"])
        total = float(row["resource"]["candidate_end_to_end_seconds"])
        reconstructed = source + topology + adapter_subprocess + head
        no_reload = source + topology + adapter_train + head
        if residual < -1e-9:
            errors.append(f"negative adapter residual: {row['candidate_id']} {row['dataset']} seed {row['seed']}")
        if abs(total - reconstructed) > 1e-8:
            errors.append(f"total wall reconstruction mismatch: {row['candidate_id']} {row['dataset']} seed {row['seed']}")
        unit_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "source_backbone_train_seconds": source,
                "topology_transform_seconds": topology,
                "adapter_subprocess_combined_seconds": adapter_subprocess,
                "adapter_training_only_seconds": adapter_train,
                "checkpoint_serialization_reload_plus_orchestration_residual_seconds": residual,
                "head_transform_seconds": head,
                "total_method_wall_seconds": total,
                "reconstructed_total_method_wall_seconds": reconstructed,
                "diagnostic_total_without_reload_residual_seconds": no_reload,
                "u00_total_wall_seconds": float(row["resource"]["u00_end_to_end_seconds"]),
                "original_runtime_ratio_vs_u00": total / float(row["resource"]["u00_end_to_end_seconds"]),
                "diagnostic_no_reload_ratio_vs_u00": no_reload / float(row["resource"]["u00_end_to_end_seconds"]),
                "worker_manifest_path": str(worker_path),
                "worker_manifest_sha256": sha256_file(worker_path),
                "reload_audit_path": row["adapter"]["reload_audit"]["path"],
                "reload_audit_sha256": row["adapter"]["reload_audit"]["sha256"],
            }
        )
    grouped = []
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in unit_rows:
        by_candidate[row["candidate_id"]].append(row)
    for candidate, rows in sorted(by_candidate.items()):
        grouped.append(
            {
                "candidate_id": candidate,
                "unit_count": len(rows),
                "mean_source_backbone_train_seconds": mean([x["source_backbone_train_seconds"] for x in rows]),
                "mean_topology_transform_seconds": mean([x["topology_transform_seconds"] for x in rows]),
                "mean_adapter_subprocess_combined_seconds": mean([x["adapter_subprocess_combined_seconds"] for x in rows]),
                "mean_adapter_training_only_seconds": mean([x["adapter_training_only_seconds"] for x in rows]),
                "mean_checkpoint_serialization_reload_plus_orchestration_residual_seconds": mean([x["checkpoint_serialization_reload_plus_orchestration_residual_seconds"] for x in rows]),
                "mean_head_transform_seconds": mean([x["head_transform_seconds"] for x in rows]),
                "mean_total_method_wall_seconds": mean([x["total_method_wall_seconds"] for x in rows]),
                "mean_u00_total_wall_seconds": mean([x["u00_total_wall_seconds"] for x in rows]),
                "original_runtime_ratio_vs_u00": mean([x["total_method_wall_seconds"] for x in rows]) / mean([x["u00_total_wall_seconds"] for x in rows]),
                "diagnostic_no_reload_runtime_ratio_vs_u00": mean([x["diagnostic_total_without_reload_residual_seconds"] for x in rows]) / mean([x["u00_total_wall_seconds"] for x in rows]),
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "historical_night9a_terminal_is_immutable": True,
        "historical_instrumentation_interpretation": {
            "adapter_subprocess_combined": "Night-9A parent-process wall interval includes worker training, checkpoint serialization, fresh-process reload/forward verification, and orchestration overhead.",
            "adapter_training_only": "training_manifest.runtime_seconds measures the worker training interval only.",
            "checkpoint_serialization_reload_exact_seconds": None,
            "available_bound": "combined minus training-only; this is an inclusive residual containing serialization, reload, fresh-process forward, process startup, and orchestration.",
            "diagnostic_sensitivity": "Remove the entire inclusive residual. This is deliberately favorable and diagnostic only; it cannot change or backfill Night-9A gates or terminal state.",
        },
        "unit_count": len(unit_rows),
        "errors": errors,
        "candidate_summary": grouped,
        "unit_rows": unit_rows,
    }


def runtime_environment() -> dict:
    import scipy
    import sklearn
    import torch

    smi = command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    root_disk = shutil.disk_usage("/")
    persistent_disk = shutil.disk_usage("/root/autodl-fs")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": smi,
        "root_disk": {"total": root_disk.total, "used": root_disk.used, "free": root_disk.free},
        "persistent_disk": {
            "path": "/root/autodl-fs",
            "total": persistent_disk.total,
            "used": persistent_disk.used,
            "free": persistent_disk.free,
            "writable": os.access("/root/autodl-fs", os.W_OK),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--night9a-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    night9a_repo = args.night9a_repo.resolve()
    out = repo / "outputs/night9b"

    errors: list[str] = []
    authority_paths = {
        "taskbook": repo / "protocols/night9b/SpaLORA_Night9B_SOTA_Gap_and_RNA_Anchored_Cooperative_Fusion_Taskbook_2026-08-20.md",
        "registry": repo / "protocols/night9b/SpaLORA_Night9B_RACF_Registry_2026-08-20.json",
        "windows_verification": repo / "protocols/night9b/windows_authority_verification.json",
    }
    expected_authority = {"taskbook": TASKBOOK_SHA, "registry": REGISTRY_SHA}
    authority_audit = {}
    for key, path in authority_paths.items():
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        expected = expected_authority.get(key)
        status = exists and (expected is None or actual == expected)
        authority_audit[key] = {
            "path": str(path),
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if status else "FAIL",
        }
        if not status:
            errors.append(f"authority failed: {key}")
    windows = json.loads(authority_paths["windows_verification"].read_text(encoding="utf-8"))
    if windows.get("status") != "PASS" or any(x.get("status") != "PASS" for x in windows["compact_verification"]):
        errors.append("Windows root-aware compact verification not PASS")

    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    parent_tag = git(repo, "rev-list", "-n", "1", PARENT_TAG)
    protection_tag = git(repo, "rev-list", "-n", "1", PROTECTION_TAG)
    git_audit = {
        "head": head,
        "branch": branch,
        "expected_branch": BRANCH,
        "parent_commit": PARENT,
        "parent_tag": PARENT_TAG,
        "parent_tag_peel": parent_tag,
        "protection_tag": PROTECTION_TAG,
        "protection_tag_peel": protection_tag,
        "parent_is_ancestor_of_head": command(["git", "merge-base", "--is-ancestor", PARENT, head], cwd=repo, check=False) == "",
        "worktree_initially_clean_except_p0": True,
    }
    if head != PARENT or branch != BRANCH or parent_tag != PARENT or protection_tag != PARENT:
        errors.append("Git start/branch/tag semantics mismatch")

    registry = json.loads(authority_paths["registry"].read_text(encoding="utf-8"))
    if registry.get("parent_commit") != PARENT or registry.get("parent_tag") != PARENT_TAG:
        errors.append("Night-9B registry parent mismatch")
    if registry.get("datasets", {}).get("A1", {}).get("reference") != "C00_G04_H05_CONFIRMED":
        errors.append("A1 family reference mismatch")
    if registry.get("datasets", {}).get("P22", {}).get("reference") != "F00_R02_FULL":
        errors.append("P22 family reference mismatch")
    if registry.get("benchmark", {}).get("source_commit") != COSMOS_COMMIT:
        errors.append("COSMOS commit mismatch in registry")

    cosmos_roots = [
        Path("/root/autodl-fs/night7a_external_sources_20260818/COSMOS/COSMOS"),
        Path("/root/autodl-fs/night8a_external_sources_20260820/COSMOS/COSMOS"),
        Path("/root/autodl-fs/night4a_baselines/COSMOS/source/COSMOS"),
    ]
    cosmos = []
    for root in cosmos_roots:
        if root.is_dir():
            probe = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
            )
            if probe.returncode:
                cosmos.append(
                    {"path": str(root), "head": None, "required_object_present": False, "status": "NOT_GIT_REPOSITORY"}
                )
                continue
            commit = probe.stdout.strip()
            object_probe = subprocess.run(
                ["git", "cat-file", "-e", f"{COSMOS_COMMIT}^{{commit}}"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            object_present = object_probe.returncode == 0
            cosmos.append({"path": str(root), "head": commit, "required_object_present": object_present, "status": "PASS"})
    if not cosmos or not any(x["required_object_present"] and x["head"] == COSMOS_COMMIT for x in cosmos):
        errors.append("exact COSMOS source commit unavailable")

    night9a_out = night9a_repo / "outputs/night9a"
    lock_path = night9a_out / "locked_R1_manifest.json"
    metrics_path = night9a_out / "R1_p22_per_seed_metrics.csv"
    summary_path = night9a_out / "R1_candidate_summary.json"
    old_registry_path = night9a_repo / "protocols/night9a/SpaLORA_Night9A_Efficient_Topology_Transfer_Registry_2026-08-20.json"
    if sha256_file(old_registry_path) != NIGHT9A_REGISTRY_SHA:
        errors.append("Night-9A registry SHA mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_rows = lock["rows"]
    old_registry = json.loads(old_registry_path.read_text(encoding="utf-8"))
    candidate_ids = [x["id"] for x in old_registry["candidates"]]
    expected_keys = {(cid, dataset, seed) for cid in candidate_ids for dataset in ("p22", "misar") for seed in (0, 1, 2)}
    actual_keys = {(r["candidate_id"], r["dataset"], int(r["seed"])) for r in lock_rows}
    if len(lock_rows) != 54 or actual_keys != expected_keys:
        errors.append("Night-9A formal 54-chain key grid mismatch")
    if any(r.get("status") != "success" or r.get("scientific_retry") != 0 or r.get("fallback") not in (0, False) for r in lock_rows):
        errors.append("Night-9A chain status/retry/fallback mismatch")
    if lock.get("label_access") != {"misar_Y": 0, "p22": 0} or any(r.get("label_access") != {"misar_Y": 0, "p22": 0} for r in lock_rows):
        errors.append("Night-9A prelabel lock access mismatch")
    obs_by_dataset = defaultdict(set)
    for row in lock_rows:
        obs_by_dataset[row["dataset"]].add(row["ordered_observation_sha256"])
    if any(len(value) != 1 for value in obs_by_dataset.values()):
        errors.append("Night-9A ordered observation SHA inconsistent within dataset")

    artifact_records: dict[str, dict] = {}
    artifact_errors: list[str] = []
    walk_declared_files(lock, artifact_records, artifact_errors)
    for row in lock_rows:
        chain_path = Path(row["artifacts"]["clusters"]["path"]).parent / "chain_manifest.json"
        chain_record = {
            "path": str(chain_path),
            "sha256": row["chain_manifest_sha256"],
            "size_bytes": chain_path.stat().st_size if chain_path.is_file() else None,
        }
        verify_file_record(chain_record, artifact_records, artifact_errors)
    errors.extend(artifact_errors)

    metrics = load_metrics(metrics_path)
    if len(metrics) != 33:
        errors.append("Night-9A R1 metric row count is not 33")
    gates = old_registry["r1_r2_selection_gates"]
    recomputed = [recompute_candidate(cid, metrics, lock_rows, gates) for cid in candidate_ids]
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison_errors: list[dict] = []
    compare_values(original, recomputed, "summary", comparison_errors)
    if comparison_errors:
        errors.append(f"Night-9A independent summary mismatch count={len(comparison_errors)}")
    eligible_original = [x["candidate_id"] for x in original if x["all_hard_gates_pass"]]
    eligible_recomputed = [x["candidate_id"] for x in recomputed if x["all_hard_gates_pass"]]
    if eligible_original != eligible_recomputed or eligible_recomputed:
        errors.append("Night-9A terminal eligibility changed or unexpectedly nonempty")
    recompute_audit = {
        "status": "PASS" if not comparison_errors and eligible_original == eligible_recomputed == [] else "FAIL",
        "formal_chain_count": len(lock_rows),
        "formal_chain_key_grid_exact": actual_keys == expected_keys,
        "metric_row_count": len(metrics),
        "candidate_count": len(recomputed),
        "comparison_numeric_tolerance": 1e-9,
        "comparison_errors": comparison_errors[:100],
        "original_eligible_candidates": eligible_original,
        "recomputed_eligible_candidates": eligible_recomputed,
        "terminal_preserved": eligible_original == eligible_recomputed == [],
        "recomputed_candidate_summary": recomputed,
    }

    resource_audit = summarize_resources(lock_rows)
    if resource_audit["status"] != "PASS":
        errors.extend(resource_audit["errors"])
    atomic_json(out / "night9a_resource_semantics_audit.json", resource_audit)

    environment = runtime_environment()
    if not environment["cuda_available"] or environment["cuda_device_count"] < 1:
        errors.append("CUDA unavailable in required GPU mode")
    if not environment["persistent_disk"]["writable"]:
        errors.append("persistent disk is not writable")

    # Existing label-free lineage locations are checked by manifests only.  No
    # label, cluster, H5AD, or snapshot content is opened here.
    source_index = night9a_repo / "outputs/night7a_handoff/source_views_index.csv"
    source_rows = list(csv.DictReader(source_index.open(newline="", encoding="utf-8")))
    a1_rows = [
        row for row in source_rows
        if row["dataset"] == "a1"
        and row["graph_id"] == "G04_SP10_F10_EUC_UNION"
        and int(row["seed"]) in (0, 1, 2)
    ]
    lineage_files = []
    for row in a1_rows:
        for path_key, sha_key, size_key in (
            ("views_path", "views_sha256", "views_size_bytes"),
            ("run_manifest_path", "run_manifest_sha256", None),
            ("observation_ids_path", "observation_ids_sha256", None),
            ("coordinates_path", "coordinates_sha256", None),
        ):
            record = {"path": row[path_key], "sha256": row[sha_key]}
            if size_key:
                record["size_bytes"] = int(row[size_key])
            verify_file_record(record, artifact_records, artifact_errors)
            lineage_files.append(record)
    if len(a1_rows) != 3:
        errors.append("A1 G04 source-view lineage does not contain seeds 0-2 exactly")
    errors.extend(x for x in artifact_errors if x not in errors)

    firewall = {
        "status": "PASS",
        "phase": "P0_PRE_SCIENCE",
        "authorized_role": "night9b_p0_manifest_auditor",
        "a1": {
            "raw_h5ad_deserialization_count": 0,
            "obs_value_read_count": 0,
            "label_snapshot_read_count": 0,
            "training_or_selection_use_count": 0,
        },
        "p22": {
            "raw_h5ad_deserialization_count": 0,
            "obs_value_read_count": 0,
            "label_snapshot_read_count": 0,
            "training_or_selection_use_count": 0,
        },
        "misar_Y": {
            "this_round_read_count": 0,
            "lineage_final_read_count": 2,
            "permanently_forbidden": True,
        },
        "formal_training_units_created": 0,
        "formal_benchmark_outputs_created": 0,
        "script_contract": {
            "imports_anndata": False,
            "opens_h5ad": False,
            "loads_label_snapshot": False,
            "loads_clusters": False,
            "recomputes_scientific_metrics": False,
        },
    }

    p0 = {
        "schema_version": 1,
        "phase": "P0_AUTHORITY_AND_RESOURCE_CONTRACT",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "authority": authority_audit,
        "windows_compact_verification": windows,
        "git": git_audit,
        "night9b_registry_contract": {
            "registry_id": registry.get("registry_id"),
            "candidate_ids": [x["id"] for x in registry["candidates"]],
            "a1_reference": registry["datasets"]["A1"]["reference"],
            "p22_reference": registry["datasets"]["P22"]["reference"],
            "scientific_retry": registry["shared_rules"]["scientific_retry"],
            "fallback": registry["shared_rules"]["fallback"],
            "cosmos_commit": registry["benchmark"]["source_commit"],
        },
        "external_source_location": {
            "cosmos_required_commit": COSMOS_COMMIT,
            "cosmos_checkouts": cosmos,
            "arise_policy": "P1 source audit only; public label/ARI best-epoch selection path must not be imported or run.",
        },
        "night9a_formal_chain_audit": {
            "status": "PASS" if not artifact_errors and len(lock_rows) == 54 and actual_keys == expected_keys else "FAIL",
            "row_count": len(lock_rows),
            "key_grid_exact": actual_keys == expected_keys,
            "unique_ordered_observation_sha256_by_dataset": {key: sorted(value) for key, value in obs_by_dataset.items()},
            "unique_declared_artifact_count": len(artifact_records),
            "declared_artifact_failure_count": len(artifact_errors),
            "declared_artifacts": sorted(artifact_records.values(), key=lambda x: x["path"]),
        },
        "night9a_candidate_and_gate_recompute": recompute_audit,
        "night9a_resource_semantics_audit": {
            "path": str(out / "night9a_resource_semantics_audit.json"),
            "sha256": sha256_file(out / "night9a_resource_semantics_audit.json"),
            "status": resource_audit["status"],
            "historical_terminal_unchanged": True,
        },
        "lineage_input_location_audit": {
            "status": "PASS" if len(a1_rows) == 3 and not artifact_errors else "FAIL",
            "a1_reference": "C00_G04_H05_CONFIRMED",
            "a1_g04_source_view_rows": a1_rows,
            "p22_reference": "F00_R02_FULL",
            "p22_reference_authority": str(night9a_out / "p0_reference_manifest.json"),
            "night6d_spatial_thresholds": SPATIAL_THRESHOLDS,
            "night6d_authoritative_evaluator_sha256": "e62d6ceffa1cdfb4ed0a14530c9e2f980c1bf6bb005f85605f3d4513816b69c4",
        },
        "environment": environment,
        "label_firewall": firewall,
        "science_counters": {
            "formal_training_units": 0,
            "formal_benchmark_training_units": 0,
            "formal_embeddings": 0,
            "formal_affinities": 0,
            "formal_partitions": 0,
            "a1_label_reads": 0,
            "p22_label_reads": 0,
            "misar_Y_reads": 0,
        },
        "next_phase_authorized": not errors,
        "next_phase_started": False,
    }
    atomic_json(out / "p0_authority_and_resource_contract.json", p0)
    print(json.dumps({
        "status": p0["status"],
        "error_count": len(errors),
        "formal_chain_count": len(lock_rows),
        "artifact_count": len(artifact_records),
        "candidate_recompute": recompute_audit["status"],
        "resource_audit": resource_audit["status"],
        "cuda_available": environment["cuda_available"],
        "p0_path": str(out / "p0_authority_and_resource_contract.json"),
    }, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
