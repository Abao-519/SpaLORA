#!/usr/bin/env python3
"""Run the 360 fixed Night-7A label-free sparse consensus transforms."""
from __future__ import annotations

import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import (  # noqa: E402
    CANDIDATE_ORDER, DATASETS, G00, G04, VIEWS,
    affinity_audit, array_sha, atomic_json, atomic_sparse,
    candidate_affinity, canonical_json_sha, canonical_partition,
    parse_registry, partition_sha, run_spectral, sha256_file, sparse_sha,
)

OUT = REPO / "outputs/night7a_handoff"
RAW = Path("/root/autodl-fs/night7a_consensus_20260818")
REG = REPO / "protocols/night7a/SpaLORA_Night7A_Consensus_Registry_2026-08-18.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def load_base(dataset: str, seed: int) -> tuple[dict, list[str], dict]:
    root = RAW / "base" / dataset / f"seed_{seed}"
    manifest_path = root / "base_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ids = list(map(str, manifest["ids"]))
    affinities = {}
    reliability = {}
    for graph_id in (G00, G04):
        affinities[graph_id] = [
            sp.load_npz(root / f"A_{graph_id}_{key}.npz").tocsr()
            for key in VIEWS
        ]
        reliability[graph_id] = np.load(root / f"reliability_{graph_id}.npy",
                                        allow_pickle=False)
    base = {
        "affinities": affinities, "reliability": reliability,
        "S_G00": sp.load_npz(root / "S_G00.npz").tocsr(),
        "S_G04": sp.load_npz(root / "S_G04.npz").tocsr(),
        "T_spatial": sp.load_npz(root / "T_spatial.npz").tocsr(),
    }
    for filename, meta in manifest["artifacts"].items():
        path = Path(meta["path"])
        if not path.is_file() or path.stat().st_size != meta["size_bytes"] or sha256_file(path) != meta["file_sha256"]:
            raise RuntimeError(f"base artifact changed: {path}")
    return base, ids, manifest


def save_clusters(path: Path, ids: list[str], labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame({"observation_id": ids, "cluster": labels.astype(np.int64)}).to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("Night-7A transform requires CPU-only isolation")
    registry = json.loads(REG.read_text())
    candidates = parse_registry(registry)
    p0 = json.loads((OUT / "p0_semantic_contract.json").read_text())
    if p0["status"] != "PASS" or p0["h05_exact_partition_parity"] != "30/30 x 2":
        raise RuntimeError("P0 semantic contract is not locked PASS")
    if p0["registry_sha256"] != sha256_file(REG):
        raise RuntimeError("registry changed after P0")
    prediction_index = pd.read_csv(OUT / "source_prediction_index.csv")
    historical = {
        (row.dataset, int(row.seed), row.graph_id, row.head_id):
            row.canonical_partition_sha256
        for row in prediction_index.itertuples(index=False)
    }
    code_commit = git("rev-parse", "HEAD")
    rows = []
    resumed_complete_cells = 0
    ordinal = 0
    start_all = time.perf_counter()
    for dataset in DATASETS:
        seeds = range(5) if dataset in {"a1", "tonsil"} else range(10)
        for seed in seeds:
            base, ids, base_manifest = load_base(dataset, seed)
            input_affinity_shas = {
                f"{graph_id}/{key}": sparse_sha(matrix)
                for graph_id in (G00, G04)
                for key, matrix in zip(VIEWS, base["affinities"][graph_id])
            }
            for candidate_id in CANDIDATE_ORDER:
                ordinal += 1
                target = RAW / "formal" / dataset / f"seed_{seed}" / candidate_id / "attempt_001"
                if target.exists():
                    # A completed atomic audit may be reused after an SSH/client
                    # interruption.  No transform is rerun and no byte is
                    # overwritten.  A partial cell remains fail-closed for an
                    # explicitly recorded infrastructure correction.
                    audit_path = target / "candidate_audit.json"
                    if not audit_path.is_file():
                        raise RuntimeError(
                            f"partial formal cell requires a recorded correction; "
                            f"refusing overwrite: {target}"
                        )
                    record = json.loads(audit_path.read_text())
                    expected_identity = {
                        "ordinal": ordinal, "dataset": dataset, "seed": seed,
                        "candidate_id": candidate_id, "attempt": 1,
                        "candidate_config_sha256": canonical_json_sha(candidates[candidate_id]),
                        "registry_sha256": sha256_file(REG), "code_commit": code_commit,
                    }
                    if any(record.get(key) != value for key, value in expected_identity.items()):
                        raise RuntimeError(f"completed formal cell identity mismatch: {target}")
                    if record.get("status") not in {"success", "scientific_numerical_failure"}:
                        raise RuntimeError(f"unexpected completed formal status: {target}")
                    for artifact in record.get("artifacts", {}).values():
                        path = Path(artifact["path"])
                        if (not path.is_file() or path.stat().st_size != artifact["size_bytes"] or
                                sha256_file(path) != artifact["sha256"]):
                            raise RuntimeError(f"completed formal artifact changed: {path}")
                    weights = record.get("local_reliability_weights")
                    if weights:
                        path = Path(weights["path"])
                        if (not path.is_file() or path.stat().st_size != weights["size_bytes"] or
                                sha256_file(path) != weights["sha256"]):
                            raise RuntimeError(f"completed reliability weights changed: {path}")
                    record["candidate_audit_path"] = str(audit_path)
                    record["candidate_audit_sha256"] = sha256_file(audit_path)
                    rows.append(record)
                    resumed_complete_cells += 1
                    continue
                target.mkdir(parents=True)
                started = time.perf_counter()
                record = {
                    "ordinal": ordinal, "dataset": dataset, "seed": seed,
                    "candidate_id": candidate_id, "attempt": 1,
                    "correction_id": None,
                    "candidate_config_sha256": canonical_json_sha(candidates[candidate_id]),
                    "registry_sha256": sha256_file(REG), "code_commit": code_commit,
                    "base_manifest_path": str(RAW / "base" / dataset / f"seed_{seed}" / "base_manifest.json"),
                    "base_manifest_sha256": sha256_file(RAW / "base" / dataset / f"seed_{seed}" / "base_manifest.json"),
                    "input_views_sha256": [item["views_sha256"] for item in base_manifest["source_pair"]],
                    "six_affinity_canonical_sha256": input_affinity_shas,
                    "S_G00_sha256": sparse_sha(base["S_G00"]),
                    "S_G04_sha256": sparse_sha(base["S_G04"]),
                    "fallback": False, "label_access": False,
                    "gpu_allocation_mib": 0.0,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "python": platform.python_version(),
                    "spectral_warnings": [],
                    "artifacts": {},
                    "environment": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__,
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                    },
                }
                try:
                    matrix, extra = candidate_affinity(candidate_id, base, ids)
                    audit = affinity_audit(matrix)
                    affinity_path = target / "consensus_affinity.npz"
                    atomic_sparse(affinity_path, matrix)
                    record["affinity_audit"] = audit
                    record["artifacts"]["consensus_affinity.npz"] = {
                        "path": str(affinity_path),
                        "size_bytes": affinity_path.stat().st_size,
                        "sha256": sha256_file(affinity_path),
                        "canonical_sparse_sha256": sparse_sha(matrix),
                    }
                    if candidate_id == "C07_DUAL_LOCAL_RELIABILITY":
                        weights = np.asarray(extra.pop("g00_weights"), dtype=np.float64)
                        weights_path = target / "local_reliability_weights.csv"
                        weights_tmp = weights_path.with_name(weights_path.name + ".tmp")
                        pd.DataFrame({"observation_id": ids, "g00_weight": weights,
                                      "g04_weight": 1.0 - weights}).to_csv(weights_tmp, index=False)
                        os.replace(weights_tmp, weights_path)
                        record["local_reliability_weights"] = {
                            "path": str(weights_path), "size_bytes": weights_path.stat().st_size,
                            "sha256": sha256_file(weights_path),
                        }
                    record["candidate_diagnostics"] = extra
                    if (not audit["finite"] or audit["symmetry_max_error"] > 1e-12 or
                            audit["diagonal_max_abs"] > 1e-12 or audit["zero_degree_count"] > 0):
                        raise FloatingPointError(f"invalid consensus affinity audit: {audit}")
                    caught = []
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            labels, spectral_runtime, peak_rss = run_spectral(matrix, dataset)
                    finally:
                        record["spectral_warnings"] = [
                            {"category": item.category.__name__, "message": str(item.message)}
                            for item in caught
                        ]
                    clusters_path = target / "clusters.csv"
                    save_clusters(clusters_path, ids, labels)
                    record.update({
                        "status": "success",
                        "spectral_runtime_seconds": spectral_runtime,
                        "runtime_seconds": time.perf_counter() - started,
                        "process_peak_rss_mib": peak_rss,
                        "cluster_count": int(len(np.unique(labels))),
                        "canonical_partition_sha256": partition_sha(labels),
                    })
                    record["artifacts"]["clusters.csv"] = {
                        "path": str(clusters_path),
                        "size_bytes": clusters_path.stat().st_size,
                        "sha256": sha256_file(clusters_path),
                    }
                    if candidate_id in {"C00_G04_H05_CONFIRMED", "C01_G00_H05"}:
                        graph_id = G04 if candidate_id.startswith("C00") else G00
                        expected = historical[(dataset, seed, graph_id,
                                               "H05_EQUAL3_AFFINITY_SPECTRAL")]
                        record["authoritative_h05_partition_sha256"] = expected
                        record["authoritative_h05_partition_exact"] = (
                            record["canonical_partition_sha256"] == expected
                        )
                        if not record["authoritative_h05_partition_exact"]:
                            raise RuntimeError(f"formal H05 partition parity failed {dataset}/{seed}/{graph_id}")
                except (
                    FloatingPointError,
                    np.linalg.LinAlgError,
                    sp.linalg.ArpackNoConvergence,
                    ValueError,
                ) as exc:
                    record.update({
                        "status": "scientific_numerical_failure",
                        "runtime_seconds": time.perf_counter() - started,
                        "process_peak_rss_mib": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss / 1024.0,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                except Exception as exc:
                    # File/provenance errors, implementation bugs, parity
                    # failures, and infrastructure exceptions must not be
                    # laundered into a scientific negative result.
                    record.update({
                        "status": "implementation_or_infrastructure_failure",
                        "runtime_seconds": time.perf_counter() - started,
                        "process_peak_rss_mib": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss / 1024.0,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    atomic_json(target / "candidate_audit.json", record)
                    raise
                audit_path = target / "candidate_audit.json"
                atomic_json(audit_path, record)
                record["candidate_audit_path"] = str(audit_path)
                record["candidate_audit_sha256"] = sha256_file(audit_path)
                rows.append(record)
    if ordinal != 360 or len(rows) != 360:
        raise RuntimeError(f"formal transform cardinality mismatch: {ordinal}/{len(rows)}")
    success = sum(row["status"] == "success" for row in rows)
    failure = 360 - success
    parity = [row for row in rows if row["candidate_id"] in {
        "C00_G04_H05_CONFIRMED", "C01_G00_H05"
    }]
    if len(parity) != 60 or not all(row.get("authoritative_h05_partition_exact") for row in parity):
        raise RuntimeError("30x2 H05 parity did not survive formal transform lock")
    manifest = {
        "schema_version": 1, "status": "LOCKED",
        "fixed_order": "dataset [a1,tonsil,d1,p22], seed ascending, candidate_order",
        "registry_sha256": sha256_file(REG), "code_commit": code_commit,
        "planned_formal_transforms": 360, "formal_transform_attempts": 360,
        "implementation_corrections": 0, "total_transform_attempts": 360,
        "success_count": success, "scientific_numerical_failure_count": failure,
        "locked_before_label_access": True,
        "development_per_spot_ground_truth_deserializations": 0,
        "fresh_external_per_spot_label_reads": 0,
        "scientific_training": 0, "checkpoint_forward": 0, "diffusion": 0,
        "gpu_allocation_mib": 0.0,
        "h05_formal_partition_parity": "30/30 x 2",
        "runtime_seconds": time.perf_counter() - start_all,
        "resumed_completed_cells_without_rerun": resumed_complete_cells,
        "transforms": rows,
    }
    path = OUT / "locked_consensus_transform_manifest.json"
    atomic_json(path, manifest)
    os.chmod(path, 0o444)
    atomic_json(OUT / "budget_and_access_audit_pre_evaluation.json", {
        "status": "PASS", "scientific_training": 0, "checkpoint_forward": 0,
        "diffusion": 0, "formal_transforms": 360,
        "scientific_numerical_failures": failure, "correction_attempts": 0,
        "total_transform_attempts": 360, "formal_benchmark_runs": 0,
        "development_label_reads": 0, "fresh_external_label_reads": 0,
        "gpu_allocation_mib": 0.0,
        "transform_manifest_sha256": sha256_file(path),
    })
    print(json.dumps({"status": "LOCKED", "success": success,
                      "scientific_numerical_failures": failure,
                      "attempts": 360}, sort_keys=True))


if __name__ == "__main__":
    main()
