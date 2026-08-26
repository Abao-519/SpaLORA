#!/usr/bin/env python3
"""Label-free Night-19B sparse-operator candidate producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from SpaLORA.night19b_csad import (
    arm_operator,
    build_operator_bank,
    graph_sha256,
    sha256_array,
    spectral_partition,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csr_from_archive(archive: dict[str, np.ndarray], prefix: str) -> sp.csr_matrix:
    required = {f"{prefix}__data", f"{prefix}__indices", f"{prefix}__indptr", f"{prefix}__shape"}
    if not required.issubset(archive):
        raise ValueError(f"carrier lacks {prefix}")
    return sp.csr_matrix(
        (archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
        shape=tuple(int(x) for x in archive[f"{prefix}__shape"]),
    )


def load_contract(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "night19b-csad-formula-freeze-v1":
        raise ValueError("formula contract schema mismatch")
    if value.get("producer_label_reads") != 0:
        raise ValueError("formula contract label-flow mismatch")
    return value


def load_numeric_carrier(path: Path, graph_scales: set[int]):
    base = {"ids", "view1", "view2", "retained"}
    graph_keys = {
        f"graph{scale}__{suffix}"
        for scale in graph_scales
        for suffix in ("data", "indices", "indptr", "shape")
    }
    accessed = sorted(base | graph_keys)
    with np.load(path, allow_pickle=False) as archive:
        discovered = sorted(archive.files)
        missing = set(accessed) - set(discovered)
        if missing:
            raise ValueError(f"carrier misses allow-listed numeric keys: {sorted(missing)}")
        carrier = {key: np.asarray(archive[key]) for key in accessed}
    return carrier, discovered, accessed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-id", action="append")
    parser.add_argument("--arm", action="append")
    parser.add_argument("--endpoint-seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    carrier_path, contract_path = Path(args.carrier), Path(args.contract)
    contract = load_contract(contract_path)
    profiles = [row for row in contract["profiles"] if not args.profile_id or row["config_id"] in args.profile_id]
    arms = [arm for arm in contract["arms"] if not args.arm or arm in args.arm]
    if not profiles or not arms:
        raise ValueError("empty profile or arm selection")
    if args.profile_id and {x["config_id"] for x in profiles} != set(args.profile_id):
        raise ValueError("unknown requested profile")
    if args.arm and set(arms) != set(args.arm):
        raise ValueError("unknown requested arm")
    carrier, discovered_keys, accessed_keys = load_numeric_carrier(
        carrier_path, {int(row["spatial_scale"]) for row in profiles}
    )
    required = {"ids", "view1", "view2", "retained"}
    if not required.issubset(carrier):
        raise ValueError("carrier lacks numeric authority")
    n = int(carrier["ids"].size)
    if carrier["view1"].shape[0] != n or carrier["view2"].shape[0] != n or carrier["retained"].shape[0] != n:
        raise ValueError("carrier row authority mismatch")

    candidate_ids = []
    partitions = []
    eigenvalues = []
    diagnostics = []
    with threadpool_limits(limits=1):
        for profile in profiles:
            scale = int(profile["spatial_scale"])
            graph = csr_from_archive(carrier, f"graph{scale}")
            if graph.shape != (n, n):
                raise ValueError("registered graph shape mismatch")
            bank_started = time.time()
            bank = build_operator_bank(
                carrier["view1"], carrier["view2"], graph,
                int(profile["feature_neighbors"]), int(profile["spatial_topk"]), float(profile["self_loop"]),
            )
            bank_seconds = time.time() - bank_started
            for arm in arms:
                candidate_started = time.time()
                operator, operator_diag = arm_operator(
                    arm, bank, carrier["ids"], int(profile["product_topk"]),
                    float(profile["conflict_strength"]), float(profile["conflict_floor"]),
                )
                partition, _embedding, values, endpoint_diag = spectral_partition(
                    operator, int(args.k), int(args.k) + int(profile["spectral_extra_dim"]), int(args.endpoint_seed)
                )
                candidate_id = f"{profile['config_id']}__{arm}__E{args.endpoint_seed}"
                candidate_ids.append(candidate_id)
                partitions.append(partition)
                padded = np.full(int(args.k) + max(int(x["spectral_extra_dim"]) for x in profiles), np.nan)
                padded[: values.size] = values
                eigenvalues.append(padded)
                sizes = np.bincount(partition, minlength=int(args.k)).astype(int)
                diagnostics.append({
                    "candidate_id": candidate_id,
                    "config": profile,
                    "arm": arm,
                    "spatial_input_graph_sha256": graph_sha256(graph),
                    "operator_bank": bank.diagnostics,
                    "operator": operator_diag,
                    "endpoint": endpoint_diag,
                    "cluster_sizes": sizes.tolist(),
                    "min_cluster_size": int(sizes.min()),
                    "exact_k": bool(np.unique(partition).size == int(args.k)),
                    "bank_seconds": float(bank_seconds),
                    "candidate_seconds": float(time.time() - candidate_started),
                })
    partition_array = np.stack(partitions).astype(np.int32)
    eigenvalue_array = np.stack(eigenvalues).astype(np.float64)
    artifact_path = output / "artifact.npz"
    np.savez_compressed(
        artifact_path,
        ids=carrier["ids"],
        candidate_ids=np.asarray(candidate_ids, dtype="U"),
        partitions=partition_array,
        eigenvalues=eigenvalue_array,
    )
    with np.load(artifact_path, allow_pickle=False) as reloaded:
        if not np.array_equal(reloaded["ids"], carrier["ids"]):
            raise RuntimeError("artifact ID reload mismatch")
        if not np.array_equal(reloaded["partitions"], partition_array):
            raise RuntimeError("artifact partition reload mismatch")
        if not np.array_equal(reloaded["candidate_ids"], np.asarray(candidate_ids)):
            raise RuntimeError("artifact candidate reload mismatch")
        if not np.array_equal(reloaded["eigenvalues"], eigenvalue_array, equal_nan=True):
            raise RuntimeError("artifact eigenvalue reload mismatch")
    manifest = {
        "schema": "night19b-csad-producer-v1",
        "lane": args.lane,
        "n": n,
        "k": int(args.k),
        "view1_shape": list(carrier["view1"].shape),
        "view2_shape": list(carrier["view2"].shape),
        "retained_shape": list(carrier["retained"].shape),
        "ordered_ids_sha256": sha256_array(carrier["ids"]),
        "carrier_sha256": file_sha256(carrier_path),
        "carrier_discovered_keys": discovered_keys,
        "carrier_accessed_keys": accessed_keys,
        "contract_sha256": file_sha256(contract_path),
        "taskbook_sha256": contract["taskbook_sha256"],
        "core_source_sha256": file_sha256(Path(__file__).resolve().parents[2] / "SpaLORA" / "night19b_csad.py"),
        "producer_source_sha256": file_sha256(Path(__file__).resolve()),
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "partition_sha256": [sha256_array(row) for row in partition_array],
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_reload": "PASS",
        "diagnostics": diagnostics,
        "producer_label_reads": 0,
        "carrier_annotation_arrays_accessed": 0,
        "annotation_columns_accessed": [],
        "dense_observation_by_observation_count": 0,
        "wall_seconds": float(time.time() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "gpu_seconds": 0.0,
        "thread_limits": {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "threadpool_limits": 1},
    }
    (output / "producer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("lane", "n", "k", "candidate_count", "artifact_sha256", "wall_seconds", "peak_rss_mib")}, indent=2))


if __name__ == "__main__":
    main()
