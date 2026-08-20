#!/usr/bin/env python3
"""One no-retry transform cell for the locked Night-8B recovery."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha
from SpaLORA.night7a_consensus import canonical_partition
from SpaLORA.night8b_head_recovery import (
    HEAD_CONFIG_SHA256, HEAD_ID, OBSERVATION_SHA256, canonical_affinity,
    run_twice,
)

RAW = Path("/root/autodl-fs/night8b_head_recovery_20260820")
OUT = REPO / "outputs/night8b_head_recovery"
ORIGINAL = Path("/root/autodl-fs/night8b_raw_runs_20260820")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("HR_U00", "HR_F00"), required=True)
    parser.add_argument("--seed", type=int, choices=range(10), required=True)
    args = parser.parse_args()
    lock = json.loads((OUT / "recovery_input_view_manifest.json").read_text())
    row = next(x for x in lock["rows"] if x["method"] == args.method and int(x["seed"]) == args.seed)
    target = RAW / "partitions" / args.method / f"seed_{args.seed}"
    if target.exists():
        raise RuntimeError(f"no-retry transform refuses existing target {target}")
    target.mkdir(parents=True, exist_ok=False)
    source = Path(row["path"])
    if sha256_file(source) != row["file_sha256"]:
        raise RuntimeError("locked affinity file SHA mismatch")
    affinity = sp.load_npz(source).tocsr()
    if sparse_sha(affinity) != row["canonical_sparse_sha256"]:
        raise RuntimeError("locked affinity canonical SHA mismatch")
    started = time.perf_counter()
    labels, audit = run_twice(affinity)
    wall = time.perf_counter() - started
    ids = (ORIGINAL / "formal/adapter/inputs/seed_0/observation_ids.txt").read_text().splitlines()
    if len(ids) != len(labels):
        raise RuntimeError("observation count mismatch")
    observation_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if observation_sha != OBSERVATION_SHA256:
        raise RuntimeError("observation order SHA mismatch")
    clusters_tmp = target / "clusters.csv.tmp"
    with clusters_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_id", "cluster"])
        writer.writerows(zip(ids, map(int, labels)))
    clusters = target / "clusters.csv"
    os.replace(clusters_tmp, clusters)
    canonical_labels = canonical_partition(labels)
    manifest = {
        "schema_version": 1, "status": "success", "method": args.method,
        "seed": args.seed, "K": int(np.unique(canonical_labels).size),
        "head_id": HEAD_ID, "head_config_sha256": HEAD_CONFIG_SHA256,
        "head_function_arguments": ["affinity"],
        "input_path": str(source), "input_file_sha256": row["file_sha256"],
        "input_canonical_sparse_sha256": sparse_sha(affinity),
        "post_symmetry_canonical_sparse_sha256": sparse_sha(canonical_affinity(affinity)),
        "ordered_observation_sha256": observation_sha,
        "canonical_partition_sha256": array_sha(canonical_labels),
        "clusters_path": str(clusters), "clusters_file_sha256": sha256_file(clusters),
        "determinism_audit": audit, "runtime_seconds": float(wall),
        "recovery_head_effective_seconds": audit["first_head_seconds"],
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_used": False, "label_access": False, "training": 0,
        "adapter": 0, "affinity_rebuild": 0, "scientific_retry": False,
        "fallback": False,
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    atomic_json(target / "transform_manifest.json", manifest)
    print(json.dumps({"status": "success", "method": args.method,
                      "seed": args.seed, "seconds": wall,
                      "partition_sha256": manifest["canonical_partition_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
