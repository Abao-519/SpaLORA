"""Materialize and hash one Night-22A geometry ceiling bank without labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night22a_geometry import array_sha, generate_geometry_bank


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph(archive, prefix: str = "graph0") -> sp.csr_matrix:
    return sp.csr_matrix(
        (archive[prefix + "__data"], archive[prefix + "__indices"], archive[prefix + "__indptr"]),
        shape=tuple(int(x) for x in archive[prefix + "__shape"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-key", default="representation")
    parser.add_argument("--representation-source", required=True)
    parser.add_argument("--parent-bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    carrier_path = Path(args.carrier)
    embedding_path = Path(args.embedding)
    parent_bank_path = Path(args.parent_bank)
    parent_manifest_path = parent_bank_path.with_suffix(".json")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest["lane"] != args.lane or parent_manifest["representation_source"] != args.representation_source:
        raise RuntimeError("parent bank authority mismatch")
    if file_sha(embedding_path) != parent_manifest["embedding_file_sha256"]:
        raise RuntimeError("embedding file SHA differs from Night-21C authority")
    if file_sha(carrier_path) != parent_manifest["carrier_sha256"]:
        raise RuntimeError("carrier SHA differs from Night-21C authority")

    with np.load(embedding_path, allow_pickle=False) as archive:
        discovered_embedding_keys = list(archive.files)
        ids = np.asarray(archive["ids"])
        representation = np.asarray(archive[args.embedding_key], dtype=np.float32)
    with np.load(carrier_path, allow_pickle=False) as archive:
        discovered_carrier_keys = list(archive.files)
        carrier_ids = np.asarray(archive["ids"])
        spatial_graph = load_graph(archive)
    if not np.array_equal(ids, carrier_ids):
        raise RuntimeError("embedding/carrier ordered ID mismatch")
    if array_sha(ids) != parent_manifest["ordered_ids_sha256"]:
        raise RuntimeError("ordered ID SHA differs from Night-21C authority")
    if array_sha(representation) != parent_manifest["embedding_array_sha256"]:
        raise RuntimeError("embedding array SHA differs from Night-21C authority")
    if not np.isfinite(representation).all():
        raise RuntimeError("non-finite representation")

    bank = generate_geometry_bank(representation, spatial_graph, args.k)
    np.savez_compressed(output, ids=ids, candidate_ids=bank.candidate_ids, partitions=bank.partitions)
    with np.load(output, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["partitions"], bank.partitions):
            raise RuntimeError("fresh artifact reload mismatch")

    manifest = {
        "schema": "night22a-geometry-bank-v1",
        "lane": args.lane,
        "k": int(args.k),
        "representation_source": args.representation_source,
        "embedding_path": str(embedding_path),
        "embedding_file_sha256": file_sha(embedding_path),
        "embedding_array_sha256": array_sha(representation),
        "carrier_path": str(carrier_path),
        "carrier_sha256": file_sha(carrier_path),
        "parent_bank_path": str(parent_bank_path),
        "parent_bank_sha256": file_sha(parent_bank_path),
        "parent_manifest_sha256": file_sha(parent_manifest_path),
        "ordered_ids_sha256": array_sha(ids),
        "candidate_ids": bank.candidate_ids.tolist(),
        "candidate_partition_sha256": {
            name: array_sha(partition) for name, partition in zip(bank.candidate_ids.tolist(), bank.partitions)
        },
        "candidate_bank_sha256": file_sha(output),
        "geometry_diagnostics": bank.diagnostics,
        "discovered_embedding_keys": discovered_embedding_keys,
        "accessed_embedding_keys": ["ids", args.embedding_key],
        "discovered_carrier_keys": discovered_carrier_keys,
        "accessed_carrier_keys": [
            "ids", "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape"
        ],
        "labels_read": 0,
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
