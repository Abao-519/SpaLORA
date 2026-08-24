#!/usr/bin/env python3
"""Build label-free Night-16G candidate evidence and optional scout evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from SpaLORA.night16g_basin_selector import (
    BasinSelectorConfig,
    candidate_similarity,
    molecular_separation,
    partition_sha256,
    robust_standardize,
    select_basin,
    spatial_evidence,
)


def load_csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
        shape=tuple(int(x) for x in archive[f"{prefix}__shape"]),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    with np.load(args.carrier, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype("U")
        views = {
            "retained": robust_standardize(archive["retained"]),
            "view1": robust_standardize(archive["view1"]),
            "view2": robust_standardize(archive["view2"]),
        }
        graphs = [load_csr(archive, f"graph{i}") for i in range(3)]
    with np.load(args.partition_bank, allow_pickle=False) as archive:
        partition_ids = np.asarray(archive["ids"]).astype("U")
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
    if not np.array_equal(ids, partition_ids):
        raise ValueError("carrier/partition ordered-ID mismatch")
    producer = json.loads(Path(args.producer_json).read_text())
    passed = sorted(
        (row for row in producer["rows"] if row["status"] == "PASS"),
        key=lambda row: int(row["partition_index"]),
    )
    if len(passed) != len(partitions):
        raise ValueError("producer row/partition mismatch")
    rows: list[dict[str, object]] = []
    for index, (partition, metadata) in enumerate(zip(partitions, passed)):
        unique = np.unique(partition)
        if len(unique) != args.k:
            raise ValueError("candidate does not have exact K")
        molecular = {name: molecular_separation(view, partition) for name, view in views.items()}
        topology = [spatial_evidence(graph, partition) for graph in graphs]
        sizes = np.bincount(partition, minlength=args.k)
        molecular_joint = float(
            np.exp(np.mean(np.log(np.maximum([molecular[n]["explained"] for n in views], 1e-12))))
        )
        topology_joint = float(np.mean([value["excess"] for value in topology]))
        row: dict[str, object] = {
            "candidate_id": metadata["candidate_id"],
            "candidate_sha256": partition_sha256(partition),
            "start_id": metadata["start_id"],
            "arm": metadata["arm"],
            "molecular_joint": molecular_joint,
            "topology_joint": topology_joint,
            "microcluster_score": float(min(sizes) / max(len(partition) / args.k, 1.0)),
            "size_entropy": float(-np.sum((sizes / sizes.sum()) * np.log(np.maximum(sizes / sizes.sum(), 1e-12))) / np.log(args.k)),
            "min_cluster_size": int(min(sizes)),
            "path_id": metadata.get("path_id", ""),
            "path_index": metadata.get("path_index", ""),
            "path_lambda": metadata.get("path_lambda", ""),
        }
        for name, values in molecular.items():
            for key, value in values.items():
                row[f"{name}_{key}"] = value
        for scale, values in enumerate(topology):
            row[f"graph{scale}_agreement"] = values["agreement"]
            row[f"graph{scale}_excess"] = values["excess"]
        rows.append(row)

    selected, enriched, diagnostics = select_basin(
        rows,
        partitions,
        BasinSelectorConfig(
            basin_ari_threshold=args.basin_threshold,
            persistent_ari_threshold=args.persistence_threshold,
            molecular_weight=args.molecular_weight,
            topology_weight=args.topology_weight,
            persistence_weight=args.persistence_weight,
            risk_weight=args.risk_weight,
            representative_evidence_weight=args.representative_evidence_weight,
        ),
    )
    similarity = candidate_similarity(partitions)
    controls = {
        "plain_medoid": int(np.argmax((similarity.sum(axis=1) - 1.0) / max(len(partitions) - 1, 1))),
        "minimum_retained_inertia": int(np.argmin([row["retained_within_per_observation"] for row in rows])),
        "maximum_retained_ch": int(np.argmax([row["retained_ch"] for row in rows])),
        "maximum_joint_molecular": int(np.argmax([row["molecular_joint"] for row in rows])),
        "maximum_spatial_coherence": int(np.argmax([row["topology_joint"] for row in rows])),
        "fixed_direct_energy": next(
            index for index, row in enumerate(rows)
            if row["arm"] == "DIRECT_BASE" and int(passed[index]["start_index"]) == 0
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output.with_suffix(".features.csv"), enriched)
    np.savez_compressed(
        output,
        ids=ids,
        partitions=partitions,
        candidate_ids=np.asarray([row["candidate_id"] for row in rows]),
        selected_index=np.asarray(selected, dtype=np.int32),
    )
    manifest = {
        "schema": "night16g-label-free-basin-scout-v1",
        "lane": args.lane,
        "k": int(args.k),
        "n": len(ids),
        "candidate_count": len(rows),
        "selected": {"index": selected, **enriched[selected]},
        "controls": {
            name: {"index": index, "candidate_id": rows[index]["candidate_id"], "candidate_sha256": rows[index]["candidate_sha256"]}
            for name, index in controls.items()
        },
        "diagnostics": diagnostics,
        "producer_label_reads": 0,
        "dense_observation_by_observation_count": 0,
        "thread_limit": 1,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basin-threshold", type=float, default=0.82)
    parser.add_argument("--persistence-threshold", type=float, default=0.90)
    parser.add_argument("--molecular-weight", type=float, default=1.0)
    parser.add_argument("--topology-weight", type=float, default=1.0)
    parser.add_argument("--persistence-weight", type=float, default=1.0)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--representative-evidence-weight", type=float, default=0.35)
    with threadpool_limits(limits=1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
