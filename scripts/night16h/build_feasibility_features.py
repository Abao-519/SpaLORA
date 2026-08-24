#!/usr/bin/env python3
"""Attach universal sparse-graph feasibility evidence to a locked bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night16h_feasible_selector import (
    FEASIBILITY_MODES,
    feasibility_record,
    is_feasible,
    prepare_graph,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def graph(archive: np.lib.npyio.NpzFile, index: int) -> sp.csr_matrix:
    prefix = f"graph{index}__"
    return sp.csr_matrix(
        (archive[prefix + "data"], archive[prefix + "indices"], archive[prefix + "indptr"]),
        shape=tuple(int(value) for value in archive[prefix + "shape"]),
    )


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    feature_path = Path(args.features)
    bank_path = Path(args.partition_bank)
    carrier_path = Path(args.carrier)
    records = list(csv.DictReader(feature_path.open(encoding="utf-8")))
    with np.load(bank_path, allow_pickle=False) as locked:
        ids = locked["ids"].astype("U")
        partitions = np.asarray(locked["partitions"], dtype=np.int32)
        candidate_ids = locked["candidate_ids"].astype("U")
    with np.load(carrier_path, allow_pickle=False) as carrier:
        carrier_ids = carrier["ids"].astype("U")
        graphs = [prepare_graph(graph(carrier, index)) for index in range(3)]
    if not np.array_equal(ids, carrier_ids):
        raise ValueError("carrier/bank ordered-ID mismatch")
    if candidate_ids.tolist() != [row["candidate_id"] for row in records]:
        raise ValueError("feature/bank candidate order mismatch")
    enriched = []
    for row, partition in zip(records, partitions):
        evidence = feasibility_record(graphs, partition, args.k, prepared_graphs=True)
        enriched.append(
            {
                **row,
                "structural_exact_k": evidence["exact_k"],
                "structural_empty_cluster_count": evidence["empty_cluster_count"],
                "structural_min_cluster_size": evidence["min_cluster_size"],
                "structural_cluster_sizes": json.dumps(evidence["cluster_sizes"], separators=(",", ":")),
                "per_scale_min_internal_edges": json.dumps(evidence["per_scale_min_internal_edges"], separators=(",", ":")),
                "per_scale_min_internal_weight": json.dumps(evidence["per_scale_min_internal_weight"], separators=(",", ":")),
                "per_scale_min_supported_node_fraction": json.dumps(evidence["per_scale_min_supported_node_fraction"], separators=(",", ":")),
                "smallest_scale_degree_threshold": evidence["smallest_scale_degree_threshold"],
                "smallest_scale_min_internal_edges": evidence["smallest_scale_min_internal_edges"],
                "any_scale_each_cluster_has_internal_edge": evidence["any_scale_each_cluster_has_internal_edge"],
                **{f"feasible_{mode}": is_feasible(evidence, mode) for mode in FEASIBILITY_MODES},
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(enriched[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(enriched)
    manifest = {
        "schema": "night16h-locked-feasibility-features-v1",
        "lane": args.lane,
        "k": args.k,
        "n": len(ids),
        "candidate_count": len(records),
        "feasible_counts": {
            mode: sum(str(row[f"feasible_{mode}"]).lower() == "true" for row in enriched)
            for mode in FEASIBILITY_MODES
        },
        "feature_sha256": sha256(feature_path),
        "partition_bank_sha256": sha256(bank_path),
        "carrier_sha256": sha256(carrier_path),
        "output_sha256": sha256(output),
        "producer_label_reads": 0,
        "dense_observation_by_observation_count": 0,
        "thread_limit": 1,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--features", required=True)
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--output", required=True)
    with threadpool_limits(1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
