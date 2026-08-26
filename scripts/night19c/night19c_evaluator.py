#!/usr/bin/env python3
"""Independent post-lock evaluator for Night-19C placenta partitions."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

from SpaLORA.night17b_sfrd import encode_partition, sha256_array
from SpaLORA.night19c_zero_start_transfer import file_sha256
from scripts.night19a.night19a_d0_evaluator import aligned_changed_count, categorical_spatial, load_reference
from scripts.night19b.night19b_evaluator import internal_edge_counts
from scripts.night19c.night19c_producer import csr_from_carrier, load_numeric_carrier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if file_sha256(args.artifact) != producer["artifact_sha256"]:
        raise ValueError("locked producer artifact SHA mismatch before annotation access")
    with np.load(args.artifact, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"])
        run_ids = archive["run_ids"].astype(str)
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
        representation_sha = archive["representation_sha256"].astype(str)
    if run_ids.tolist() != producer["run_ids"]:
        raise ValueError("run ID authority mismatch before annotation access")
    if [sha256_array(row) for row in partitions] != producer["partition_sha256"]:
        raise ValueError("partition authority mismatch before annotation access")
    carrier, discovered, accessed = load_numeric_carrier(Path(args.carrier))
    if not np.array_equal(ids, carrier["ids"]):
        raise ValueError("ordered IDs mismatch before annotation access")
    labels, mask = load_reference("h5ad", Path(args.reference), ids, args.label_key, "label_mask")
    truth = encode_partition(labels[mask])
    if mask.sum() != ids.size or np.unique(truth).size != 10:
        raise ValueError("placenta ALL_1662 K10 reference contract mismatch")
    graph = csr_from_carrier(carrier)
    zero_index = int(np.flatnonzero(run_ids == "ZERO_RESIDUAL")[0])
    rows = []
    for run_id, partition, rep_sha in zip(run_ids, partitions, representation_sha):
        encoded = encode_partition(partition)
        prediction = encode_partition(encoded[mask])
        sizes = np.bincount(encoded, minlength=10)
        agreement, moran, geary = categorical_spatial(encoded, graph)
        internal = internal_edge_counts(encoded, graph, 10)
        rows.append({
            "lane": "PLACENTA_K10", "run_id": run_id,
            "ari": adjusted_rand_score(truth, prediction),
            "nmi": normalized_mutual_info_score(truth, prediction),
            "ami": adjusted_mutual_info_score(truth, prediction),
            "fmi": fowlkes_mallows_score(truth, prediction),
            "neighbor_agreement": agreement, "moran_macro_ovr": moran, "geary_macro_ovr": geary,
            "min_cluster_size": int(sizes.min()),
            "cluster_sizes": json.dumps(sizes.astype(int).tolist(), separators=(",", ":")),
            "min_internal_spatial_edges": int(internal.min()),
            "changed_spots_vs_zero": aligned_changed_count(partitions[zero_index], encoded),
            "representation_sha256": rep_sha, "partition_sha256": sha256_array(encoded),
            "evaluator_label_reads": 1,
        })
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    audit = {
        "schema": "night19c-placenta-evaluator-v1", "lane": "PLACENTA_K10",
        "artifact_verified_before_annotation_access": True,
        "partition_count": len(rows), "n_evaluated": int(mask.sum()), "reference_k": 10,
        "reference_path": str(Path(args.reference).resolve()),
        "reference_sha256": file_sha256(args.reference),
        "ordered_label_sha256": sha256_array(np.asarray(labels[mask]).astype("U")),
        "producer_label_reads": 0, "evaluator_label_reads": 1,
        "carrier_discovered_keys": discovered, "carrier_accessed_keys": accessed,
        "metrics_sha256": file_sha256(output),
        "wall_seconds": float(time.time() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".evaluator.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
