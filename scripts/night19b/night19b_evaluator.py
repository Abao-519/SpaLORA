#!/usr/bin/env python3
"""Independent post-lock evaluator for Night-19B candidates."""

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

from SpaLORA.night17b_sfrd import encode_partition
from SpaLORA.night19b_csad import sha256_array
from scripts.night19a.night19a_d0_evaluator import categorical_spatial, load_reference
from scripts.night19b.night19b_producer import csr_from_archive, file_sha256, load_numeric_carrier


def internal_edge_counts(partition: np.ndarray, graph: sp.csr_matrix, k: int):
    labels = encode_partition(partition)
    upper = sp.triu(graph, k=1).tocoo()
    counts = np.zeros(k, dtype=np.int64)
    same = labels[upper.row] == labels[upper.col]
    for cluster in labels[upper.row[same]]:
        counts[int(cluster)] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-kind", choices=("npz", "h5ad"), required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--mask-key", default="label_mask")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    artifact_path = Path(args.artifact)
    if file_sha256(artifact_path) != producer["artifact_sha256"]:
        raise ValueError("locked artifact SHA mismatch before annotation access")
    with np.load(artifact_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"])
        candidate_ids = archive["candidate_ids"].astype(str)
        partitions = np.asarray(archive["partitions"])
    if candidate_ids.tolist() != producer["candidate_ids"]:
        raise ValueError("candidate authority mismatch")
    if [sha256_array(row) for row in partitions] != producer["partition_sha256"]:
        raise ValueError("partition authority mismatch")
    carrier, discovered_keys, accessed_keys = load_numeric_carrier(Path(args.carrier), {0})
    if not np.array_equal(ids, carrier["ids"]):
        raise ValueError("ordered ID authority mismatch")
    # Annotation access begins only after every candidate and partition SHA passed.
    labels, mask = load_reference(args.reference_kind, Path(args.reference), ids, args.label_key, args.mask_key)
    truth = encode_partition(labels[mask])
    if np.unique(truth).size != int(producer["k"]):
        raise ValueError("reference K differs from registered K")
    graph0 = csr_from_archive(carrier, "graph0")
    rows = []
    for candidate_id, raw in zip(candidate_ids, partitions):
        partition = encode_partition(raw)
        if np.unique(partition).size != int(producer["k"]):
            raise ValueError("candidate violates exact K")
        prediction = encode_partition(partition[mask])
        sizes = np.bincount(partition, minlength=int(producer["k"])).astype(int)
        agreement, moran, geary = categorical_spatial(partition, graph0)
        internal = internal_edge_counts(partition, graph0, int(producer["k"]))
        profile, arm, endpoint = candidate_id.split("__")
        rows.append({
            "lane": producer["lane"], "candidate_id": candidate_id, "config_id": profile, "arm": arm,
            "endpoint_seed": int(endpoint[1:]), "ari": adjusted_rand_score(truth, prediction),
            "nmi": normalized_mutual_info_score(truth, prediction),
            "ami": adjusted_mutual_info_score(truth, prediction), "fmi": fowlkes_mallows_score(truth, prediction),
            "neighbor_agreement": agreement, "moran_macro_ovr": moran, "geary_macro_ovr": geary,
            "min_cluster_size": int(sizes.min()), "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
            "min_internal_spatial_edges": int(internal.min()),
            "internal_spatial_edges": json.dumps(internal.tolist(), separators=(",", ":")),
            "partition_sha256": sha256_array(partition), "evaluator_label_reads": 1,
        })
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = {
        "schema": "night19b-csad-evaluator-v1", "lane": producer["lane"],
        "artifact_verified_before_annotation_access": True, "candidate_count": len(rows),
        "reference_path": str(Path(args.reference).resolve()), "reference_sha256": file_sha256(Path(args.reference)),
        "ordered_label_sha256": sha256_array(np.asarray(labels[mask]).astype("U")),
        "n_evaluated": int(np.sum(mask)), "reference_k": int(np.unique(truth).size),
        "metrics_sha256": file_sha256(output), "producer_label_reads": 0, "evaluator_label_reads": 1,
        "carrier_discovered_keys": discovered_keys, "carrier_accessed_keys": accessed_keys,
        "wall_seconds": float(time.time() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".evaluator.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
