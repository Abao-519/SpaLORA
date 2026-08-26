#!/usr/bin/env python3
"""Independent post-lock evaluator for Night-19A Stage-A partitions."""

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

from SpaLORA.night17b_sfrd import csr_from_carrier, encode_partition, sha256_array
from scripts.night19a.night19a_d0_evaluator import aligned_changed_count, categorical_spatial, load_reference
from scripts.night19a.night19a_d0_producer import file_sha256


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
    artifact_path = Path(args.artifact)
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if file_sha256(artifact_path) != producer["artifact_sha256"]:
        raise ValueError("locked artifact SHA mismatch before annotation load")
    with np.load(artifact_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"])
        profile_ids = archive["profile_ids"].astype(str)
        partitions = np.asarray(archive["partitions"])
    if [sha256_array(value) for value in partitions] != producer["partition_sha256"]:
        raise ValueError("partition authority mismatch before annotation load")
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(ids, carrier["ids"]):
        raise ValueError("artifact/carrier IDs differ")
    labels, mask = load_reference(args.reference_kind, Path(args.reference), ids, args.label_key, args.mask_key)
    truth = encode_partition(labels[mask])
    if np.unique(truth).size != int(producer["k"]):
        raise ValueError("reference K differs")
    graph = csr_from_carrier(carrier, "graph0")
    anchor = encode_partition(partitions[0])
    rows = []
    for index, profile in enumerate(profile_ids):
        partition = encode_partition(partitions[index])
        prediction = encode_partition(partition[mask])
        if np.unique(partition).size != int(producer["k"]):
            raise ValueError("partition violates exact K")
        agreement, moran, geary = categorical_spatial(partition, graph)
        sizes = np.bincount(partition, minlength=int(producer["k"])).astype(int)
        rows.append({
            "lane": producer["lane"], "training_seed": int(producer["training_seed"]), "profile": profile,
            "ari": adjusted_rand_score(truth, prediction), "nmi": normalized_mutual_info_score(truth, prediction),
            "ami": adjusted_mutual_info_score(truth, prediction), "fmi": fowlkes_mallows_score(truth, prediction),
            "neighbor_agreement": agreement, "moran_macro_ovr": moran, "geary_macro_ovr": geary,
            "min_cluster_size": int(sizes.min()), "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
            "partition_sha256": sha256_array(partition), "changed_spots_vs_strong_start": aligned_changed_count(anchor, partition),
            "evaluator_label_reads": 1,
        })
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = {
        "schema": "night19a-stage-a-evaluator-v1", "lane": producer["lane"], "training_seed": int(producer["training_seed"]),
        "locked_artifact_verified_before_annotation_load": True, "reference_path": str(Path(args.reference).resolve()),
        "reference_sha256": file_sha256(Path(args.reference)), "ordered_label_sha256": sha256_array(np.asarray(labels[mask]).astype("U")),
        "n_evaluated": int(mask.sum()), "reference_k": int(np.unique(truth).size), "metrics_sha256": file_sha256(output),
        "producer_label_reads": 0, "evaluator_label_reads": 1, "wall_seconds": time.time() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output.with_suffix(".evaluator.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
