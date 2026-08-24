#!/usr/bin/env python3
"""Independent evaluator for a locked human-hippocampus producer artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from scripts.night16e.human_hippocampus_producer import ordered_id_sha256, sparse_spatial_graph
from scripts.night16e.night16e_evaluator import categorical_spatial_metrics, cluster_sizes_full_eval


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.producer_json).read_text())
    with np.load(args.partition_bank, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype(str)
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
    reference = ad.read_h5ad(args.reference)
    reference_ids = np.asarray(reference.obs_names.astype(str))
    if set(ids) != set(reference_ids):
        raise ValueError("producer/reference spot-ID sets differ")
    lookup = {identifier: index for index, identifier in enumerate(reference_ids)}
    order = np.asarray([lookup[identifier] for identifier in ids], dtype=np.int64)
    if not np.array_equal(reference_ids[order], ids):
        raise RuntimeError("explicit reference-ID alignment failed")
    if args.label_column not in reference.obs:
        raise KeyError(args.label_column)
    label_series = reference.obs.iloc[order][args.label_column]
    mask = np.asarray(label_series.notna(), dtype=bool)
    if not np.any(mask):
        raise ValueError("official reference contains no non-missing labels")
    truth = np.asarray(label_series[mask].astype(str))
    unique_labels, label_counts = np.unique(truth, return_counts=True)
    if len(unique_labels) != int(args.k):
        raise ValueError(
            f"official cleaned reference K mismatch: observed {len(unique_labels)}, expected {args.k}"
        )
    label_payload = b"\0".join(
        identifier.encode("utf-8") + b"=" + label.encode("utf-8")
        for identifier, label in zip(ids[mask], truth)
    )
    ordered_label_sha256 = hashlib.sha256(label_payload).hexdigest()
    coordinates = np.asarray(reference.obsm["spatial"], dtype=np.float64)[order]
    graph = sparse_spatial_graph(coordinates, 8)
    rows: list[dict[str, object]] = []
    for source in manifest["rows"]:
        partition = partitions[int(source["partition_index"])]
        sizes_full, sizes_eval = cluster_sizes_full_eval(partition, mask, args.k)
        moran, geary, agreement = categorical_spatial_metrics(partition, graph)
        rows.append(
            {
                "candidate_id": source["candidate_id"],
                "profile_id": source["profile_id"],
                "variant": source["variant"],
                "partition_sha256": source["partition_sha256"],
                "absolute_ari": float(adjusted_rand_score(truth, partition[mask])),
                "absolute_nmi": float(normalized_mutual_info_score(truth, partition[mask])),
                "ami": float(adjusted_mutual_info_score(truth, partition[mask])),
                "fmi": float(fowlkes_mallows_score(truth, partition[mask])),
                "homogeneity": float(homogeneity_score(truth, partition[mask])),
                "v_measure": float(v_measure_score(truth, partition[mask])),
                "morans_i_macro": moran,
                "gearys_c_macro": geary,
                "neighbor_agreement": agreement,
                "n_total": int(len(ids)),
                "n_evaluated": int(np.sum(mask)),
                "k": int(args.k),
                "cluster_sizes_full": json.dumps([int(x) for x in sizes_full]),
                "cluster_sizes_eval": json.dumps([int(x) for x in sizes_eval]),
                "min_cluster_size_full": int(sizes_full.min()),
                "min_cluster_size_eval": int(sizes_eval.min()),
                "ordered_id_sha256": ordered_id_sha256(ids),
                "ordered_label_sha256": ordered_label_sha256,
                "reference_type": "OFFICIAL_MANUAL_ANATOMICAL_GROUND_TRUTH_CARRIER",
                "reference_column": args.label_column,
                "reference_mask": (
                    "ALL_OFFICIAL_PROCESSED_PIXELS"
                    if bool(np.all(mask))
                    else "NON_MISSING_OFFICIAL_REFERENCE_ONLY"
                ),
                "evaluator_label_reads": 1,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "schema": "night16e-human-hippocampus-independent-evaluator-v1",
                "reference_type": "OFFICIAL_MANUAL_ANATOMICAL_GROUND_TRUTH_CARRIER",
                "reference_column": args.label_column,
                "n_total": int(len(ids)),
                "n_nonmissing_reference": int(np.sum(mask)),
                "missing_reference_count": int(np.sum(~mask)),
                "reference_mask": (
                    "ALL_OFFICIAL_PROCESSED_PIXELS"
                    if bool(np.all(mask))
                    else "NON_MISSING_OFFICIAL_REFERENCE_ONLY"
                ),
                "cleaned_reference_k": int(len(unique_labels)),
                "category_counts": {
                    str(label): int(count) for label, count in zip(unique_labels, label_counts)
                },
                "ordered_label_sha256": ordered_label_sha256,
                "ordered_id_sha256": ordered_id_sha256(ids),
                "evaluator_label_reads": 1,
                "producer_label_reads": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--label-column", default="true_label")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
