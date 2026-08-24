#!/usr/bin/env python3
"""Label-free diagnostic for frozen melanoma KMeans starts; never selects formal output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def run(args: argparse.Namespace) -> None:
    with np.load(args.carrier, allow_pickle=False) as archive:
        retained = np.asarray(archive["retained"], dtype=np.float64)
        identifiers = np.asarray(archive["start_ids"]).astype("U")
        partitions = np.asarray(archive["start_partitions"], dtype=np.int32)
    selected = [index for index, value in enumerate(identifiers) if value.startswith("KMEANS_RETAINED_S")]
    rows = []
    for index in selected:
        labels = partitions[index]
        centers = np.stack([retained[labels == group].mean(axis=0) for group in np.unique(labels)])
        inertia = float(np.sum((retained - centers[labels]) ** 2))
        peers = [
            float(adjusted_rand_score(labels, partitions[other]))
            for other in selected
            if other != index
        ]
        rows.append(
            {
                "start_id": str(identifiers[index]),
                "inertia": inertia,
                "silhouette": float(silhouette_score(retained, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(retained, labels)),
                "davies_bouldin": float(davies_bouldin_score(retained, labels)),
                "mean_pairwise_partition_ari": float(np.mean(peers)),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selectors = {
        "minimum_inertia": min(rows, key=lambda row: (row["inertia"], row["start_id"]))["start_id"],
        "maximum_silhouette": max(rows, key=lambda row: (row["silhouette"], row["start_id"]))["start_id"],
        "maximum_calinski_harabasz": max(rows, key=lambda row: (row["calinski_harabasz"], row["start_id"]))["start_id"],
        "minimum_davies_bouldin": min(rows, key=lambda row: (row["davies_bouldin"], row["start_id"]))["start_id"],
        "maximum_partition_centrality": max(rows, key=lambda row: (row["mean_pairwise_partition_ari"], row["start_id"]))["start_id"],
    }
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "night16f-melanoma-label-free-start-diagnostic-v1",
                "formal_selector_changed": False,
                "diagnostic_only": True,
                "candidate_count": len(rows),
                "criterion_winners": selectors,
                "all_criteria_select_same_start": len(set(selectors.values())) == 1,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
