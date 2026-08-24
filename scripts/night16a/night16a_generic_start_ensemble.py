#!/usr/bin/env python3
"""Build label-free medoid/consensus starts from fixed generic generators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def sha(value):
    value = np.ascontiguousarray(value); h = hashlib.sha256(); h.update(value.dtype.str.encode()); h.update(np.asarray(value.shape, np.int64).tobytes()); h.update(value.tobytes()); return h.hexdigest()


def align(reference, candidate, k):
    overlap = np.zeros((k, k), dtype=np.int64)
    np.add.at(overlap, (reference, candidate), 1)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.arange(k, dtype=np.int32); mapping[cols] = rows
    return mapping[candidate]


def consensus(partitions, weights, k):
    count = len(partitions)
    agreement = np.eye(count, dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            agreement[i, j] = agreement[j, i] = adjusted_rand_score(partitions[i], partitions[j])
    centrality = np.median(agreement, axis=1)
    reference_index = int(np.argmax(centrality))
    reference = partitions[reference_index]
    aligned = np.stack([align(reference, item, k) for item in partitions])
    votes = np.zeros((len(reference), k), dtype=np.float64)
    for item, weight in zip(aligned, weights):
        votes[np.arange(len(reference)), item] += float(weight)
    result = np.argmax(votes, axis=1).astype(np.int32)
    if len(np.unique(result)) != k:
        result = reference.copy()
    return result, reference, centrality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.probe / "start_bank_probe_ledger.csv")
    archive = np.load(args.probe / "start_bank_partitions.npz", allow_pickle=False)
    rows, payload = [], {}
    for lane, group in frame.groupby("lane", sort=False):
        items = []
        index = 0
        while f"{lane}__p{index:03d}" in archive:
            name = str(archive[f"{lane}__n{index:03d}"][0])
            if name.startswith(("kmeans::", "gmm_diag::", "gmm_tied::")):
                row = group[group.candidate == name].iloc[0]
                if float(row.min_cluster_relative_to_equal) >= 0.02:
                    items.append((name, np.asarray(archive[f"{lane}__p{index:03d}"], np.int32), row))
            index += 1
        subsets = {
            "ALL_GENERIC": items,
            "MOLECULAR_ONLY": [x for x in items if not any(token in x[0] for token in ("morph_handcrafted", "morph_resnet", "molecular_morph_coord"))],
            "RETAINED_ONLY": [x for x in items if "::retained::" in x[0]],
            "MULTIVIEW_MOLECULAR": [x for x in items if any(token in x[0] for token in ("::retained::", "::view1::", "::view2::", "::molecular_fused::"))],
        }
        for subset_name, subset in subsets.items():
            if len(subset) < 2:
                continue
            partitions = [x[1] for x in subset]
            descriptor = np.asarray([
                [float(x[2].graph_same_fraction), float(x[2].normalized_cluster_entropy),
                 float(x[2].min_cluster_relative_to_equal), float(x[2]["separation__retained"]),
                 float(x[2]["separation__view1"]), float(x[2]["separation__view2"])]
                for x in subset
            ])
            med = np.median(descriptor, axis=0); scale = np.maximum(1.4826 * np.median(abs(descriptor - med), axis=0), np.std(descriptor, axis=0))
            z = np.clip((descriptor - med) / np.maximum(scale, 1e-8), -5, 5)
            schemes = {
                "EQUAL": np.ones(len(subset)),
                "STRUCTURAL": np.exp(np.clip(z @ np.asarray([.15, .20, .20, .20, .125, .125]), -3, 3)),
            }
            for scheme, weights in schemes.items():
                result, medoid, centrality = consensus(partitions, weights, int(group.k.iloc[0]))
                for kind, partition in (("CONSENSUS", result), ("MEDOID", medoid)):
                    key = f"{lane}__{subset_name}__{scheme}__{kind}"
                    payload[key] = partition
                    rows.append({
                        "lane": lane, "candidate": key, "subset": subset_name, "weight_scheme": scheme,
                        "ensemble_kind": kind, "member_count": len(subset), "partition_key": key,
                        "partition_sha256": sha(partition), "median_member_centrality": float(np.median(centrality)),
                    })
    np.savez_compressed(args.output / "generic_ensemble_partitions.npz", **payload)
    with (args.output / "generic_ensemble_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "manifest.json").write_text(json.dumps({"rows": len(rows), "label_reads": 0}, indent=2) + "\n")


if __name__ == "__main__":
    main()
