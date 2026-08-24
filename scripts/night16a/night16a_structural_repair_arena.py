#!/usr/bin/env python3
"""Development-only structural repair arena for a frozen partition.

This runner is deliberately separated from the automatic consumer.  Public
references are exposed only after a repaired partition is produced and are
used to diagnose whether a generic microcluster repair is worth retaining.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


LANE_DATASET = {
    "A1": "A1", "D1": "D1", "tonsil_s1": "tonsil_s1", "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3", "P22": "P22", "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1", "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
}


def encode(value):
    return np.unique(np.asarray(value), return_inverse=True)[1].astype(np.int32)


def reduce(value, dim):
    value = StandardScaler().fit_transform(np.asarray(value, dtype=np.float32))
    dim = min(int(dim), value.shape[1], value.shape[0] - 1)
    if dim < value.shape[1]:
        value = PCA(n_components=dim, svd_solver="full").fit_transform(value)
    return StandardScaler().fit_transform(value).astype(np.float32)


def sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    h.update(value.tobytes())
    return h.hexdigest()


def repair(initial, feature, k, relative_threshold, split_quantile, dispersion_mode):
    partition = encode(initial)
    threshold = max(2, int(np.ceil(relative_threshold * len(partition) / k)))
    while True:
        sizes = np.bincount(partition, minlength=len(np.unique(partition)))
        tiny = np.flatnonzero(sizes < threshold)
        if not len(tiny):
            break
        surviving = np.flatnonzero(sizes >= threshold)
        if len(surviving) < 2:
            raise RuntimeError("fewer than two surviving clusters")
        centers = np.stack([feature[partition == group].mean(axis=0) for group in surviving])
        for group in tiny:
            indices = np.flatnonzero(partition == group)
            distance = np.mean((feature[indices, None, :] - centers[None, :, :]) ** 2, axis=2)
            partition[indices] = surviving[np.argmin(distance, axis=1)]
        partition = encode(partition)
    removed = k - len(np.unique(partition))
    while len(np.unique(partition)) < k:
        choices = []
        for group in np.unique(partition):
            indices = np.flatnonzero(partition == group)
            if len(indices) < 2 * threshold:
                continue
            centered = feature[indices] - feature[indices].mean(axis=0)
            score = float(np.sum(centered ** 2) if dispersion_mode == "sse" else np.mean(centered ** 2))
            choices.append((score, len(indices), int(group), indices, centered))
        if not choices:
            raise RuntimeError("no cluster can be split under the minimum-size rule")
        _, _, _, indices, centered = max(choices)
        direction = PCA(n_components=1, svd_solver="full").fit_transform(centered).reshape(-1)
        order = np.argsort(direction, kind="mergesort")
        cut = int(np.clip(round(split_quantile * len(indices)), threshold, len(indices) - threshold))
        partition[indices[order[:cut]]] = int(partition.max()) + 1
        partition = encode(partition)
    return partition, threshold, removed


def lane_semantics(data, lane):
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"]), np.ones(len(data["ids"]), bool)
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)
    return int(data["k_primary"][0]), np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--banks", type=Path, required=True)
    p.add_argument("--initials", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lanes", nargs="+", required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, payload = [], {}
    for lane in args.lanes:
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        initial = np.load(args.initials / f"{lane}.npy", allow_pickle=False)
        k, labels, mask = lane_semantics(data, lane)
        truth = encode(labels[mask])
        features = {
            "retained": reduce(bank[f"{lane}__retained_embedding"], 48),
            "view1": reduce(data["view1"], 32),
            "view2": reduce(data["view2"], 32),
            "molecular": reduce(np.concatenate((bank[f"{lane}__retained_embedding"], data["view1"], data["view2"]), axis=1), 64),
        }
        for feature_name, feature in features.items():
            for threshold in (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
                for quantile in (0.35, 0.40, 0.50, 0.60, 0.65):
                    for mode in ("sse", "mean"):
                        try:
                            partition, absolute_threshold, removed = repair(initial, feature, k, threshold, quantile, mode)
                            status, failure = "PASS", ""
                        except Exception as exc:
                            partition, absolute_threshold, removed = encode(initial), 0, 0
                            status, failure = "FAILED", f"{type(exc).__name__}: {exc}"
                        sizes = np.bincount(partition, minlength=k)
                        key = f"{lane}__p{len(rows):05d}"
                        payload[key] = partition.astype(np.int32)
                        rows.append({
                            "lane": lane, "feature": feature_name, "relative_threshold": threshold,
                            "absolute_threshold": absolute_threshold, "split_quantile": quantile,
                            "dispersion_mode": mode, "removed_clusters": removed, "status": status,
                            "failure": failure, "absolute_ari": float(adjusted_rand_score(truth, partition[mask])),
                            "absolute_nmi": float(normalized_mutual_info_score(truth, partition[mask])),
                            "min_cluster_size": int(sizes.min()), "cluster_sizes": json.dumps(sizes.tolist()),
                            "partition_sha256": sha(partition), "partition_key": key,
                        })
    columns = sorted({key for row in rows for key in row})
    with (args.output / "structural_repair_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output / "partitions.npz", **payload)
    (args.output / "summary.json").write_text(json.dumps({"rows": len(rows), "failed": sum(r["status"] != "PASS" for r in rows)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
