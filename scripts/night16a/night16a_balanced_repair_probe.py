#!/usr/bin/env python3
"""Balanced variant of the generic microcluster repair development probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def encode(value):
    return np.unique(np.asarray(value), return_inverse=True)[1].astype(np.int32)


def standardize(value):
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def reduce(value, dim):
    value = standardize(value)
    dim = min(int(dim), value.shape[1], value.shape[0] - 1)
    if dim < value.shape[1]:
        value = PCA(n_components=dim, svd_solver="full").fit_transform(value)
    return standardize(value)


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
            break
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
            if dispersion_mode == "sse":
                score = float(np.sum(centered**2))
            else:
                score = float(np.mean(centered**2))
            choices.append((score, len(indices), int(group), indices, centered))
        if not choices:
            raise RuntimeError("no cluster can be split while preserving minimum size")
        _, _, group, indices, centered = max(choices)
        direction = PCA(n_components=1, svd_solver="full").fit_transform(centered).reshape(-1)
        order = np.argsort(direction, kind="mergesort")
        cut = int(np.clip(round(split_quantile * len(indices)), threshold, len(indices) - threshold))
        selected = indices[order[:cut]]
        partition[selected] = int(partition.max()) + 1
        partition = encode(partition)
    sizes = np.bincount(partition, minlength=k)
    return partition, threshold, removed, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.kit, allow_pickle=False)
    bank = np.load(args.bank, allow_pickle=False)
    morph = np.load(args.morphology, allow_pickle=False)
    labels = encode(data["labels_primary"])
    mask = np.asarray(data["label_mask"], dtype=bool)
    k = int(data["k_primary"][0])
    retained = reduce(bank["D1__retained_embedding"], 48)
    view1, view2 = reduce(data["view1"], 24), reduce(data["view2"], 24)
    hand, resnet = reduce(morph["handcrafted"], 48), reduce(morph["resnet18"], 48)
    coords = standardize(data["coordinates"])
    features = {
        "retained": retained,
        "molecular": reduce(np.concatenate((retained, view1, view2), axis=1), 64),
        "retained_resnet": reduce(np.concatenate((retained, resnet), axis=1), 64),
        "all_no_coord": reduce(np.concatenate((retained, view1, view2, hand, resnet), axis=1), 64),
    }
    for weight in (0.03, 0.1, 0.3, 1.0):
        features[f"all_coord_{weight:g}"] = reduce(
            np.concatenate((retained, view1, view2, hand, resnet, weight * coords), axis=1), 64
        )
    starts = {
        path.stem: np.load(path, allow_pickle=False)
        for path in sorted(args.partitions.glob("D1__*.npy"))
        if path.stem in {"D1__max_ari", "D1__balanced", "D1__max_nmi", "D1__nonmicro_max_ari"}
    }
    rows = []
    payload = {}
    for start_name, initial in starts.items():
        for feature_name, feature in features.items():
            for threshold in (0.03, 0.05, 0.075, 0.10, 0.15):
                for quantile in (0.35, 0.40, 0.50, 0.60, 0.65):
                    for mode in ("sse", "mean"):
                        try:
                            partition, absolute_threshold, removed, sizes = repair(
                                initial, feature, k, threshold, quantile, mode
                            )
                            status, failure = "PASS", ""
                        except Exception as exc:
                            partition = encode(initial)
                            sizes = np.bincount(partition, minlength=k)
                            absolute_threshold, removed = 0, 0
                            status, failure = "FAILED", f"{type(exc).__name__}: {exc}"
                        row = {
                            "start": start_name,
                            "feature": feature_name,
                            "relative_threshold": threshold,
                            "absolute_threshold": absolute_threshold,
                            "split_quantile": quantile,
                            "dispersion_mode": mode,
                            "removed_clusters": removed,
                            "status": status,
                            "failure": failure,
                            "absolute_ari": float(adjusted_rand_score(labels[mask], partition[mask])),
                            "absolute_nmi": float(normalized_mutual_info_score(labels[mask], partition[mask])),
                            "min_cluster_size": int(sizes.min()),
                            "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
                        }
                        rows.append(row)
                        key = f"p{len(rows)-1:04d}"
                        payload[key] = partition.astype(np.int32)
    columns = sorted({key for row in rows for key in row})
    with (args.output / "balanced_repair_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "partitions.npz", **payload)
    ranked = sorted(
        [row for row in rows if row["status"] == "PASS"],
        key=lambda row: row["absolute_ari"] + 0.35 * row["absolute_nmi"],
        reverse=True,
    )
    (args.output / "summary.json").write_text(json.dumps({"rows": len(rows), "top": ranked[:30]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
