#!/usr/bin/env python3
"""Development probe for deterministic microcluster repair.

Tiny clusters are detected from N/K only, reassigned to nearest surviving
prototype, and the highest-dispersion surviving clusters are split until K is
restored.  Public references are used only after every repaired partition has
been emitted.  This is an internal score-frontier probe, not the frozen
self-calibrating selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def encode(value):
    return np.unique(np.asarray(value), return_inverse=True)[1].astype(np.int32)


def standardize(value):
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def reduce(value, dim):
    value = standardize(value)
    dim = min(dim, value.shape[1], value.shape[0] - 1)
    if dim < value.shape[1]:
        value = PCA(n_components=dim, svd_solver="full").fit_transform(value)
    return standardize(value)


def sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def repair(initial, feature, k, relative_threshold, seed):
    initial = encode(initial)
    partition = initial.copy()
    threshold = max(2, int(np.ceil(float(relative_threshold) * len(partition) / k)))
    sizes = np.bincount(partition, minlength=k)
    tiny = np.flatnonzero(sizes < threshold)
    surviving = np.flatnonzero(sizes >= threshold)
    if not len(tiny) or len(surviving) < 2:
        return partition, {"threshold": threshold, "tiny_count": len(tiny), "split_count": 0}
    centers = np.stack([feature[partition == group].mean(axis=0) for group in surviving])
    tiny_mask = np.isin(partition, tiny)
    nearest = np.argmin(
        np.mean((feature[tiny_mask, None, :] - centers[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    partition[tiny_mask] = surviving[nearest]
    partition = encode(partition)
    split_count = 0
    rng = np.random.default_rng(seed)
    while len(np.unique(partition)) < k:
        groups = np.unique(partition)
        dispersion = []
        for group in groups:
            indices = np.flatnonzero(partition == group)
            center = feature[indices].mean(axis=0)
            score = float(np.sum((feature[indices] - center) ** 2))
            dispersion.append((score, len(indices), int(group), indices))
        _, _, group, indices = max(dispersion)
        split_seed = int(rng.integers(0, 2**31 - 1))
        assignment = KMeans(n_clusters=2, random_state=split_seed, n_init=20).fit_predict(feature[indices])
        new_label = int(partition.max()) + 1
        partition[indices[assignment == 1]] = new_label
        partition = encode(partition)
        split_count += 1
    return partition, {
        "threshold": threshold,
        "tiny_count": len(tiny),
        "split_count": split_count,
    }


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
    view1 = reduce(data["view1"], 24)
    view2 = reduce(data["view2"], 24)
    handcrafted = reduce(morph["handcrafted"], 48)
    resnet = reduce(morph["resnet18"], 48)
    coordinates = standardize(data["coordinates"])
    blocks = {
        "retained": retained,
        "molecular": reduce(np.concatenate((retained, view1, view2), axis=1), 64),
        "retained_handcrafted": reduce(np.concatenate((retained, handcrafted), axis=1), 64),
        "retained_resnet": reduce(np.concatenate((retained, resnet), axis=1), 64),
        "all_no_coord": reduce(np.concatenate((retained, view1, view2, handcrafted, resnet), axis=1), 64),
    }
    for coord_weight in (0.03, 0.1, 0.3, 1.0):
        blocks[f"all_coord_{coord_weight:g}"] = reduce(
            np.concatenate((retained, view1, view2, handcrafted, resnet, coord_weight * coordinates), axis=1), 64
        )
    starts = {path.stem: np.load(path, allow_pickle=False) for path in sorted(args.partitions.glob("D1__*.npy"))}
    rows = []
    payload = {}
    for start_name, initial in starts.items():
        for block_name, feature in blocks.items():
            for threshold in (0.01, 0.02, 0.03, 0.05, 0.075, 0.1):
                for seed in range(5):
                    partition, diagnostics = repair(initial, feature, k, threshold, seed)
                    sizes = np.bincount(partition, minlength=k)
                    row = {
                        "start": start_name,
                        "feature": block_name,
                        "relative_threshold": threshold,
                        "seed": seed,
                        "absolute_ari": float(adjusted_rand_score(labels[mask], partition[mask])),
                        "absolute_nmi": float(normalized_mutual_info_score(labels[mask], partition[mask])),
                        "min_cluster_size": int(sizes.min()),
                        "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
                        "partition_sha256": sha(partition),
                        **diagnostics,
                    }
                    rows.append(row)
                    payload[row["partition_sha256"]] = partition.astype(np.int32)
    columns = sorted({key for row in rows for key in row})
    with (args.output / "microcluster_repair_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: (row["min_cluster_size"] >= 0.01 * len(labels), row["absolute_ari"] + 0.35 * row["absolute_nmi"]), reverse=True)
    keep = {row["partition_sha256"]: payload[row["partition_sha256"]] for row in ranked[:20]}
    np.savez_compressed(args.output / "top_partitions.npz", **keep)
    (args.output / "summary.json").write_text(json.dumps({"rows": len(rows), "top": ranked[:20]}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
