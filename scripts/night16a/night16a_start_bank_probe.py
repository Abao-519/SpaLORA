#!/usr/bin/env python3
"""Build a label-independent multi-start bank and audit selector features.

All candidate partitions and selector features are created before the public
reference vector is accessed by ``evaluate_reference``.  The evaluator is kept
in this development-only runner so that correlations can be studied without
pretending that an oracle configuration is an automatic output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


LANE_DATASET = {
    "A1": "A1",
    "D1": "D1",
    "tonsil_s1": "tonsil_s1",
    "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3",
    "P22": "P22",
    "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1",
    "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
}

LANE_K = {
    "A1": 10,
    "D1": 10,
    "tonsil_s1": 4,
    "tonsil_s2": 4,
    "tonsil_s3": 4,
    "P22": 9,
    "P22_3DOT_K18": 18,
    "MISAR_E15_5_S1": 7,
    "MISAR_E15_5_S1_K12": 12,
}


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def csr_from_archive(archive, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            archive[f"{prefix}__data"],
            archive[f"{prefix}__indices"],
            archive[f"{prefix}__indptr"],
        ),
        shape=tuple(map(int, archive[f"{prefix}__shape"])),
        dtype=np.float32,
    )


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value, dtype=str), return_inverse=True)[1].astype(np.int32)


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler(copy=True).fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def reduce_full(value: np.ndarray, dim: int) -> np.ndarray:
    value = standardize(value)
    dim = min(int(dim), value.shape[1], value.shape[0] - 1)
    if dim < value.shape[1]:
        value = PCA(n_components=dim, svd_solver="full").fit_transform(value)
    return standardize(value)


def lane_semantics(data, lane: str):
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"]), np.ones(len(data["ids"]), dtype=bool)
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], dtype=bool)
    return int(np.asarray(data["k_primary"])[0]), np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], dtype=bool)


def assigned_separation(value: np.ndarray, partition: np.ndarray, k: int) -> tuple[float, float]:
    value = np.asarray(value, dtype=np.float32)
    partition = np.asarray(partition, dtype=np.int32)
    global_center = value.mean(axis=0, keepdims=True)
    total = float(np.mean(np.sum((value - global_center) ** 2, axis=1)))
    centers = []
    for group in range(k):
        members = value[partition == group]
        if not len(members):
            return -1.0, -1.0
        centers.append(members.mean(axis=0))
    centers = np.asarray(centers, dtype=np.float32)
    distance = np.mean((value[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assigned = distance[np.arange(len(value)), partition]
    nearest = np.partition(distance, 1, axis=1)[:, :2]
    margin = (nearest[:, 1] - nearest[:, 0]) / np.maximum(nearest[:, 1], 1e-6)
    separation = 1.0 - float(np.mean(assigned)) / max(total / max(value.shape[1], 1), 1e-8)
    return float(separation), float(np.mean(margin))


def graph_same_fraction(graph: sp.spmatrix, partition: np.ndarray) -> float:
    coo = sp.triu(sp.csr_matrix(graph), k=1, format="coo")
    if not coo.nnz:
        return 0.0
    return float(np.average(partition[coo.row] == partition[coo.col], weights=np.maximum(coo.data, 1e-8)))


def cluster_balance(partition: np.ndarray, k: int) -> tuple[float, float, list[int]]:
    sizes = np.bincount(partition, minlength=k).astype(np.int64)
    probability = sizes / max(int(sizes.sum()), 1)
    positive = probability[probability > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / max(float(np.log(k)), 1e-8)
    return entropy, float(sizes.min() / max(len(partition) / k, 1.0)), sizes.tolist()


def evaluate_reference(labels, mask, partition) -> dict[str, float]:
    truth = encode(np.asarray(labels)[mask])
    observed = np.asarray(partition, dtype=np.int32)[mask]
    return {
        "absolute_ari": float(adjusted_rand_score(truth, observed)),
        "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
        "ami": float(adjusted_mutual_info_score(truth, observed)),
        "fmi": float(fowlkes_mallows_score(truth, observed)),
    }


def unique_partition(candidates: list[tuple[str, np.ndarray]], name: str, partition: np.ndarray, k: int) -> None:
    partition = encode(np.asarray(partition))
    if len(np.unique(partition)) != k:
        return
    digest = array_sha256(partition)
    if any(array_sha256(existing) == digest for _, existing in candidates):
        return
    candidates.append((name, partition))


def load_existing_partitions(args, lane: str, k: int, n: int) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    dataset = LANE_DATASET[lane]
    bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
    prefix = lane
    for name, partition in zip(bank[f"{prefix}__teacher_names"], bank[f"{prefix}__teacher_partitions"]):
        unique_partition(candidates, f"teacher::{str(name)}", partition, k)
    unique_partition(candidates, "bank::medoid", bank[f"{prefix}__medoid"], k)
    unique_partition(candidates, "bank::consensus", bank[f"{prefix}__consensus"], k)
    night15f = args.night15f_partitions / f"{lane}.npy"
    if night15f.exists():
        unique_partition(candidates, "authority::night15f", np.load(night15f, allow_pickle=False), k)

    profile_locations = {
        "A1": args.night15g_work / "profile_replay_rev3_a1" / "partitions",
        "D1": args.night15g_work / "morphology_head_replay_d1_rev1" / "partitions",
        "tonsil_s3": args.night15g_work / "profile_replay_rev3_tonsil" / "partitions",
    }
    location = profile_locations.get(lane)
    if location and location.exists():
        for path in sorted(location.glob("*.npy")):
            partition = np.load(path, allow_pickle=False)
            if len(partition) == n:
                unique_partition(candidates, f"night15g::{path.stem}", partition, k)
    return candidates


def feature_blocks(data, bank, lane: str, morphology_root: Path) -> dict[str, np.ndarray]:
    blocks = {
        "retained": reduce_full(bank[f"{lane}__retained_embedding"], 32),
        "view1": reduce_full(data["view1"], min(24, data["view1"].shape[1])),
        "view2": reduce_full(data["view2"], min(24, data["view2"].shape[1])),
    }
    blocks["molecular_fused"] = reduce_full(np.concatenate((blocks["view1"], blocks["view2"]), axis=1), 40)
    dataset = LANE_DATASET[lane]
    path = morphology_root / f"{dataset}_morphology_views.npz"
    if path.exists():
        morph = np.load(path, allow_pickle=False)
        blocks["morph_handcrafted"] = reduce_full(morph["handcrafted"], 32)
        blocks["morph_resnet"] = reduce_full(morph["resnet18"], 32)
        blocks["coordinates"] = standardize(data["coordinates"])
        blocks["molecular_morph_coord"] = reduce_full(
            np.concatenate(
                ((blocks["retained"] / np.sqrt(blocks["retained"].shape[1])),
                 (0.5 * blocks["morph_resnet"] / np.sqrt(blocks["morph_resnet"].shape[1])),
                 (0.03 * blocks["coordinates"] / np.sqrt(blocks["coordinates"].shape[1]))),
                axis=1,
            ),
            64,
        )
    return blocks


def generated_starts(blocks: dict[str, np.ndarray], k: int) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    for block_name, value in blocks.items():
        if block_name == "coordinates":
            continue
        for seed in (0, 1, 2):
            partition = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(value)
            unique_partition(candidates, f"kmeans::{block_name}::s{seed}", partition, k)
        for covariance in ("diag", "tied"):
            for seed in (0, 1):
                partition = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance,
                    random_state=seed,
                    n_init=3,
                    reg_covar=1e-5,
                ).fit_predict(value)
                unique_partition(candidates, f"gmm_{covariance}::{block_name}::s{seed}", partition, k)
    return candidates


def audit_candidates(
    lane: str,
    candidates: list[tuple[str, np.ndarray]],
    blocks: dict[str, np.ndarray],
    graph: sp.spmatrix,
    k: int,
) -> list[dict]:
    partitions = [partition for _, partition in candidates]
    centrality = []
    for index, partition in enumerate(partitions):
        similarity = [adjusted_rand_score(partition, other) for j, other in enumerate(partitions) if j != index]
        centrality.append(float(np.median(similarity)) if similarity else 1.0)
    rows = []
    for index, (name, partition) in enumerate(candidates):
        entropy, min_relative, sizes = cluster_balance(partition, k)
        row = {
            "lane": lane,
            "candidate": name,
            "partition_sha256": array_sha256(partition),
            "k": k,
            "cluster_sizes": json.dumps(sizes, separators=(",", ":")),
            "min_cluster_size": min(sizes),
            "normalized_cluster_entropy": entropy,
            "min_cluster_relative_to_equal": min_relative,
            "graph_same_fraction": graph_same_fraction(graph, partition),
            "partition_centrality_median_ari": centrality[index],
        }
        for block_name, value in blocks.items():
            if block_name == "coordinates":
                continue
            separation, margin = assigned_separation(value, partition, k)
            row[f"separation__{block_name}"] = separation
            row[f"margin__{block_name}"] = margin
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--night15f-partitions", type=Path, required=True)
    parser.add_argument("--night15g-work", type=Path, required=True)
    parser.add_argument("--morphology-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", nargs="*", default=list(LANE_DATASET))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_rows = []
    partitions_payload = {}
    summary = {"status": "PASS", "lanes": {}, "labels_read_after_candidate_generation": True}
    for lane in args.lanes:
        started = time.perf_counter()
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False, mmap_mode="r")
        graph = csr_from_archive(data, "graph")
        k, labels, mask = lane_semantics(data, lane)
        blocks = feature_blocks(data, bank, lane, args.morphology_root)
        candidates = load_existing_partitions(args, lane, k, len(data["ids"]))
        for name, partition in generated_starts(blocks, k):
            unique_partition(candidates, name, partition, k)
        rows = audit_candidates(lane, candidates, blocks, graph, k)
        # Public references are deliberately accessed only after the complete
        # candidate/selector-feature bank exists.
        for row, (_, partition) in zip(rows, candidates):
            row.update(evaluate_reference(labels, mask, partition))
            row["wall_seconds_lane"] = time.perf_counter() - started
        all_rows.extend(rows)
        for index, (name, partition) in enumerate(candidates):
            partitions_payload[f"{lane}__p{index:03d}"] = partition.astype(np.int32)
            partitions_payload[f"{lane}__n{index:03d}"] = np.asarray([name])
        summary["lanes"][lane] = {
            "candidate_count": len(candidates),
            "tensor_shapes": {name: list(value.shape) for name, value in blocks.items()},
            "graph_shape": list(graph.shape),
            "graph_nnz": int(graph.nnz),
            "k": k,
            "total_observations": len(data["ids"]),
            "evaluated_observations": int(mask.sum()),
            "wall_seconds": time.perf_counter() - started,
        }
    write_csv(args.output / "start_bank_probe_ledger.csv", all_rows)
    np.savez_compressed(args.output / "start_bank_partitions.npz", **partitions_payload)
    (args.output / "start_bank_probe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
