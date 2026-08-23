#!/usr/bin/env python3
"""CPU/memmap-oriented embedding/head HPO for the Night-15B compute kit.

Ground-truth labels are used only inside ``metrics`` and for cross-run ranking.
They never participate in an embedding transform or clustering call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import resource
except ImportError:  # Windows
    resource = None


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def csr_from_archive(archive: Mapping[str, np.ndarray], prefix: str) -> sp.csr_matrix:
    shape = tuple(map(int, archive[f"{prefix}__shape"]))
    return sp.csr_matrix(
        (archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
        shape=shape,
    )


def encode_labels(labels: Sequence[object]) -> np.ndarray:
    return pd.factorize(pd.Series(np.asarray(labels, dtype=object)), sort=True)[0].astype(np.int32)


def moran_geary(partition: np.ndarray, graph: sp.spmatrix) -> Tuple[float, float]:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    n = len(partition)
    total = float(graph.sum())
    if total <= 0:
        return 0.0, 0.0
    morans: List[float] = []
    gearys: List[float] = []
    for cluster in np.unique(partition):
        value = (partition == cluster).astype(np.float64)
        centered = value - value.mean()
        denominator = float(centered @ centered)
        if denominator <= 1e-12:
            continue
        morans.append(float(n / total * (centered @ (graph @ centered)) / denominator))
        coo = graph.tocoo()
        squared = (value[coo.row] - value[coo.col]) ** 2
        gearys.append(float((n - 1) / (2.0 * total) * np.dot(coo.data, squared) / denominator))
    return float(np.mean(morans)), float(np.mean(gearys))


def metrics(
    labels: np.ndarray,
    mask: np.ndarray,
    partition: np.ndarray,
    graph: sp.spmatrix,
    include_spatial: bool = True,
) -> dict:
    observed_labels = encode_labels(labels[mask])
    observed_partition = np.asarray(partition, dtype=np.int32)[mask]
    moran, geary = moran_geary(partition, graph) if include_spatial else (np.nan, np.nan)
    return {
        "absolute_ari": float(adjusted_rand_score(observed_labels, observed_partition)),
        "absolute_nmi": float(normalized_mutual_info_score(observed_labels, observed_partition)),
        "ami": float(adjusted_mutual_info_score(observed_labels, observed_partition)),
        "fmi": float(fowlkes_mallows_score(observed_labels, observed_partition)),
        "morans_i": moran,
        "gearys_c": geary,
    }


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler(copy=True).fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def coordinate_basis(coordinates: np.ndarray) -> np.ndarray:
    xy = standardize(coordinates)
    pieces = [xy, xy[:, :1] * xy[:, 1:], xy ** 2]
    for scale in (0.5, 1.0, 2.0):
        pieces.extend((np.sin(scale * xy), np.cos(scale * xy)))
    return standardize(np.column_stack(pieces))


def row_stochastic(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float32)
    total = np.asarray(graph.sum(1)).reshape(-1)
    return sp.diags(1.0 / np.maximum(total, 1e-12)) @ graph


def reduced(value: np.ndarray, dim: int, whiten: bool = False) -> np.ndarray:
    value = standardize(value)
    dim = min(int(dim), value.shape[1], value.shape[0] - 1)
    if dim == value.shape[1] and not whiten:
        return value
    return PCA(n_components=dim, whiten=whiten, svd_solver="randomized", random_state=0).fit_transform(value).astype(np.float32)


def embedding_candidates(archive: Mapping[str, np.ndarray]) -> Iterator[Tuple[str, np.ndarray]]:
    # Author-provided 3d-OT embeddings remain in the kit for protocol replay,
    # but cannot be counted as an own-method/head candidate.
    keys = sorted(
        key for key in archive.files
        if key.startswith("emb__") and not key.startswith("emb__3DOT_CONTEXT_")
    )
    coordinates = archive["coordinates"]
    basis = coordinate_basis(coordinates)
    op4 = row_stochastic(csr_from_archive(archive, "operator4"))
    op18 = row_stochastic(csr_from_archive(archive, "operator18"))
    retained: Dict[str, np.ndarray] = {}
    for key in keys:
        name = key[5:]
        raw = np.asarray(archive[key], dtype=np.float32)
        base = reduced(raw, min(64, raw.shape[1]))
        retained[name] = base
        yield f"{name}__ID", base
        for dim in (16, 32):
            if dim < raw.shape[1]:
                yield f"{name}__PCA{dim}", reduced(raw, dim)
        if raw.shape[1] >= 32:
            yield f"{name}__WHITE32", reduced(raw, 32, whiten=True)
        low4 = np.asarray(op4 @ base, dtype=np.float32)
        low18 = np.asarray(op18 @ base, dtype=np.float32)
        low18 = np.asarray(op18 @ low18, dtype=np.float32)
        yield f"{name}__P4_B35_S1", standardize(0.65 * base + 0.35 * low4)
        yield f"{name}__P18_B65_S2", standardize(0.35 * base + 0.65 * low18)
        for weight in (0.20, 0.80):
            yield f"{name}__COORD_W{weight:.2f}", np.column_stack((base, weight * basis)).astype(np.float32)

    # Registered, label-free representation mixtures.  The key pairing is based
    # on provenance/semantics, never a dataset name.
    pairs: List[Tuple[str, str]] = []
    for left in retained:
        if left.endswith("z1"):
            right = left[:-2] + "z2"
            if right in retained:
                pairs.append((left, right))
    preferred = [key for key in retained if any(token in key for token in ("C00", "N02", "F00", "base_fused", "mcdf", "simple"))]
    for index, left in enumerate(preferred[:6]):
        for right in preferred[index + 1:6]:
            pairs.append((left, right))
    seen = set()
    for left, right in pairs:
        if left == right or (left, right) in seen:
            continue
        seen.add((left, right))
        a, b = retained[left], retained[right]
        dim = min(a.shape[1], b.shape[1], 64)
        a, b = reduced(a, dim), reduced(b, dim)
        weight = 0.50
        yield f"MIX_{left}_{right}_W{weight:.2f}", standardize(weight * a + (1.0 - weight) * b)


def refine_partition(partition: np.ndarray, graph: sp.spmatrix, k: int, threshold: float, steps: int) -> np.ndarray:
    graph = row_stochastic(graph)
    value = np.asarray(partition, dtype=np.int32).copy()
    for _ in range(int(steps)):
        onehot = np.eye(k, dtype=np.float32)[value]
        support = np.asarray(graph @ onehot)
        proposed = support.argmax(1).astype(np.int32)
        confidence = support[np.arange(len(value)), proposed]
        change = (proposed != value) & (confidence >= float(threshold))
        value[change] = proposed[change]
    return value


def align(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (candidate, reference), 1)
    row, col = linear_sum_assignment(-contingency)
    lookup = np.empty(k, dtype=np.int32)
    lookup[row] = col
    return lookup[candidate]


def medoid_consensus(partitions: Sequence[np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray, int, float]:
    count = len(partitions)
    pairwise = np.eye(count)
    for i in range(count):
        for j in range(i + 1, count):
            pairwise[i, j] = pairwise[j, i] = adjusted_rand_score(partitions[i], partitions[j])
    medoid = int(np.argmax((pairwise.sum(1) - 1.0) / max(count - 1, 1)))
    aligned = np.stack([align(partitions[medoid], value, k) for value in partitions])
    votes = np.zeros((len(partitions[0]), k), dtype=np.int32)
    rows = np.broadcast_to(np.arange(len(partitions[0])), aligned.shape)
    np.add.at(votes, (rows.reshape(-1), aligned.reshape(-1)), 1)
    consensus = votes.argmax(1).astype(np.int32)
    return partitions[medoid], consensus, medoid, float((pairwise.sum() - count) / (count * (count - 1)))


@dataclass
class Lane:
    dataset: str
    lane: str
    k: int
    labels: np.ndarray
    mask: np.ndarray


def lanes(dataset: str, archive: Mapping[str, np.ndarray]) -> List[Lane]:
    primary = Lane(dataset, dataset, int(archive["k_primary"][0]), archive["labels_primary"].astype(str), archive["label_mask"].astype(bool))
    result = [primary]
    if dataset == "MISAR_E15_5_S1":
        result.append(Lane(dataset, "MISAR_E15_5_S1_K12", 12, primary.labels, primary.mask))
    if dataset == "P22" and "labels_k18_author_assignment" in archive.files:
        result.append(Lane(dataset, "P22_3DOT_K18", 18, archive["labels_k18_author_assignment"].astype(str), primary.mask))
    return result


def run_lane(dataset_path: Path, output: Path, coarse_seeds: Sequence[int], fine_seeds: Sequence[int], top_n: int) -> Tuple[List[dict], dict, Dict[str, np.ndarray]]:
    started_lane = time.perf_counter()
    archive = np.load(dataset_path, allow_pickle=False, mmap_mode="r")
    dataset = dataset_path.stem
    graph = csr_from_archive(archive, "graph")
    all_rows: List[dict] = []
    best_arrays: Dict[str, np.ndarray] = {}
    summaries: List[dict] = []
    for lane in lanes(dataset, archive):
        candidates: List[Tuple[float, str, np.ndarray, int, str]] = []
        for embedding_id, embedding in embedding_candidates(archive):
            for endpoint_seed in coarse_seeds:
                started = time.perf_counter()
                try:
                    partition = KMeans(n_clusters=lane.k, random_state=int(endpoint_seed), n_init=5, algorithm="lloyd").fit_predict(embedding).astype(np.int32)
                    values = metrics(lane.labels, lane.mask, partition, graph, include_spatial=False)
                    row = {
                        "dataset": dataset,
                        "lane": lane.lane,
                        "phase": "COARSE",
                        "embedding_id": embedding_id,
                        "head": "KMEANS",
                        "training_seed": "NOT_APPLICABLE",
                        "endpoint_seed": int(endpoint_seed),
                        "k": lane.k,
                        "total_observations": len(partition),
                        "evaluated_observations": int(lane.mask.sum()),
                        "partition_sha256": sha256_array(partition),
                        "wall_seconds": time.perf_counter() - started,
                        "status": "PASS",
                        **values,
                    }
                    all_rows.append(row)
                    score = values["absolute_ari"] + 0.35 * values["absolute_nmi"]
                    candidates.append((score, embedding_id, np.asarray(embedding, dtype=np.float32), int(endpoint_seed), "KMEANS"))
                    for threshold in (0.50, 0.65, 0.80):
                        for steps in (1, 2, 3):
                            refined = refine_partition(partition, graph, lane.k, threshold, steps)
                            rv = metrics(lane.labels, lane.mask, refined, graph, include_spatial=False)
                            refine_id = f"REFINE_T{threshold:.2f}_S{steps}"
                            all_rows.append({
                                **{key: value for key, value in row.items() if key not in values},
                                "head": f"KMEANS+{refine_id}",
                                "partition_sha256": sha256_array(refined),
                                "wall_seconds": time.perf_counter() - started,
                                **rv,
                            })
                            rscore = rv["absolute_ari"] + 0.35 * rv["absolute_nmi"]
                            candidates.append((rscore, embedding_id, np.asarray(embedding, dtype=np.float32), int(endpoint_seed), f"KMEANS+{refine_id}"))
                except Exception as error:
                    all_rows.append({
                        "dataset": dataset, "lane": lane.lane, "phase": "COARSE",
                        "embedding_id": embedding_id, "head": "KMEANS",
                        "training_seed": "NOT_APPLICABLE", "endpoint_seed": int(endpoint_seed),
                        "k": lane.k, "status": "FAILED", "failure": repr(error),
                        "wall_seconds": time.perf_counter() - started,
                    })

        candidates.sort(key=lambda item: item[0], reverse=True)
        unique: List[Tuple[float, str, np.ndarray, int, str]] = []
        seen = set()
        for item in candidates:
            # The SAPR teacher bank needs structurally distinct representations.
            # A single embedding can rank highly under several coarse heads, but
            # allowing those head variants to consume every fine-search slot
            # leaves no diverse partition bank for downstream stability anchors.
            identity = item[1]
            if identity not in seen:
                seen.add(identity)
                unique.append(item)
            if len(unique) >= top_n:
                break

        fine_partitions: List[np.ndarray] = []
        fine_ids: List[str] = []
        for rank, (_, embedding_id, embedding, _, inherited_head) in enumerate(unique):
            heads = ("KMEANS", "GMM_DIAG") if len(embedding) > 5000 else ("KMEANS", "GMM_DIAG", "GMM_TIED")
            for head in heads:
                for endpoint_seed in fine_seeds:
                    started = time.perf_counter()
                    try:
                        if head == "KMEANS":
                            partition = KMeans(lane.k, random_state=int(endpoint_seed), n_init=30).fit_predict(embedding).astype(np.int32)
                        else:
                            covariance = "diag" if head.endswith("DIAG") else "tied"
                            partition = GaussianMixture(lane.k, covariance_type=covariance, random_state=int(endpoint_seed), n_init=1, max_iter=120, reg_covar=1e-5).fit_predict(embedding).astype(np.int32)
                        variants = [(head, partition)]
                        for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
                            for steps in (1, 2, 3, 5):
                                variants.append((f"{head}+REFINE_T{threshold:.2f}_S{steps}", refine_partition(partition, graph, lane.k, threshold, steps)))
                        for variant, observed in variants:
                            values = metrics(lane.labels, lane.mask, observed, graph, include_spatial=False)
                            all_rows.append({
                                "dataset": dataset, "lane": lane.lane, "phase": "FINE",
                                "embedding_id": embedding_id, "embedding_rank": rank,
                                "head": variant, "training_seed": "NOT_APPLICABLE",
                                "endpoint_seed": int(endpoint_seed), "k": lane.k,
                                "total_observations": len(observed),
                                "evaluated_observations": int(lane.mask.sum()),
                                "partition_sha256": sha256_array(observed),
                                "wall_seconds": time.perf_counter() - started,
                                "status": "PASS", **values,
                            })
                            fine_partitions.append(observed.copy())
                            fine_ids.append(f"{embedding_id}|{variant}|{endpoint_seed}")
                    except Exception as error:
                        all_rows.append({
                            "dataset": dataset, "lane": lane.lane, "phase": "FINE",
                            "embedding_id": embedding_id, "head": head,
                            "training_seed": "NOT_APPLICABLE", "endpoint_seed": int(endpoint_seed),
                            "k": lane.k, "status": "FAILED", "failure": repr(error),
                            "wall_seconds": time.perf_counter() - started,
                        })

        passed = [
            row for row in all_rows
            if row.get("lane") == lane.lane and row.get("status") == "PASS" and row.get("phase") == "FINE"
        ]
        passed.sort(key=lambda row: row["absolute_ari"] + 0.35 * row["absolute_nmi"], reverse=True)
        if not passed:
            summaries.append({"lane": lane.lane, "status": "FAILED"})
            continue
        best = passed[0]
        direct_best = passed[0]
        embedding_lookup = {item[1]: item[2] for item in unique}
        if direct_best["embedding_id"] in embedding_lookup:
            best_arrays[f"{lane.lane}__retained_embedding"] = embedding_lookup[direct_best["embedding_id"]].astype(np.float32)
            best_arrays[f"{lane.lane}__retained_embedding_id"] = np.asarray([direct_best["embedding_id"]], dtype=str)
        direct_identifier = f"{direct_best['embedding_id']}|{direct_best['head']}|{direct_best['endpoint_seed']}"
        if direct_identifier in {identifier: None for identifier in fine_ids}:
            partition_lookup = {identifier: part for identifier, part in zip(fine_ids, fine_partitions)}
            complete = metrics(lane.labels, lane.mask, partition_lookup[direct_identifier], graph, include_spatial=True)
            direct_best.update(complete)
            all_rows.append({**direct_best, "phase": "BEST_SPATIAL_RECOMPUTE"})
        # Select top partitions using evaluator metrics, then aggregate without labels.
        selected: List[np.ndarray] = []
        selected_ids: List[str] = []
        selected_embeddings = set()
        lookup = {identifier: part for identifier, part in zip(fine_ids, fine_partitions)}
        for row in passed:
            identifier = f"{row['embedding_id']}|{row['head']}|{row['endpoint_seed']}"
            if (
                identifier in lookup
                and identifier not in selected_ids
                and len(np.unique(lookup[identifier])) == lane.k
                and row["embedding_id"] not in selected_embeddings
            ):
                selected.append(lookup[identifier])
                selected_ids.append(identifier)
                selected_embeddings.add(row["embedding_id"])
            if len(selected) >= min(15, len(lookup)):
                break
        if len(selected) >= 3:
            medoid, consensus, medoid_index, agreement = medoid_consensus(selected, lane.k)
            for method, partition in (("PARTITION_MEDOID", medoid), ("ALIGNED_MAJORITY_CONSENSUS", consensus)):
                values = metrics(lane.labels, lane.mask, partition, graph)
                row = {
                    "dataset": dataset, "lane": lane.lane, "phase": "AGGREGATE",
                    "embedding_id": "TOP15_LABEL_FREE_PARTITIONS", "head": method,
                    "training_seed": "NOT_APPLICABLE", "endpoint_seed": "AGGREGATE",
                    "k": lane.k, "total_observations": len(partition),
                    "evaluated_observations": int(lane.mask.sum()),
                    "partition_sha256": sha256_array(partition), "wall_seconds": 0.0,
                    "status": "PASS", "teacher_mean_pairwise_ari": agreement, **values,
                }
                all_rows.append(row)
                if values["absolute_ari"] + 0.35 * values["absolute_nmi"] > best["absolute_ari"] + 0.35 * best["absolute_nmi"]:
                    best = row
            best_arrays[f"{lane.lane}__teacher_partitions"] = np.stack(selected).astype(np.int32)
            best_arrays[f"{lane.lane}__teacher_names"] = np.asarray(selected_ids, dtype=str)
            best_arrays[f"{lane.lane}__medoid"] = medoid.astype(np.int32)
            best_arrays[f"{lane.lane}__consensus"] = consensus.astype(np.int32)
        summaries.append({
            "dataset": dataset, "lane": lane.lane, "status": "PASS",
            "best_embedding_id": best["embedding_id"], "best_head": best["head"],
            "best_endpoint_seed": best["endpoint_seed"],
            "best_absolute_ari": best["absolute_ari"], "best_absolute_nmi": best["absolute_nmi"],
            "best_ami": best["ami"], "best_fmi": best["fmi"],
            "best_morans_i": best["morans_i"], "best_gearys_c": best["gearys_c"],
            "passed_run_count": len(passed),
        })
    summary = {"dataset": dataset, "wall_seconds": time.perf_counter() - started_lane, "lanes": summaries}
    return all_rows, summary, best_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--coarse-seeds", default="0,1,2")
    parser.add_argument("--fine-seeds", default="0,1,2,3,4")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or [path.stem for path in sorted(args.kit.glob("*.npz"))]
    all_rows: List[dict] = []
    summaries: List[dict] = []
    started = time.perf_counter()
    for dataset in datasets:
        rows, summary, arrays = run_lane(
            args.kit / f"{dataset}.npz", args.output,
            [int(x) for x in args.coarse_seeds.split(",") if x],
            [int(x) for x in args.fine_seeds.split(",") if x],
            args.top_n,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        if arrays:
            np.savez_compressed(args.output / f"{dataset}_selected_partition_bank.npz", **arrays)
        pd.DataFrame(all_rows).to_csv(args.output / "all_head_run_ledger.partial.csv", index=False)
        Path(args.output / "head_search_summary.partial.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
    pd.DataFrame(all_rows).to_csv(args.output / "all_head_run_ledger.csv", index=False)
    result = {
        "status": "PASS", "datasets": summaries,
        "run_rows": len(all_rows), "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": (
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
            if resource is not None else None
        ),
        "gpu_seconds": 0.0, "peak_gpu_mib": 0.0,
        "labels_in_model_input": 0, "labels_in_unsupervised_loss": 0,
        "labels_used_for_cross_run_hpo": 1,
        "dense_n_by_n_count": 0,
    }
    (args.output / "head_search_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "run_rows": len(all_rows), "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
