"""Deterministic reference metrics for Night-10A evaluation-only backfill.

This file is intentionally independent of SpaLORA training code.  It can be
copied into the server worktree or used as an oracle in tests.  It never reads
files, labels, or checkpoints by itself.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        calinski_harabasz_score,
        completeness_score,
        davies_bouldin_score,
        fowlkes_mallows_score,
        homogeneity_score,
        mutual_info_score,
        normalized_mutual_info_score,
        silhouette_score,
        v_measure_score,
    )

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised by minimal local runtimes
    SKLEARN_AVAILABLE = False


def _one_dimensional(values: np.ndarray | Iterable[object], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def supervised_clustering_metrics(
    reference_labels: np.ndarray | Iterable[object],
    predicted_labels: np.ndarray | Iterable[object],
) -> dict[str, float]:
    """Return the label-agreement panel used by recent spatial-omics papers."""

    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for supervised clustering metrics")
    reference = _one_dimensional(reference_labels, "reference_labels")
    predicted = _one_dimensional(predicted_labels, "predicted_labels")
    if reference.shape != predicted.shape:
        raise ValueError(f"label shapes differ: {reference.shape} vs {predicted.shape}")
    return {
        "ari": float(adjusted_rand_score(reference, predicted)),
        "nmi": float(normalized_mutual_info_score(reference, predicted)),
        "ami": float(adjusted_mutual_info_score(reference, predicted)),
        "mi": float(mutual_info_score(reference, predicted)),
        "fmi": float(fowlkes_mallows_score(reference, predicted)),
        "homogeneity": float(homogeneity_score(reference, predicted)),
        "completeness": float(completeness_score(reference, predicted)),
        "v_measure": float(v_measure_score(reference, predicted)),
    }


def embedding_cluster_metrics(
    embedding: np.ndarray,
    predicted_labels: np.ndarray | Iterable[object],
    *,
    silhouette_sample_size: int | None = 5000,
    random_seed: int = 0,
) -> dict[str, float]:
    """Return label-free geometry metrics on one fixed embedding.

    The optional silhouette subsample is deterministic and is only a runtime
    control. Davies-Bouldin and Calinski-Harabasz are evaluated on all rows.
    """

    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for embedding cluster metrics")
    matrix = _matrix(embedding, "embedding")
    predicted = _one_dimensional(predicted_labels, "predicted_labels")
    if len(matrix) != len(predicted):
        raise ValueError("embedding and predicted_labels row counts differ")
    cluster_count = np.unique(predicted).size
    if cluster_count < 2 or cluster_count >= len(predicted):
        raise ValueError(
            f"internal metrics require 2..n-1 clusters, got {cluster_count}"
        )
    sample_size = None
    if silhouette_sample_size is not None and len(matrix) > silhouette_sample_size:
        sample_size = int(silhouette_sample_size)
    return {
        "silhouette": float(
            silhouette_score(
                matrix,
                predicted,
                metric="euclidean",
                sample_size=sample_size,
                random_state=int(random_seed),
            )
        ),
        "davies_bouldin": float(davies_bouldin_score(matrix, predicted)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, predicted)),
    }


def _row_l2(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, np.finfo(np.float64).eps)


def _directional_retrieval(
    query: np.ndarray,
    target: np.ndarray,
    *,
    top_k: tuple[int, ...],
    block_size: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Return one-indexed paired ranks and top-k hit counts.

    A paired target has rank 1 + the number of strictly larger cosine
    similarities. Exact ties are therefore handled deterministically and
    optimistically; the tie policy is exposed here rather than left to an
    unstable sort implementation.
    """

    n = len(query)
    ranks = np.empty(n, dtype=np.int64)
    hits = {k: 0 for k in top_k}
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        similarities = query[start:stop] @ target.T
        paired = similarities[np.arange(stop - start), np.arange(start, stop)]
        block_ranks = 1 + np.sum(similarities > paired[:, None], axis=1)
        ranks[start:stop] = block_ranks
        for k in top_k:
            hits[k] += int(np.sum(block_ranks <= k))
    return ranks, hits


def paired_cross_modal_metrics(
    modality_1_embedding: np.ndarray,
    modality_2_embedding: np.ndarray,
    *,
    top_k: tuple[int, ...] = (1, 5, 10),
    block_size: int = 1024,
) -> dict[str, float]:
    """Return symmetric FOSCTTM and paired cross-modal retrieval metrics."""

    first = _row_l2(_matrix(modality_1_embedding, "modality_1_embedding"))
    second = _row_l2(_matrix(modality_2_embedding, "modality_2_embedding"))
    if first.shape != second.shape:
        raise ValueError(f"paired embedding shapes differ: {first.shape} vs {second.shape}")
    if len(first) < 2:
        raise ValueError("paired retrieval requires at least two observations")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    normalized_top_k = tuple(sorted({int(k) for k in top_k if 0 < int(k) <= len(first)}))
    if not normalized_top_k:
        raise ValueError("top_k must contain a value between 1 and n")

    rank_12, hit_12 = _directional_retrieval(
        first, second, top_k=normalized_top_k, block_size=int(block_size)
    )
    rank_21, hit_21 = _directional_retrieval(
        second, first, top_k=normalized_top_k, block_size=int(block_size)
    )
    denominator = float(len(first) - 1)
    result = {
        "foscttm_mod1_to_mod2": float(np.mean((rank_12 - 1) / denominator)),
        "foscttm_mod2_to_mod1": float(np.mean((rank_21 - 1) / denominator)),
        "foscttm_mean": float(
            0.5
            * (
                np.mean((rank_12 - 1) / denominator)
                + np.mean((rank_21 - 1) / denominator)
            )
        ),
        "median_paired_rank_mod1_to_mod2": float(np.median(rank_12)),
        "median_paired_rank_mod2_to_mod1": float(np.median(rank_21)),
    }
    for k in normalized_top_k:
        result[f"recall_at_{k}_mod1_to_mod2"] = hit_12[k] / float(len(first))
        result[f"recall_at_{k}_mod2_to_mod1"] = hit_21[k] / float(len(first))
        result[f"recall_at_{k}_mean"] = (hit_12[k] + hit_21[k]) / float(2 * len(first))
    return result
