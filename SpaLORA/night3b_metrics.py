"""Evaluation-only spatial metrics added for Night-3B."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def symmetric_knn_adjacency(coordinates: np.ndarray, k: int) -> sp.csr_matrix:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or len(coordinates) < 2:
        raise ValueError("coordinates must contain at least two observations")
    k = min(max(1, int(k)), len(coordinates) - 1)
    indices = NearestNeighbors(n_neighbors=k + 1).fit(coordinates).kneighbors(return_distance=False)
    rows = np.repeat(np.arange(len(coordinates)), k)
    cols = indices[:, 1:].reshape(-1)
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(coordinates), len(coordinates)))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def geary_c(values: np.ndarray, adjacency) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    graph = adjacency if sp.issparse(adjacency) else sp.csr_matrix(np.asarray(adjacency, dtype=np.float64))
    graph = graph.tocsr().astype(np.float64)
    if graph.shape != (len(values), len(values)):
        raise ValueError("adjacency shape mismatch")
    total_weight = float(graph.sum())
    denominator = float(np.sum((values - values.mean()) ** 2))
    if total_weight <= 0 or denominator <= 0:
        return float("nan")
    rows, cols = graph.nonzero()
    weights = np.asarray(graph[rows, cols]).reshape(-1)
    numerator = float(np.sum(weights * (values[rows] - values[cols]) ** 2))
    return float((len(values) - 1) * numerator / (2.0 * total_weight * denominator))


def mean_one_vs_rest_geary(labels: np.ndarray, adjacency) -> Tuple[float, dict]:
    labels = np.asarray(labels)
    per_cluster = {}
    for cluster in np.unique(labels):
        value = geary_c((labels == cluster).astype(np.float64), adjacency)
        per_cluster[str(cluster)] = value
    finite = [value for value in per_cluster.values() if np.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan")), per_cluster


__all__ = ["symmetric_knn_adjacency", "geary_c", "mean_one_vs_rest_geary"]
