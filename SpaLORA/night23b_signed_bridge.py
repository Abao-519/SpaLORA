"""Carrier-preserving sparse signed-relation consumers for Night-23B."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def robust_columns(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    median = np.median(value, axis=0)
    mad = np.median(np.abs(value - median), axis=0) * 1.4826
    std = np.std(value, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return (value - median) / scale


def row_unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def _stable_node_v0(degree: np.ndarray) -> np.ndarray:
    degree = np.asarray(degree, dtype=np.float64)
    rank = (rankdata(degree, method="average") - 0.5) / max(len(degree), 1)
    output = 1.0 + 0.1 * rank
    return output / np.linalg.norm(output)


def _farthest_first_centers(value: np.ndarray, k: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    order = np.lexsort(tuple(value[:, col] for col in reversed(range(value.shape[1]))))
    selected = [int(order[0])]
    min_distance = np.sum((value - value[selected[0]]) ** 2, axis=1)
    for _ in range(1, k):
        best_distance = float(min_distance.max())
        tied = np.flatnonzero(np.isclose(min_distance, best_distance, rtol=1e-12, atol=1e-14))
        if not len(tied):
            raise RuntimeError("deterministic center selection failed")
        tied_order = np.lexsort(tuple(value[tied, col] for col in reversed(range(value.shape[1]))))
        chosen = int(tied[int(tied_order[0])])
        selected.append(chosen)
        min_distance = np.minimum(min_distance, np.sum((value - value[chosen]) ** 2, axis=1))
    return value[np.asarray(selected)].copy()


def deterministic_exact_kmeans(value: np.ndarray, k: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    centers = _farthest_first_centers(value, k)
    partition = KMeans(n_clusters=k, init=centers, n_init=1, algorithm="lloyd", random_state=2302).fit_predict(value)
    if len(np.unique(partition)) != k:
        raise RuntimeError("exact-K consumer failure")
    return partition.astype(np.int32)


def tri_state_from_thresholds(score: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64)
    if not (0 <= lower < upper <= 1) or not np.all(np.isfinite(score)):
        raise RuntimeError("invalid tri-state thresholds/scores")
    negative = score <= lower
    positive = score >= upper
    unknown = ~(negative | positive)
    if np.any(negative & positive) or not np.all(negative | positive | unknown):
        raise RuntimeError("tri-state exclusivity/completeness failure")
    return positive, negative, unknown


def signed_relation_embedding(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    positive_weight: np.ndarray,
    negative_weight: np.ndarray,
    dimensions: int,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    positive_weight = np.asarray(positive_weight, dtype=np.float64)
    negative_weight = np.asarray(negative_weight, dtype=np.float64)
    if not (len(rows) == len(cols) == len(positive_weight) == len(negative_weight)):
        raise RuntimeError("signed edge length mismatch")
    if np.any(rows >= cols) or np.any(rows < 0) or np.any(cols >= n):
        raise RuntimeError("signed edge authority mismatch")
    if np.any(positive_weight < 0) or np.any(negative_weight < 0) or not np.all(
        np.isfinite(positive_weight + negative_weight)
    ):
        raise RuntimeError("invalid signed edge weights")
    if np.any((positive_weight > 0) & (negative_weight > 0)):
        raise RuntimeError("edge cannot attract and repel simultaneously")
    signed = sp.coo_matrix((positive_weight - negative_weight, (rows, cols)), shape=(n, n))
    signed = (signed + signed.T).tocsr()
    absolute = sp.coo_matrix((positive_weight + negative_weight, (rows, cols)), shape=(n, n))
    absolute = (absolute + absolute.T).tocsr()
    degree = np.asarray(absolute.sum(axis=1)).reshape(-1)
    inv = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = sp.diags(inv) @ signed @ sp.diags(inv)
    dimensions = min(int(dimensions), n - 2)
    if dimensions <= 0 or normalized.nnz == 0:
        raise RuntimeError("signed relation embedding is empty")
    _, vectors = eigsh(
        normalized,
        k=dimensions,
        which="LA",
        tol=1e-7,
        maxiter=max(5000, n * 3),
        v0=_stable_node_v0(degree),
    )
    return row_unit(vectors)


def carrier_embedding(carrier: np.ndarray, dimensions: int = 32) -> np.ndarray:
    carrier = robust_columns(carrier)
    dimensions = min(int(dimensions), carrier.shape[1], len(carrier) - 1)
    value = PCA(n_components=dimensions, svd_solver="full").fit_transform(carrier)
    return row_unit(value)


def carrier_preserving_signed_partition(
    carrier: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    positive_weight: np.ndarray,
    negative_weight: np.ndarray,
    k: int,
    relation_scale: float,
    carrier_dimensions: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    carrier_block = carrier_embedding(carrier, dimensions=carrier_dimensions)
    if relation_scale == 0:
        fused = carrier_block
    else:
        relation_block = signed_relation_embedding(
            len(carrier_block), rows, cols, positive_weight, negative_weight, dimensions=k
        )
        fused = np.column_stack([carrier_block, float(relation_scale) * relation_block])
    partition = deterministic_exact_kmeans(fused, k)
    return partition, fused.astype(np.float32)


def signed_edge_energy(partition: np.ndarray, rows: np.ndarray, cols: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    partition = np.asarray(partition)
    same = partition[rows] == partition[cols]
    return float(np.sum(np.asarray(positive)[~same]) + np.sum(np.asarray(negative)[same]))


def edge_disagreement(partition: np.ndarray, teacher_relation: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    observed = np.asarray(partition)[rows] == np.asarray(partition)[cols]
    return float(np.mean(observed != np.asarray(teacher_relation, dtype=bool)))


def positive_connectivity_audit(
    n: int, rows: np.ndarray, cols: np.ndarray, teacher_relation: np.ndarray, teacher_partition: np.ndarray
) -> dict:
    teacher_relation = np.asarray(teacher_relation, dtype=bool)
    positive = sp.coo_matrix((np.ones(int(teacher_relation.sum())), (rows[teacher_relation], cols[teacher_relation])), shape=(n, n))
    positive = (positive + positive.T).tocsr()
    component_count, component = connected_components(positive, directed=False, return_labels=True)
    degree = np.asarray(positive.getnnz(axis=1)).reshape(-1)
    cluster_splits = {}
    cross_cluster_component = False
    for label in np.unique(teacher_partition):
        node = np.flatnonzero(teacher_partition == label)
        cluster_splits[str(label)] = int(len(np.unique(component[node])))
    for value in np.unique(component):
        if len(np.unique(teacher_partition[component == value])) > 1:
            cross_cluster_component = True
    return {
        "positive_component_count": int(component_count),
        "isolated_node_count": int(np.sum(degree == 0)),
        "teacher_cluster_split_counts": cluster_splits,
        "cross_teacher_cluster_positive_component": bool(cross_cluster_component),
    }


def recovery_metrics(teacher: np.ndarray, partition: np.ndarray, rows: np.ndarray, cols: np.ndarray, relation: np.ndarray) -> dict:
    _, counts = np.unique(partition, return_counts=True)
    return {
        "teacher_recovery_ari": float(adjusted_rand_score(teacher, partition)),
        "teacher_recovery_nmi": float(normalized_mutual_info_score(teacher, partition)),
        "edge_disagreement": edge_disagreement(partition, relation, rows, cols),
        "observed_k": int(len(counts)),
        "min_cluster_size": int(counts.min()),
        "partition_sha256": array_sha(np.asarray(partition, dtype=np.int32)),
    }

