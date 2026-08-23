"""Sparse direct-clustering primitives for Night-15C.

The model-facing functions in this module never accept dataset names or ground
truth labels.  Public labels are restricted to the evaluator in the runner.
All graph operations are sparse and scale with registered graph edges.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


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
        (
            archive[f"{prefix}__data"],
            archive[f"{prefix}__indices"],
            archive[f"{prefix}__indptr"],
        ),
        shape=shape,
        dtype=np.float32,
    )


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler(copy=True).fit_transform(
        np.asarray(value, dtype=np.float32)
    ).astype(np.float32)


def reduced(value: np.ndarray, dim: int) -> np.ndarray:
    return reduced_controlled(value, dim, solver="randomized")


def reduced_controlled(value: np.ndarray, dim: int, solver: str) -> np.ndarray:
    value = standardize(value)
    dim = min(int(dim), value.shape[1], value.shape[0] - 1)
    if solver == "none":
        return value
    if solver not in ("randomized", "full"):
        raise ValueError(solver)
    if dim < value.shape[1]:
        value = PCA(
            n_components=dim,
            random_state=0,
            svd_solver=solver,
        ).fit_transform(value)
    return standardize(value)


def symmetric_binary(graph: sp.spmatrix) -> sp.csr_matrix:
    value = sp.csr_matrix(graph, dtype=np.float32)
    value = value.maximum(value.T)
    value.setdiag(0)
    value.eliminate_zeros()
    value.data[:] = 1.0
    return value


def row_stochastic(graph: sp.spmatrix, floor: float = 1e-8) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float32)
    total = np.asarray(graph.sum(axis=1)).reshape(-1)
    return (sp.diags(1.0 / np.maximum(total, float(floor))) @ graph).tocsr()


def _edge_distance(value: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    delta = np.asarray(value[row] - value[col], dtype=np.float32)
    distance = np.einsum("ij,ij->i", delta, delta, optimize=True)
    positive = distance[distance > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    return distance / max(scale, 1e-12)


def multiscale_edge_conductance(
    graph: sp.spmatrix,
    view1: np.ndarray,
    view2: np.ndarray,
    operator4: sp.spmatrix,
    operator18: sp.spmatrix,
    mode: str = "cross_persistence",
    tau: float = 1.0,
) -> sp.csr_matrix:
    """Build sparse edge conductance from two modality views.

    ``cross_persistence`` requires both modalities to support an edge across
    raw, short-range and long-range views.  ``cross_agreement`` additionally
    vetoes modality conflict.  No dense observation-by-observation object is
    materialized.
    """

    base = symmetric_binary(graph)
    upper = sp.triu(base, k=1).tocoo()
    row, col = upper.row, upper.col
    if mode == "uniform":
        weight = np.ones(len(row), dtype=np.float32)
    else:
        p4 = row_stochastic(operator4)
        p18 = row_stochastic(operator18)
        v1 = standardize(view1)
        v2 = standardize(view2)
        scales1 = (v1, np.asarray(p4 @ v1), np.asarray(p18 @ v1))
        scales2 = (v2, np.asarray(p4 @ v2), np.asarray(p18 @ v2))
        d1 = np.stack([_edge_distance(x, row, col) for x in scales1])
        d2 = np.stack([_edge_distance(x, row, col) for x in scales2])
        s1 = np.exp(-np.mean(d1, axis=0) / max(float(tau), 1e-6))
        s2 = np.exp(-np.mean(d2, axis=0) / max(float(tau), 1e-6))
        raw1 = np.exp(-d1[0] / max(float(tau), 1e-6))
        raw2 = np.exp(-d2[0] / max(float(tau), 1e-6))
        if mode == "cross_max":
            # Either modality may preserve a true boundary-interior edge.
            weight = np.maximum(raw1, raw2)
        elif mode == "cross_min":
            weight = np.minimum(s1, s2)
        elif mode == "cross_geom":
            weight = np.sqrt(s1 * s2)
        elif mode == "cross_agreement":
            weight = np.sqrt(s1 * s2) * np.maximum(0.0, 1.0 - np.abs(s1 - s2))
        elif mode == "cross_persistence":
            raw_support = np.sqrt(np.exp(-d1[0] / tau) * np.exp(-d2[0] / tau))
            persistent = np.sqrt(s1 * s2)
            weight = np.sqrt(raw_support * persistent)
        else:
            raise ValueError(f"unknown edge mode: {mode}")
        weight = np.maximum(weight, 1e-4).astype(np.float32)
    result = sp.coo_matrix((weight, (row, col)), shape=base.shape)
    return (result + result.T).tocsr()


def raw_bimodal_edge_conductance(
    graph: sp.spmatrix,
    view1: np.ndarray,
    view2: np.ndarray,
    mode: str,
    tau: float = 1.0,
    dim: int = 16,
    solver: str = "randomized",
) -> sp.csr_matrix:
    """Raw two-modality sparse conductance used by the exact Potts audit.

    Each modality is independently standardized/PCA-reduced.  Edge similarity
    is evaluated only on registered graph edges. ``either_similar`` takes the
    maximum and ``both_similar`` the minimum of modality similarities.
    """

    value = sp.csr_matrix(graph, dtype=np.float32).tocoo(copy=True)
    value.setdiag(0)
    value.eliminate_zeros()
    first = reduced_controlled(view1, min(int(dim), view1.shape[1]), solver=solver)
    second = reduced_controlled(view2, min(int(dim), view2.shape[1]), solver=solver)
    distance1 = np.mean((first[value.row] - first[value.col]) ** 2, axis=1)
    distance2 = np.mean((second[value.row] - second[value.col]) ** 2, axis=1)
    scale1 = max(float(np.median(distance1[distance1 > 0])), 1e-6)
    scale2 = max(float(np.median(distance2[distance2 > 0])), 1e-6)
    similarity1 = np.exp(-distance1 / (scale1 * max(float(tau), 1e-6)))
    similarity2 = np.exp(-distance2 / (scale2 * max(float(tau), 1e-6)))
    if mode == "spatial":
        weight = np.ones_like(similarity1)
    elif mode == "either_similar":
        weight = np.maximum(similarity1, similarity2)
    elif mode == "both_similar":
        weight = np.minimum(similarity1, similarity2)
    elif mode == "geomean":
        weight = np.sqrt(similarity1 * similarity2)
    else:
        raise ValueError(mode)
    return sp.csr_matrix(
        (weight.astype(np.float32), (value.row, value.col)), shape=value.shape
    )


def prototype_unary(
    embedding: np.ndarray,
    partition: np.ndarray,
    k: int,
    covariance: str = "diag",
) -> np.ndarray:
    """Return label-free prototype negative log-distance unary costs."""

    value = standardize(embedding)
    labels = np.asarray(partition, dtype=np.int32)
    unary = np.empty((len(value), int(k)), dtype=np.float32)
    global_var = np.var(value, axis=0) + 1e-3
    for cluster in range(int(k)):
        members = value[labels == cluster]
        if len(members) == 0:
            unary[:, cluster] = np.finfo(np.float32).max / 100.0
            continue
        center = members.mean(axis=0)
        if covariance == "diag":
            variance = np.var(members, axis=0) + 0.10 * global_var + 1e-3
            cost = np.mean((value - center) ** 2 / variance, axis=1)
            cost += 0.05 * float(np.mean(np.log(variance)))
        elif covariance == "spherical":
            variance = float(np.mean(np.var(members, axis=0)) + 1e-3)
            cost = np.mean((value - center) ** 2, axis=1) / variance
        else:
            raise ValueError(covariance)
        unary[:, cluster] = cost.astype(np.float32)
    unary -= unary.min(axis=1, keepdims=True)
    positive = unary[unary > 1e-8]
    scale = float(np.median(positive)) if positive.size else 1.0
    return unary / max(scale, 1e-6)


def centroid_unary_row_scaled(
    embedding: np.ndarray,
    partition: np.ndarray,
    k: int,
) -> np.ndarray:
    """Centroid-distance unary normalized within each observation.

    Recomputing this quantity after each update turns ICM into an alternating
    prototype/partition energy descent.  It remains label-free.
    """

    value = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(partition, dtype=np.int32)
    centers = []
    for cluster in range(int(k)):
        members = value[labels == cluster]
        if not len(members):
            raise ValueError("empty cluster")
        centers.append(members.mean(axis=0))
    centers = np.stack(centers)
    distance = np.mean((value[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    scale = np.std(distance, axis=1, keepdims=True)
    return ((distance - distance.min(axis=1, keepdims=True)) / np.maximum(scale, 1e-6)).astype(np.float32)


def align_partition(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    contingency = np.zeros((int(k), int(k)), dtype=np.int64)
    np.add.at(contingency, (np.asarray(candidate, int), np.asarray(reference, int)), 1)
    row, col = linear_sum_assignment(-contingency)
    lookup = np.arange(int(k), dtype=np.int32)
    lookup[row] = col
    return lookup[np.asarray(candidate, dtype=np.int32)]


def partition_bank_unary(
    partitions: Sequence[np.ndarray],
    graph: sp.spmatrix,
    k: int,
    weighting: str = "agreement_spatial",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Construct sparse/local-confidence weighted ensemble unary costs."""

    parts = [np.asarray(part, dtype=np.int32) for part in partitions]
    count = len(parts)
    if count < 2:
        raise ValueError("at least two partitions are required")
    pairwise = np.eye(count, dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            pairwise[i, j] = pairwise[j, i] = adjusted_rand_score(parts[i], parts[j])
    medoid = int(np.argmax((pairwise.sum(axis=1) - 1.0) / max(count - 1, 1)))
    aligned = np.stack([align_partition(parts[medoid], part, k) for part in parts])
    transition = row_stochastic(symmetric_binary(graph))
    global_agreement = np.maximum(0.05, (pairwise.sum(axis=1) - 1.0) / max(count - 1, 1))
    spatial = np.empty((count, aligned.shape[1]), dtype=np.float32)
    for index, part in enumerate(aligned):
        support = np.asarray(transition @ np.eye(k, dtype=np.float32)[part])
        spatial[index] = support[np.arange(len(part)), part]
    if weighting == "uniform":
        global_weight = np.ones(count, dtype=np.float32)
        local_weight = np.ones_like(spatial)
    elif weighting == "agreement":
        global_weight = global_agreement.astype(np.float32)
        local_weight = np.ones_like(spatial)
    elif weighting == "agreement_spatial":
        global_weight = global_agreement.astype(np.float32)
        local_weight = 0.25 + 0.75 * spatial
    else:
        raise ValueError(weighting)
    votes = np.zeros((aligned.shape[1], int(k)), dtype=np.float32)
    for index, part in enumerate(aligned):
        weight = global_weight[index] * local_weight[index]
        np.add.at(votes, (np.arange(len(part)), part), weight)
    probability = votes / np.maximum(votes.sum(axis=1, keepdims=True), 1e-8)
    unary = -np.log(np.maximum(probability, 1e-6)).astype(np.float32)
    consensus = probability.argmax(axis=1).astype(np.int32)
    diagnostics = {
        "global_weights": global_weight,
        "mean_local_weights": local_weight.mean(axis=1),
        "mean_pairwise_ari": np.asarray(
            [(pairwise.sum() - count) / max(count * (count - 1), 1)], dtype=np.float32
        ),
        "medoid_index": np.asarray([medoid], dtype=np.int32),
    }
    return unary, consensus, diagnostics


def potts_icm(
    unary: np.ndarray,
    graph: sp.spmatrix,
    initial: np.ndarray,
    pairwise_strength: float,
    iterations: int,
) -> np.ndarray:
    """Synchronous sparse Potts updates using unary and pairwise terms."""

    k = unary.shape[1]
    transition = row_stochastic(graph)
    labels = np.asarray(initial, dtype=np.int32).copy()
    for _ in range(int(iterations)):
        support = np.asarray(transition @ np.eye(k, dtype=np.float32)[labels])
        updated = np.argmin(unary - float(pairwise_strength) * support, axis=1).astype(np.int32)
        if np.array_equal(updated, labels):
            break
        labels = updated
    return labels


def dynamic_prototype_icm(
    embedding: np.ndarray,
    graph: sp.spmatrix,
    initial: np.ndarray,
    k: int,
    pairwise_strength: float,
    iterations: int,
    row_floor: float = 1e-8,
) -> Tuple[np.ndarray, int, bool]:
    """Alternating centroid unary + sparse Potts ICM with cardinality guard."""

    transition = row_stochastic(graph, floor=row_floor)
    labels = np.asarray(initial, dtype=np.int32).copy()
    collapse_guard = False
    completed = 0
    for step in range(int(iterations)):
        unary = centroid_unary_row_scaled(embedding, labels, k)
        support = np.asarray(transition @ np.eye(k, dtype=np.float32)[labels])
        proposed = np.argmin(
            unary - float(pairwise_strength) * support, axis=1
        ).astype(np.int32)
        if len(np.unique(proposed)) != int(k):
            collapse_guard = True
            break
        completed = step + 1
        if np.array_equal(proposed, labels):
            break
        labels = proposed
    return labels, completed, collapse_guard


def potts_mean_field(
    unary: np.ndarray,
    graph: sp.spmatrix,
    initial: np.ndarray,
    pairwise_strength: float,
    iterations: int,
    temperature_start: float = 1.0,
    temperature_end: float = 0.25,
) -> np.ndarray:
    """Annealed sparse mean-field approximation for a Potts energy."""

    k = unary.shape[1]
    transition = row_stochastic(graph)
    q = 0.5 * softmax(-unary / max(temperature_start, 1e-4), axis=1)
    q += 0.5 * np.eye(k, dtype=np.float32)[np.asarray(initial, dtype=np.int32)]
    for step in range(int(iterations)):
        fraction = step / max(int(iterations) - 1, 1)
        temperature = temperature_start * (temperature_end / temperature_start) ** fraction
        logits = (-unary + float(pairwise_strength) * np.asarray(transition @ q)) / max(temperature, 1e-4)
        updated = softmax(logits, axis=1).astype(np.float32)
        q = 0.35 * q + 0.65 * updated
    return q.argmax(axis=1).astype(np.int32)


def spectral_partition(
    graph: sp.spmatrix,
    k: int,
    seed: int,
    n_init: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse normalized-affinity spectral clustering."""

    affinity = sp.csr_matrix(graph, dtype=np.float64)
    affinity = affinity.maximum(affinity.T)
    affinity.setdiag(np.maximum(affinity.diagonal(), 1e-3))
    degree = np.asarray(affinity.sum(axis=1)).reshape(-1)
    inv = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = (sp.diags(inv) @ affinity @ sp.diags(inv)).tocsr()
    _, vectors = eigsh(normalized, k=int(k), which="LA", tol=1e-4, maxiter=5000)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    partition = KMeans(int(k), random_state=int(seed), n_init=int(n_init)).fit_predict(vectors)
    return partition.astype(np.int32), vectors.astype(np.float32)


def pseudo_fisher_transform(
    embedding: np.ndarray,
    partition: np.ndarray,
    k: int,
    power: float = 0.5,
    mix: float = 0.5,
) -> np.ndarray:
    """Feature-wise pseudo-class Fisher metric without ground-truth labels."""

    value = standardize(embedding)
    labels = np.asarray(partition, dtype=np.int32)
    global_mean = value.mean(axis=0)
    within = np.zeros(value.shape[1], dtype=np.float64)
    between = np.zeros(value.shape[1], dtype=np.float64)
    total = 0
    for cluster in range(int(k)):
        members = value[labels == cluster]
        if len(members) == 0:
            continue
        mean = members.mean(axis=0)
        within += np.sum((members - mean) ** 2, axis=0)
        between += len(members) * (mean - global_mean) ** 2
        total += len(members)
    ratio = between / np.maximum(within, 1e-6)
    ratio /= max(float(np.median(ratio[ratio > 0])) if np.any(ratio > 0) else 1.0, 1e-6)
    weight = np.clip(ratio, 0.05, 20.0) ** float(power)
    transformed = value * ((1.0 - float(mix)) + float(mix) * weight)
    return standardize(transformed)


@dataclass(frozen=True)
class EnergyConfig:
    family: str
    unary_source: str
    edge_mode: str
    edge_tau: float
    pairwise_strength: float
    iterations: int
    covariance: str = "diag"
