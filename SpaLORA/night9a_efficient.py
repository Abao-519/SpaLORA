"""Label-free primitives for Night-9A efficient topology transfer.

The module deliberately contains no dataset loader, label path, evaluator, or
metric-based router.  Dataset-specific cache locations and locked cluster
cardinalities live in the execution script; every scientific primitive here is
identity blind and accepts only arrays, sparse operators, or an explicitly
registered candidate record.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .night6c_pipeline import array_sha, sparse_sha
from .night7a_consensus import canonical_partition


VIEW_KEYS = ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused")
HEAD_ID = "DATASET_LOCKED_EIGEN_KMEANS100"
HEAD_BASE_CONFIG = {
    "algorithm": "symmetric_normalized_laplacian_eigenvectors_then_kmeans",
    "affinity_dtype": "float64_csr",
    "affinity_symmetrization": "0.5*(A+A.T)",
    "diagonal": 0.0,
    "degree_policy": "fail_if_nonfinite_or_leq_zero",
    "eigensolver": {
        "function": "scipy.sparse.linalg.eigsh",
        "which": "SM",
        "v0": "numpy.linspace(1,2,N,dtype=float64)",
        "tol": 1e-10,
        "maxiter": "max(10000,N*20)",
    },
    "eigenvector_sign": "largest_absolute_pivot_positive",
    "row_l2_floor": 1e-12,
    "kmeans": {
        "n_init": 100,
        "random_state": 2020,
        "algorithm": "lloyd",
    },
    "head_id": HEAD_ID,
    "fallback": False,
}


def canonical_json_sha(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_l2(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise RuntimeError("projection input must be a finite matrix")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    result = value / np.maximum(norms, 1e-12)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("row-L2 projection produced non-finite values")
    return result


def canonical_operator(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Canonical Night-9A parameter-free propagation operator."""
    value = matrix.tocsr().astype(np.float64)
    if value.shape[0] != value.shape[1]:
        raise RuntimeError("projection operator must be square")
    if not np.isfinite(value.data).all() or np.any(value.data < 0.0):
        raise RuntimeError("projection operator must be finite and nonnegative")
    value.setdiag(0.0)
    value.eliminate_zeros()
    value.sort_indices()
    degree = np.asarray(value.sum(axis=1), dtype=np.float64).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0.0):
        raise RuntimeError("projection operator has nonpositive degree")
    value = (sp.diags(1.0 / degree) @ value).tocsr()
    value.eliminate_zeros()
    value.sort_indices()
    return value


def load_projection_operators(graph_dir: Path) -> Mapping[str, sp.csr_matrix]:
    support = {}
    for key in ("adj_spatial_omics1", "adj_feature_omics1",
                "adj_spatial_omics2", "adj_feature_omics2"):
        path = graph_dir / f"{key}_support.npz"
        if not path.is_file():
            raise RuntimeError(f"missing graph support: {path}")
        support[key] = canonical_operator(sp.load_npz(path))
    omics1 = canonical_operator((support["adj_spatial_omics1"] +
                                 support["adj_feature_omics1"]) * 0.5)
    omics2 = canonical_operator((support["adj_spatial_omics2"] +
                                 support["adj_feature_omics2"]) * 0.5)
    fused = canonical_operator((omics1 + omics2) * 0.5)
    return {
        "emb_latent_omics1": omics1,
        "emb_latent_omics2": omics2,
        "SpaLORA_fused": fused,
    }


def projection_views(candidate_id: str, source_views: Mapping[str, np.ndarray],
                     target_ops: Mapping[str, sp.csr_matrix],
                     source_ops: Mapping[str, sp.csr_matrix]) -> Mapping[str, np.ndarray]:
    """Apply one of the four preregistered projection formulas."""
    if candidate_id not in {
        "E05_SGC1_RESIDUAL50", "E06_SGC2_MEAN",
        "E07_DELTA_TOPOLOGY_RESIDUAL50", "E08_DELTA_TOPOLOGY_RESIDUAL100",
    }:
        raise KeyError(candidate_id)
    result = {}
    for key in VIEW_KEYS:
        h = np.asarray(source_views[key], dtype=np.float64)
        p00 = target_ops[key]
        p04 = source_ops[key]
        if p00.shape[0] != h.shape[0] or p04.shape[0] != h.shape[0]:
            raise RuntimeError("projection observation count mismatch")
        if candidate_id == "E05_SGC1_RESIDUAL50":
            value = 0.5 * h + 0.5 * (p00 @ h)
        elif candidate_id == "E06_SGC2_MEAN":
            once = p00 @ h
            value = (h + once + p00 @ once) / 3.0
        elif candidate_id == "E07_DELTA_TOPOLOGY_RESIDUAL50":
            value = h + 0.5 * ((p00 @ h) - (p04 @ h))
        else:
            value = h + (p00 @ h) - (p04 @ h)
        result[key] = row_l2(value)
    return result


def canonical_affinity(matrix: sp.spmatrix) -> sp.csr_matrix:
    value = matrix.tocsr().astype(np.float64)
    value = ((value + value.T) * 0.5).tocsr()
    value.setdiag(0.0)
    value.eliminate_zeros()
    value.sort_indices()
    if value.shape[0] != value.shape[1]:
        raise RuntimeError("affinity must be square")
    if not np.isfinite(value.data).all():
        raise RuntimeError("affinity contains non-finite values")
    return value


def head_config(k: int) -> dict:
    config = json.loads(json.dumps(HEAD_BASE_CONFIG))
    config["eigensolver"]["k"] = int(k)
    config["kmeans"]["n_clusters"] = int(k)
    config["dataset_locked_K"] = int(k)
    return config


def _canonicalize_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            result[:, column] *= -1.0
    return result


def eigen_kmeans100(affinity: sp.spmatrix, k: int) -> tuple[np.ndarray, dict]:
    """Run the sole Night-9A head with only the prelocked K as an argument."""
    started = time.perf_counter()
    value = canonical_affinity(affinity)
    if value.shape[0] <= int(k):
        raise RuntimeError("HEAD_NUMERICS: insufficient observations")
    degree = np.asarray(value.sum(axis=1), dtype=np.float64).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0.0):
        raise RuntimeError("HEAD_NUMERICS: nonpositive degree")
    inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = sp.diags(inv_sqrt) @ value @ sp.diags(inv_sqrt)
    laplacian = (sp.eye(value.shape[0], dtype=np.float64, format="csr") -
                 normalized.tocsr())
    laplacian.sort_indices()
    _eigenvalues, vectors = eigsh(
        laplacian, k=int(k), which="SM",
        v0=np.linspace(1.0, 2.0, value.shape[0], dtype=np.float64),
        tol=1e-10, maxiter=max(10000, value.shape[0] * 20),
    )
    vectors = _canonicalize_eigenvector_signs(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.maximum(norms, 1e-12)
    labels = KMeans(n_clusters=int(k), n_init=100, random_state=2020,
                    algorithm="lloyd").fit_predict(normalized_vectors)
    labels = canonical_partition(labels)
    if np.unique(labels).size != int(k):
        raise RuntimeError("HEAD_NUMERICS: K is not exact")
    return labels, {
        "head_id": HEAD_ID,
        "head_config": head_config(int(k)),
        "head_config_sha256": canonical_json_sha(head_config(int(k))),
        "runtime_seconds": float(time.perf_counter() - started),
        "canonical_affinity_sha256": sparse_sha(value),
        "canonical_partition_sha256": array_sha(labels),
    }


def exact_partition_equal(first: Sequence[int], second: Sequence[int]) -> bool:
    return bool(np.array_equal(canonical_partition(np.asarray(first)),
                               canonical_partition(np.asarray(second))))


def _topk_indices(matrix: sp.csr_matrix, row: int, k: int) -> np.ndarray:
    start, stop = matrix.indptr[row], matrix.indptr[row + 1]
    indices = matrix.indices[start:stop]
    values = matrix.data[start:stop]
    if len(values) <= k:
        order = np.lexsort((indices, -values))
    else:
        selected = np.argpartition(values, -k)[-k:]
        order = selected[np.lexsort((indices[selected], -values[selected]))]
    return indices[order[:k]]


def affinity_fidelity(candidate: sp.spmatrix, teacher: sp.spmatrix,
                      candidate_partition: Sequence[int],
                      teacher_partition: Sequence[int]) -> dict:
    a = canonical_affinity(candidate)
    b = canonical_affinity(teacher)
    if a.shape != b.shape:
        raise RuntimeError("teacher/candidate affinity shape mismatch")
    jaccard = {}
    for k in (10, 20):
        values = []
        for row in range(a.shape[0]):
            left = set(map(int, _topk_indices(a, row, k)))
            right = set(map(int, _topk_indices(b, row, k)))
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
        jaccard[str(k)] = float(np.mean(values))
    difference = a - b
    a_norm = float(np.sqrt(a.multiply(a).sum()))
    b_norm = float(np.sqrt(b.multiply(b).sum()))
    diff_norm = float(np.sqrt(difference.multiply(difference).sum()))
    inner = float(a.multiply(b).sum())
    first = canonical_partition(np.asarray(candidate_partition))
    second = canonical_partition(np.asarray(teacher_partition))
    a_sizes = np.sort(np.bincount(first))
    b_sizes = np.sort(np.bincount(second))
    width = max(len(a_sizes), len(b_sizes))
    a_sizes = np.pad(a_sizes, (width - len(a_sizes), 0))
    b_sizes = np.pad(b_sizes, (width - len(b_sizes), 0))
    return {
        "partition_ari": float(adjusted_rand_score(second, first)),
        "partition_nmi": float(normalized_mutual_info_score(second, first)),
        "partition_exact_up_to_permutation": exact_partition_equal(first, second),
        "topk_neighbor_jaccard": jaccard,
        "normalized_frobenius_distance": diff_norm / max(b_norm, 1e-12),
        "cosine_distance": 1.0 - inner / max(a_norm * b_norm, 1e-12),
        "cluster_size_distribution_l1": float(np.abs(a_sizes - b_sizes).sum() /
                                               (2.0 * len(first))),
    }


def align_partition(predicted: Sequence[int], reference: Sequence[int]) -> np.ndarray:
    """Diagnostic Hungarian alignment; metrics remain permutation invariant."""
    pred = canonical_partition(np.asarray(predicted))
    ref = canonical_partition(np.asarray(reference))
    width = max(int(pred.max()) + 1, int(ref.max()) + 1)
    contingency = np.zeros((width, width), dtype=np.int64)
    np.add.at(contingency, (pred, ref), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(row): int(col) for row, col in zip(rows, cols)}
    return np.asarray([mapping.get(int(value), int(value)) for value in pred])


__all__ = [
    "HEAD_ID", "VIEW_KEYS", "affinity_fidelity", "canonical_affinity",
    "canonical_json_sha", "canonical_operator", "eigen_kmeans100",
    "exact_partition_equal", "head_config", "load_projection_operators",
    "projection_views", "row_l2", "sha256_file",
]
