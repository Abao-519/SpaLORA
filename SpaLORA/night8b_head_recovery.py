"""Label-free primitives for the Night-8B uniform-head recovery.

This module intentionally contains no dataset loader, evaluator, label path,
method router, or metric implementation.  The sole scientific entry point,
``recovery_eigen_kmeans100``, accepts only an affinity matrix so every locked
cell executes the identical head.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from .night6c_pipeline import array_sha, sparse_sha
from .night7a_consensus import canonical_partition


HEAD_ID = "RECOVERY_EIGEN_KMEANS100"
N_CLUSTERS = 12
OBSERVATION_SHA256 = "9f0514cee55d307a0ff81d44ffffc2da742dbe2d02b849576b7ef5903743dd1b"
HEAD_CONFIG: Dict[str, Any] = {
    "algorithm": "symmetric_normalized_laplacian_eigenvectors_then_kmeans",
    "affinity_dtype": "float64_csr",
    "affinity_symmetrization": "0.5*(A+A.T)",
    "diagonal": 0.0,
    "degree_policy": "fail_if_nonfinite_or_leq_zero",
    "eigensolver": {
        "function": "scipy.sparse.linalg.eigsh",
        "k": 12,
        "which": "SM",
        "v0": "numpy.linspace(1,2,N,dtype=float64)",
        "tol": 1e-10,
        "maxiter": "max(10000,N*20)",
    },
    "eigenvector_sign": "largest_absolute_pivot_positive",
    "row_l2_floor": 1e-12,
    "kmeans": {
        "n_clusters": 12,
        "n_init": 100,
        "random_state": 2020,
        "algorithm": "lloyd",
    },
    "head_id": HEAD_ID,
    "fallback": False,
}


def canonical_json_sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


HEAD_CONFIG_SHA256 = canonical_json_sha(HEAD_CONFIG)


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


def _canonicalize_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            result[:, column] *= -1.0
    return result


def recovery_eigen_kmeans100(affinity: sp.spmatrix) -> np.ndarray:
    """Run the sole locked head; the signature deliberately has one argument."""
    value = canonical_affinity(affinity)
    if value.shape[0] <= N_CLUSTERS:
        raise RuntimeError("RECOVERY_BLOCKED_HEAD_NUMERICS: insufficient observations")
    degree = np.asarray(value.sum(axis=1), dtype=np.float64).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0.0):
        raise RuntimeError("RECOVERY_BLOCKED_HEAD_NUMERICS: nonpositive degree")
    inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = sp.diags(inv_sqrt) @ value @ sp.diags(inv_sqrt)
    laplacian = (sp.eye(value.shape[0], dtype=np.float64, format="csr")
                 - normalized.tocsr())
    laplacian.sort_indices()
    _eigenvalues, vectors = eigsh(
        laplacian,
        k=N_CLUSTERS,
        which="SM",
        v0=np.linspace(1.0, 2.0, value.shape[0], dtype=np.float64),
        tol=1e-10,
        maxiter=max(10000, value.shape[0] * 20),
    )
    vectors = _canonicalize_eigenvector_signs(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.maximum(norms, 1e-12)
    labels = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=100,
        random_state=2020,
        algorithm="lloyd",
    ).fit_predict(normalized_vectors)
    labels = canonical_partition(labels)
    if np.unique(labels).size != N_CLUSTERS:
        raise RuntimeError("RECOVERY_BLOCKED_HEAD_NUMERICS: K is not exactly 12")
    return labels


def run_twice(affinity: sp.spmatrix) -> Tuple[np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    first = recovery_eigen_kmeans100(affinity)
    first_seconds = time.perf_counter() - started
    second_started = time.perf_counter()
    second = recovery_eigen_kmeans100(affinity)
    second_seconds = time.perf_counter() - second_started
    first_sha = array_sha(canonical_partition(first))
    second_sha = array_sha(canonical_partition(second))
    if first_sha != second_sha or not np.array_equal(first, second):
        raise RuntimeError("RECOVERY_BLOCKED_HEAD_NUMERICS: determinism mismatch")
    return first, {
        "first_head_seconds": float(first_seconds),
        "second_determinism_seconds": float(second_seconds),
        "total_seconds": float(first_seconds + second_seconds),
        "first_partition_sha256": first_sha,
        "second_partition_sha256": second_sha,
        "deterministic_exact": True,
        "canonical_affinity_sha256": sparse_sha(canonical_affinity(affinity)),
        "head_config_sha256": HEAD_CONFIG_SHA256,
    }
