"""Sparse cross-modal spatial alternating diffusion for Night-19B.

The implementation never constructs a dense observation-by-observation matrix.
RNA, ATAC and registered spatial operators remain CSR throughout.  Every sparse
product is deterministically top-k pruned before the next multiplication.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


EPS = 1e-12


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def graph_sha256(graph: sp.spmatrix) -> str:
    value = sp.csr_matrix(graph)
    value.sort_indices()
    digest = hashlib.sha256()
    for array in (value.data, value.indices, value.indptr, np.asarray(value.shape, dtype=np.int64)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    mean = np.mean(x, axis=0, keepdims=True)
    scale = np.std(x, axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return ((x - mean) / scale).astype(np.float32)


def row_normalize(graph: sp.spmatrix) -> sp.csr_matrix:
    value = sp.csr_matrix(graph, dtype=np.float64)
    value.eliminate_zeros()
    if value.nnz and (not np.all(np.isfinite(value.data)) or np.min(value.data) < 0):
        raise ValueError("operator contains nonfinite or negative weights")
    mass = np.asarray(value.sum(axis=1)).ravel()
    if np.any(mass <= EPS):
        raise ValueError("operator has an empty row")
    result = sp.diags(1.0 / mass) @ value
    result = sp.csr_matrix(result)
    result.sort_indices()
    return result


def deterministic_topk(graph: sp.spmatrix, k: int) -> sp.csr_matrix:
    """Keep the largest k entries in each row, resolving ties by column ID."""

    value = sp.csr_matrix(graph, dtype=np.float64)
    value.sum_duplicates()
    value.eliminate_zeros()
    rows = []
    cols = []
    data = []
    for row in range(value.shape[0]):
        start, end = value.indptr[row], value.indptr[row + 1]
        row_cols = value.indices[start:end]
        row_data = value.data[start:end]
        valid = np.isfinite(row_data) & (row_data > 0)
        row_cols, row_data = row_cols[valid], row_data[valid]
        if row_data.size > int(k):
            order = np.lexsort((row_cols, -row_data))[: int(k)]
            row_cols, row_data = row_cols[order], row_data[order]
        order = np.argsort(row_cols, kind="mergesort")
        row_cols, row_data = row_cols[order], row_data[order]
        rows.extend([row] * row_cols.size)
        cols.extend(row_cols.tolist())
        data.extend(row_data.tolist())
    output = sp.csr_matrix((data, (rows, cols)), shape=value.shape, dtype=np.float64)
    output.sum_duplicates()
    output.sort_indices()
    return output


def mutual_knn_operator(value: np.ndarray, neighbors: int, self_loop: float) -> sp.csr_matrix:
    x = standardize(value)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    x = x / norm
    count = min(int(neighbors) + 1, x.shape[0])
    model = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=1)
    distances, indices = model.fit(x).kneighbors(x, return_distance=True)
    rows = np.repeat(np.arange(x.shape[0]), count)
    cols = indices.reshape(-1)
    dist = distances.reshape(-1)
    keep = rows != cols
    rows, cols, dist = rows[keep], cols[keep], dist[keep]
    positive = dist[dist > 1e-10]
    fallback = float(np.median(positive)) if positive.size else 1.0
    # Zelnik-Manor style local scaling: each node uses the distance to its
    # furthest registered kNN neighbour.  A global positive-distance median is
    # used only for a genuinely zero local radius (duplicate observations).
    local_scale = distances[:, -1].astype(np.float64)
    local_scale[local_scale <= 1e-10] = max(fallback, 1e-6)
    denominator = np.maximum(local_scale[rows] * local_scale[cols], 1e-12)
    weights = np.exp(-np.square(dist) / denominator)
    directed = sp.csr_matrix((weights, (rows, cols)), shape=(x.shape[0], x.shape[0]))
    mutual = directed.minimum(directed.T)
    # A bounded self-loop prevents rows with no mutual neighbour from vanishing.
    mutual = mutual + float(self_loop) * sp.eye(x.shape[0], format="csr")
    return row_normalize(mutual)


def spatial_operator(graph: sp.spmatrix, topk: int, self_loop: float) -> sp.csr_matrix:
    value = sp.csr_matrix(graph, dtype=np.float64)
    value = value.maximum(value.T)
    value.setdiag(0)
    value.eliminate_zeros()
    if value.nnz == 0:
        raise ValueError("registered spatial graph is empty")
    scale = float(np.median(value.data[value.data > 0]))
    value.data = np.clip(value.data / max(scale, EPS), 1e-4, 1e4)
    value = deterministic_topk(value, int(topk))
    value = value.maximum(value.T)
    value = value + float(self_loop) * sp.eye(value.shape[0], format="csr")
    return row_normalize(value)


def sparse_chain(operators: Sequence[sp.csr_matrix], topk: int) -> Tuple[sp.csr_matrix, Sequence[Mapping[str, object]]]:
    if not operators:
        raise ValueError("empty operator chain")
    current = operators[0]
    ledger = []
    for index, operator in enumerate(operators[1:], start=1):
        current = sp.csr_matrix(current @ operator)
        before = int(current.nnz)
        current = deterministic_topk(current, int(topk))
        current = row_normalize(current)
        ledger.append({"multiplication_index": index, "nnz_before_prune": before, "nnz_after_prune": int(current.nnz)})
    return current, ledger


def row_cosine(left: sp.csr_matrix, right: sp.csr_matrix) -> np.ndarray:
    numerator = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_norm = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
    right_norm = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
    return np.clip(numerator / np.maximum(left_norm * right_norm, EPS), 0.0, 1.0)


def row_weighted_jaccard(left: sp.csr_matrix, right: sp.csr_matrix) -> np.ndarray:
    numerator = np.asarray(left.minimum(right).sum(axis=1)).ravel()
    denominator = np.asarray(left.maximum(right).sum(axis=1)).ravel()
    return np.clip(numerator / np.maximum(denominator, EPS), 0.0, 1.0)


def conflict_reliability(
    rna: sp.csr_matrix, atac: sp.csr_matrix, spatial: sp.csr_matrix
) -> Tuple[np.ndarray, Mapping[str, object]]:
    molecular = 0.5 * (row_cosine(rna, atac) + row_weighted_jaccard(rna, atac))
    spatial_support = 0.5 * (row_cosine(rna, spatial) + row_cosine(atac, spatial))
    reliability = np.sqrt(np.clip(molecular * spatial_support, 0.0, 1.0))
    diagnostics = {
        "molecular_agreement_min_median_max": [float(np.min(molecular)), float(np.median(molecular)), float(np.max(molecular))],
        "spatial_support_min_median_max": [float(np.min(spatial_support)), float(np.median(spatial_support)), float(np.max(spatial_support))],
        "reliability_min_median_max": [float(np.min(reliability)), float(np.median(reliability)), float(np.max(reliability))],
        "reliability_unique_count": int(np.unique(reliability).size),
    }
    return reliability, diagnostics


def apply_conflict_self_return(
    operator: sp.csr_matrix, reliability: np.ndarray, strength: float, floor: float
) -> Tuple[sp.csr_matrix, Mapping[str, object]]:
    value = sp.csr_matrix(operator, dtype=np.float64).copy()
    value.setdiag(0)
    value.eliminate_zeros()
    node_factor = np.clip(float(floor) + (1.0 - float(floor)) * reliability, 0.0, 1.0)
    coo = value.tocoo()
    edge_factor = np.sqrt(node_factor[coo.row] * node_factor[coo.col])
    edge_factor = (1.0 - float(strength)) + float(strength) * edge_factor
    accepted = sp.csr_matrix((coo.data * edge_factor, (coo.row, coo.col)), shape=value.shape)
    accepted_mass = np.asarray(accepted.sum(axis=1)).ravel()
    if np.any(accepted_mass > 1.0 + 1e-8):
        # Input is row stochastic up to floating error.
        accepted = sp.diags(1.0 / np.maximum(accepted_mass, 1.0)) @ accepted
        accepted_mass = np.asarray(accepted.sum(axis=1)).ravel()
    rejected = np.clip(1.0 - accepted_mass, 0.0, 1.0)
    output = accepted + sp.diags(rejected)
    output = row_normalize(output)
    return output, {
        "accepted_mass_min_median_max": [float(np.min(accepted_mass)), float(np.median(accepted_mass)), float(np.max(accepted_mass))],
        "rejected_mass_min_median_max": [float(np.min(rejected)), float(np.median(rejected)), float(np.max(rejected))],
        "row_mass_error": float(np.max(np.abs(np.asarray(output.sum(axis=1)).ravel() - 1.0))),
    }


def symmetric_nonnegative(operator: sp.spmatrix, topk: int) -> sp.csr_matrix:
    value = sp.csr_matrix(operator, dtype=np.float64)
    value = 0.5 * (value + value.T)
    value = deterministic_topk(value, int(topk))
    value = value.maximum(value.T)
    value.eliminate_zeros()
    if value.nnz == 0 or np.min(value.data) < 0 or not np.all(np.isfinite(value.data)):
        raise ValueError("invalid symmetric nonnegative operator")
    return value


def stable_permutation(ids: np.ndarray, salt: str) -> np.ndarray:
    keys = []
    for index, value in enumerate(np.asarray(ids).astype(str)):
        digest = hashlib.sha256((salt + "\0" + value).encode("utf-8")).digest()
        keys.append((digest, index))
    order = np.asarray([index for _, index in sorted(keys)], dtype=np.int64)
    if np.array_equal(order, np.arange(order.size)):
        order = np.roll(order, 1)
    return order


def permute_operator(operator: sp.csr_matrix, order: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(operator[order][:, order])


@dataclass(frozen=True)
class OperatorBank:
    rna: sp.csr_matrix
    atac: sp.csr_matrix
    spatial: sp.csr_matrix
    concatenated: sp.csr_matrix
    reliability: np.ndarray
    diagnostics: Mapping[str, object]


def build_operator_bank(
    view1: np.ndarray,
    view2: np.ndarray,
    spatial_graph: sp.spmatrix,
    feature_neighbors: int,
    spatial_topk: int,
    self_loop: float,
) -> OperatorBank:
    rna = mutual_knn_operator(view1, feature_neighbors, self_loop)
    atac = mutual_knn_operator(view2, feature_neighbors, self_loop)
    block1 = standardize(view1) / np.sqrt(max(1, view1.shape[1]))
    block2 = standardize(view2) / np.sqrt(max(1, view2.shape[1]))
    concatenated = mutual_knn_operator(np.concatenate([block1, block2], axis=1), feature_neighbors, self_loop)
    spatial = spatial_operator(spatial_graph, spatial_topk, self_loop)
    reliability, reliability_diagnostics = conflict_reliability(rna, atac, spatial)
    diagnostics = {
        "rna_graph_sha256": graph_sha256(rna), "atac_graph_sha256": graph_sha256(atac),
        "spatial_graph_sha256": graph_sha256(spatial), "concatenated_graph_sha256": graph_sha256(concatenated),
        "rna_nnz": int(rna.nnz), "atac_nnz": int(atac.nnz), "spatial_nnz": int(spatial.nnz),
        "concatenated_nnz": int(concatenated.nnz), "conflict_reliability": reliability_diagnostics,
    }
    return OperatorBank(rna, atac, spatial, concatenated, reliability, diagnostics)


def arm_operator(
    arm: str,
    bank: OperatorBank,
    ids: np.ndarray,
    topk: int,
    conflict_strength: float,
    conflict_floor: float,
) -> Tuple[sp.csr_matrix, Mapping[str, object]]:
    rna, atac, spatial = bank.rna, bank.atac, bank.spatial
    diagnostics: Dict[str, object] = {"arm": arm, "multiplications": []}
    if arm == "RNA_ONLY_DIFFUSION":
        raw = rna
    elif arm == "ATAC_ONLY_DIFFUSION":
        raw = atac
    elif arm == "SPATIAL_ONLY_DIFFUSION":
        raw = spatial
    elif arm == "CONCATENATED_FEATURE_KNN":
        raw = bank.concatenated
    elif arm == "SIMPLE_OPERATOR_AVERAGE":
        raw = (rna + atac + spatial) / 3.0
    elif arm == "CLASSICAL_ALTERNATING_RA":
        forward, ledger_f = sparse_chain((rna, atac), topk)
        reverse, ledger_r = sparse_chain((atac, rna), topk)
        raw = 0.5 * (forward + reverse)
        diagnostics["multiplications"] = [*ledger_f, *ledger_r]
    else:
        use_rna, use_atac = rna, atac
        reliability = bank.reliability
        if arm == "CSAD_MODALITY_EDGE_PERMUTED":
            order_r = stable_permutation(ids, "NIGHT19B_RNA")
            order_a = stable_permutation(ids, "NIGHT19B_ATAC")
            use_rna, use_atac = permute_operator(rna, order_r), permute_operator(atac, order_a)
            reliability, perm_diag = conflict_reliability(use_rna, use_atac, spatial)
            diagnostics["permutation_rna_sha256"] = sha256_array(order_r)
            diagnostics["permutation_atac_sha256"] = sha256_array(order_a)
            diagnostics["permuted_reliability"] = perm_diag
        forward, ledger_f = sparse_chain((use_rna, spatial, use_atac), topk)
        reverse, ledger_r = sparse_chain((use_atac, spatial, use_rna), topk)
        diagnostics["multiplications"] = [*ledger_f, *ledger_r]
        if arm == "SPATIALLY_ANCHORED_ALTERNATING":
            raw = forward
        else:
            raw = 0.5 * (forward + reverse)
        if arm in ("CSAD_FULL", "CSAD_MODALITY_EDGE_PERMUTED"):
            raw, gate_diag = apply_conflict_self_return(raw, reliability, conflict_strength, conflict_floor)
            diagnostics["conflict_gate"] = gate_diag
        elif arm not in ("CSAD_CONFLICT_DISABLED", "SPATIALLY_ANCHORED_ALTERNATING"):
            raise ValueError(f"unknown arm: {arm}")
    output = symmetric_nonnegative(raw, topk)
    symmetry = output - output.T
    diagnostics.update({
        "shape": list(output.shape), "nnz": int(output.nnz), "operator_sha256": graph_sha256(output),
        "symmetry_max_abs": float(np.max(np.abs(symmetry.data))) if symmetry.nnz else 0.0,
        "weight_min": float(np.min(output.data)), "weight_max": float(np.max(output.data)),
        "finite": bool(np.all(np.isfinite(output.data))), "nonnegative": bool(np.min(output.data) >= 0.0),
    })
    return output, diagnostics


def spectral_partition(
    operator: sp.csr_matrix, k: int, spectral_dim: int, endpoint_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    degree = np.asarray(operator.sum(axis=1)).ravel()
    if np.any(degree <= EPS):
        raise ValueError("symmetric operator contains a zero-degree observation")
    normalized = sp.diags(1.0 / np.sqrt(degree)) @ operator @ sp.diags(1.0 / np.sqrt(degree))
    normalized = 0.5 * (normalized + normalized.T)
    dim = min(max(int(k), int(spectral_dim)), operator.shape[0] - 2)
    v0 = np.linspace(1.0, 2.0, operator.shape[0], dtype=np.float64)
    values, vectors = eigsh(normalized, k=dim, which="LA", v0=v0, tol=1e-7, maxiter=10000)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    embedding = (vectors / norms).astype(np.float32)
    partition = KMeans(n_clusters=int(k), n_init=20, random_state=int(endpoint_seed), algorithm="lloyd").fit_predict(embedding)
    if np.unique(partition).size != int(k):
        raise RuntimeError("spectral endpoint violates exact K")
    diagnostics = {
        "eigenvalues": values.tolist(), "spectral_dim": int(dim),
        "embedding_sha256": sha256_array(embedding), "partition_sha256": sha256_array(partition.astype(np.int32)),
        "normalized_operator_sha256": graph_sha256(normalized),
    }
    return partition.astype(np.int32), embedding, values, diagnostics
