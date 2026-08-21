"""Night-10A REV2 quality-calibrated cross-modal residual denoising.

This module is label-free and dataset-identity blind. It consumes paired,
frozen views plus a sparse spatial graph. REV2 adds the frozen, rectangular
orthogonal reference harmonizer while retaining the exact REV1 QCRD formulas.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

EPS = 1e-8
MASK_FRACTION = 0.15
LOSS_WEIGHTS = {"align": 1.0, "mask": 1.0, "anchor": 0.25,
                "correction": 0.05, "boundary": 0.25, "mnn": 0.10}
MASKED_CANDIDATES = {
    "Q03_MASKED_RESIDUAL", "Q04_SPOT_GATED_MASKED_RESIDUAL",
    "Q05_BOUNDARY_GATED_RESIDUAL", "Q06_COORDINATE_PRIOR_RESIDUAL",
    "Q07_CONFIDENCE_MNN_RESIDUAL",
}
GLOBAL_CANDIDATES = {"Q01_GLOBAL_QUALITY_BLEND", "Q03_MASKED_RESIDUAL"}
ALL_TRAINABLE_CANDIDATES = {
    "Q01_GLOBAL_QUALITY_BLEND", "Q02_SPOT_QUALITY_BLEND", *MASKED_CANDIDATES,
}


def row_normalize(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), np.float32(EPS))


def standardize(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    return (value - value.mean(axis=0, keepdims=True)) / np.maximum(value.std(axis=0, keepdims=True), EPS)


def canonical_array_sha256(x: np.ndarray) -> str:
    value = np.ascontiguousarray(x)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(canonical_array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def canonical_sparse_sha256(matrix: sp.spmatrix) -> str:
    value = matrix.tocsr().astype(np.float64); value.sort_indices()
    digest = hashlib.sha256(); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.indptr.astype(np.int64).tobytes())
    digest.update(value.indices.astype(np.int64).tobytes()); digest.update(value.data.tobytes())
    return digest.hexdigest()


def _row_l2_normalize_float64(x: np.ndarray) -> np.ndarray:
    """Registered CPU-float64 normalization used only by the harmonizer."""
    value = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, np.float64(EPS))


@dataclass(frozen=True)
class FrozenReferenceHarmonizer:
    """Outcome-blind frozen reference alignment and its audit evidence."""
    mode: str
    zf_aligned: np.ndarray
    projection: Optional[np.ndarray]
    audit: Dict[str, object]


def frozen_reference_harmonizer(
        first: np.ndarray, second: np.ndarray, fused_reference_raw: np.ndarray,
        first_order_sha256: Optional[str] = None,
        second_order_sha256: Optional[str] = None,
        reference_order_sha256: Optional[str] = None,
        audited_row_count: int = 256) -> FrozenReferenceHarmonizer:
    """Build the exact REV2 identity or rectangular-orthogonal harmonizer.

    The signature deliberately has no dataset identifier.  Identity returns
    ``fused_reference_raw`` itself: no copy, cast, SVD, or renormalization.
    Expansion computes P in CPU float64 and casts only the final aligned array.
    """
    if not isinstance(fused_reference_raw, np.ndarray):
        raise TypeError("authoritative fused reference must be a numpy array")
    z1 = np.asarray(first); z2 = np.asarray(second); zf_raw = fused_reference_raw
    for name, value in (("first", z1), ("second", z2), ("fused_reference_raw", zf_raw)):
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"{name} must have a numeric dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    if not (len(z1) == len(z2) == len(zf_raw)):
        raise ValueError("view/reference cardinality mismatch")
    if len(z1) <= 0:
        raise ValueError("empty paired spot set is unsupported")
    d_view = int(z1.shape[1]); d_second = int(z2.shape[1]); d_ref = int(zf_raw.shape[1])
    if d_view <= 0 or d_second <= 0 or d_ref <= 0:
        raise ValueError("non-positive feature dimension is unsupported")
    if d_view != d_second:
        raise ValueError("private views must have identical feature dimensions")
    order_values = (first_order_sha256, second_order_sha256, reference_order_sha256)
    if any(value is not None for value in order_values):
        if any(not isinstance(value, str) or len(value) != 64 for value in order_values):
            raise ValueError("all three ordered-spot SHA values are required")
        if len(set(order_values)) != 1:
            raise ValueError("ordered spot identity mismatch")

    raw_sha = canonical_array_sha256(zf_raw)
    common = {
        "d_view": d_view, "d_ref": d_ref, "n_spots": int(len(z1)),
        "zf_raw_canonical_sha256": raw_sha,
        "ordered_spot_sha256": first_order_sha256,
        "dataset_name_routing": False,
    }
    if d_ref == d_view:
        audit = {
            **common, "mode": "identity", "projection_canonical_sha256": None,
            "zf_aligned_canonical_sha256": canonical_array_sha256(zf_raw),
            "identity_same_object": True, "identity_sha_exact": True,
        }
        return FrozenReferenceHarmonizer("identity", zf_raw, None, audit)
    if d_ref > d_view:
        raise ValueError("reference contraction is unsupported by the REV2 contract")

    x = _row_l2_normalize_float64(zf_raw)
    target = _row_l2_normalize_float64(np.float64(0.5) *
                                       (np.asarray(z1, dtype=np.float64) +
                                        np.asarray(z2, dtype=np.float64)))
    covariance = x.T @ target
    if not np.isfinite(covariance).all():
        raise ValueError("non-finite harmonizer cross-covariance")
    u, singular_values, vt = np.linalg.svd(covariance, full_matrices=False)
    if not (np.isfinite(u).all() and np.isfinite(singular_values).all() and
            np.isfinite(vt).all()):
        raise ValueError("non-finite harmonizer factorization")
    maximum = float(singular_values.max(initial=0.0))
    rank = int(np.sum(singular_values > maximum * 1e-10)) if maximum > 0 else 0
    if rank != d_ref:
        raise ValueError(f"rank-deficient harmonizer covariance: rank={rank}, expected={d_ref}")
    projection = np.ascontiguousarray(u @ vt, dtype=np.float64)
    orthogonality_error = float(np.max(np.abs(
        projection @ projection.T - np.eye(d_ref, dtype=np.float64))))
    if not np.isfinite(orthogonality_error) or orthogonality_error > 1e-10:
        raise ValueError("rectangular harmonizer orthogonality gate failed")
    projected = x @ projection
    count = max(1, min(int(audited_row_count), len(x)))
    subset = np.unique(np.linspace(0, len(x) - 1, count, dtype=np.int64))
    before_subset = x[subset] @ x[subset].T
    after_subset = projected[subset] @ projected[subset].T
    subset_error = float(np.max(np.abs(before_subset - after_subset)))
    if not np.isfinite(subset_error) or subset_error > 1e-10:
        raise ValueError("audited reference geometry gate failed")
    # Certify every pair without allocating a forbidden dense N-by-N matrix.
    residual = projection @ projection.T - np.eye(d_ref, dtype=np.float64)
    full_bound = float(np.max(np.sum(np.abs(x), axis=1)) ** 2 * np.max(np.abs(residual)))
    if not np.isfinite(full_bound) or full_bound > 1e-8:
        raise ValueError("full reference geometry bound failed")
    aligned64 = _row_l2_normalize_float64(projected)
    if not np.isfinite(aligned64).all():
        raise ValueError("non-finite aligned reference")
    aligned = np.ascontiguousarray(aligned64.astype(np.float32, copy=False))
    audit = {
        **common, "mode": "rectangular_expansion", "rank": rank,
        "rank_rtol": 1e-10, "singular_values": singular_values.tolist(),
        "projection_shape": list(projection.shape),
        "projection_canonical_sha256": canonical_array_sha256(projection),
        "orthogonality_max_abs": orthogonality_error,
        "geometry_audited_row_count": int(len(subset)),
        "geometry_subset_max_abs": subset_error,
        "geometry_full_max_abs_certified_bound": full_bound,
        "zf_aligned_canonical_sha256": canonical_array_sha256(aligned),
    }
    return FrozenReferenceHarmonizer("rectangular_expansion", aligned, projection, audit)


def binary_spatial_graph(graph: sp.spmatrix, n: int) -> sp.csr_matrix:
    value = graph.tocsr().astype(np.float64)
    if value.shape != (n, n): raise ValueError("graph/view cardinality mismatch")
    value = value.maximum(value.T).tocsr(); value.setdiag(0); value.eliminate_zeros()
    value.data[:] = 1.0
    return value


def sparse_neighbor_mean(graph: sp.spmatrix, x: np.ndarray) -> np.ndarray:
    """Sparse neighbor mean with the registered zero-degree self rule."""
    value = np.asarray(x, dtype=np.float64); support = binary_spatial_graph(graph, len(value))
    degree = np.asarray(support.sum(axis=1)).ravel(); result = np.asarray(support @ value)
    positive = degree > 0; result[positive] /= degree[positive, None]; result[~positive] = value[~positive]
    return result


def neighbor_entropy(graph: sp.spmatrix, partition: np.ndarray, k_clusters: int) -> np.ndarray:
    labels = np.asarray(partition); support = binary_spatial_graph(graph, len(labels))
    result = np.zeros(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        neighbors = support.indices[support.indptr[i]:support.indptr[i + 1]]; degree = len(neighbors)
        count = min(int(k_clusters), degree + 1)
        denominator = math.log(count) if count > 1 else 0.0
        if denominator <= 0: continue
        values = np.concatenate((labels[neighbors], labels[i:i + 1]))
        counts = np.unique(values, return_counts=True)[1].astype(np.float64)
        probability = counts / counts.sum()
        result[i] = float(-np.sum(probability * np.log(probability)) / denominator)
    return result


def reciprocal_mnn_support(first: np.ndarray, second: np.ndarray, k: int = 10
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse reciprocal cross-view pairs and endpoint support fractions."""
    a, b = row_normalize(first), row_normalize(second)
    if a.shape != b.shape: raise ValueError("paired views must have identical shape")
    n = len(a); registered_k = int(k); query_k = max(1, min(registered_k, n))
    ab = NearestNeighbors(n_neighbors=query_k, metric="cosine", algorithm="brute", n_jobs=1).fit(b)
    ba = NearestNeighbors(n_neighbors=query_k, metric="cosine", algorithm="brute", n_jobs=1).fit(a)
    idx_ab = ab.kneighbors(a, return_distance=False); idx_ba = ba.kneighbors(b, return_distance=False)
    reverse = [set(row.tolist()) for row in idx_ba]; rows, cols = [], []
    for i, candidates in enumerate(idx_ab):
        for j in candidates:
            if i in reverse[int(j)]: rows.append(i); cols.append(int(j))
    rows_a = np.asarray(rows, dtype=np.int64); cols_a = np.asarray(cols, dtype=np.int64)
    support1 = np.bincount(rows_a, minlength=n).astype(np.float64) / max(registered_k, 1)
    support2 = np.bincount(cols_a, minlength=n).astype(np.float64) / max(registered_k, 1)
    return rows_a, cols_a, np.clip(support1, 0, 1), np.clip(support2, 0, 1)


def modality_global_features(embedding: np.ndarray, partition: np.ndarray,
                             graph: sp.spmatrix) -> np.ndarray:
    z = row_normalize(embedding).astype(np.float64); labels = np.asarray(partition)
    scaled = standardize(z); sample_size = min(len(z), 5000)
    silhouette = float(silhouette_score(scaled, labels, metric="euclidean",
                                        sample_size=sample_size, random_state=0))
    db = -math.log1p(max(float(davies_bouldin_score(scaled, labels)), 0.0))
    ch = math.log1p(max(float(calinski_harabasz_score(scaled, labels)), 0.0))
    support = binary_spatial_graph(graph, len(z)); degree = np.asarray(support.sum(axis=1)).ravel()
    positive = degree > 0
    if positive.any():
        equal_counts = np.zeros(len(z), dtype=np.float64); rows, cols = support.nonzero()
        np.add.at(equal_counts, rows, labels[rows] == labels[cols])
        consistency = float(np.mean(equal_counts[positive] / degree[positive]))
    else: consistency = 0.0
    residual = np.linalg.norm(z - sparse_neighbor_mean(support, z), axis=1)
    graph_residual = -math.log1p(float(np.mean(residual)))
    return np.asarray([silhouette, db, ch, consistency, graph_residual], dtype=np.float64)


@dataclass(frozen=True)
class FrozenQuality:
    global_features_1: np.ndarray; global_features_2: np.ndarray
    global_contrasts: np.ndarray; global_weights: np.ndarray
    local_residual_1: np.ndarray; local_residual_2: np.ndarray
    entropy_1: np.ndarray; entropy_2: np.ndarray
    support_1: np.ndarray; support_2: np.ndarray
    spot_logits: np.ndarray; spot_weights: np.ndarray
    disagreement: np.ndarray; fused_residual: np.ndarray
    boundary_risk: np.ndarray; base_gate: np.ndarray
    mnn_rows: np.ndarray; mnn_cols: np.ndarray
    zero_degree_count: int; diagnostics: Dict[str, object]


def frozen_quality(first: np.ndarray, second: np.ndarray, aligned_reference: np.ndarray,
                   partition1: np.ndarray, partition2: np.ndarray,
                   spatial_graph: sp.spmatrix, k_clusters: int,
                   mnn_k: int = 10) -> FrozenQuality:
    z1, z2, zf = (row_normalize(first), row_normalize(second),
                  row_normalize(aligned_reference))
    p1, p2 = np.asarray(partition1), np.asarray(partition2)
    if not (len(z1) == len(z2) == len(zf) == len(p1) == len(p2)):
        raise ValueError("view/partition cardinality mismatch")
    graph = binary_spatial_graph(spatial_graph, len(z1)); degree = np.asarray(graph.sum(axis=1)).ravel()
    e1 = modality_global_features(z1, p1, graph); e2 = modality_global_features(z2, p2, graph)
    contrasts = np.clip((e1 - e2) / (np.abs(e1) + np.abs(e2) + EPS), -1.0, 1.0)
    global_delta = float(np.mean(contrasts)); global_w1 = 1.0 / (1.0 + math.exp(-global_delta))
    global_weights = np.asarray([global_w1, 1.0 - global_w1], dtype=np.float64)
    mean1, mean2, meanf = (sparse_neighbor_mean(graph, x) for x in (z1, z2, zf))
    r1 = np.linalg.norm(z1 - mean1, axis=1); r2 = np.linalg.norm(z2 - mean2, axis=1)
    scale = max(float(np.median(np.concatenate((r1, r2)))), EPS)
    h1 = neighbor_entropy(graph, p1, k_clusters); h2 = neighbor_entropy(graph, p2, k_clusters)
    rows, cols, support1, support2 = reciprocal_mnn_support(z1, z2, mnn_k)
    logits = np.column_stack([
        math.log(global_weights[0] + EPS) - r1 / scale - 0.5 * h1 + 0.5 * support1,
        math.log(global_weights[1] + EPS) - r2 / scale - 0.5 * h2 + 0.5 * support2])
    stable = logits - logits.max(axis=1, keepdims=True); spot_weights = np.exp(stable)
    spot_weights /= spot_weights.sum(axis=1, keepdims=True)
    disagreement = np.clip((1.0 - np.sum(z1 * z2, axis=1)) / 2.0, 0.0, 1.0)
    rf = np.linalg.norm(zf - meanf, axis=1); lo, hi = np.quantile(rf, [0.05, 0.95])
    rn = np.clip((rf - lo) / (hi - lo + EPS), 0.0, 1.0)
    boundary = np.clip(0.5 * rn + 0.25 * (h1 + h2), 0.0, 1.0)
    base_gate = np.clip(0.5 * np.abs(spot_weights[:, 0] - spot_weights[:, 1])
                        + 0.25 * disagreement + 0.25 * np.maximum(support1, support2), 0.0, 1.0)
    base_gate = np.clip(base_gate * (1.0 - 0.25 * (h1 + h2)), 0.0, 1.0)
    arrays = [e1, e2, contrasts, global_weights, r1, r2, h1, h2, support1, support2,
              logits, spot_weights, disagreement, rf, boundary, base_gate, rows, cols]
    frozen_sha = hashlib.sha256("".join(canonical_array_sha256(x) for x in arrays).encode("ascii")).hexdigest()
    diagnostics = {"global_feature_order": ["silhouette", "davies_bouldin", "calinski_harabasz", "spatial_local_consistency", "graph_local_residual"],
                   "global_delta": global_delta, "global_weights": global_weights.tolist(),
                   "mnn_k": int(mnn_k), "mnn_edge_count": int(len(rows)),
                   "zero_degree_count": int(np.sum(degree == 0)),
                   "spatial_graph_sha256": canonical_sparse_sha256(graph),
                   "frozen_quality_sha256": frozen_sha}
    return FrozenQuality(e1, e2, contrasts, global_weights, r1, r2, h1, h2,
                         support1, support2, logits, spot_weights, disagreement,
                         rf, boundary, base_gate, rows, cols, int(np.sum(degree == 0)), diagnostics)


class QCRDAdapter(torch.nn.Module):
    """The one registered trainable class for both modality families."""
    def __init__(self, dim: int, coord_features: int = 0, hidden_width: int = 64,
                 rank: int = 16, dropout: float = 0.1):
        super().__init__(); self.dim = int(dim); self.coord_features = int(coord_features)
        self.input = torch.nn.Linear(3 * self.dim + self.coord_features, int(hidden_width))
        self.activation = torch.nn.GELU(); self.dropout = torch.nn.Dropout(float(dropout))
        self.down = torch.nn.Linear(int(hidden_width), int(rank), bias=False)
        self.up = torch.nn.Linear(int(rank), self.dim, bias=False); torch.nn.init.zeros_(self.up.weight)

    def forward(self, student_input: torch.Tensor, teacher: torch.Tensor,
                zf_aligned: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        pieces = [student_input, teacher.detach(), zf_aligned.detach()]
        if self.coord_features:
            if coords is None or coords.shape[1] != self.coord_features:
                raise ValueError("registered coordinate feature shape mismatch")
            pieces.append(coords.detach())
        return self.up(self.down(self.dropout(self.activation(self.input(torch.cat(pieces, dim=1))))))


def fourier_coordinates(coords: np.ndarray, frequencies: Sequence[int] = (1, 2, 4, 8)) -> np.ndarray:
    value = np.asarray(coords, dtype=np.float32)
    value = (value - value.mean(axis=0, keepdims=True)) / np.maximum(value.std(axis=0, keepdims=True), 1e-6)
    pieces = []
    for frequency in frequencies: pieces.extend((np.sin(float(frequency) * value), np.cos(float(frequency) * value)))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def mask_dimension_count(dimension: int) -> int:
    dimension = int(dimension); count = max(1, int(math.floor(MASK_FRACTION * dimension + 0.5)))
    return min(count, dimension - 1) if dimension > 1 else 1


def deterministic_mask(candidate_id: str, input_artifact_sha256: str,
                       scientific_seed: int, epoch: int, n: int, d: int) -> np.ndarray:
    if candidate_id not in MASKED_CANDIDATES: return np.zeros((int(n), int(d)), dtype=bool)
    message = "night10a-qcrd-mask-v1|{}|{}|{}|{}".format(candidate_id, input_artifact_sha256, int(scientific_seed), int(epoch))
    seed = int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed); count = mask_dimension_count(d); mask = np.zeros((int(n), int(d)), dtype=bool)
    for row in range(int(n)): mask[row, rng.choice(int(d), size=count, replace=False)] = True
    return mask


def _torch_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1, eps=EPS)


def candidate_weights_and_gate(quality: FrozenQuality, candidate_id: str,
                               dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if candidate_id not in ALL_TRAINABLE_CANDIDATES: raise ValueError("unregistered trainable candidate")
    spot = torch.as_tensor(quality.spot_weights, dtype=dtype, device=device)
    global_weights = torch.as_tensor(quality.global_weights, dtype=dtype, device=device).view(1, 2).expand_as(spot)
    weights = global_weights if candidate_id in GLOBAL_CANDIDATES else spot
    if candidate_id in GLOBAL_CANDIDATES:
        gate = torch.full((len(spot), 1), float(abs(quality.global_weights[0] - quality.global_weights[1])), dtype=dtype, device=device)
    else: gate = torch.as_tensor(quality.base_gate, dtype=dtype, device=device).view(-1, 1)
    if candidate_id == "Q05_BOUNDARY_GATED_RESIDUAL":
        gate = gate * (1.0 - torch.as_tensor(quality.boundary_risk, dtype=dtype, device=device).view(-1, 1))
    return weights, gate.clamp(0.0, 1.0)


def qcrd_forward(model: QCRDAdapter, first: torch.Tensor, second: torch.Tensor,
                 zf_aligned: torch.Tensor, quality: FrozenQuality,
                 candidate_id: str, coord_features: Optional[torch.Tensor] = None,
                 mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if first.ndim != 2 or second.ndim != 2 or zf_aligned.ndim != 2:
        raise ValueError("QCRD views must be two-dimensional")
    if first.shape != second.shape or first.shape != zf_aligned.shape:
        raise ValueError("QCRD requires harmonized equal-shaped views")
    if first.shape[1] != model.dim:
        raise ValueError("adapter dimension does not match harmonized view dimension")
    if not (torch.isfinite(first).all() and torch.isfinite(second).all() and
            torch.isfinite(zf_aligned).all()):
        raise ValueError("QCRD received non-finite views")
    weights, gate = candidate_weights_and_gate(quality, candidate_id, first.dtype, first.device)
    teacher = weights[:, :1] * first + weights[:, 1:] * second
    clean_student = weights[:, 1:] * first + weights[:, :1] * second
    model_student = clean_student if mask is None else clean_student.masked_fill(mask, 0.0)
    raw_delta = model(model_student, teacher, zf_aligned, coord_features)
    delta = raw_delta / raw_delta.norm(dim=1, keepdim=True).clamp_min(1.0); correction = 0.25 * gate * delta
    first_c = _torch_normalize(first + (1.0 - weights[:, :1]) * correction)
    second_c = _torch_normalize(second + (1.0 - weights[:, 1:]) * correction)
    fused_c = _torch_normalize((first_c + second_c + zf_aligned) / 3.0)
    return {"weights": weights, "gate": gate, "teacher": teacher, "clean_student": clean_student,
            "model_student": model_student, "raw_delta": raw_delta, "delta": delta,
            "correction": correction, "z1c": first_c, "z2c": second_c, "zc": fused_c,
            "student_corrected": _torch_normalize(clean_student + correction),
            "pred_student": _torch_normalize(model_student + correction)}


def corrected_views(model: QCRDAdapter, first: torch.Tensor, second: torch.Tensor,
                    zf_aligned: torch.Tensor, quality: FrozenQuality,
                    candidate_id: str, coord_features: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    result = qcrd_forward(model, first, second, zf_aligned, quality, candidate_id, coord_features, mask=None)
    return result["z1c"], result["z2c"], result["zc"], result["correction"]


def qcrd_loss_components(forward: Mapping[str, torch.Tensor], first: torch.Tensor,
                         second: torch.Tensor, zf_aligned: torch.Tensor,
                         quality: FrozenQuality, candidate_id: str,
                         spatial_graph: sp.spmatrix, reference_partition: np.ndarray,
                         mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    gate = forward["gate"].view(-1)
    cosine = 1.0 - F.cosine_similarity(forward["student_corrected"], forward["teacher"].detach(), dim=1, eps=EPS)
    align = torch.sum(gate * cosine) / (torch.sum(gate) + EPS)
    if candidate_id in MASKED_CANDIDATES:
        if mask is None: raise ValueError("registered masked candidate requires a mask")
        mask_float = mask.to(dtype=first.dtype)
        numerator = torch.sum(mask_float * (forward["pred_student"] - forward["clean_student"].detach()) ** 2, dim=1)
        denominator = torch.sum(mask_float * forward["clean_student"].detach() ** 2, dim=1) + EPS
        mask_loss = torch.mean(numerator / denominator)
    else:
        if mask is not None and bool(mask.any()): raise ValueError("non-masked candidate received a nonempty mask")
        mask_loss = align.new_zeros(())
    anchor = 0.5 * torch.mean((1.0 - F.cosine_similarity(forward["z1c"], first, dim=1, eps=EPS))
                              + (1.0 - F.cosine_similarity(forward["z2c"], second, dim=1, eps=EPS)))
    correction = torch.mean((forward["correction"].norm(dim=1) / 0.25) ** 2)
    graph = binary_spatial_graph(spatial_graph, len(first)); upper = sp.triu(graph, k=1).tocoo()
    pf = np.asarray(reference_partition); keep = pf[upper.row] != pf[upper.col]
    if np.any(keep):
        rows = torch.as_tensor(upper.row[keep], dtype=torch.long, device=first.device)
        cols = torch.as_tensor(upper.col[keep], dtype=torch.long, device=first.device)
        new_cos = torch.sum(forward["zc"][rows] * forward["zc"][cols], dim=1)
        old_cos = torch.sum(zf_aligned[rows] * zf_aligned[cols], dim=1)
        boundary = torch.mean(F.relu(new_cos - old_cos) ** 2)
    else: boundary = align.new_zeros(())
    if candidate_id == "Q07_CONFIDENCE_MNN_RESIDUAL" and len(quality.mnn_rows):
        rows = torch.as_tensor(quality.mnn_rows, dtype=torch.long, device=first.device)
        cols = torch.as_tensor(quality.mnn_cols, dtype=torch.long, device=first.device)
        h1 = torch.as_tensor(quality.entropy_1[quality.mnn_rows], dtype=first.dtype, device=first.device)
        h2 = torch.as_tensor(quality.entropy_2[quality.mnn_cols], dtype=first.dtype, device=first.device)
        b1 = torch.as_tensor(quality.boundary_risk[quality.mnn_rows], dtype=first.dtype, device=first.device)
        b2 = torch.as_tensor(quality.boundary_risk[quality.mnn_cols], dtype=first.dtype, device=first.device)
        edge_weight = torch.sqrt((1.0 - h1).clamp_min(0) * (1.0 - h2).clamp_min(0)) * (1.0 - b1) * (1.0 - b2)
        pair_cos = 1.0 - F.cosine_similarity(forward["z1c"][rows], forward["z2c"][cols], dim=1, eps=EPS)
        mnn = torch.sum(edge_weight * pair_cos) / (torch.sum(edge_weight) + EPS)
    else: mnn = align.new_zeros(())
    total = (LOSS_WEIGHTS["align"] * align + LOSS_WEIGHTS["mask"] * mask_loss
             + LOSS_WEIGHTS["anchor"] * anchor + LOSS_WEIGHTS["correction"] * correction
             + LOSS_WEIGHTS["boundary"] * boundary
             + (LOSS_WEIGHTS["mnn"] * mnn if candidate_id == "Q07_CONFIDENCE_MNN_RESIDUAL" else 0.0))
    return {"align": align, "mask": mask_loss, "anchor": anchor,
            "correction": correction, "boundary": boundary, "mnn": mnn, "total": total}


def state_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO(); torch.save(state, buffer); return buffer.getvalue()


__all__ = ["ALL_TRAINABLE_CANDIDATES", "EPS", "FrozenQuality",
           "FrozenReferenceHarmonizer", "GLOBAL_CANDIDATES",
           "LOSS_WEIGHTS", "MASKED_CANDIDATES", "MASK_FRACTION", "QCRDAdapter",
           "binary_spatial_graph", "candidate_weights_and_gate", "canonical_array_sha256",
           "canonical_sparse_sha256", "canonical_state_sha256", "corrected_views",
           "deterministic_mask", "fourier_coordinates", "frozen_quality",
           "frozen_reference_harmonizer",
           "mask_dimension_count", "modality_global_features", "neighbor_entropy",
           "qcrd_forward", "qcrd_loss_components", "reciprocal_mnn_support",
           "row_normalize", "sparse_neighbor_mean", "standardize"]
