"""Night-10A quality-calibrated cross-modal residual denoising.

The module consumes frozen, label-free views.  It deliberately has no dataset
identifier input and constructs only sparse neighborhood objects.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    den = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(den, np.finfo(np.float32).eps)


def canonical_array_sha256(x: np.ndarray) -> str:
    a = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def canonical_state_sha256(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8") + b"\0")
        h.update(canonical_array_sha256(value).encode("ascii"))
    return h.hexdigest()


def sparse_neighbor_mean(graph: sp.spmatrix, x: np.ndarray) -> np.ndarray:
    g = graph.tocsr().astype(np.float64)
    if g.shape != (len(x), len(x)):
        raise ValueError("graph/view cardinality mismatch")
    degree = np.asarray(g.sum(axis=1)).ravel()
    degree = np.maximum(degree, np.finfo(np.float64).eps)
    return np.asarray(g @ np.asarray(x, dtype=np.float64)) / degree[:, None]


def reciprocal_mnn_support(
    first: np.ndarray, second: np.ndarray, k: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse reciprocal cross-view pairs and paired-spot support.

    Nearest-neighbor queries are block/tree based in sklearn; no dense N by N
    distance matrix is materialized.
    """
    a, b = row_normalize(first), row_normalize(second)
    if a.shape != b.shape:
        raise ValueError("paired views must have identical shape")
    n = len(a)
    k = max(1, min(int(k), n))
    ab = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=1).fit(b)
    ba = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=1).fit(a)
    idx_ab = ab.kneighbors(a, return_distance=False)
    idx_ba = ba.kneighbors(b, return_distance=False)
    reverse = [set(row.tolist()) for row in idx_ba]
    rows, cols = [], []
    for i, row in enumerate(idx_ab):
        for j in row:
            if i in reverse[int(j)]:
                rows.append(i)
                cols.append(int(j))
    paired = np.zeros(n, dtype=np.float32)
    for i, j in zip(rows, cols):
        if i == j:
            paired[i] = 1.0
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), paired


def _geometry_evidence(x: np.ndarray, partition: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    sample = 5000 if len(x) > 5000 else None
    sil = float(silhouette_score(x, partition, metric="euclidean", sample_size=sample, random_state=0))
    db = float(davies_bouldin_score(x, partition))
    ch = float(calinski_harabasz_score(x, partition))
    return np.asarray([sil, -math.log1p(max(db, 0.0)), math.log1p(max(ch, 0.0))], dtype=np.float64)


@dataclass(frozen=True)
class FrozenQuality:
    global_weights: np.ndarray
    spot_weights: np.ndarray
    gate: np.ndarray
    boundary_risk: np.ndarray
    mnn_rows: np.ndarray
    mnn_cols: np.ndarray
    diagnostics: Dict[str, object]


def frozen_quality(
    first: np.ndarray,
    second: np.ndarray,
    fused: np.ndarray,
    partition: np.ndarray,
    spatial_graph: sp.spmatrix,
    mnn_k: int = 10,
) -> FrozenQuality:
    a, b, f = row_normalize(first), row_normalize(second), row_normalize(fused)
    partition = np.asarray(partition)
    if not (len(a) == len(b) == len(f) == len(partition)):
        raise ValueError("view/partition cardinality mismatch")
    g1, g2 = _geometry_evidence(a, partition), _geometry_evidence(b, partition)
    delta = np.clip(np.mean((g1 - g2) / np.maximum(np.abs(g1) + np.abs(g2), 1e-8)), -6.0, 6.0)
    global_w1 = 1.0 / (1.0 + math.exp(-float(delta)))
    global_weights = np.asarray([global_w1, 1.0 - global_w1], dtype=np.float32)
    ma, mb, mf = sparse_neighbor_mean(spatial_graph, a), sparse_neighbor_mean(spatial_graph, b), sparse_neighbor_mean(spatial_graph, f)
    r1 = np.linalg.norm(a - ma, axis=1)
    r2 = np.linalg.norm(b - mb, axis=1)
    disagreement = 1.0 - np.sum(a * b, axis=1)
    rows, cols, paired_support = reciprocal_mnn_support(a, b, mnn_k)
    scale = np.maximum(np.median(np.concatenate([r1, r2])), 1e-8)
    logits = np.column_stack([-r1 / scale, -r2 / scale])
    logits -= logits.max(axis=1, keepdims=True)
    spot_weights = np.exp(logits)
    spot_weights /= spot_weights.sum(axis=1, keepdims=True)
    fused_residual = np.linalg.norm(f - mf, axis=1)
    lo, hi = np.quantile(fused_residual, [0.05, 0.95])
    boundary = np.clip((fused_residual - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    gate = np.clip(np.abs(spot_weights[:, 0] - spot_weights[:, 1]) + 0.25 * disagreement + 0.25 * paired_support, 0.0, 1.0)
    arrays = [global_weights, spot_weights.astype(np.float32), gate.astype(np.float32), boundary.astype(np.float32), rows, cols]
    diag = {
        "global_evidence_modality_1": g1.tolist(),
        "global_evidence_modality_2": g2.tolist(),
        "global_weights": global_weights.tolist(),
        "mnn_edge_count": int(len(rows)),
        "paired_mnn_support_mean": float(paired_support.mean()),
        "frozen_quality_sha256": hashlib.sha256("".join(canonical_array_sha256(x) for x in arrays).encode("ascii")).hexdigest(),
    }
    return FrozenQuality(global_weights, spot_weights.astype(np.float32), gate.astype(np.float32), boundary.astype(np.float32), rows, cols, diag)


class QCRDAdapter(torch.nn.Module):
    """One architecture for both modality families; no dataset routing hook."""

    def __init__(self, dim: int, coord_features: int = 0, hidden_width: int = 64, rank: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.coord_features = int(coord_features)
        in_dim = 3 * self.dim + self.coord_features
        self.input = torch.nn.Linear(in_dim, int(hidden_width))
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(float(dropout))
        self.down = torch.nn.Linear(int(hidden_width), int(rank), bias=False)
        self.up = torch.nn.Linear(int(rank), self.dim, bias=False)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor, context: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        pieces = [student, teacher, context]
        if self.coord_features:
            if coords is None or coords.shape[1] != self.coord_features:
                raise ValueError("registered coordinate feature shape mismatch")
            pieces.append(coords)
        return self.up(self.down(self.dropout(self.activation(self.input(torch.cat(pieces, dim=1))))))


def fourier_coordinates(coords: np.ndarray, frequencies: int = 4) -> np.ndarray:
    c = np.asarray(coords, dtype=np.float32)
    c = (c - c.mean(axis=0, keepdims=True)) / np.maximum(c.std(axis=0, keepdims=True), 1e-6)
    out = []
    for power in range(int(frequencies)):
        scale = float(2 ** power)
        out.extend([np.sin(scale * c), np.cos(scale * c)])
    return np.concatenate(out, axis=1).astype(np.float32)


def _torch_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)


def corrected_views(
    model: QCRDAdapter,
    first: torch.Tensor,
    second: torch.Tensor,
    fused_reference: torch.Tensor,
    context: torch.Tensor,
    quality: FrozenQuality,
    candidate_id: str,
    coord_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sw = torch.as_tensor(quality.spot_weights, dtype=first.dtype, device=first.device)
    gw = torch.as_tensor(quality.global_weights, dtype=first.dtype, device=first.device).view(1, 2).expand_as(sw)
    weights = gw if candidate_id in {"Q01_GLOBAL_QUALITY_BLEND", "Q03_MASKED_RESIDUAL"} else sw
    teacher = weights[:, :1] * first + weights[:, 1:] * second
    student = weights[:, 1:] * first + weights[:, :1] * second
    delta = model(student, teacher.detach(), context, coord_features)
    gate = torch.as_tensor(quality.gate, dtype=first.dtype, device=first.device).view(-1, 1)
    if candidate_id == "Q01_GLOBAL_QUALITY_BLEND":
        gate = gate.new_full(gate.shape, float(abs(quality.global_weights[0] - quality.global_weights[1])))
    if candidate_id == "Q05_BOUNDARY_GATED_RESIDUAL":
        boundary = torch.as_tensor(quality.boundary_risk, dtype=first.dtype, device=first.device).view(-1, 1)
        gate = gate * (1.0 - boundary)
    gate = gate.clamp(0.0, 1.0) * 0.25
    delta = delta / delta.norm(dim=1, keepdim=True).clamp_min(1.0)
    first_c = _torch_normalize(first + gate * (1.0 - weights[:, :1]) * delta)
    second_c = _torch_normalize(second + gate * (1.0 - weights[:, 1:]) * delta)
    fused = _torch_normalize((first_c + second_c + fused_reference) / 3.0)
    return first_c, second_c, fused, gate * delta


def state_bytes(state: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()

