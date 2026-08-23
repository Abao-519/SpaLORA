"""Night-14B clean-room RNA+ATAC edge-state model and sparse heads.

The module contains no dataset registry and accepts only tensors, sparse edge
lists, coordinates, identifiers, and explicit numerical configuration.  Public
labels are intentionally kept in the runner/evaluator and never reach these
model or filtering functions.

TSPR is a working engineering name.  Its three edge states are soft evidence
states, not biological labels: mutually supported interior, mutually rejected
boundary, and cross-modal conflict.  Propagation uses only the interior mass;
boundary and conflict mass retain the identity path.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F

from .night14a_tcf import (
    UnifiedGraphAutoencoder,
    loss_components,
    state_sha256,
    unsupervised_loss,
    weighted_neighbor_mean,
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def initialize_cuda_device(device: torch.device) -> None:
    """Materialize CUDA before resetting allocator peaks on PyTorch 2.0."""
    if device.type != "cuda":
        raise RuntimeError("Night-14B full training requires CUDA")
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)


def canonical_csr(value: sp.spmatrix) -> sp.csr_matrix:
    result = value.tocsr().astype(np.float64)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    return result


def row_stochastic(value: sp.spmatrix, self_loop: float = 0.0) -> sp.csr_matrix:
    result = canonical_csr(value)
    if float(self_loop) > 0.0:
        result = result + sp.eye(result.shape[0], format="csr") * float(self_loop)
    degree = np.asarray(result.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("operator contains an isolated observation")
    return canonical_csr(sp.diags(1.0 / degree) @ result)


def row_stochastic_with_identity_abstention(value: sp.spmatrix) -> Tuple[sp.csr_matrix, int]:
    """Normalize accepted edges while making edge-less rows exact identity.

    A selective operator may reject every spatial edge incident to one spot.
    Such a spot must retain its own representation; silently shrinking it toward
    zero would turn abstention into an unregistered high-pass operation.
    """
    result = canonical_csr(value)
    degree = np.asarray(result.sum(axis=1)).ravel()
    isolated = np.flatnonzero(degree <= 0)
    if len(isolated):
        patch = sp.coo_matrix(
            (np.ones(len(isolated), dtype=np.float64), (isolated, isolated)),
            shape=result.shape,
        ).tocsr()
        result = canonical_csr(result + patch)
    return row_stochastic(result), int(len(isolated))


def spatial_operator(coordinates: np.ndarray, k: int) -> sp.csr_matrix:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [N,2]")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates contain non-finite values")
    n = len(coordinates)
    k = int(k)
    if k <= 0 or k >= n:
        raise ValueError("invalid spatial neighborhood size")
    neighbor = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbor.fit(coordinates)
    distance, index = neighbor.kneighbors(coordinates)
    row = np.repeat(np.arange(n, dtype=np.int64), k)
    col = index[:, 1:].reshape(-1).astype(np.int64)
    dist = distance[:, 1:].reshape(-1)
    positive = dist[dist > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    weight = np.exp(-0.5 * (dist / max(scale, 1e-12)) ** 2)
    graph = sp.coo_matrix((weight, (row, col)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return row_stochastic(graph)


def _row_l2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def empirical_edge_states(
    z1: np.ndarray,
    z2: np.ndarray,
    operator: sp.spmatrix,
    interior_quantile: float,
    boundary_quantile: float,
    conflict_quantile: float,
    floor: float = 0.0,
) -> Tuple[sp.csr_matrix, Dict[str, float]]:
    """Build a rank-calibrated, dataset-identity-blind TSPR operator.

    Ranks calibrate the two modality similarities before they are compared.
    An edge propagates only when both ranks clear ``interior_quantile`` and
    their rank disagreement is below ``conflict_quantile``.  Both-low edges
    are recorded as boundary evidence; discordant edges are recorded as
    conflict.  ``floor`` is an explicit propagation floor for ablations.
    """
    if not (0.0 <= boundary_quantile < interior_quantile <= 1.0):
        raise ValueError("edge-state quantiles are not ordered")
    if not (0.0 <= conflict_quantile <= 1.0 and 0.0 <= floor <= 1.0):
        raise ValueError("edge-state scalar outside [0,1]")
    graph = canonical_csr(operator)
    row, col = graph.nonzero()
    a = _row_l2(z1)
    b = _row_l2(z2)
    if a.shape[0] != graph.shape[0] or b.shape[0] != graph.shape[0]:
        raise ValueError("edge-state observation mismatch")
    support1 = np.sum(a[row] * a[col], axis=1)
    support2 = np.sum(b[row] * b[col], axis=1)
    n = max(1, len(row))
    rank1 = (rankdata(support1, method="average") - 0.5) / n
    rank2 = (rankdata(support2, method="average") - 0.5) / n
    minimum = np.minimum(rank1, rank2)
    maximum = np.maximum(rank1, rank2)
    disagreement = np.abs(rank1 - rank2)
    interior = (minimum >= float(interior_quantile)) & (
        disagreement <= float(conflict_quantile)
    )
    boundary = maximum <= float(boundary_quantile)
    conflict = (~interior) & (~boundary) & (
        disagreement > float(conflict_quantile)
    )
    # Smooth confidence inside the accepted state prevents binary edge ties.
    span = max(1e-8, 1.0 - float(interior_quantile))
    confidence = np.clip((minimum - float(interior_quantile)) / span, 0.0, 1.0)
    trust = np.maximum(float(floor), interior.astype(np.float64) * (0.5 + 0.5 * confidence))
    weighted = sp.coo_matrix(
        (graph.data * trust, (row, col)), shape=graph.shape
    ).tocsr()
    weighted = weighted.maximum(weighted.T)
    # Exact identity fallback is represented by the caller when no edge survives.
    nonzero = int(np.count_nonzero(trust > 0))
    diagnostics = {
        "edge_count": int(len(row)),
        "interior_fraction": float(np.mean(interior)),
        "boundary_fraction": float(np.mean(boundary)),
        "conflict_fraction": float(np.mean(conflict)),
        "abstain_fraction": float(np.mean(~interior)),
        "trust_mean": float(np.mean(trust)),
        "surviving_directed_edges": nonzero,
        "dense_n_by_n_count": 0,
    }
    if weighted.nnz == 0:
        return sp.csr_matrix(graph.shape, dtype=np.float64), diagnostics
    normalized, isolated = row_stochastic_with_identity_abstention(weighted)
    diagnostics["identity_abstention_rows"] = isolated
    return normalized, diagnostics


def diffuse(
    value: np.ndarray,
    operator: sp.spmatrix,
    beta: float,
    steps: int,
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    beta = float(beta)
    if not 0.0 <= beta <= 1.0 or int(steps) < 0:
        raise ValueError("invalid diffusion configuration")
    result = value.copy()
    for _ in range(int(steps)):
        result = ((1.0 - beta) * result + beta * operator.dot(result)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("diffusion produced non-finite values")
    return result


def multiscale_filter(
    value: np.ndarray,
    operators: Sequence[sp.spmatrix],
    weights: Sequence[float],
) -> np.ndarray:
    if len(operators) + 1 != len(weights):
        raise ValueError("weights must include identity plus every operator")
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("multiscale weights must be a convex combination")
    base = np.asarray(value, dtype=np.float32)
    result = weights[0] * base
    for weight, operator in zip(weights[1:], operators):
        result = result + float(weight) * operator.dot(base)
    return np.asarray(result, dtype=np.float32)


def anchored_majority_refine(
    initial: np.ndarray,
    operator: sp.spmatrix,
    n_clusters: int,
    anchor: float,
    iterations: int,
) -> np.ndarray:
    """Known-K spatial Potts head with an immutable molecular anchor."""
    initial = np.asarray(initial, dtype=np.int64)
    unique = np.unique(initial)
    if len(unique) != int(n_clusters):
        raise ValueError("initial partition does not contain registered K")
    lookup = {int(value): i for i, value in enumerate(sorted(map(int, unique)))}
    encoded = np.asarray([lookup[int(x)] for x in initial], dtype=np.int64)
    eye = np.eye(int(n_clusters), dtype=np.float32)
    base = eye[encoded]
    current = encoded.copy()
    for _ in range(int(iterations)):
        score = operator.dot(eye[current]) + float(anchor) * base
        updated = np.argmax(score, axis=1).astype(np.int64)
        if len(np.unique(updated)) != int(n_clusters):
            break
        if np.array_equal(updated, current):
            break
        current = updated
    return current


class UnifiedEdgeStateModel(nn.Module):
    """Night-14A core plus an in-forward three-state propagation layer."""

    MODES = {"IDENTITY", "FIXED_LOW", "SUPPORT_ONLY", "TSPR"}

    def __init__(self, input1: int, input2: int, config: Mapping[str, object]) -> None:
        super().__init__()
        self.config = dict(config)
        self.mode = str(config["edge_mode"])
        if self.mode not in self.MODES:
            raise ValueError("unknown edge-state mode")
        self.base = UnifiedGraphAutoencoder(input1, input2, config["base_config"])
        latent = int(config["base_config"]["latent_dim"])
        self.output_norm = nn.LayerNorm(latent)
        initial = float(config["initial_low_strength"])
        initial = min(max(initial, 1e-4), 1.0 - 1e-4)
        self.low_logit = nn.Parameter(torch.tensor(np.log(initial / (1.0 - initial)), dtype=torch.float32))
        high = float(config.get("initial_high_strength", 0.0))
        self.high_strength = nn.Parameter(torch.tensor(high, dtype=torch.float32))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        output = self.base(x1, x2, edge_index, edge_weight)
        z1, z2, fused = output["z1"], output["z2"], output["fused"]
        row, col = edge_index[0], edge_index[1]
        support1 = (F.cosine_similarity(z1[row], z1[col], dim=1) + 1.0) * 0.5
        support2 = (F.cosine_similarity(z2[row], z2[col], dim=1) + 1.0) * 0.5
        joint = torch.sqrt((support1 * support2).clamp_min(0.0))
        conflict = torch.abs(support1 - support2)
        if self.mode == "IDENTITY":
            trust = torch.zeros_like(joint)
        elif self.mode == "FIXED_LOW":
            trust = torch.ones_like(joint)
        elif self.mode == "SUPPORT_ONLY":
            trust = joint
        else:
            interior = float(self.config["support_scale"]) * (
                joint - float(self.config["support_center"])
            ) - float(self.config["conflict_scale"]) * conflict
            boundary = float(self.config["boundary_scale"]) * (
                float(self.config["boundary_center"]) - joint
            ) - 0.5 * float(self.config["conflict_scale"]) * conflict
            conflict_logit = float(self.config["conflict_scale"]) * (
                conflict - float(self.config["conflict_center"])
            )
            state = torch.softmax(torch.stack((interior, boundary, conflict_logit), dim=1), dim=1)
            trust = state[:, 0]
        low = weighted_neighbor_mean(fused, edge_index, edge_weight * trust)
        uniform = weighted_neighbor_mean(fused, edge_index, edge_weight)
        beta = torch.sigmoid(self.low_logit)
        gamma = torch.tanh(self.high_strength)
        if self.mode == "IDENTITY":
            edge_fused = fused
        else:
            edge_fused = self.output_norm(
                fused + beta * (low - fused) + gamma * (fused - uniform)
            )
        output = dict(output)
        output.update({
            "edge_fused": edge_fused,
            "edge_trust": trust,
            "joint_support": joint,
            "edge_conflict": conflict,
            "low_strength": beta,
            "high_strength": gamma,
            "edge_recon1": self.base.decoder1(edge_fused),
            "edge_recon2": self.base.decoder2(edge_fused),
        })
        return output


def edge_state_loss(
    model: UnifiedEdgeStateModel,
    output: Mapping[str, torch.Tensor],
    x1: torch.Tensor,
    x2: torch.Tensor,
    edge_index: torch.Tensor,
    weights: Mapping[str, object],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    components = loss_components(output, x1, x2, edge_index)
    base_loss, audit = unsupervised_loss(
        model.base, components, model.config["base_config"]["loss_weights"]
    )
    edge_reconstruction = 0.5 * (
        F.mse_loss(output["edge_recon1"], x1)
        + F.mse_loss(output["edge_recon2"], x2)
    )
    row, col = edge_index[0], edge_index[1]
    smooth = (
        output["edge_trust"]
        * (output["edge_fused"][row] - output["edge_fused"][col]).square().mean(dim=1)
    ).mean()
    retain = (
        (1.0 - output["edge_trust"])
        * torch.abs(
            F.cosine_similarity(output["edge_fused"][row], output["edge_fused"][col], dim=1)
            - F.cosine_similarity(output["fused"][row], output["fused"][col], dim=1)
        )
    ).mean()
    total = (
        base_loss
        + float(weights["edge_reconstruction"]) * edge_reconstruction
        + float(weights["trusted_smoothness"]) * smooth
        + float(weights["rejected_edge_retention"]) * retain
    )
    audit.update({
        "edge_reconstruction": float(edge_reconstruction.detach().cpu()),
        "trusted_smoothness": float(smooth.detach().cpu()),
        "rejected_edge_retention": float(retain.detach().cpu()),
        "low_strength": float(output["low_strength"].detach().cpu()),
        "high_strength": float(output["high_strength"].detach().cpu()),
        "edge_trust_mean": float(output["edge_trust"].mean().detach().cpu()),
        "total_with_edge_state": float(total.detach().cpu()),
    })
    return total, audit


__all__ = [
    "UnifiedEdgeStateModel",
    "anchored_majority_refine",
    "array_sha256",
    "canonical_sha256",
    "diffuse",
    "edge_state_loss",
    "empirical_edge_states",
    "initialize_cuda_device",
    "multiscale_filter",
    "row_stochastic",
    "spatial_operator",
    "state_sha256",
]
