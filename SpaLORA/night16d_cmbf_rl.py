"""Trainable tri-state cross-modal boundary-field residual representation.

The core consumes only numeric modality views, a sparse spatial graph, a
label-free teacher partition/start bank, and a numeric configuration.  It has
no dataset or annotation argument.  Every registered sparse edge is assigned
support, consensus-boundary, and modality-conflict mass.  These states drive
different representation operators: support low-pass, boundary high-pass,
and conflict-private preservation.  Rejected edge mass and teacher trust
anchor the residual to the strong input representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CMBFRLConfig:
    latent_dim: int = 32
    hidden_dim: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    training_steps: int = 80
    teacher_scale: float = 3.0
    content_scale: float = 0.25
    residual_scale: float = 0.20
    unary_temperature: float = 1.0
    boundary_margin: float = 0.70
    lambda_reconstruct: float = 1.0
    lambda_support: float = 0.20
    lambda_boundary: float = 0.20
    lambda_conflict: float = 0.10
    lambda_anchor: float = 1.0
    lambda_variance: float = 0.05
    lambda_gate_prior: float = 0.10
    endpoint_iterations: int = 1
    endpoint_core_quantile: float = 0.50
    endpoint_trust_threshold: float = 0.60
    endpoint_move_margin: float = 0.01
    teacher_source: str = "synthetic"
    rank_mode: str = "global"
    operation_mode: str = "full"
    trust_enabled: bool = True
    shuffle_edge_states: bool = False

    def validate(self) -> None:
        if self.latent_dim < 8 or self.hidden_dim < 8:
            raise ValueError("latent_dim and hidden_dim must be at least 8")
        if self.training_steps < 0 or self.endpoint_iterations < 0:
            raise ValueError("step counts must be nonnegative")
        if self.rank_mode not in {"global", "node_local"}:
            raise ValueError("rank_mode must be global or node_local")
        if self.teacher_source not in {"synthetic", "retained"}:
            raise ValueError("teacher_source must be synthetic or retained")
        if self.operation_mode not in {
            "teacher", "teacher_head", "generic", "support", "support_boundary",
            "support_conflict", "full", "full_no_trust",
        }:
            raise ValueError("unsupported operation_mode")
        positive = {
            "learning_rate": self.learning_rate,
            "teacher_scale": self.teacher_scale,
            "unary_temperature": self.unary_temperature,
            "boundary_margin": self.boundary_margin,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = {
            "weight_decay": self.weight_decay,
            "content_scale": self.content_scale,
            "residual_scale": self.residual_scale,
            "lambda_reconstruct": self.lambda_reconstruct,
            "lambda_support": self.lambda_support,
            "lambda_boundary": self.lambda_boundary,
            "lambda_conflict": self.lambda_conflict,
            "lambda_anchor": self.lambda_anchor,
            "lambda_variance": self.lambda_variance,
            "lambda_gate_prior": self.lambda_gate_prior,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 <= float(self.endpoint_core_quantile) <= 1.0:
            raise ValueError("endpoint_core_quantile must be in [0,1]")
        if not 0.0 <= float(self.endpoint_trust_threshold) <= 1.0:
            raise ValueError("endpoint_trust_threshold must be in [0,1]")
        if not np.isfinite(self.endpoint_move_margin) or float(self.endpoint_move_margin) < 0:
            raise ValueError("endpoint_move_margin must be finite and nonnegative")


@dataclass(frozen=True)
class TriStateField:
    graph: sp.csr_matrix
    row: np.ndarray
    col: np.ndarray
    base_weight: np.ndarray
    rank1: np.ndarray
    rank2: np.ndarray
    support: np.ndarray
    boundary: np.ndarray
    conflict: np.ndarray
    rejected_mass: np.ndarray
    node_support: np.ndarray
    node_boundary: np.ndarray
    node_conflict: np.ndarray
    start_stability: np.ndarray
    prototype_confidence: np.ndarray
    trust: np.ndarray


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def config_sha256(config: CMBFRLConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def encode_partition(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1:
        raise ValueError("partition must be one-dimensional")
    return np.unique(value, return_inverse=True)[1].astype(np.int32)


def canonical_graph(graph: sp.spmatrix, n: int) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    if graph.shape != (n, n):
        raise ValueError("graph shape does not match observations")
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sum_duplicates()
    graph.sort_indices()
    if graph.nnz == 0 or not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError("invalid sparse graph")
    return graph


def robust_standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("view must be finite and two-dimensional")
    median = np.median(value, axis=0, keepdims=True)
    mad = np.median(np.abs(value - median), axis=0, keepdims=True)
    fallback = np.std(value, axis=0, keepdims=True)
    scale = np.where(mad > 1e-8, 1.4826 * mad, np.maximum(fallback, 1e-8))
    return np.clip((value - median) / scale, -12.0, 12.0).astype(np.float32)


def _edge_change(value: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((value[row] - value[col]) ** 2, axis=1)).astype(np.float64)


def _edge_rank(value: np.ndarray, indptr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "global":
        return ((rankdata(value, method="average") - 0.5) / len(value)).astype(np.float32)
    result = np.zeros(len(value), dtype=np.float32)
    for node in range(len(indptr) - 1):
        begin, end = int(indptr[node]), int(indptr[node + 1])
        if end == begin:
            continue
        result[begin:end] = (rankdata(value[begin:end], method="average") - 0.5) / (end - begin)
    return result


def _align_partition(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = encode_partition(reference)
    candidate = encode_partition(candidate)
    table = np.zeros((int(reference.max()) + 1, int(candidate.max()) + 1), dtype=np.int64)
    np.add.at(table, (reference, candidate), 1)
    r, c = linear_sum_assignment(-table)
    mapping = {int(cc): int(rr) for rr, cc in zip(r, c)}
    for group in range(table.shape[1]):
        mapping.setdefault(group, int(np.argmax(table[:, group])))
    return np.asarray([mapping[int(x)] for x in candidate], dtype=np.int32)


def _start_stability(initial: np.ndarray, bank: np.ndarray | None) -> np.ndarray:
    if bank is None:
        return np.ones(len(initial), dtype=np.float32)
    bank = np.asarray(bank)
    if bank.ndim != 2 or bank.shape[1] != len(initial):
        raise ValueError("start bank shape mismatch")
    aligned = [_align_partition(initial, row) for row in bank]
    aligned.append(initial)
    return np.mean(np.stack(aligned) == initial[None, :], axis=0).astype(np.float32)


def _prototype_confidence(view1: np.ndarray, view2: np.ndarray, initial: np.ndarray) -> np.ndarray:
    k = int(initial.max()) + 1
    rows = np.arange(len(initial))
    distances = []
    for view in (view1, view2):
        centers = np.stack([np.median(view[initial == group], axis=0) for group in range(k)])
        dist = np.mean((view[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        dist /= max(float(np.median(dist)), 1e-8)
        distances.append(dist)
    merged = 0.5 * (distances[0] + distances[1])
    current = merged[rows, initial]
    alternative = merged.copy()
    alternative[rows, initial] = np.inf
    margin = np.min(alternative, axis=1) - current
    scale = 1.4826 * float(np.median(np.abs(margin - np.median(margin))))
    z = np.clip(margin / max(scale, 1e-8), -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _node_weighted_mean(graph: sp.csr_matrix, edge: np.ndarray) -> np.ndarray:
    weighted = graph.copy()
    weighted.data = graph.data * np.asarray(edge, dtype=np.float64)
    denom = np.asarray(graph.sum(axis=1)).reshape(-1)
    value = np.asarray(weighted.sum(axis=1)).reshape(-1)
    return np.divide(value, denom, out=np.zeros_like(value), where=denom > 0).astype(np.float32)


def build_tri_state_field(
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.spmatrix,
    initial: np.ndarray,
    start_bank: np.ndarray | None,
    rank_mode: str,
    shuffle_seed: int | None = None,
) -> TriStateField:
    initial = encode_partition(initial)
    view1 = robust_standardize(view1)
    view2 = robust_standardize(view2)
    if len(view1) != len(view2) or len(view1) != len(initial):
        raise ValueError("observation mismatch")
    graph = canonical_graph(graph, len(initial))
    row = np.repeat(np.arange(len(initial), dtype=np.int64), np.diff(graph.indptr))
    col = graph.indices.astype(np.int64, copy=False)
    rank1 = _edge_rank(_edge_change(view1, row, col), graph.indptr, rank_mode)
    rank2 = _edge_rank(_edge_change(view2, row, col), graph.indptr, rank_mode)
    support = (1.0 - rank1) * (1.0 - rank2)
    boundary = rank1 * rank2
    conflict = rank1 * (1.0 - rank2) + rank2 * (1.0 - rank1)
    total = np.maximum(support + boundary + conflict, 1e-12)
    support, boundary, conflict = support / total, boundary / total, conflict / total
    if shuffle_seed is not None:
        permutation = np.random.default_rng(int(shuffle_seed)).permutation(len(support))
        support, boundary, conflict = support[permutation], boundary[permutation], conflict[permutation]
    accepted = graph.data * support
    base_mass = np.add.reduceat(graph.data, graph.indptr[:-1])
    accepted_mass = np.add.reduceat(accepted, graph.indptr[:-1])
    rejected = np.divide(base_mass - accepted_mass, base_mass, out=np.zeros_like(base_mass), where=base_mass > 0)
    stability = _start_stability(initial, start_bank)
    prototype = _prototype_confidence(view1, view2, initial)
    node_support = _node_weighted_mean(graph, support)
    node_boundary = _node_weighted_mean(graph, boundary)
    node_conflict = _node_weighted_mean(graph, conflict)
    trust = np.power(
        np.maximum(stability, 1e-6)
        * np.maximum(prototype, 1e-6)
        * np.maximum(1.0 - node_boundary, 1e-6)
        * np.maximum(1.0 - node_conflict, 1e-6),
        0.25,
    ).astype(np.float32)
    return TriStateField(
        graph=graph,
        row=row,
        col=col,
        base_weight=graph.data.astype(np.float32),
        rank1=rank1,
        rank2=rank2,
        support=support.astype(np.float32),
        boundary=boundary.astype(np.float32),
        conflict=conflict.astype(np.float32),
        rejected_mass=rejected.astype(np.float32),
        node_support=node_support,
        node_boundary=node_boundary,
        node_conflict=node_conflict,
        start_stability=stability,
        prototype_confidence=prototype,
        trust=np.clip(trust, 0.0, 1.0),
    )


def teacher_representation(
    view1: np.ndarray,
    view2: np.ndarray,
    initial: np.ndarray,
    latent_dim: int,
    teacher_scale: float,
    content_scale: float,
) -> np.ndarray:
    initial = encode_partition(initial)
    k = int(initial.max()) + 1
    if k >= latent_dim:
        raise ValueError("latent_dim must exceed K")
    fused = np.concatenate([robust_standardize(view1), robust_standardize(view2)], axis=1).astype(np.float64)
    covariance = fused.T @ fused / max(len(fused) - 1, 1)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalue)[::-1][:latent_dim]
    content = fused @ eigenvector[:, order]
    content = robust_standardize(content)
    if content.shape[1] < latent_dim:
        content = np.pad(content, ((0, 0), (0, latent_dim - content.shape[1])))
    code = np.zeros((k, latent_dim), dtype=np.float32)
    code[:, :k] = np.eye(k, dtype=np.float32) - np.float32(1.0 / k)
    return (float(teacher_scale) * code[initial] + float(content_scale) * content[:, :latent_dim]).astype(np.float32)


def retained_teacher_representation(
    retained: np.ndarray,
    initial: np.ndarray,
    latent_dim: int,
    teacher_scale: float,
    content_scale: float,
) -> np.ndarray:
    """Use an exact retained embedding as the teacher content source.

    Only deterministic robust scaling and dimension padding/reduction are
    applied.  The input array hash is audited by the producer, so this cannot
    silently substitute a same-named reconstruction.
    """
    retained = robust_standardize(retained)
    initial = encode_partition(initial)
    k = int(initial.max()) + 1
    if k >= latent_dim:
        raise ValueError("latent_dim must exceed K")
    if retained.shape[1] > latent_dim:
        covariance = retained.astype(np.float64).T @ retained.astype(np.float64) / max(len(retained) - 1, 1)
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalue)[::-1][:latent_dim]
        content = retained @ eigenvector[:, order]
    else:
        content = np.pad(retained, ((0, 0), (0, latent_dim - retained.shape[1])))
    code = np.zeros((k, latent_dim), dtype=np.float32)
    code[:, :k] = np.eye(k, dtype=np.float32) - np.float32(1.0 / k)
    return (float(teacher_scale) * code[initial] + float(content_scale) * content[:, :latent_dim]).astype(np.float32)


def _weighted_message(
    x: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    n = x.shape[0]
    total = torch.zeros_like(x)
    total.index_add_(0, row, weight[:, None] * x[col])
    degree = torch.zeros(n, dtype=x.dtype, device=x.device)
    degree.index_add_(0, row, weight)
    mean = total / degree.clamp_min(1e-12)[:, None]
    return torch.where((degree > 1e-12)[:, None], mean, fallback)


class CMBFResidualLearner(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, config: CMBFRLConfig):
        super().__init__()
        d, h = int(config.latent_dim), int(config.hidden_dim)
        self.adapter1 = nn.Linear(int(input_dim1), d)
        self.adapter2 = nn.Linear(int(input_dim2), d)
        self.conflict_projector = nn.Sequential(nn.Linear(3 * d, h), nn.GELU(), nn.Linear(h, d))
        self.gate = nn.Sequential(nn.Linear(6, h // 2), nn.GELU(), nn.Linear(h // 2, 4))
        self.alpha = nn.Sequential(nn.Linear(6, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        self.residual = nn.Sequential(nn.Linear(4 * d, h), nn.GELU(), nn.Linear(h, d))
        self.decoder1 = nn.Linear(d, int(input_dim1))
        self.decoder2 = nn.Linear(d, int(input_dim2))
        self.config = config

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        teacher: torch.Tensor,
        node_statistics: torch.Tensor,
        trust: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        base: torch.Tensor,
        support: torch.Tensor,
        boundary: torch.Tensor,
        conflict: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p1 = F.layer_norm(self.adapter1(x1), (self.config.latent_dim,))
        p2 = F.layer_norm(self.adapter2(x2), (self.config.latent_dim,))
        shared = 0.5 * (p1 + p2)
        m_support = _weighted_message(shared, row, col, base * support, shared)
        b_mean = _weighted_message(shared, row, col, base * boundary, shared)
        m_boundary = shared - b_mean
        c1 = _weighted_message(p1, row, col, base * conflict, p1)
        c2 = _weighted_message(p2, row, col, base * conflict, p2)
        m_conflict = self.conflict_projector(torch.cat([p1 - p2, c1, c2], dim=1))
        gates = torch.softmax(self.gate(node_statistics), dim=1)
        mode = self.config.operation_mode
        use_support = mode in {"support", "support_boundary", "support_conflict", "full", "full_no_trust"}
        use_boundary = mode in {"support_boundary", "full", "full_no_trust"}
        use_conflict = mode in {"support_conflict", "full", "full_no_trust"}
        r_self = gates[:, 0:1] * (shared - teacher)
        r_support = gates[:, 1:2] * (m_support - shared) if use_support else torch.zeros_like(shared)
        r_boundary = gates[:, 2:3] * m_boundary if use_boundary else torch.zeros_like(shared)
        r_conflict = gates[:, 3:4] * m_conflict if use_conflict else torch.zeros_like(shared)
        relation = r_support + r_boundary + r_conflict
        raw_residual = self.residual(torch.cat([teacher, r_self, relation, p1 - p2], dim=1))
        residual = F.layer_norm(raw_residual, (self.config.latent_dim,))
        alpha = torch.sigmoid(self.alpha(node_statistics)) * float(self.config.residual_scale)
        if self.config.trust_enabled and mode != "full_no_trust":
            alpha = alpha * (1.0 - trust[:, None])
        z = teacher + alpha * residual
        return {
            "z": z,
            "p1": p1,
            "p2": p2,
            "gates": gates,
            "alpha": alpha,
            "support_message": m_support,
            "boundary_message": m_boundary,
            "conflict_message": m_conflict,
            "reconstruction1": self.decoder1(z),
            "reconstruction2": self.decoder2(z),
            "private_reconstruction1": self.decoder1(p1),
            "private_reconstruction2": self.decoder2(p2),
        }


def operation_loss(
    output: dict[str, torch.Tensor],
    x1: torch.Tensor,
    x2: torch.Tensor,
    teacher: torch.Tensor,
    trust: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    base: torch.Tensor,
    support: torch.Tensor,
    boundary: torch.Tensor,
    conflict: torch.Tensor,
    rank1: torch.Tensor,
    rank2: torch.Tensor,
    config: CMBFRLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    z, p1, p2 = output["z"], output["p1"], output["p2"]
    rec = F.mse_loss(output["reconstruction1"], x1) + F.mse_loss(output["reconstruction2"], x2)
    rec = rec + 0.5 * (F.mse_loss(output["private_reconstruction1"], x1) + F.mse_loss(output["private_reconstruction2"], x2))
    edge_z = torch.mean((z[row] - z[col]) ** 2, dim=1)
    support_loss = torch.sum(base * support * edge_z) / torch.sum(base * support).clamp_min(1e-12)
    distance = torch.sqrt(edge_z + 1e-8)
    boundary_loss = torch.sum(base * boundary * F.relu(float(config.boundary_margin) - distance) ** 2) / torch.sum(base * boundary).clamp_min(1e-12)
    d1 = torch.sqrt(torch.mean((p1[row] - p1[col]) ** 2, dim=1) + 1e-8)
    d2 = torch.sqrt(torch.mean((p2[row] - p2[col]) ** 2, dim=1) + 1e-8)
    private_loss = torch.sum(base * conflict * ((d1 - rank1) ** 2 + (d2 - rank2) ** 2)) / torch.sum(base * conflict).clamp_min(1e-12)
    anchor = torch.mean(trust[:, None] * (z - teacher) ** 2)
    std = torch.sqrt(torch.var(z, dim=0, unbiased=False) + 1e-4)
    variance = torch.mean(F.relu(0.5 - std))
    prior = torch.stack(
        [
            torch.clamp(1.0 - trust, min=1e-4),
            torch.clamp(output["support_message"].pow(2).mean(1).sqrt(), min=1e-4),
            torch.clamp(output["boundary_message"].pow(2).mean(1).sqrt(), min=1e-4),
            torch.clamp(output["conflict_message"].pow(2).mean(1).sqrt(), min=1e-4),
        ],
        dim=1,
    )
    # The structural prior prevents the learned gate from silently collapsing
    # to a pure self path.  It is label-free and only uses trust plus the
    # magnitudes of the three relationship-specific operators.
    prior = prior / prior.sum(1, keepdim=True).clamp_min(1e-12)
    gate_prior = torch.mean(torch.sum(prior * (torch.log(prior) - torch.log(output["gates"].clamp_min(1e-8))), dim=1))
    mode = config.operation_mode
    loss = float(config.lambda_reconstruct) * rec + float(config.lambda_anchor) * anchor + float(config.lambda_variance) * variance
    loss = loss + float(config.lambda_gate_prior) * gate_prior
    if mode in {"support", "support_boundary", "support_conflict", "full", "full_no_trust"}:
        loss = loss + float(config.lambda_support) * support_loss
    if mode in {"support_boundary", "full", "full_no_trust"}:
        loss = loss + float(config.lambda_boundary) * boundary_loss
    if mode in {"support_conflict", "full", "full_no_trust"}:
        loss = loss + float(config.lambda_conflict) * private_loss
    detail = {
        "total": float(loss.detach().cpu()),
        "reconstruct": float(rec.detach().cpu()),
        "support": float(support_loss.detach().cpu()),
        "boundary": float(boundary_loss.detach().cpu()),
        "conflict_private": float(private_loss.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
        "variance": float(variance.detach().cpu()),
        "gate_prior": float(gate_prior.detach().cpu()),
    }
    return loss, detail


def graph_from_csr_arrays(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: Iterable[int]) -> sp.csr_matrix:
    shape = tuple(int(x) for x in shape)
    return canonical_graph(sp.csr_matrix((data, indices, indptr), shape=shape), shape[0])
