"""Sparse spaMGCN-style backbone with matched signed-relation distillation.

This is a clean-room, source-faithful *sparse port*, not an official spaMGCN
replay.  The public backbone supplies the two per-view AE/graph encoders,
multi-order graph propagation and late fusion.  Dense N x N adjacency losses
from the public implementation are replaced by registered-edge positives and
deterministic sparse negatives.  MSRD is the only project-specific module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


ARM_NAMES = (
    "T0_STRONG_CARRIER",
    "B0_BACKBONE_ONLY",
    "B1_BACKBONE_POINTWISE_ANCHOR",
    "B2_POSITIVE_RELATION_ONLY",
    "B3_BOUNDARY_RELATION_ONLY",
    "FULL_SIGNED_RELATIONAL_DISTILLATION",
)


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def robust_standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    median = np.median(value, axis=0, keepdims=True)
    mad = np.median(np.abs(value - median), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    result = (value - median) / scale
    result = np.clip(result, -8.0, 8.0)
    if not np.isfinite(result).all():
        raise ValueError("non-finite standardized view")
    return result.astype(np.float32)


def row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return (value / np.maximum(norm, 1e-8)).astype(np.float32)


def prepare_graph(graph: sp.csr_matrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square")
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz == 0 or np.any(graph.data < 0) or not np.isfinite(graph.data).all():
        raise ValueError("graph must be finite, nonnegative, and non-empty")
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = (sp.diags(inv_sqrt) @ graph @ sp.diags(inv_sqrt)).tocsr()
    normalized = normalized + sp.eye(graph.shape[0], format="csr", dtype=np.float64)
    degree2 = np.asarray(normalized.sum(axis=1)).ravel()
    normalized = (sp.diags(1.0 / np.maximum(degree2, 1e-12)) @ normalized).tocsr()
    return normalized.astype(np.float32)


def upper_edges(graph: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = sp.triu(sp.csr_matrix(graph).maximum(sp.csr_matrix(graph).T), k=1).tocoo()
    order = np.lexsort((upper.col, upper.row))
    return (
        np.asarray(upper.row[order], dtype=np.int64),
        np.asarray(upper.col[order], dtype=np.int64),
        np.asarray(upper.data[order], dtype=np.float32),
    )


def scipy_to_torch_sparse(graph: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = graph.tocoo()
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def _cosine_on_edges(view: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    normalized = row_normalize(view)
    return np.sum(normalized[rows] * normalized[cols], axis=1).astype(np.float64)


@dataclass(frozen=True)
class RelationEvidence:
    positive_rows: np.ndarray
    positive_cols: np.ndarray
    positive_weights: np.ndarray
    boundary_rows: np.ndarray
    boundary_cols: np.ndarray
    boundary_weights: np.ndarray
    positive_threshold: float
    boundary_threshold: float
    positive_fraction: float
    boundary_fraction: float


def relation_evidence(
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.csr_matrix,
    positive_quantile: float = 0.70,
    boundary_quantile: float = 0.30,
) -> RelationEvidence:
    """Create label-free signed relations on registered sparse spatial edges.

    Positive edges require high carrier similarity and non-low support in both
    raw modality views. Boundary edges require low carrier similarity and
    simultaneous low support in both views. Ambiguous/conflicting edges abstain.
    """
    rows, cols, base = upper_edges(graph)
    if rows.size == 0:
        raise ValueError("no registered edges")
    carrier_cos = _cosine_on_edges(retained, rows, cols)
    view1_cos = _cosine_on_edges(view1, rows, cols)
    view2_cos = _cosine_on_edges(view2, rows, cols)
    q_pos = float(np.quantile(carrier_cos, positive_quantile))
    q_boundary = float(np.quantile(carrier_cos, boundary_quantile))
    v1_mid = float(np.median(view1_cos))
    v2_mid = float(np.median(view2_cos))
    positive = (carrier_cos >= q_pos) & (view1_cos >= v1_mid) & (view2_cos >= v2_mid)
    boundary = (carrier_cos <= q_boundary) & (view1_cos <= v1_mid) & (view2_cos <= v2_mid)
    if int(positive.sum()) < max(8, int(0.005 * rows.size)):
        raise ValueError("positive relation support is numerically degenerate")
    if int(boundary.sum()) < max(8, int(0.005 * rows.size)):
        raise ValueError("boundary relation support is numerically degenerate")
    pos_scale = max(1e-8, 1.0 - q_pos)
    neg_scale = max(1e-8, q_boundary + 1.0)
    pos_w = base[positive] * np.clip((carrier_cos[positive] - q_pos) / pos_scale, 0.05, 1.0)
    neg_w = base[boundary] * np.clip((q_boundary - carrier_cos[boundary]) / neg_scale, 0.05, 1.0)
    pos_w = (pos_w / max(float(np.mean(pos_w)), 1e-8)).astype(np.float32)
    neg_w = (neg_w / max(float(np.mean(neg_w)), 1e-8)).astype(np.float32)
    return RelationEvidence(
        rows[positive], cols[positive], pos_w,
        rows[boundary], cols[boundary], neg_w,
        q_pos, q_boundary,
        float(positive.mean()), float(boundary.mean()),
    )


def deterministic_negative_pairs(n: int, graph: sp.csr_matrix, count: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, _ = upper_edges(graph)
    occupied = set((int(i), int(j)) for i, j in zip(rows, cols))
    rng = np.random.RandomState(seed)
    result = []
    attempts = 0
    target = max(1, int(count))
    while len(result) < target and attempts < target * 50:
        i = int(rng.randint(0, n)); j = int(rng.randint(0, n))
        attempts += 1
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) in occupied:
            continue
        occupied.add((i, j)); result.append((i, j))
    if len(result) < target:
        raise RuntimeError("could not construct sparse negative pairs")
    array = np.asarray(result, dtype=np.int64)
    return array[:, 0], array[:, 1]


@dataclass(frozen=True)
class MSRDConfig:
    config_id: str
    family: str
    steps: int = 300
    hidden_dim: int = 96
    graph_order: int = 4
    sigma: float = 0.5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    reconstruction_weight: float = 1.0
    sparse_graph_weight: float = 0.10
    cross_modal_weight: float = 0.05
    point_anchor_weight: float = 0.20
    positive_relation_weight: float = 0.25
    boundary_relation_weight: float = 0.25
    boundary_margin: float = 0.15
    positive_quantile: float = 0.70
    boundary_quantile: float = 0.30
    temperature: float = 0.20

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class MultiOrderGraphEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, order: int):
        super().__init__()
        self.order = int(order)
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.attn1 = torch.nn.Linear(hidden_dim, max(8, hidden_dim // 2))
        self.attn2 = torch.nn.Linear(max(8, hidden_dim // 2), 1)
        self.output = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current = F.leaky_relu(self.input(x), 0.2)
        orders = []
        for _ in range(self.order):
            current = torch.sparse.mm(adjacency, current)
            orders.append(current)
        stacked = torch.stack(orders, dim=1)
        summaries = stacked.mean(dim=0)
        scores = self.attn2(torch.tanh(self.attn1(summaries))).squeeze(-1)
        alpha = torch.softmax(scores, dim=0)
        combined = (stacked * alpha.view(1, -1, 1)).sum(dim=1)
        return self.output(combined), alpha


class ViewEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, order: int):
        super().__init__()
        self.ae = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.graph = MultiOrderGraphEncoder(input_dim, hidden_dim, latent_dim, order)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, sigma: float):
        ae = self.ae(x)
        graph, alpha = self.graph(x, adjacency)
        return (1.0 - sigma) * ae + sigma * graph, ae, graph, alpha


class SparseMGCNPort(torch.nn.Module):
    def __init__(self, view1_dim: int, view2_dim: int, latent_dim: int, config: MSRDConfig):
        super().__init__()
        self.config = config
        self.view1 = ViewEncoder(view1_dim, config.hidden_dim, latent_dim, config.graph_order)
        self.view2 = ViewEncoder(view2_dim, config.hidden_dim, latent_dim, config.graph_order)
        self.fusion = torch.nn.Linear(2 * latent_dim, latent_dim)
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, config.hidden_dim), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(config.hidden_dim, view1_dim),
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, config.hidden_dim), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(config.hidden_dim, view2_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, adjacency: torch.Tensor):
        z1, ae1, graph1, alpha1 = self.view1(x1, adjacency, self.config.sigma)
        z2, ae2, graph2, alpha2 = self.view2(x2, adjacency, self.config.sigma)
        fused = self.fusion(torch.cat([z1, z2], dim=1))
        return {
            "z": fused, "z1": z1, "z2": z2,
            "ae1": ae1, "ae2": ae2, "graph1": graph1, "graph2": graph2,
            "alpha1": alpha1, "alpha2": alpha2,
            "x1_hat": self.decoder1(fused), "x2_hat": self.decoder2(fused),
        }


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / torch.clamp(torch.sum(weight), min=1e-8)


def _edge_cosine(z: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(z, p=2, dim=1, eps=1e-8)
    return torch.sum(normalized[rows] * normalized[cols], dim=1)


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _losses(
    output: Mapping[str, torch.Tensor], x1: torch.Tensor, x2: torch.Tensor,
    retained: torch.Tensor, positive: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    boundary: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    graph_positive: Tuple[torch.Tensor, torch.Tensor],
    graph_negative: Tuple[torch.Tensor, torch.Tensor], config: MSRDConfig, arm: str,
) -> Dict[str, torch.Tensor]:
    z = output["z"]
    reconstruction = F.mse_loss(output["x1_hat"], x1) + F.mse_loss(output["x2_hat"], x2)
    pos_graph = _edge_cosine(z, graph_positive[0], graph_positive[1])
    neg_graph = _edge_cosine(z, graph_negative[0], graph_negative[1])
    sparse_graph = -F.logsigmoid(pos_graph / config.temperature).mean() - F.logsigmoid(-neg_graph / config.temperature).mean()
    cross_modal = (1.0 - F.cosine_similarity(output["z1"], output["z2"], dim=1)).mean()
    point_anchor = F.mse_loss(F.normalize(z, dim=1), F.normalize(retained, dim=1))
    pos_cos = _edge_cosine(z, positive[0], positive[1])
    positive_relation = _weighted_mean(1.0 - pos_cos, positive[2])
    boundary_cos = _edge_cosine(z, boundary[0], boundary[1])
    boundary_relation = _weighted_mean(F.relu(boundary_cos - config.boundary_margin).square(), boundary[2])
    total = (config.reconstruction_weight * reconstruction
             + config.sparse_graph_weight * sparse_graph
             + config.cross_modal_weight * cross_modal)
    if arm == "B1_BACKBONE_POINTWISE_ANCHOR":
        total = total + config.point_anchor_weight * point_anchor
    if arm in {"B2_POSITIVE_RELATION_ONLY", "FULL_SIGNED_RELATIONAL_DISTILLATION"}:
        total = total + config.positive_relation_weight * positive_relation
    if arm in {"B3_BOUNDARY_RELATION_ONLY", "FULL_SIGNED_RELATIONAL_DISTILLATION"}:
        total = total + config.boundary_relation_weight * boundary_relation
    return {
        "total": total, "reconstruction": reconstruction, "sparse_graph": sparse_graph,
        "cross_modal": cross_modal, "point_anchor": point_anchor,
        "positive_relation": positive_relation, "boundary_relation": boundary_relation,
    }


def build_model_and_inputs(
    view1: np.ndarray, view2: np.ndarray, retained: np.ndarray,
    graph: sp.csr_matrix, config: MSRDConfig, seed: int, device: str,
):
    set_determinism(seed)
    dev = torch.device(device)
    x1_np = robust_standardize(view1)
    x2_np = robust_standardize(view2)
    retained_np = robust_standardize(retained)
    adjacency_np = prepare_graph(graph)
    model = SparseMGCNPort(x1_np.shape[1], x2_np.shape[1], retained_np.shape[1], config).to(dev)
    inputs = (
        torch.as_tensor(x1_np, device=dev), torch.as_tensor(x2_np, device=dev),
        torch.as_tensor(retained_np, device=dev), scipy_to_torch_sparse(adjacency_np, dev),
    )
    return model, inputs, retained_np


def train_msrd(
    view1: np.ndarray, view2: np.ndarray, retained: np.ndarray, graph: sp.csr_matrix,
    config: MSRDConfig, arm: str, seed: int, device: str = "cuda",
):
    if arm not in ARM_NAMES[1:]:
        raise ValueError("train_msrd requires a trainable arm")
    model, inputs, retained_np = build_model_and_inputs(view1, view2, retained, graph, config, seed, device)
    x1, x2, retained_t, adjacency = inputs
    evidence = relation_evidence(retained_np, x1.detach().cpu().numpy(), x2.detach().cpu().numpy(), graph,
                                 config.positive_quantile, config.boundary_quantile)
    graph_rows, graph_cols, _ = upper_edges(graph)
    neg_rows, neg_cols = deterministic_negative_pairs(len(retained_np), graph, len(graph_rows), seed + 991)
    def edge_tensors(rows, cols, weights=None):
        result = [torch.as_tensor(rows, dtype=torch.long, device=x1.device), torch.as_tensor(cols, dtype=torch.long, device=x1.device)]
        if weights is not None:
            result.append(torch.as_tensor(weights, dtype=torch.float32, device=x1.device))
        return tuple(result)
    positive = edge_tensors(evidence.positive_rows, evidence.positive_cols, evidence.positive_weights)
    boundary = edge_tensors(evidence.boundary_rows, evidence.boundary_cols, evidence.boundary_weights)
    graph_positive = edge_tensors(graph_rows, graph_cols)
    graph_negative = edge_tensors(neg_rows, neg_cols)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    before = _state_sha(model)
    snapshots = []
    checkpoints = {0, 9, 49, 99, 199, config.steps - 1}
    loss_history = []
    max_grad_norm = 0.0
    for step in range(config.steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(x1, x2, adjacency)
        losses = _losses(output, x1, x2, retained_t, positive, boundary, graph_positive, graph_negative, config, arm)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("non-finite training loss")
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        max_grad_norm = max(max_grad_norm, float(grad_norm.detach().cpu()))
        optimizer.step()
        loss_history.append(float(losses["total"].detach().cpu()))
        if step in checkpoints:
            snapshots.append({"step": step + 1, **{key: float(value.detach().cpu()) for key, value in losses.items()}})
    model.eval()
    with torch.no_grad():
        representation = model(x1, x2, adjacency)["z"].detach().cpu().numpy().astype(np.float32)
    after = _state_sha(model)
    window = min(50, len(loss_history) // 2)
    early = float(np.mean(loss_history[-2 * window:-window]))
    late = float(np.mean(loss_history[-window:]))
    diagnostics = {
        "optimizer_steps": config.steps, "parameter_state_sha_before": before,
        "parameter_state_sha_after": after, "parameter_changed": before != after,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "max_gradient_norm": max_grad_norm, "loss_snapshots": snapshots,
        "loss_relative_change_last_windows": (late - early) / max(abs(early), 1e-12),
        "positive_relation_count": int(len(evidence.positive_rows)),
        "boundary_relation_count": int(len(evidence.boundary_rows)),
        "positive_relation_fraction": evidence.positive_fraction,
        "boundary_relation_fraction": evidence.boundary_fraction,
        "positive_threshold": evidence.positive_threshold,
        "boundary_threshold": evidence.boundary_threshold,
        "representation_sha256": sha256_array(representation),
        "input_shapes": {"view1": list(view1.shape), "view2": list(view2.shape), "retained": list(retained.shape)},
        "graph_shape": list(graph.shape), "graph_nnz": int(graph.nnz), "dense_nxn_allocated": False,
    }
    if not diagnostics["parameter_changed"] or diagnostics["max_gradient_norm"] <= 0:
        raise RuntimeError("optimizer did not update parameters")
    return representation, {key: value.detach().cpu() for key, value in model.state_dict().items()}, diagnostics


def reload_msrd(
    view1: np.ndarray, view2: np.ndarray, retained: np.ndarray, graph: sp.csr_matrix,
    config: MSRDConfig, seed: int, state: Mapping[str, torch.Tensor], device: str = "cpu",
) -> np.ndarray:
    model, inputs, _ = build_model_and_inputs(view1, view2, retained, graph, config, seed, device)
    incompatible = model.load_state_dict(dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint reload failed")
    model.eval()
    with torch.no_grad():
        return model(inputs[0], inputs[1], inputs[3])["z"].detach().cpu().numpy().astype(np.float32)


def common_kmeans_endpoint(representation: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    representation = np.asarray(representation, dtype=np.float32)
    partition = KMeans(n_clusters=int(k), n_init=20, random_state=int(seed)).fit_predict(representation)
    if np.unique(partition).size != int(k):
        raise RuntimeError("endpoint did not produce exact K")
    return partition.astype(np.int32)

