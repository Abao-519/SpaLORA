"""Direct N x K geometry junction used by the Night-22A method P0.

The module intentionally does not learn a new node embedding.  Its trainable
state is the partition itself, cluster-conditioned graph-mixture weights, and
an elliptical emission model on a locked representation.  This keeps the
representation/endpoint attribution explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from SpaLORA.night22a_geometry import mass_normalize, self_tuning_knn, standardize, undirected_no_diag


@dataclass(frozen=True)
class JunctionConfig:
    steps: int = 240
    learning_rate: float = 0.025
    emission_weight: float = 0.35
    graph_weight: float = 1.0
    conditional_entropy_weight: float = 0.025
    volume_floor_weight: float = 2.0
    volume_floor_fraction: float = 0.20
    anchor_weight: float = 0.12
    graph_mixture_entropy_weight: float = 0.01
    logit_bound: float = 8.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def upper_edges(graph: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = sp.triu(undirected_no_diag(graph), k=1).tocoo()
    if upper.nnz == 0:
        raise ValueError("graph has no undirected edges")
    if not np.isfinite(upper.data).all() or np.any(upper.data <= 0):
        raise ValueError("graph edges must be finite and positive")
    return (
        upper.row.astype(np.int64),
        upper.col.astype(np.int64),
        upper.data.astype(np.float64),
    )


def graph_bank(
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    spatial: sp.spmatrix,
    neighbors: int = 12,
    secondary_neighbors: int = 24,
) -> Dict[str, sp.csr_matrix]:
    """Return equal-total-mass graphs with common semantics across families."""
    graphs = {
        f"retained_feature_k{int(neighbors)}": self_tuning_knn(retained, neighbors),
        f"retained_feature_k{int(secondary_neighbors)}": self_tuning_knn(retained, secondary_neighbors),
        f"view1_feature_k{int(neighbors)}": self_tuning_knn(view1, neighbors),
        f"view2_feature_k{int(neighbors)}": self_tuning_knn(view2, neighbors),
        "registered_spatial": undirected_no_diag(spatial),
    }
    return {name: mass_normalize(graph) for name, graph in graphs.items()}


def initialize_emission(x: np.ndarray, start: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x = standardize(x).astype(np.float32)
    start = np.asarray(start, dtype=np.int64)
    centers: List[np.ndarray] = []
    scales: List[np.ndarray] = []
    global_scale = np.maximum(np.std(x, axis=0), 0.25)
    for label in range(int(k)):
        subset = x[start == label]
        if subset.size == 0:
            raise ValueError("start partition contains an empty cluster")
        centers.append(np.mean(subset, axis=0))
        scales.append(np.maximum(np.std(subset, axis=0), 0.20 * global_scale))
    return np.stack(centers).astype(np.float32), np.stack(scales).astype(np.float32)


class GeometryJunction(nn.Module):
    """Direct soft-partition optimizer with cluster-conditioned sparse geometry."""

    def __init__(
        self,
        x: np.ndarray,
        start: np.ndarray,
        k: int,
        graphs: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        config: JunctionConfig,
        arm: str,
    ) -> None:
        super().__init__()
        if arm not in {
            "EMISSION_ONLY",
            "SHARED_GRAPH_ONLY",
            "CLUSTER_GRAPH_ONLY",
            "ADDITIVE_SHARED",
            "FULL",
        }:
            raise ValueError(f"unknown arm: {arm}")
        self.arm = arm
        self.k = int(k)
        self.config = config
        xz = standardize(x).astype(np.float32)
        # Numeric inputs and sparse graph arrays are reconstructed from locked
        # authorities on replay.  They are non-persistent buffers so a strict
        # checkpoint contains trainable state only instead of duplicating the
        # carrier for every candidate.
        self.register_buffer("x", torch.from_numpy(xz), persistent=False)
        start = np.asarray(start, dtype=np.int64)
        if np.unique(start).size != self.k:
            raise ValueError("start does not have exact K")
        start_onehot = np.eye(self.k, dtype=np.float32)[start]
        self.register_buffer("start_onehot", torch.from_numpy(start_onehot), persistent=False)
        initial_logits = np.full((len(start), self.k), -3.0, dtype=np.float32)
        initial_logits[np.arange(len(start)), start] = 3.0
        self.logits = nn.Parameter(torch.from_numpy(initial_logits))
        centers, scales = initialize_emission(xz, start, self.k)
        self.centers = nn.Parameter(torch.from_numpy(centers))
        self.log_scales = nn.Parameter(torch.from_numpy(np.log(scales)))
        self.graph_logits = nn.Parameter(torch.zeros(self.k, len(graphs), dtype=torch.float32))

        self.graph_names: List[str] = []
        for index, (rows, cols, weights) in enumerate(graphs):
            if len(rows) != len(cols) or len(rows) != len(weights):
                raise ValueError("edge-array length mismatch")
            if np.any(rows == cols) or np.any(rows < 0) or np.any(cols < 0):
                raise ValueError("invalid graph endpoints")
            self.register_buffer(
                f"rows_{index}", torch.from_numpy(np.asarray(rows, dtype=np.int64)), persistent=False
            )
            self.register_buffer(
                f"cols_{index}", torch.from_numpy(np.asarray(cols, dtype=np.int64)), persistent=False
            )
            self.register_buffer(
                f"weights_{index}", torch.from_numpy(np.asarray(weights, dtype=np.float32)), persistent=False
            )
            degree = np.zeros(len(start), dtype=np.float32)
            np.add.at(degree, rows, weights)
            np.add.at(degree, cols, weights)
            self.register_buffer(f"degree_{index}", torch.from_numpy(degree), persistent=False)
            self.graph_names.append(str(index))

    def q(self) -> torch.Tensor:
        return torch.softmax(torch.clamp(self.logits, -self.config.logit_bound, self.config.logit_bound), dim=1)

    def emission_loss(self, q: torch.Tensor) -> torch.Tensor:
        scales = torch.exp(torch.clamp(self.log_scales, -3.0, 3.0))
        residual = (self.x[:, None, :] - self.centers[None, :, :]) / scales[None, :, :]
        nll = 0.5 * residual.square().mean(dim=2) + torch.log(scales).mean(dim=1)[None, :]
        return torch.sum(q * nll) / q.shape[0]

    def association_matrix(self, q: torch.Tensor) -> torch.Tensor:
        associations: List[torch.Tensor] = []
        for index in range(len(self.graph_names)):
            rows = getattr(self, f"rows_{index}")
            cols = getattr(self, f"cols_{index}")
            weights = getattr(self, f"weights_{index}")
            degree = getattr(self, f"degree_{index}")
            numerator = 2.0 * torch.sum(weights[:, None] * q[rows] * q[cols], dim=0)
            denominator = torch.sum(degree[:, None] * q, dim=0).clamp_min(1e-9)
            associations.append(numerator / denominator)
        return torch.stack(associations, dim=1)

    def graph_loss(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        association = self.association_matrix(q)
        if self.arm in {"SHARED_GRAPH_ONLY", "ADDITIVE_SHARED"}:
            mixture = torch.full_like(association, 1.0 / association.shape[1])
        else:
            mixture = torch.softmax(self.graph_logits, dim=1)
        loss = 1.0 - torch.mean(torch.sum(mixture * association, dim=1))
        return loss, mixture

    def loss(self, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = self.q()
        emission = self.emission_loss(q)
        graph, mixture = self.graph_loss(q)
        conditional_entropy = -torch.mean(torch.sum(q * torch.log(q.clamp_min(1e-9)), dim=1))
        mass = q.mean(dim=0)
        floor = self.config.volume_floor_fraction / self.k
        volume_floor = torch.mean(torch.relu(floor - mass).square())
        anchor_scale = max(0.0, 1.0 - float(step) / max(1.0, 0.60 * self.config.steps))
        anchor = torch.mean(torch.sum((q - self.start_onehot).square(), dim=1))
        mixture_entropy = -torch.mean(torch.sum(mixture * torch.log(mixture.clamp_min(1e-9)), dim=1))

        emission_weight = (
            self.config.emission_weight if self.arm in {"EMISSION_ONLY", "ADDITIVE_SHARED", "FULL"} else 0.0
        )
        graph_weight = (
            self.config.graph_weight
            if self.arm in {"SHARED_GRAPH_ONLY", "CLUSTER_GRAPH_ONLY", "ADDITIVE_SHARED", "FULL"}
            else 0.0
        )
        total = (
            emission_weight * emission
            + graph_weight * graph
            + self.config.conditional_entropy_weight * conditional_entropy
            + self.config.volume_floor_weight * volume_floor
            + self.config.anchor_weight * anchor_scale * anchor
        )
        if self.arm in {"CLUSTER_GRAPH_ONLY", "FULL"}:
            # A small negative entropy coefficient avoids a brittle single-graph
            # collapse while still allowing cluster-specific geometry.
            total -= self.config.graph_mixture_entropy_weight * mixture_entropy
        diagnostics = {
            "total": total.detach(),
            "emission": emission.detach(),
            "graph": graph.detach(),
            "conditional_entropy": conditional_entropy.detach(),
            "volume_floor": volume_floor.detach(),
            "anchor": anchor.detach(),
            "mixture_entropy": mixture_entropy.detach(),
            "min_soft_mass": mass.min().detach(),
        }
        return total, diagnostics


def exact_k_repair(partition: np.ndarray, probabilities: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    partition = np.asarray(partition, dtype=np.int32).copy()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    repairs = 0
    for missing in sorted(set(range(int(k))) - set(np.unique(partition).tolist())):
        counts = np.bincount(partition, minlength=int(k))
        donors = np.flatnonzero(counts[partition] > 1)
        if donors.size == 0:
            raise RuntimeError("cannot repair empty cluster")
        chosen = int(donors[np.argmax(probabilities[donors, missing])])
        partition[chosen] = int(missing)
        repairs += 1
    if np.unique(partition).size != int(k):
        raise RuntimeError("exact-K repair failed")
    return partition, repairs


def hard_partition(model: GeometryJunction) -> Tuple[np.ndarray, np.ndarray, int]:
    with torch.no_grad():
        probabilities = model.q().detach().cpu().numpy()
    partition = probabilities.argmax(axis=1).astype(np.int32)
    partition, repairs = exact_k_repair(partition, probabilities, model.k)
    return partition, probabilities, repairs
