"""Night-15B stability-anchored prototype residual (SAPR).

The module is intentionally small and clean-room.  It consumes reduced modality
views, a retained strong embedding, a sparse spatial graph, and a bank of
label-free partitions.  Ground-truth labels are not accepted by any public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - local endpoint-only runner
    torch = None
    nn = object
    F = None


def _contiguous(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    unique = np.unique(values)
    if len(unique) != k:
        raise ValueError(f"partition has {len(unique)} clusters; expected {k}")
    lookup = {int(value): index for index, value in enumerate(unique.tolist())}
    return np.asarray([lookup[int(value)] for value in values], dtype=np.int64)


def align_partition(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    """Align candidate cluster IDs to a reference using maximum overlap."""
    reference = _contiguous(reference, k)
    candidate = _contiguous(candidate, k)
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (candidate, reference), 1)
    row, col = linear_sum_assignment(-contingency)
    mapping = np.empty(k, dtype=np.int64)
    mapping[row] = col
    return mapping[candidate]


@dataclass(frozen=True)
class StabilityAnchors:
    consensus: np.ndarray
    confidence: np.ndarray
    boundary: np.ndarray
    interior_weight: np.ndarray
    aligned_partitions: np.ndarray
    medoid_index: int
    mean_pairwise_ari: float
    cardinality_repair_count: int


def build_stability_anchors(
    partitions: Sequence[np.ndarray], graph: sp.spmatrix, k: int
) -> StabilityAnchors:
    """Create label-free stability and boundary evidence from partitions."""
    if len(partitions) < 3:
        raise ValueError("at least three label-free partitions are required")
    values = [_contiguous(np.asarray(item), k) for item in partitions]
    n = len(values[0])
    if any(len(item) != n for item in values):
        raise ValueError("partition observation counts differ")
    pairwise = np.eye(len(values), dtype=np.float64)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            pairwise[i, j] = pairwise[j, i] = adjusted_rand_score(values[i], values[j])
    medoid = int(np.argmax((pairwise.sum(axis=1) - 1.0) / max(len(values) - 1, 1)))
    aligned = np.stack(
        [align_partition(values[medoid], item, k) for item in values], axis=0
    )
    votes = np.zeros((n, k), dtype=np.int32)
    rows = np.broadcast_to(np.arange(n), aligned.shape)
    np.add.at(votes, (rows.reshape(-1), aligned.reshape(-1)), 1)
    consensus = votes.argmax(axis=1).astype(np.int64)
    repair_count = 0
    # Majority voting can erase a small cluster even though every registered
    # teacher contains all K clusters.  Restore non-empty cardinality using the
    # medoid's own members and the largest vote advantage; no labels are used.
    for missing in sorted(set(range(k)) - set(np.unique(consensus).tolist())):
        candidates = np.flatnonzero(aligned[medoid] == missing)
        if len(candidates) == 0:
            raise ValueError(f"teacher medoid has no member for cluster {missing}")
        current = consensus[candidates]
        advantage = votes[candidates, missing] - votes[candidates, current]
        chosen = int(candidates[int(np.argmax(advantage))])
        consensus[chosen] = missing
        repair_count += 1
    confidence = (votes.max(axis=1) / float(len(values))).astype(np.float32)

    graph = sp.csr_matrix(graph, dtype=np.float32)
    graph.setdiag(0)
    graph.eliminate_zeros()
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1)
    same = np.zeros(n, dtype=np.float32)
    for cluster in range(k):
        indicator = (consensus == cluster).astype(np.float32)
        same += indicator * np.asarray(graph @ indicator).reshape(-1)
    boundary = np.divide(
        row_sum - same,
        np.maximum(row_sum, 1e-12),
        out=np.zeros(n, dtype=np.float32),
    ).astype(np.float32)
    interior = (confidence * (1.0 - boundary)).astype(np.float32)
    return StabilityAnchors(
        consensus=consensus,
        confidence=confidence,
        boundary=boundary,
        interior_weight=interior,
        aligned_partitions=aligned.astype(np.int32),
        medoid_index=medoid,
        mean_pairwise_ari=float((pairwise.sum() - len(values)) / (len(values) * (len(values) - 1))),
        cardinality_repair_count=repair_count,
    )


def scipy_csr_to_torch(matrix: sp.spmatrix, device: "torch.device") -> "torch.Tensor":
    matrix = sp.coo_matrix(matrix, dtype=np.float32)
    indices = torch.as_tensor(np.vstack((matrix.row, matrix.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(matrix.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=device).coalesce()


if torch is not None:

    class SAPRCore(nn.Module):
        """Shared residual/prototype core with modality-specific input adapters."""

        def __init__(
            self,
            base_dim: int,
            view1_dim: int,
            view2_dim: int,
            latent_dim: int,
            n_clusters: int,
            hidden_dim: int = 128,
            dropout: float = 0.05,
        ) -> None:
            super().__init__()
            self.base_adapter = nn.Linear(base_dim, latent_dim)
            self.view1_adapter = nn.Linear(view1_dim, latent_dim)
            self.view2_adapter = nn.Linear(view2_dim, latent_dim)
            self.residual = nn.Sequential(
                nn.Linear(latent_dim * 4 + 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.prototypes = nn.Parameter(torch.empty(n_clusters, latent_dim))
            self.log_residual_scale = nn.Parameter(torch.tensor(-2.0))
            self.log_temperature = nn.Parameter(torch.tensor(-1.7))
            self.n_clusters = int(n_clusters)
            self.latent_dim = int(latent_dim)
            nn.init.orthogonal_(self.base_adapter.weight)
            nn.init.orthogonal_(self.view1_adapter.weight)
            nn.init.orthogonal_(self.view2_adapter.weight)
            nn.init.zeros_(self.base_adapter.bias)
            nn.init.zeros_(self.view1_adapter.bias)
            nn.init.zeros_(self.view2_adapter.bias)
            nn.init.normal_(self.prototypes, std=0.05)

        @torch.no_grad()
        def initialize_prototypes(
            self,
            base: torch.Tensor,
            consensus: torch.Tensor,
            interior_weight: torch.Tensor,
        ) -> None:
            projected = F.normalize(self.base_adapter(base), dim=1)
            for cluster in range(self.n_clusters):
                mask = consensus == cluster
                if not bool(mask.any()):
                    raise ValueError(f"empty stability anchor cluster {cluster}")
                weight = interior_weight[mask].clamp_min(1e-3)
                center = (projected[mask] * weight[:, None]).sum(0) / weight.sum()
                self.prototypes[cluster].copy_(F.normalize(center, dim=0))

        def forward(
            self,
            base: torch.Tensor,
            view1: torch.Tensor,
            view2: torch.Tensor,
            graph: torch.Tensor,
            confidence: torch.Tensor,
            boundary: torch.Tensor,
            residual_enabled: bool = True,
        ) -> Mapping[str, torch.Tensor]:
            b = F.normalize(self.base_adapter(base), dim=1)
            z1 = F.normalize(self.view1_adapter(view1), dim=1)
            z2 = F.normalize(self.view2_adapter(view2), dim=1)
            local = torch.sparse.mm(graph, b) - b
            evidence = torch.stack((confidence, boundary), dim=1)
            raw = self.residual(torch.cat((b, z1, z2, local, evidence), dim=1))
            # Stable interiors are protected; disagreement/boundary points may move.
            gate = (0.5 * (1.0 - confidence) + 0.5 * boundary).clamp(0.0, 1.0)
            scale = F.softplus(self.log_residual_scale)
            delta = gate[:, None] * scale * raw if residual_enabled else torch.zeros_like(raw)
            embedding = F.normalize(b + delta, dim=1)
            prototype = F.normalize(self.prototypes, dim=1)
            temperature = F.softplus(self.log_temperature).clamp_min(0.03)
            logits = embedding @ prototype.T / temperature
            logits1 = z1 @ prototype.T / temperature
            logits2 = z2 @ prototype.T / temperature
            return {
                "embedding": embedding,
                "delta": delta,
                "gate": gate,
                "logits": logits,
                "logits1": logits1,
                "logits2": logits2,
                "temperature": temperature,
            }


    def sapr_loss(
        output: Mapping[str, torch.Tensor],
        consensus: torch.Tensor,
        confidence: torch.Tensor,
        boundary: torch.Tensor,
        graph: torch.Tensor,
        weights: Mapping[str, float],
    ) -> Mapping[str, torch.Tensor]:
        ce = F.cross_entropy(output["logits"], consensus, reduction="none")
        anchor = (ce * confidence).sum() / confidence.sum().clamp_min(1.0)
        trust_weight = (confidence * (1.0 - boundary)).clamp(0.0, 1.0)
        trust = ((output["delta"].square().sum(1)) * trust_weight).sum() / trust_weight.sum().clamp_min(1.0)
        target_prob = output["logits"].detach().softmax(1)
        cross_view = 0.5 * (
            F.kl_div(output["logits1"].log_softmax(1), target_prob, reduction="batchmean")
            + F.kl_div(output["logits2"].log_softmax(1), target_prob, reduction="batchmean")
        )
        probability = output["logits"].softmax(1)
        smooth = (probability - torch.sparse.mm(graph, probability)).square().sum(1)
        boundary_consistency = (smooth * (0.25 + 0.75 * boundary)).mean()
        marginal = probability.mean(0).clamp_min(1e-8)
        balance = (marginal * (marginal.log() + np.log(float(probability.shape[1])))).sum()
        total = (
            float(weights.get("anchor", 1.0)) * anchor
            + float(weights.get("trust", 1.0)) * trust
            + float(weights.get("cross_view", 0.0)) * cross_view
            + float(weights.get("boundary", 0.0)) * boundary_consistency
            + float(weights.get("balance", 0.0)) * balance
        )
        return {
            "total": total,
            "anchor": anchor,
            "trust": trust,
            "cross_view": cross_view,
            "boundary": boundary_consistency,
            "balance": balance,
        }


def module_semantics() -> Mapping[str, object]:
    return {
        "ground_truth_argument_count": 0,
        "dataset_name_argument_count": 0,
        "dense_n_by_n_count": 0,
        "teacher_semantics": "aligned label-free partition bank",
        "novelty_claim_boundary": (
            "stability-derived interior trust region plus boundary-targeted residual; "
            "prototype sharpening itself is prior art and is not claimed"
        ),
    }
