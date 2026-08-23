"""Night-13C endpoint-robustness and trainable unified-core primitives.

Model APIs intentionally accept tensors and sparse operators only. Dataset,
tissue, assay-family, labels and evaluation metrics stay outside this module.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def canonical_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def aligned_partition_change(reference: Sequence[int], value: Sequence[int]) -> float:
    """Fraction changed after optimal label permutation; no biological labels."""
    left = np.asarray(reference, dtype=np.int64)
    right = np.asarray(value, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("partition shape mismatch")
    a = np.unique(left, return_inverse=True)[1]
    b = np.unique(right, return_inverse=True)[1]
    table = np.zeros((int(a.max()) + 1, int(b.max()) + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    rows, cols = linear_sum_assignment(-table)
    matched = int(table[rows, cols].sum())
    return float(1.0 - matched / len(left))


def consensus_medoid(partitions: Sequence[np.ndarray]) -> Tuple[np.ndarray, int, float]:
    """Sparse-safe consensus: partition maximizing mean pairwise ARI."""
    values = [np.asarray(x, dtype=np.int64) for x in partitions]
    if not values or any(x.shape != values[0].shape for x in values):
        raise ValueError("invalid consensus partitions")
    agreement = np.eye(len(values), dtype=np.float64)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            score = adjusted_rand_score(values[i], values[j])
            agreement[i, j] = agreement[j, i] = score
    means = agreement.mean(axis=1)
    index = int(np.argmax(means))
    return values[index].copy(), index, float(means[index])


def centroid_margin(distances: np.ndarray, partition: np.ndarray) -> float:
    distances = np.asarray(distances, dtype=np.float64)
    partition = np.asarray(partition, dtype=np.int64)
    assigned = distances[np.arange(len(partition)), partition]
    second = np.partition(distances, 1, axis=1)[:, 1]
    return float(np.mean((second - assigned) / np.maximum(second, 1e-12)))


def torch_sparse(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long,
                              device=device)
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape,
                                   device=device).coalesce()


@dataclass(frozen=True)
class ResidualVariants:
    identity: np.ndarray
    p4_only: np.ndarray
    p18_only: np.ndarray
    b10_mixture: np.ndarray
    multiscale_direction: np.ndarray
    discrepancy: float
    gate: float
    beta: float


def deterministic_residual_variants(
    embedding: np.ndarray,
    fine: sp.spmatrix,
    broad: sp.spmatrix,
    threshold: float = 0.66,
    slope: float = 40.0,
    max_residual: float = 0.70,
) -> ResidualVariants:
    """Exact float32 Night-13B graph-residual semantics without clustering."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = torch.as_tensor(np.asarray(embedding), dtype=torch.float32, device=device)
    p4 = torch_sparse(fine, device)
    p18 = torch_sparse(broad, device)
    with torch.no_grad():
        normalized = F.normalize(z, p=2, dim=1, eps=1e-12)
        fine_normalized = F.normalize(torch.sparse.mm(p4, normalized), p=2,
                                      dim=1, eps=1e-12)
        discrepancy = (1.0 - (normalized * fine_normalized).sum(dim=1)).mean()
        gate = torch.sigmoid(float(slope) * (discrepancy - float(threshold)))
        beta = float(max_residual) * gate
        fine_z = torch.sparse.mm(p4, z)
        broad_z = torch.sparse.mm(p18, z)
        multiscale = (1.0 - gate) * fine_z + gate * broad_z
        fused = (1.0 - beta) * z + beta * multiscale
    convert = lambda x: x.detach().cpu().numpy().astype(np.float32)
    return ResidualVariants(
        identity=convert(z), p4_only=convert(fine_z), p18_only=convert(broad_z),
        b10_mixture=convert(fused), multiscale_direction=convert(multiscale),
        discrepancy=float(discrepancy.cpu()), gate=float(gate.cpu()),
        beta=float(beta.cpu()),
    )


def fixed_beta_residual(identity: np.ndarray, multiscale: np.ndarray,
                        beta: float) -> np.ndarray:
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta outside [0,1]")
    left = torch.as_tensor(identity, dtype=torch.float32)
    right = torch.as_tensor(multiscale, dtype=torch.float32)
    return ((1.0 - float(beta)) * left + float(beta) * right).numpy().astype(np.float32)


class SharedAdapters(nn.Module):
    """Dimension adapters plus a tied core shared by both modalities."""

    def __init__(self, input1: int, input2: int, latent: int):
        super().__init__()
        self.adapter1 = nn.Linear(input1, latent)
        self.adapter2 = nn.Linear(input2, latent)
        self.shared = nn.Sequential(nn.Linear(latent, latent), nn.GELU(),
                                    nn.LayerNorm(latent))
        self.decoder1 = nn.Linear(latent, input1)
        self.decoder2 = nn.Linear(latent, input2)

    def encode(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.shared(self.adapter1(x1)), self.shared(self.adapter2(x2))


def variance_floor(*values: torch.Tensor) -> torch.Tensor:
    terms = [F.relu(1.0 - torch.sqrt(x.var(dim=0) + 1e-4)).mean() for x in values]
    return torch.stack(terms).mean()


class TrainableUnifiedCore(nn.Module):
    """Three mechanism-distinct, family-blind unsupervised fusion cores."""

    MECHANISMS = {"EDGE_RELIABILITY", "MODALITY_DROPOUT_CROSS_RECON",
                  "SHARED_PRIVATE_ORTHOGONAL"}

    def __init__(self, input1: int, input2: int, latent: int, mechanism: str,
                 residual: float = 0.20):
        super().__init__()
        if mechanism not in self.MECHANISMS:
            raise ValueError("unknown mechanism")
        self.mechanism = mechanism
        self.residual = float(residual)
        self.base = SharedAdapters(input1, input2, latent)
        if mechanism == "EDGE_RELIABILITY":
            self.edge_mlp = nn.Sequential(nn.Linear(4, 16), nn.GELU(),
                                          nn.Linear(16, 1))
        elif mechanism == "SHARED_PRIVATE_ORTHOGONAL":
            self.private1 = nn.Linear(latent, latent)
            self.private2 = nn.Linear(latent, latent)
            self.private_decoder1 = nn.Linear(latent, input1)
            self.private_decoder2 = nn.Linear(latent, input2)

    @staticmethod
    def _edge_features(h1: torch.Tensor, h2: torch.Tensor,
                       edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edges
        s1 = F.cosine_similarity(h1[row], h1[col], dim=1)
        s2 = F.cosine_similarity(h2[row], h2[col], dim=1)
        features = torch.stack(((s1 + s2) / 2.0, torch.minimum(s1, s2),
                                torch.maximum(s1, s2), (s1 - s2).abs()), dim=1)
        target = torch.clamp((s1.detach() + s2.detach()) / 2.0, 0.0, 1.0)
        return features, target

    @staticmethod
    def _propagate(base: torch.Tensor, edges: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
        row, col = edges
        n = base.shape[0]
        degree = torch.zeros(n, device=base.device).index_add_(0, row, weights)
        summed = torch.zeros_like(base).index_add_(0, row, weights[:, None] * base[col])
        return summed / degree.clamp_min(1e-6)[:, None]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                edges: torch.Tensor, dropout_mask: Tuple[bool, bool] = (False, False)) -> Dict[str, torch.Tensor]:
        h1, h2 = self.base.encode(x1, x2)
        if dropout_mask[0]:
            h1_used = torch.zeros_like(h1)
        else:
            h1_used = h1
        if dropout_mask[1]:
            h2_used = torch.zeros_like(h2)
        else:
            h2_used = h2
        base = 0.5 * (h1_used + h2_used)
        edge_target = torch.empty(0, device=x1.device)
        edge_reliability = torch.empty(0, device=x1.device)
        node_reliability = torch.empty(0, device=x1.device)
        if self.mechanism == "EDGE_RELIABILITY":
            edge_features, edge_target = self._edge_features(h1, h2, edges)
            edge_reliability = torch.sigmoid(self.edge_mlp(edge_features)).squeeze(1)
            propagated = self._propagate(base, edges, edge_reliability)
            row = edges[0]
            count = torch.zeros(base.shape[0], device=base.device).index_add_(
                0, row, torch.ones_like(edge_reliability))
            node_reliability = torch.zeros(base.shape[0], device=base.device).index_add_(
                0, row, edge_reliability) / count.clamp_min(1.0)
            beta = self.residual * node_reliability[:, None]
            fused = (1.0 - beta) * base + beta * propagated
        elif self.mechanism == "SHARED_PRIVATE_ORTHOGONAL":
            raw1 = self.base.adapter1(x1)
            raw2 = self.base.adapter2(x2)
            private1 = F.layer_norm(self.private1(raw1), (raw1.shape[1],))
            private2 = F.layer_norm(self.private2(raw2), (raw2.shape[1],))
            fused = base
        else:
            fused = base
        private1 = private1 if self.mechanism == "SHARED_PRIVATE_ORTHOGONAL" else torch.empty(0, device=x1.device)
        private2 = private2 if self.mechanism == "SHARED_PRIVATE_ORTHOGONAL" else torch.empty(0, device=x1.device)
        recon1 = self.base.decoder1(h1)
        recon2 = self.base.decoder2(h2)
        if self.mechanism == "SHARED_PRIVATE_ORTHOGONAL":
            recon1 = recon1 + self.private_decoder1(private1)
            recon2 = recon2 + self.private_decoder2(private2)
        return {"h1": h1, "h2": h2, "fused": fused,
                "recon1_h1": recon1, "recon2_h2": recon2,
                "cross1": self.base.decoder1(h2), "cross2": self.base.decoder2(h1),
                "fused1": self.base.decoder1(fused), "fused2": self.base.decoder2(fused),
                "private1": private1, "private2": private2,
                "edge_target": edge_target, "edge_reliability": edge_reliability,
                "node_reliability": node_reliability}

    def loss(self, output: Mapping[str, torch.Tensor], x1: torch.Tensor,
             x2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h1, h2 = output["h1"], output["h2"]
        within = F.mse_loss(output["recon1_h1"], x1) + F.mse_loss(output["recon2_h2"], x2)
        cross = F.mse_loss(output["cross1"], x1) + F.mse_loss(output["cross2"], x2)
        agreement = (1.0 - F.cosine_similarity(h1, h2, dim=1)).mean()
        variance = variance_floor(h1, h2, output["fused"])
        if self.mechanism == "EDGE_RELIABILITY":
            edge = F.mse_loss(output["edge_reliability"], output["edge_target"])
            total = within + 0.50 * cross + 0.20 * agreement + 0.20 * variance + 0.50 * edge
        elif self.mechanism == "MODALITY_DROPOUT_CROSS_RECON":
            edge = torch.zeros((), device=x1.device)
            fused_reconstruction = (F.mse_loss(output["fused1"], x1) +
                                    F.mse_loss(output["fused2"], x2))
            total = (0.35 * within + cross + fused_reconstruction +
                     0.20 * agreement + 0.20 * variance)
        else:
            p1, p2 = output["private1"], output["private2"]
            centered_h1, centered_p1 = h1 - h1.mean(0), p1 - p1.mean(0)
            centered_h2, centered_p2 = h2 - h2.mean(0), p2 - p2.mean(0)
            cov1 = centered_h1.T @ centered_p1 / max(1, h1.shape[0] - 1)
            cov2 = centered_h2.T @ centered_p2 / max(1, h2.shape[0] - 1)
            edge = 0.5 * (cov1.square().mean() + cov2.square().mean())
            total = within + 0.50 * cross + 0.15 * agreement + 0.20 * variance + 0.10 * edge
        return total, {"within": within, "cross": cross, "agreement": agreement,
                       "variance": variance, "mechanism_specific": edge}


__all__ = [
    "TrainableUnifiedCore", "ResidualVariants", "aligned_partition_change",
    "canonical_sha256", "centroid_margin", "consensus_medoid",
    "deterministic_residual_variants", "fixed_beta_residual", "seed_everything",
]
