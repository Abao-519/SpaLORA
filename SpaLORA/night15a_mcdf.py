"""Night-15A identity-blind score-source and MCDF primitives.

The functions in this module accept arrays, sparse operators and explicit
numerical settings only.  Dataset identifiers and public benchmark labels are
kept in runners/evaluators, never in representation construction or fitting.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .night14a_tcf import (
    UnifiedGraphAutoencoder,
    loss_components,
    node_edge_mean,
    unsupervised_loss,
    weighted_neighbor_mean,
)


CONTROL_IDS = {
    "COORDINATE_ONLY",
    "RNA_ONLY",
    "ATAC_ONLY",
    "RNA_PLUS_COORDINATES",
    "ATAC_PLUS_COORDINATES",
    "FUSED_FULL",
    "FUSED_WITHOUT_COORDINATES",
    "FUSED_WITHOUT_GRAPH_FILTER",
    "FUSED_WITHOUT_SPATIAL_REFINEMENT",
    "NIGHT14B_FROZEN_BEST",
}


def array_sha256(value: np.ndarray) -> str:
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def row_l2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def coordinate_features(coordinates: np.ndarray, basis: str) -> np.ndarray:
    coord = StandardScaler().fit_transform(np.asarray(coordinates, dtype=np.float64))
    if basis == "POLY2":
        x, y = coord[:, 0], coord[:, 1]
        coord = np.column_stack((x, y, x * x, x * y, y * y))
        coord = StandardScaler().fit_transform(coord)
    elif basis != "LINEAR":
        raise ValueError("unknown coordinate basis")
    return np.asarray(coord, dtype=np.float32)


def _source_for_representation(
    views: Mapping[str, np.ndarray], representation: str
) -> np.ndarray:
    if representation == "RNA":
        return row_l2(views["emb_latent_omics1"])
    if representation == "ATAC":
        return row_l2(views["emb_latent_omics2"])
    if representation == "FUSED":
        return row_l2(views["SpaLORA_fused"])
    if representation == "PRIVATE2":
        return np.concatenate(
            (
                row_l2(views["emb_latent_omics1"]) / np.sqrt(2.0),
                row_l2(views["emb_latent_omics2"]) / np.sqrt(2.0),
            ),
            axis=1,
        )
    if representation == "EQUAL3":
        return np.concatenate(
            tuple(
                row_l2(views[key]) / np.sqrt(3.0)
                for key in (
                    "emb_latent_omics1",
                    "emb_latent_omics2",
                    "SpaLORA_fused",
                )
            ),
            axis=1,
        )
    raise ValueError("unknown representation")


def molecular_features(
    views: Mapping[str, np.ndarray], representation: str, dimension: int
) -> np.ndarray:
    source = _source_for_representation(views, representation)
    dimension = min(int(dimension), source.shape[1], source.shape[0] - 1)
    if dimension < 1:
        raise ValueError("invalid PCA dimension")
    reduced = PCA(
        n_components=dimension, random_state=0, svd_solver="randomized"
    ).fit_transform(source)
    return StandardScaler().fit_transform(reduced).astype(np.float32)


def matched_control_matrix(
    control_id: str,
    filtered_views: Mapping[str, np.ndarray],
    unfiltered_views: Mapping[str, np.ndarray],
    coordinates: np.ndarray,
    representation: str,
    dimension: int,
    coordinate_basis: str,
    coordinate_weight: float,
) -> np.ndarray:
    """Create one score-source ablation without inspecting labels."""
    if control_id not in CONTROL_IDS:
        raise ValueError("unknown Night-15A score-source control")
    coord = coordinate_features(coordinates, coordinate_basis)
    if control_id == "COORDINATE_ONLY":
        return coord
    if control_id in {"RNA_ONLY", "RNA_PLUS_COORDINATES"}:
        molecular = molecular_features(filtered_views, "RNA", dimension)
    elif control_id in {"ATAC_ONLY", "ATAC_PLUS_COORDINATES"}:
        molecular = molecular_features(filtered_views, "ATAC", dimension)
    elif control_id == "FUSED_WITHOUT_GRAPH_FILTER":
        molecular = molecular_features(unfiltered_views, representation, dimension)
    else:
        molecular = molecular_features(filtered_views, representation, dimension)
    with_coordinates = control_id in {
        "RNA_PLUS_COORDINATES",
        "ATAC_PLUS_COORDINATES",
        "FUSED_FULL",
        "FUSED_WITHOUT_GRAPH_FILTER",
        "FUSED_WITHOUT_SPATIAL_REFINEMENT",
        "NIGHT14B_FROZEN_BEST",
    }
    if not with_coordinates:
        return molecular
    return np.concatenate(
        (molecular, float(coordinate_weight) * coord), axis=1
    ).astype(np.float32)


def cluster_known_k(
    value: np.ndarray,
    n_clusters: int,
    algorithm: str,
    seed: int,
    n_init: int,
) -> np.ndarray:
    """Known-K unsupervised endpoint; labels are not accepted by this API."""
    value = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("cluster input contains non-finite values")
    if algorithm == "KMEANS":
        return KMeans(
            n_clusters=int(n_clusters), random_state=int(seed), n_init=int(n_init)
        ).fit_predict(value).astype(np.int64)
    if not algorithm.startswith("GMM_"):
        raise ValueError("unknown clustering algorithm")
    covariance = algorithm.split("_", 1)[1].lower()
    return GaussianMixture(
        n_components=int(n_clusters),
        covariance_type=covariance,
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=300,
        reg_covar=1e-5,
    ).fit_predict(value).astype(np.int64)


def sparse_weighted_sum(
    operators: Mapping[str, sp.spmatrix], weights: Mapping[str, float]
) -> sp.csr_matrix:
    """Convex sparse operator mixture used by later MCDF candidates."""
    if set(operators) != set(weights):
        raise ValueError("operator/weight keys differ")
    values = np.asarray([float(weights[key]) for key in operators], dtype=np.float64)
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("expert weights must be a convex combination")
    result = None
    for key, operator in operators.items():
        term = operator.tocsr().astype(np.float64) * float(weights[key])
        result = term if result is None else result + term
    result = result.tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def _variance_loss(value: torch.Tensor) -> torch.Tensor:
    standard_deviation = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - standard_deviation).mean()


def _covariance_loss(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, value.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    return (covariance.square().sum() - diagonal.square().sum()) / value.shape[1]


def _multiscale_stream(
    value: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    one = weighted_neighbor_mean(value, edge_index, edge_weight)
    two = weighted_neighbor_mean(one, edge_index, edge_weight)
    weight = torch.softmax(logits, dim=0)
    return weight[0] * value + weight[1] * one + weight[2] * two


class MCDFUnifiedCore(nn.Module):
    """Unified trainable four-expert RNA+ATAC diffusion core.

    The API contains no dataset, path, tissue or label argument.  Both assays
    use legal input adapters followed by one shared graph/fusion rule.
    """

    MODES = {"IDENTITY", "FIXED_EQUAL", "GATED", "GATED_NO_GEOMETRY"}

    def __init__(self, input1: int, input2: int, config: Mapping[str, object]) -> None:
        super().__init__()
        self.config = dict(config)
        self.mode = str(config["mode"])
        if self.mode not in self.MODES:
            raise ValueError("unknown MCDF mode")
        self.base = UnifiedGraphAutoencoder(input1, input2, config["base_config"])
        latent = int(config["base_config"]["latent_dim"])
        hidden = int(config["gate_hidden"])
        self.gate = nn.Sequential(
            nn.Linear(4 * latent + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.scale_logits = nn.Parameter(torch.zeros(4, 3))
        initial = float(config["initial_residual_strength"])
        initial = min(max(initial, 1e-4), 1.0 - 1e-4)
        self.residual_logit = nn.Parameter(
            torch.tensor(np.log(initial / (1.0 - initial)), dtype=torch.float32)
        )
        self.output_norm = nn.LayerNorm(latent)

    @staticmethod
    def _edge_support(
        value: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        return ((F.cosine_similarity(value[row], value[col], dim=1) + 1.0) * 0.5).clamp(
            1e-4, 1.0
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        output = self.base(x1, x2, edge_index, edge_weight)
        z1, z2, fused = output["z1"], output["z2"], output["fused"]
        support1 = self._edge_support(z1, edge_index)
        support2 = self._edge_support(z2, edge_index)
        joint = torch.sqrt((support1 * support2).clamp_min(1e-8))
        expert_weights = (
            edge_weight,
            edge_weight * support1.pow(float(self.config["support_power"])),
            edge_weight * support2.pow(float(self.config["support_power"])),
            edge_weight * joint.pow(float(self.config["joint_power"])),
        )
        experts = torch.stack(
            tuple(
                _multiscale_stream(fused, edge_index, weight, self.scale_logits[index])
                for index, weight in enumerate(expert_weights)
            ),
            dim=1,
        )
        node_support1 = node_edge_mean(support1, edge_index, fused.shape[0])[:, None]
        node_support2 = node_edge_mean(support2, edge_index, fused.shape[0])[:, None]
        gate_input = torch.cat(
            (z1, z2, torch.abs(z1 - z2), z1 * z2, node_support1, node_support2), dim=1
        )
        if self.mode == "IDENTITY":
            gate = torch.zeros(
                (fused.shape[0], 4), dtype=fused.dtype, device=fused.device
            )
            gate[:, 0] = 1.0
            result = fused
        elif self.mode == "FIXED_EQUAL":
            gate = torch.full(
                (fused.shape[0], 4), 0.25, dtype=fused.dtype, device=fused.device
            )
            mixed = (experts * gate[:, :, None]).sum(dim=1)
            strength = torch.sigmoid(self.residual_logit)
            result = self.output_norm(fused + strength * (mixed - fused))
        else:
            logits = self.gate(gate_input)
            if self.mode == "GATED_NO_GEOMETRY":
                logits = logits.clone()
                logits[:, 0] = torch.finfo(logits.dtype).min
            gate = torch.softmax(logits, dim=1)
            mixed = (experts * gate[:, :, None]).sum(dim=1)
            strength = torch.sigmoid(self.residual_logit)
            result = self.output_norm(fused + strength * (mixed - fused))
        result_output = dict(output)
        result_output.update(
            {
                "mcdf": result,
                "expert_outputs": experts,
                "expert_gate": gate,
                "support1": support1,
                "support2": support2,
                "joint_support": joint,
                "mcdf_recon1": self.base.decoder1(result),
                "mcdf_recon2": self.base.decoder2(result),
                "residual_strength": torch.sigmoid(self.residual_logit),
                "scale_weights": torch.softmax(self.scale_logits, dim=1),
            }
        )
        return result_output


def mcdf_unsupervised_loss(
    model: MCDFUnifiedCore,
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
    reconstruction = 0.5 * (
        F.mse_loss(output["mcdf_recon1"], x1)
        + F.mse_loss(output["mcdf_recon2"], x2)
    )
    cross_view = 0.5 * (
        (1.0 - F.cosine_similarity(output["mcdf"], output["z1"], dim=1)).mean()
        + (1.0 - F.cosine_similarity(output["mcdf"], output["z2"], dim=1)).mean()
    )
    variance = _variance_loss(output["mcdf"])
    covariance = _covariance_loss(output["mcdf"])
    gate_mean = output["expert_gate"].mean(dim=0)
    geometry_cap = F.relu(
        gate_mean[0] - float(weights["maximum_geometry_gate"])
    ).square()
    molecular_mass = gate_mean[1:].sum()
    molecular_floor = F.relu(
        float(weights["minimum_molecular_gate"]) - molecular_mass
    ).square()
    entropy = -(
        output["expert_gate"].clamp_min(1e-8)
        * output["expert_gate"].clamp_min(1e-8).log()
    ).sum(dim=1).mean()
    entropy_floor = F.relu(float(weights["minimum_gate_entropy"]) - entropy).square()
    total = (
        base_loss
        + float(weights["mcdf_reconstruction"]) * reconstruction
        + float(weights["cross_view_consistency"]) * cross_view
        + float(weights["variance"]) * variance
        + float(weights["covariance"]) * covariance
        + float(weights["coordinate_dominance"]) * geometry_cap
        + float(weights["molecular_contribution"]) * molecular_floor
        + float(weights["gate_entropy"]) * entropy_floor
    )
    audit.update(
        {
            "mcdf_reconstruction": float(reconstruction.detach().cpu()),
            "mcdf_cross_view": float(cross_view.detach().cpu()),
            "mcdf_variance": float(variance.detach().cpu()),
            "mcdf_covariance": float(covariance.detach().cpu()),
            "geometry_gate_mean": float(gate_mean[0].detach().cpu()),
            "rna_gate_mean": float(gate_mean[1].detach().cpu()),
            "atac_gate_mean": float(gate_mean[2].detach().cpu()),
            "joint_gate_mean": float(gate_mean[3].detach().cpu()),
            "gate_entropy": float(entropy.detach().cpu()),
            "coordinate_dominance_penalty": float(geometry_cap.detach().cpu()),
            "molecular_contribution_penalty": float(molecular_floor.detach().cpu()),
            "gate_entropy_penalty": float(entropy_floor.detach().cpu()),
            "residual_strength": float(output["residual_strength"].detach().cpu()),
            "total_with_mcdf": float(total.detach().cpu()),
        }
    )
    return total, audit


__all__ = [
    "CONTROL_IDS",
    "array_sha256",
    "cluster_known_k",
    "coordinate_features",
    "matched_control_matrix",
    "MCDFUnifiedCore",
    "mcdf_unsupervised_loss",
    "molecular_features",
    "row_l2",
    "sparse_weighted_sum",
]
