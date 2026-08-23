"""Clean-room sparse unified graph autoencoder and topology-conflict filter.

Night-14A deliberately does not copy SMART (GPL-3.0) or SpaBalance
(AGPL-3.0) source.  The implementation below uses standard, independently
written graph-autoencoder primitives: modality projections, shared sparse
message passing, self/cross reconstruction, and a topology filter whose edge
weights are functions of observable cross-modal support and conflict.

The model API intentionally accepts tensors and sparse edge lists only.  It
cannot inspect a dataset, tissue, assay-family, file path, or label.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _check_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 n_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weight must have shape [E]")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= int(n_nodes)
    ):
        raise ValueError("edge index outside observation range")


def sparse_aggregate(x: torch.Tensor, edge_index: torch.Tensor,
                     edge_weight: torch.Tensor) -> torch.Tensor:
    """Row-oriented sparse aggregation without materialising dense N x N."""
    _check_edges(edge_index, edge_weight, x.shape[0])
    row, col = edge_index[0], edge_index[1]
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * edge_weight[:, None])
    return out


def weighted_neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor,
                           edge_weight: torch.Tensor,
                           fallback_to_identity: bool = True) -> torch.Tensor:
    """Weighted non-dense neighbor mean with a safe zero-degree fallback."""
    _check_edges(edge_index, edge_weight, x.shape[0])
    row, col = edge_index[0], edge_index[1]
    numerator = torch.zeros_like(x)
    denominator = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    numerator.index_add_(0, row, x[col] * edge_weight[:, None])
    denominator.index_add_(0, row, edge_weight)
    result = numerator / denominator.clamp_min(1e-8)[:, None]
    if fallback_to_identity:
        result = torch.where((denominator > 1e-8)[:, None], result, x)
    return result


def node_edge_mean(values: torch.Tensor, edge_index: torch.Tensor,
                   n_nodes: int) -> torch.Tensor:
    row = edge_index[0]
    numerator = torch.zeros(n_nodes, dtype=values.dtype, device=values.device)
    denominator = torch.zeros_like(numerator)
    numerator.index_add_(0, row, values)
    denominator.index_add_(0, row, torch.ones_like(values))
    return numerator / denominator.clamp_min(1.0)


class ModalityAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedGraphResidualBlock(nn.Module):
    """One shared graph block applied to either modality representation."""
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        neighbor = sparse_aggregate(x, edge_index, edge_weight)
        update = self.self_linear(x) + self.neighbor_linear(neighbor)
        update = self.dropout(F.gelu(self.norm(update)))
        return x + update


class UnifiedGraphAutoencoder(nn.Module):
    """Two legal input adapters followed by one shared trainable graph core."""

    BACKBONES = {"CR_SAGE_AE", "CR_BALANCED_XREC"}

    def __init__(self, input1: int, input2: int,
                 config: Mapping[str, object]) -> None:
        super().__init__()
        backbone = str(config["backbone"])
        if backbone not in self.BACKBONES:
            raise ValueError("unknown clean-room backbone")
        hidden = int(config["hidden_dim"])
        latent = int(config["latent_dim"])
        depth = int(config["depth"])
        dropout = float(config["dropout"])
        if min(input1, input2, hidden, latent, depth) <= 0:
            raise ValueError("model dimensions must be positive")
        self.backbone = backbone
        self.config = dict(config)
        self.adapter1 = ModalityAdapter(input1, hidden, dropout)
        self.adapter2 = ModalityAdapter(input2, hidden, dropout)
        self.shared_blocks = nn.ModuleList(
            [SharedGraphResidualBlock(hidden, dropout) for _ in range(depth)]
        )
        self.head1 = nn.Sequential(nn.Linear(hidden, latent), nn.LayerNorm(latent))
        self.head2 = nn.Sequential(nn.Linear(hidden, latent), nn.LayerNorm(latent))
        self.fusion_gate = nn.Sequential(
            nn.Linear(4 * latent, latent), nn.GELU(), nn.Linear(latent, 1)
        )
        self.fusion_refine = nn.Sequential(
            nn.Linear(latent, latent), nn.LayerNorm(latent), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(latent, latent),
        )
        self.fusion_norm = nn.LayerNorm(latent)
        self.decoder1 = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, input1)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, input2)
        )
        # Homoscedastic multi-task weights are used only by the balanced lane.
        self.log_task_variance = nn.Parameter(torch.zeros(3))

    def encode_modality(self, x: torch.Tensor, adapter: nn.Module,
                        head: nn.Module, edge_index: torch.Tensor,
                        edge_weight: torch.Tensor) -> torch.Tensor:
        h = adapter(x)
        for block in self.shared_blocks:
            h = block(h, edge_index, edge_weight)
        return head(h)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> Dict[str, torch.Tensor]:
        z1 = self.encode_modality(x1, self.adapter1, self.head1,
                                  edge_index, edge_weight)
        z2 = self.encode_modality(x2, self.adapter2, self.head2,
                                  edge_index, edge_weight)
        gate_input = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)
        alpha = torch.sigmoid(self.fusion_gate(gate_input))
        base = alpha * z1 + (1.0 - alpha) * z2
        fused = self.fusion_norm(base + self.fusion_refine(base))
        return {
            "z1": z1,
            "z2": z2,
            "alpha": alpha,
            "fused": fused,
            "private_recon1": self.decoder1(z1),
            "private_recon2": self.decoder2(z2),
            "fused_recon1": self.decoder1(fused),
            "fused_recon2": self.decoder2(fused),
        }


def _variance_loss(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, z.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    return (covariance.square().sum() - diagonal.square().sum()) / z.shape[1]


def loss_components(output: Mapping[str, torch.Tensor], x1: torch.Tensor,
                    x2: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
    z1, z2, fused = output["z1"], output["z2"], output["fused"]
    row, col = edge_index[0], edge_index[1]
    keep = row != col
    row, col = row[keep], col[keep]
    if row.numel() == 0:
        raise ValueError("spatial graph has no non-self edges")
    private_recon = 0.5 * (
        F.mse_loss(output["private_recon1"], x1)
        + F.mse_loss(output["private_recon2"], x2)
    )
    cross_recon = 0.5 * (
        F.mse_loss(output["fused_recon1"], x1)
        + F.mse_loss(output["fused_recon2"], x2)
    )
    alignment = (1.0 - F.cosine_similarity(z1, z2, dim=1)).mean()
    graph_smooth = (fused[row] - fused[col]).square().mean()
    support1 = F.cosine_similarity(z1[row], z1[col], dim=1)
    support2 = F.cosine_similarity(z2[row], z2[col], dim=1)
    topology_agreement = F.mse_loss(support1, support2)
    variance = (_variance_loss(z1) + _variance_loss(z2) + _variance_loss(fused)) / 3.0
    covariance = (_covariance_loss(z1) + _covariance_loss(z2) + _covariance_loss(fused)) / 3.0
    return {
        "private_recon": private_recon,
        "cross_recon": cross_recon,
        "alignment": alignment,
        "graph_smooth": graph_smooth,
        "topology_agreement": topology_agreement,
        "variance": variance,
        "covariance": covariance,
    }


def unsupervised_loss(model: UnifiedGraphAutoencoder,
                      components: Mapping[str, torch.Tensor],
                      weights: Mapping[str, object]) -> Tuple[torch.Tensor, Dict[str, float]]:
    grouped = {
        "reconstruction": (
            float(weights["private_recon"]) * components["private_recon"]
            + float(weights["cross_recon"]) * components["cross_recon"]
        ),
        "alignment": (
            float(weights["alignment"]) * components["alignment"]
            + float(weights["topology_agreement"]) * components["topology_agreement"]
        ),
        "regularization": (
            float(weights["graph_smooth"]) * components["graph_smooth"]
            + float(weights["variance"]) * components["variance"]
            + float(weights["covariance"]) * components["covariance"]
        ),
    }
    if model.backbone == "CR_BALANCED_XREC":
        values = torch.stack([grouped["reconstruction"], grouped["alignment"],
                              grouped["regularization"]])
        loss = torch.sum(torch.exp(-model.log_task_variance) * values
                         + model.log_task_variance)
    else:
        loss = sum(grouped.values())
    audit = {key: float(value.detach().cpu()) for key, value in components.items()}
    audit.update({"group_" + key: float(value.detach().cpu())
                  for key, value in grouped.items()})
    audit["total"] = float(loss.detach().cpu())
    return loss, audit


def topology_edge_evidence(z1: torch.Tensor, z2: torch.Tensor,
                           edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
    row, col = edge_index[0], edge_index[1]
    keep = row != col
    row, col = row[keep], col[keep]
    edges = torch.stack([row, col], dim=0)
    s1 = (F.cosine_similarity(z1[row], z1[col], dim=1) + 1.0) * 0.5
    s2 = (F.cosine_similarity(z2[row], z2[col], dim=1) + 1.0) * 0.5
    s1 = s1.clamp(0.0, 1.0)
    s2 = s2.clamp(0.0, 1.0)
    joint = torch.sqrt((s1 * s2).clamp_min(0.0))
    conflict = torch.abs(s1 - s2)
    return {"edge_index": edges, "support1": s1, "support2": s2,
            "joint_support": joint, "conflict": conflict}


def apply_tcf(z1: torch.Tensor, z2: torch.Tensor, fused: torch.Tensor,
              edge_index: torch.Tensor,
              config: Mapping[str, object]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply an offline TCF variant to one frozen model checkpoint.

    `variant` controls only the filter algebra.  Every variant consumes the
    same frozen z1/z2/fused tensors; callers must not retrain the backbone.
    """
    variant = str(config["variant"])
    allowed = {"IDENTITY", "FIXED_LOW", "SUPPORT_LOW", "TCF_LOW_HIGH"}
    if variant not in allowed:
        raise ValueError("unknown TCF variant")
    evidence = topology_edge_evidence(z1, z2, edge_index)
    edges = evidence["edge_index"]
    joint = evidence["joint_support"]
    conflict = evidence["conflict"]
    n_nodes = fused.shape[0]
    node_support = node_edge_mean(joint, edges, n_nodes)
    node_conflict = node_edge_mean(conflict, edges, n_nodes)
    unweighted_low = weighted_neighbor_mean(
        fused, edges, torch.ones_like(joint)
    )
    roughness_ratio = (
        (fused - unweighted_low).square().mean()
        / fused.var(unbiased=False).clamp_min(1e-8)
    )
    centered1 = evidence["support1"] - evidence["support1"].mean()
    centered2 = evidence["support2"] - evidence["support2"].mean()
    topology_correlation = (
        (centered1 * centered2).mean()
        / torch.sqrt(centered1.square().mean() * centered2.square().mean()).clamp_min(1e-8)
    ).clamp(-1.0, 1.0)
    global_disagreement = 0.5 * (1.0 - topology_correlation)
    frequency_gate = torch.sigmoid(
        float(config["global_scale"])
        * (global_disagreement - float(config["global_center"]))
        + float(config["roughness_scale"])
        * (roughness_ratio - float(config["roughness_center"]))
    )
    integrity_score = (joint.mean() - conflict.mean()).clamp(0.0, 1.0)
    integrity_scale = float(config.get("integrity_scale", 0.0))
    if integrity_scale < 0.0:
        raise ValueError("integrity_scale must be non-negative")
    integrity_gate = (
        torch.sigmoid(
            integrity_scale
            * (integrity_score - float(config.get("integrity_center", 0.0)))
        )
        if integrity_scale > 0.0
        else torch.ones_like(frequency_gate)
    )
    raw_global_gate = frequency_gate * integrity_gate
    gate_floor = float(config.get("global_gate_floor", 0.0))
    if gate_floor < 0.0 or gate_floor > 1.0:
        raise ValueError("global_gate_floor must lie in [0,1]")
    global_gate = torch.where(
        raw_global_gate >= gate_floor, raw_global_gate,
        torch.zeros_like(raw_global_gate),
    )
    if variant == "IDENTITY":
        result = fused
        trust = torch.ones_like(joint)
        beta_low = torch.zeros(n_nodes, dtype=fused.dtype, device=fused.device)
        beta_high = torch.zeros_like(beta_low)
    else:
        if variant == "FIXED_LOW":
            trust = torch.ones_like(joint)
        elif variant == "SUPPORT_LOW":
            trust = joint
        else:
            logits = (
                float(config["support_scale"]) * (joint - float(config["support_center"]))
                - float(config["conflict_scale"]) * conflict
            )
            evidence_trust = torch.sigmoid(logits)
            evidence_fraction = float(config.get("trust_evidence_fraction", 1.0))
            if evidence_fraction < 0.0 or evidence_fraction > 1.0:
                raise ValueError("trust_evidence_fraction must lie in [0,1]")
            # A convex safe residual keeps every spatial edge available while
            # letting cross-modal support/conflict down-weight questionable
            # edges.  fraction=0 is the global-gate-only ablation; fraction=1
            # is the original fully evidence-weighted filter.
            trust = evidence_fraction * evidence_trust + (1.0 - evidence_fraction)
        low = weighted_neighbor_mean(fused, edges, trust)
        if variant == "FIXED_LOW":
            beta_low = torch.full_like(node_support, float(config["max_low"]))
        else:
            node_trust = node_edge_mean(trust, edges, n_nodes)
            beta_low = float(config["max_low"]) * torch.sigmoid(
                float(config["node_scale"]) * (node_trust - float(config["node_center"]))
            )
            if variant == "TCF_LOW_HIGH":
                beta_low = beta_low * global_gate
        beta_high = torch.zeros_like(beta_low)
        if variant == "TCF_LOW_HIGH" and float(config["max_high"]) > 0:
            # Agreed low support with little conflict is a candidate boundary;
            # this optional channel preserves rather than smooths it away.
            boundary = (1.0 - node_support) - node_conflict
            beta_high = float(config["max_high"]) * torch.sigmoid(
                float(config["node_scale"]) * (boundary - float(config["high_center"]))
            )
        high = fused - unweighted_low
        result = fused + beta_low[:, None] * (low - fused) + beta_high[:, None] * high
    exact_identity_fallback = (
        variant == "TCF_LOW_HIGH"
        and float(global_gate.detach().cpu()) == 0.0
        and float(config["max_high"]) == 0.0
    )
    if exact_identity_fallback:
        result = fused
    elif variant != "IDENTITY":
        result = F.layer_norm(result, (result.shape[1],))
    diagnostics = {
        "edge_count": float(edges.shape[1]),
        "support1_mean": float(evidence["support1"].mean().detach().cpu()),
        "support2_mean": float(evidence["support2"].mean().detach().cpu()),
        "joint_support_mean": float(joint.mean().detach().cpu()),
        "conflict_mean": float(conflict.mean().detach().cpu()),
        "conflict_q90": float(torch.quantile(conflict, 0.9).detach().cpu()),
        "trust_mean": float(trust.mean().detach().cpu()),
        "trust_evidence_fraction": float(config.get("trust_evidence_fraction", 1.0)),
        "beta_low_mean": float(beta_low.mean().detach().cpu()),
        "beta_high_mean": float(beta_high.mean().detach().cpu()),
        "node_support_mean": float(node_support.mean().detach().cpu()),
        "node_conflict_mean": float(node_conflict.mean().detach().cpu()),
        "topology_correlation": float(topology_correlation.detach().cpu()),
        "global_disagreement": float(global_disagreement.detach().cpu()),
        "roughness_ratio": float(roughness_ratio.detach().cpu()),
        "frequency_gate": float(frequency_gate.detach().cpu()),
        "integrity_score": float(integrity_score.detach().cpu()),
        "integrity_gate": float(integrity_gate.detach().cpu()),
        "raw_global_gate": float(raw_global_gate.detach().cpu()),
        "global_gate_floor": gate_floor,
        "global_gate": float(global_gate.detach().cpu()),
        "exact_identity_fallback": float(exact_identity_fallback),
        "dense_n_by_n_count": 0.0,
    }
    return result, diagnostics
