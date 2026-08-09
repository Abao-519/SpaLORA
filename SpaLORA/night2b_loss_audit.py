"""Parity-locked loss variants built on frozen legacy SpaLORA inputs/model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .model import Encoder_overall
from .night1_pipeline import calculate_asr_scores, log_normalize_counts
from .preprocess import transform_adjacent_matrix
from .SpaLORA_pyG import Train_SpaLORA


VARIANTS = (
    "locked_unweighted",
    "locked_uniform_legacy_scale",
    "locked_legacy_shape_normalized",
    "locked_legacy_loss_replay",
    "locked_asr_hvg_legacy_scale_diagnostic",
)

LOSS_LOG_FIELDS = (
    "epoch",
    "raw_rna_reconstruction",
    "weighted_rna_before_global_scale",
    "global_scale_multiplier",
    "final_rna_loss",
    "final_rna_contribution",
    "raw_modality2_reconstruction",
    "final_modality2_contribution",
    "raw_corr1",
    "final_corr1_contribution",
    "raw_corr2",
    "final_corr2_contribution",
    "total_loss",
    "m_bad",
    "gene_weight_mean",
    "gene_weight_min",
    "gene_weight_max",
    "cross_omics_rna_attention",
    "rna_spatial_attention",
    "modality2_spatial_attention",
)


def required_checkpoint_epochs(epochs: int) -> List[int]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    return sorted({0, min(epochs - 1, int(round(0.10 * epochs))), min(epochs - 1, int(round(0.50 * epochs))), epochs - 1})


def legacy_bug_weight_vector(features_omics1: torch.Tensor) -> torch.Tensor:
    """Reproduce the public argsort bug exactly; this is not an abundance score."""
    if features_omics1.ndim != 2:
        raise ValueError("features_omics1 must be locations-by-genes")
    d = features_omics1.shape[1]
    average = features_omics1.mean(dim=0)
    bad_expr_percentile = torch.argsort(average, descending=False).to(features_omics1.dtype) / d
    return 1.0 + 5.0 * torch.sigmoid(-10.0 * (bad_expr_percentile - 0.25))


@dataclass
class RnaLoss:
    raw_unweighted: torch.Tensor
    weighted_before_global: torch.Tensor
    global_multiplier: torch.Tensor
    final: torch.Tensor
    m_bad: torch.Tensor
    gene_weights: torch.Tensor
    v3_identity_error: torch.Tensor


def rna_loss_dispatch(
    diff: torch.Tensor,
    features_omics1: torch.Tensor,
    variant: str,
    asr_weights: Optional[torch.Tensor] = None,
) -> RnaLoss:
    if variant not in VARIANTS:
        raise ValueError("Unknown Night-2B variant: %s" % variant)
    squared = diff ** 2
    per_gene = torch.mean(squared, dim=0)
    raw = torch.mean(per_gene)
    bad = legacy_bug_weight_vector(features_omics1)
    m_bad = torch.mean(bad)
    v2 = torch.sum(bad * per_gene) / torch.sum(bad)
    # Preserve the exact frozen reduction order for V3/P0B replay.
    v3 = torch.mean(squared * bad.unsqueeze(0))
    identity_error = torch.abs(v3 - m_bad * v2)
    if not torch.allclose(v3, m_bad * v2, atol=1e-6, rtol=1e-6):
        raise AssertionError("V3 != m_bad * normalized legacy-shape loss")

    one = torch.ones((), dtype=features_omics1.dtype, device=features_omics1.device)
    if variant == "locked_unweighted":
        weights, weighted, multiplier, final = torch.ones_like(bad), raw, one, raw
    elif variant == "locked_uniform_legacy_scale":
        weights, weighted, multiplier, final = torch.ones_like(bad), raw, m_bad, m_bad * raw
    elif variant == "locked_legacy_shape_normalized":
        weights, weighted, multiplier, final = bad, v2, one, v2
    elif variant == "locked_legacy_loss_replay":
        weights, weighted, multiplier, final = bad, v2, m_bad, v3
    else:
        if asr_weights is None:
            raise ValueError("V4 requires name-aligned ASR-v1 weights")
        weights = asr_weights.to(dtype=features_omics1.dtype, device=features_omics1.device)
        if weights.shape != bad.shape:
            raise AssertionError("ASR weight shape does not match model columns")
        weighted = torch.sum(weights * per_gene) / torch.sum(weights)
        multiplier, final = m_bad, m_bad * weighted
    return RnaLoss(raw, weighted, multiplier, final, m_bad, weights, identity_error)


def compute_loss_components(
    result: Dict[str, torch.Tensor],
    features1: torch.Tensor,
    features2: torch.Tensor,
    factors: Iterable[float],
    variant: str,
    asr_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    factors = list(factors)
    rna = rna_loss_dispatch(features1 - result["emb_recon_omics1"], features1, variant, asr_weights)
    mod2 = F.mse_loss(features2, result["emb_recon_omics2"])
    corr1 = F.mse_loss(result["emb_latent_omics1"], result["emb_latent_omics1_across_recon"])
    corr2 = F.mse_loss(result["emb_latent_omics2"], result["emb_latent_omics2_across_recon"])
    total = factors[0] * rna.final + factors[1] * mod2 + factors[2] * corr1 + factors[3] * corr2
    return {
        "raw_rna": rna.raw_unweighted,
        "weighted_rna": rna.weighted_before_global,
        "global_multiplier": rna.global_multiplier,
        "rna": rna.final,
        "mod2": mod2,
        "corr1": corr1,
        "corr2": corr2,
        "total": total,
        "m_bad": rna.m_bad,
        "gene_weights": rna.gene_weights,
        "v3_identity_error": rna.v3_identity_error,
    }


def selected_legacy_gene_names(data: dict) -> np.ndarray:
    rna = data["adata_omics1"]
    mask = rna.var["highly_variable"].to_numpy(dtype=bool)
    names = rna.var_names[mask].astype(str).to_numpy()
    if names.size != rna.obsm["raw_feat"].shape[1]:
        raise AssertionError("Legacy ordered HVG names do not match RNA model columns")
    return names


def validate_name_weight_alignment(selected_names, weight_names, weights) -> np.ndarray:
    selected = np.asarray(selected_names, dtype=str)
    named = np.asarray(weight_names, dtype=str)
    values = np.asarray(weights, dtype=np.float32)
    if not np.array_equal(selected, named) or selected.size != values.size:
        raise AssertionError("Selected gene names and weight order are not identical")
    return values


def compute_locked_asr_weights(dataset: str, cfg: dict, config: dict, data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frozen ASR-v1 label-free, then map by exact legacy HVG order."""
    raw = sc.read_h5ad(cfg["rna"])
    raw.var_names_make_unique()
    raw.obs = pd.DataFrame(index=raw.obs_names.copy())
    sc.pp.filter_genes(raw, min_cells=int(config["min_cells"]))
    locked_obs = data["adata_omics1"].obs_names.astype(str)
    raw = raw[locked_obs].copy()
    counts = raw.X.tocsr().copy() if sp.issparse(raw.X) else sp.csr_matrix(raw.X)
    xlog = log_normalize_counts(counts)

    directed = transform_adjacent_matrix(data["adata_omics1"].uns["adj_spatial"]).tocsr()
    spatial = directed.maximum(directed.T).tocsr()
    spatial.data[:] = 1.0
    spatial.eliminate_zeros()
    scores = calculate_asr_scores(counts, xlog, spatial, tau=float(config["moran_shrinkage_tau"]))
    scores.index = raw.var_names.astype(str)
    ordered = selected_legacy_gene_names(data)
    if not np.all(np.isin(ordered, scores.index)):
        raise AssertionError("ASR table is missing selected legacy genes")
    aligned_names = scores.loc[ordered].index.to_numpy(dtype=str)
    weights = 1.0 + scores.loc[ordered, "Q_score"].to_numpy(dtype=np.float32)
    weights = validate_name_weight_alignment(ordered, aligned_names, weights)
    if np.any(weights < 1.0) or np.any(weights > 2.0 + 1e-7) or not np.all(np.isfinite(weights)):
        raise AssertionError("Frozen ASR-v1 weights are outside [1,2]")
    return ordered, weights


class ParityLockedTrainer:
    """Generalized loss trainer consuming the exact frozen legacy container fields."""

    def __init__(
        self,
        data: dict,
        cfg: dict,
        variant: str,
        seed: int,
        device: torch.device,
        asr_weights: Optional[np.ndarray] = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.cfg = cfg
        self.seed = seed
        self.device = device
        self.legacy = Train_SpaLORA(
            data,
            datatype=cfg["legacy_datatype"],
            device=device,
            random_seed=seed,
            learning_rate=0.0001,
            weight_decay=0.0,
            epochs=cfg["epochs"],
            dim_output=cfg["embedding_dim"],
        )
        self.features1 = self.legacy.features_omics1
        self.features2 = self.legacy.features_omics2
        self.adjacencies = (
            self.legacy.adj_spatial_omics1,
            self.legacy.adj_feature_omics1,
            self.legacy.adj_spatial_omics2,
            self.legacy.adj_feature_omics2,
        )
        self.factors = list(self.legacy.weight_factors)
        self.epochs = int(self.legacy.epochs)
        self.embedding_dim = int(self.legacy.dim_output)
        self.gene_names = selected_legacy_gene_names(data)
        self.obs_names = data["adata_omics1"].obs_names.astype(str).to_numpy()
        self.coordinates = np.asarray(data["adata_omics1"].obsm["spatial"]).copy()
        self.asr_weights = None if asr_weights is None else torch.as_tensor(asr_weights, dtype=torch.float32, device=device)
        if self.factors != list(cfg["loss_factors"]):
            raise AssertionError("Legacy loss factors differ from frozen config")
        if self.epochs != int(cfg["epochs"]) or self.embedding_dim != int(cfg["embedding_dim"]):
            raise AssertionError("Legacy epochs/embedding dimension differ from frozen config")
        expected = legacy_bug_weight_vector(self.features1)
        if not torch.equal(expected, self.legacy.weight_vector_omics1):
            raise AssertionError("Generalized legacy-bug weights differ from frozen trainer")

    def new_model(self) -> Encoder_overall:
        return Encoder_overall(
            self.features1.shape[1], self.embedding_dim, self.features2.shape[1], self.embedding_dim
        ).to(self.device)

    def forward(self, model: Encoder_overall) -> Dict[str, torch.Tensor]:
        return model(self.features1, self.features2, *self.adjacencies)

    def components(self, result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return compute_loss_components(
            result, self.features1, self.features2, self.factors, self.variant, self.asr_weights
        )

    def train(self) -> Tuple[dict, List[dict]]:
        model = self.new_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
        checkpoints = set(required_checkpoint_epochs(self.epochs))
        logs: List[dict] = []
        model.train()
        for epoch in range(self.epochs):
            result = self.forward(model)
            components = self.components(result)
            optimizer.zero_grad()
            components["total"].backward()
            optimizer.step()
            if epoch in checkpoints:
                weights = components["gene_weights"]
                row = {
                    "epoch": epoch,
                    "raw_rna_reconstruction": float(components["raw_rna"].detach().cpu()),
                    "weighted_rna_before_global_scale": float(components["weighted_rna"].detach().cpu()),
                    "global_scale_multiplier": float(components["global_multiplier"].detach().cpu()),
                    "final_rna_loss": float(components["rna"].detach().cpu()),
                    "final_rna_contribution": float((self.factors[0] * components["rna"]).detach().cpu()),
                    "raw_modality2_reconstruction": float(components["mod2"].detach().cpu()),
                    "final_modality2_contribution": float((self.factors[1] * components["mod2"]).detach().cpu()),
                    "raw_corr1": float(components["corr1"].detach().cpu()),
                    "final_corr1_contribution": float((self.factors[2] * components["corr1"]).detach().cpu()),
                    "raw_corr2": float(components["corr2"].detach().cpu()),
                    "final_corr2_contribution": float((self.factors[3] * components["corr2"]).detach().cpu()),
                    "total_loss": float(components["total"].detach().cpu()),
                    "m_bad": float(components["m_bad"].detach().cpu()),
                    "gene_weight_mean": float(weights.mean().detach().cpu()),
                    "gene_weight_min": float(weights.min().detach().cpu()),
                    "gene_weight_max": float(weights.max().detach().cpu()),
                    "cross_omics_rna_attention": float(result["alpha"][:, 0].mean().detach().cpu()),
                    "rna_spatial_attention": float(result["alpha_omics1"][:, 0].mean().detach().cpu()),
                    "modality2_spatial_attention": float(result["alpha_omics2"][:, 0].mean().detach().cpu()),
                }
                if tuple(row) != LOSS_LOG_FIELDS:
                    raise AssertionError("Night-2B loss log schema drift")
                logs.append(row)
        if [row["epoch"] for row in logs] != required_checkpoint_epochs(self.epochs):
            raise AssertionError("Night-2B loss checkpoints are incomplete")

        model.eval()
        with torch.no_grad():
            result = self.forward(model)
        output = {
            "emb_latent_omics1": F.normalize(result["emb_latent_omics1"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "emb_latent_omics2": F.normalize(result["emb_latent_omics2"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "SpaLORA": F.normalize(result["emb_latent_combined"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "alpha_omics1": result["alpha_omics1"].cpu().numpy(),
            "alpha_omics2": result["alpha_omics2"].cpu().numpy(),
            "alpha": result["alpha"].cpu().numpy(),
        }
        return output, logs
