"""Corrected, label-free SpaLORA preprocessing and training for Night 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.neighbors import kneighbors_graph

from .model_corrected import EncoderOverallCorrected
from .preprocess import clr_normalize_each_cell, pca, pca_deterministic


@dataclass
class PreparedData:
    data: dict
    obs_names: pd.Index
    coordinates: np.ndarray
    gene_table: pd.DataFrame


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros(values.size, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1.0)


def log_normalize_counts(counts: sp.spmatrix, target_sum: float = 10000.0) -> sp.csr_matrix:
    counts = counts.tocsr().astype(np.float32, copy=True)
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("All retained observations must have positive RNA library size")
    normalized = sp.diags((target_sum / library).astype(np.float32)).dot(counts).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def symmetric_knn_graph(features: np.ndarray, k: int, metric: str = "euclidean") -> sp.csr_matrix:
    directed = kneighbors_graph(
        features,
        n_neighbors=k,
        mode="connectivity",
        metric=metric,
        include_self=False,
    ).tocsr()
    graph = directed.maximum(directed.T).tocsr()
    graph.data[:] = 1.0
    graph.eliminate_zeros()
    return graph


def normalize_graph_sparse(adjacency: sp.spmatrix) -> torch.Tensor:
    adjacency = adjacency.tocsr().astype(np.float32)
    adjacency = adjacency.maximum(adjacency.T)
    adjacency = adjacency - sp.diags(adjacency.diagonal())
    adjacency.eliminate_zeros()
    with_self = adjacency + sp.eye(adjacency.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(with_self.sum(axis=1)).ravel()
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    normalized = sp.diags(inv_sqrt).dot(with_self).dot(sp.diags(inv_sqrt)).tocoo()
    indices = torch.from_numpy(np.vstack((normalized.row, normalized.col)).astype(np.int64))
    values = torch.from_numpy(normalized.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, normalized.shape, dtype=torch.float32).coalesce()


def moran_i_sparse(xlog: sp.spmatrix, adjacency: sp.spmatrix, chunk_size: int = 512) -> np.ndarray:
    """Vectorized per-gene Moran's I without materializing an N x N matrix."""
    xlog = xlog.tocsc().astype(np.float64)
    adjacency = adjacency.tocsr().astype(np.float64)
    adjacency = adjacency - sp.diags(adjacency.diagonal())
    adjacency.eliminate_zeros()
    n_obs, n_genes = xlog.shape
    s0 = float(adjacency.sum())
    if s0 <= 0:
        raise ValueError("Moran graph has no edges")
    means = np.asarray(xlog.sum(axis=0)).ravel() / n_obs
    sum_squares = np.asarray(xlog.power(2).sum(axis=0)).ravel()
    denominator = sum_squares - n_obs * means ** 2
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    result = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        block = xlog[:, start:stop].tocsr()
        weighted = adjacency.dot(block)
        xtwx = np.asarray(block.multiply(weighted).sum(axis=0)).ravel()
        dtx = np.asarray(block.T.dot(degree)).ravel()
        mu = means[start:stop]
        numerator = xtwx - 2.0 * mu * dtx + (mu ** 2) * s0
        valid = denominator[start:stop] > 0
        block_result = np.zeros(stop - start, dtype=np.float64)
        block_result[valid] = (n_obs / s0) * numerator[valid] / denominator[start:stop][valid]
        result[start:stop] = block_result
    return result


def calculate_asr_scores(
    counts: sp.spmatrix,
    xlog: sp.spmatrix,
    spatial_graph: sp.spmatrix,
    tau: float = 20.0,
) -> pd.DataFrame:
    mean_xlog = np.asarray(xlog.mean(axis=0)).ravel()
    abundance = 1.0 - percentile_rank(mean_xlog)
    moran = moran_i_sparse(xlog, spatial_graph)
    moran_clipped = np.maximum(moran, 0.0)
    spatial_rank = percentile_rank(moran_clipped)
    detected = np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int64)
    reliability = detected / (detected + float(tau))
    q = abundance * spatial_rank * reliability
    return pd.DataFrame(
        {
            "mean_log_abundance": mean_xlog,
            "A_score": abundance,
            "morans_I": moran,
            "moran_clipped": moran_clipped,
            "S_score": spatial_rank,
            "detection_count": detected,
            "detection_rate": detected / counts.shape[0],
            "R_score": reliability,
            "Q_score": q,
        }
    )


def select_gene_mask(is_hvg: np.ndarray, q: np.ndarray, variant: str, rescue_non_hvg: int) -> Tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(is_hvg, dtype=bool).copy()
    rescued = np.zeros(selected.size, dtype=bool)
    if variant == "asr_rescue":
        non_hvg = np.flatnonzero(~selected)
        order = non_hvg[np.lexsort((non_hvg, -np.asarray(q)[non_hvg]))]
        rescue = order[:rescue_non_hvg]
        selected[rescue] = True
        rescued[rescue] = True
    return selected, rescued


def calculate_weights(scores: pd.DataFrame, variant: str, alpha: float) -> np.ndarray:
    if variant == "corrected_unweighted":
        evidence = np.zeros(len(scores), dtype=np.float64)
    elif variant == "abundance_only":
        evidence = scores["A_score"].to_numpy(dtype=np.float64)
    elif variant in ("asr_hvg", "asr_rescue"):
        evidence = scores["Q_score"].to_numpy(dtype=np.float64)
    else:
        raise ValueError("Unknown corrected variant: %s" % variant)
    weights = 1.0 + float(alpha) * evidence
    if not np.all(np.isfinite(weights)) or np.any(weights < 1.0) or np.any(weights > 1.0 + alpha + 1e-12):
        raise AssertionError("Corrected RNA weights are outside their declared bounds")
    return weights


def _load_label_free(cfg: dict) -> Tuple[ad.AnnData, ad.AnnData]:
    rna = sc.read_h5ad(cfg["rna"])
    mod2 = sc.read_h5ad(cfg["modality2"])
    rna.var_names_make_unique()
    mod2.var_names_make_unique()
    if not rna.obs_names.equals(mod2.obs_names):
        raise AssertionError("Paired modalities must have identical ordered observation IDs")
    if not np.array_equal(rna.obsm["spatial"], mod2.obsm["spatial"]):
        raise AssertionError("Paired modalities require identical spatial coordinates")
    rna.obs = pd.DataFrame(index=rna.obs_names.copy())
    mod2.obs = pd.DataFrame(index=mod2.obs_names.copy())
    return rna, mod2


def _dense_scaled(xlog: sp.spmatrix) -> np.ndarray:
    holder = ad.AnnData(xlog.copy())
    sc.pp.scale(holder)
    return np.asarray(holder.X, dtype=np.float32)


def prepare_corrected(dataset: str, cfg: dict, config: dict, variant: str) -> PreparedData:
    if variant not in ("corrected_unweighted", "abundance_only", "asr_hvg", "asr_rescue"):
        raise ValueError("Unknown corrected variant: %s" % variant)
    rna, mod2 = _load_label_free(cfg)
    original_counts = rna.X.tocsr().copy() if sp.issparse(rna.X) else sp.csr_matrix(rna.X)
    if original_counts.data.size and not np.allclose(original_counts.data, np.rint(original_counts.data)):
        raise AssertionError("RNA input must be raw count-like data")

    gene_mask = np.asarray((original_counts > 0).sum(axis=0)).ravel() >= config["min_cells"]
    rna = rna[:, gene_mask].copy()
    counts = original_counts[:, gene_mask].tocsr()
    if dataset == "p22":
        cell_mask = np.asarray((counts > 0).sum(axis=1)).ravel() >= 200
        rna = rna[cell_mask].copy()
        mod2 = mod2[cell_mask].copy()
        counts = counts[cell_mask].tocsr()
    rna.X = counts.copy()
    if not rna.obs_names.equals(mod2.obs_names):
        raise AssertionError("Filtering broke paired observation alignment")

    hvg_holder = ad.AnnData(counts.copy(), dtype=counts.dtype)
    hvg_holder.var_names = rna.var_names.copy()
    sc.pp.highly_variable_genes(hvg_holder, flavor="seurat_v3", n_top_genes=cfg["hvg"])
    is_hvg = hvg_holder.var["highly_variable"].to_numpy(dtype=bool)
    if int(is_hvg.sum()) != cfg["hvg"]:
        raise AssertionError("Unexpected HVG count")

    xlog = log_normalize_counts(counts)
    spatial_graph = symmetric_knn_graph(
        np.asarray(rna.obsm["spatial"]), cfg["spatial_neighbors"], metric="euclidean"
    )
    scores = calculate_asr_scores(
        counts,
        xlog,
        spatial_graph,
        tau=config["moran_shrinkage_tau"],
    )

    selected, rescued = select_gene_mask(
        is_hvg,
        scores["Q_score"].to_numpy(),
        variant,
        config["rescue_non_hvg"],
    )
    selected_idx = np.flatnonzero(selected)

    all_weights = calculate_weights(scores, variant, config["alpha"])
    weight_vector = all_weights[selected_idx].astype(np.float32)

    scaled = _dense_scaled(xlog[:, selected_idx])
    rna.obsm["raw_feat"] = scaled
    pca_components = 50 if dataset == "p22" else mod2.n_vars - 1
    deterministic_pca = bool(config.get("deterministic_pca", False))
    if deterministic_pca:
        rna_scores, rna_pca_metadata = pca_deterministic(
            ad.AnnData(scaled),
            n_comps=pca_components,
            svd_solver=config["pca_svd_solver"],
            random_state=config["pca_random_state"],
            return_metadata=True,
        )
        rna.obsm["feat"] = rna_scores
    else:
        rna.obsm["feat"] = pca(ad.AnnData(scaled), n_comps=pca_components)
        rna_pca_metadata = None

    if dataset == "p22":
        if "X_lsi" not in mod2.obsm:
            raise AssertionError("P22 modality 2 requires its deposited X_lsi representation")
        mod2.obsm["feat"] = np.asarray(mod2.obsm["X_lsi"], dtype=np.float32).copy()
    else:
        mod2 = clr_normalize_each_cell(mod2)
        sc.pp.scale(mod2)
        mod2.obsm["feat"] = pca(mod2, n_comps=mod2.n_vars - 1)

    feature_metric = config["feature_graph"]["metric"]
    feature_k = config["feature_graph"]["k"]
    spatial1 = spatial_graph
    spatial2 = symmetric_knn_graph(
        np.asarray(mod2.obsm["spatial"]), cfg["spatial_neighbors"], metric="euclidean"
    )
    feature1 = symmetric_knn_graph(np.asarray(rna.obsm["feat"]), feature_k, feature_metric)
    feature2 = symmetric_knn_graph(np.asarray(mod2.obsm["feat"]), feature_k, feature_metric)

    gene_table = scores.copy()
    gene_table.insert(0, "gene", rna.var_names.astype(str))
    gene_table["is_hvg"] = is_hvg
    gene_table["is_selected"] = selected
    gene_table["is_asr_rescued"] = rescued
    gene_table["final_weight"] = all_weights
    gene_table["variant"] = variant
    selected_gene_names = rna.var_names[selected_idx].astype(str).to_numpy()
    table_selected_names = gene_table.loc[gene_table["is_selected"], "gene"].to_numpy(dtype=str)
    if not np.array_equal(selected_gene_names, table_selected_names) or len(selected_gene_names) != len(weight_vector):
        raise AssertionError("Selected gene names and weight-vector order are not identical")

    # Counts remain separate and immutable; only the scaled Xlog selection enters the model.
    data = {
        "features_omics1": scaled,
        "features_omics2": np.asarray(mod2.obsm["feat"], dtype=np.float32),
        "weight_vector_omics1": weight_vector,
        "selected_gene_names": selected_gene_names,
        "adj_spatial_omics1": normalize_graph_sparse(spatial1),
        "adj_spatial_omics2": normalize_graph_sparse(spatial2),
        "adj_feature_omics1": normalize_graph_sparse(feature1),
        "adj_feature_omics2": normalize_graph_sparse(feature2),
        "counts_immutable": counts,
        "xlog_immutable": xlog,
    }
    if deterministic_pca:
        data.update({
            "rna_pca_scores": np.asarray(rna.obsm["feat"], dtype=np.float32),
            "rna_pca_explained_variance": np.asarray(
                rna_pca_metadata["explained_variance"], dtype=np.float64
            ),
            "rna_pca_explained_variance_ratio": np.asarray(
                rna_pca_metadata["explained_variance_ratio"], dtype=np.float64
            ),
            "rna_pca_metadata": {
                key: value for key, value in rna_pca_metadata.items()
                if key not in ("explained_variance", "explained_variance_ratio")
            },
        })
    return PreparedData(
        data=data,
        obs_names=rna.obs_names.copy(),
        coordinates=np.asarray(rna.obsm["spatial"]),
        gene_table=gene_table,
    )


def weighted_gene_mse(diff: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_gene = torch.mean(diff ** 2, dim=0)
    return torch.sum(weights * per_gene) / torch.sum(weights)


def train_corrected(data: dict, cfg: dict, variant: str, seed: int, device: torch.device) -> dict:
    del variant, seed  # Fixed upstream; kept in signature to make run metadata explicit.
    features1 = torch.as_tensor(data["features_omics1"], dtype=torch.float32, device=device)
    features2 = torch.as_tensor(data["features_omics2"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(data["weight_vector_omics1"], dtype=torch.float32, device=device)
    adjacencies = [
        data["adj_spatial_omics1"].to(device),
        data["adj_feature_omics1"].to(device),
        data["adj_spatial_omics2"].to(device),
        data["adj_feature_omics2"].to(device),
    ]
    model = EncoderOverallCorrected(
        features1.shape[1], cfg["embedding_dim"], features2.shape[1], cfg["embedding_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
    factors = cfg["loss_factors"]
    model.train()
    for _ in range(cfg["epochs"]):
        result = model(features1, features2, *adjacencies)
        loss_rna = weighted_gene_mse(features1 - result["emb_recon_omics1"], weights)
        loss_mod2 = F.mse_loss(features2, result["emb_recon_omics2"])
        loss_corr1 = F.mse_loss(result["emb_latent_omics1"], result["emb_latent_omics1_across_recon"])
        loss_corr2 = F.mse_loss(result["emb_latent_omics2"], result["emb_latent_omics2_across_recon"])
        loss = factors[0] * loss_rna + factors[1] * loss_mod2 + factors[2] * loss_corr1 + factors[3] * loss_corr2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        result = model(features1, features2, *adjacencies)
    return {
        "emb_latent_omics1": F.normalize(result["emb_latent_omics1"], p=2, dim=1).cpu().numpy(),
        "emb_latent_omics2": F.normalize(result["emb_latent_omics2"], p=2, dim=1).cpu().numpy(),
        "SpaLORA": F.normalize(result["emb_latent_combined"], p=2, dim=1).cpu().numpy(),
        "alpha_omics1": result["alpha_omics1"].cpu().numpy(),
        "alpha_omics2": result["alpha_omics2"].cpu().numpy(),
        "alpha": result["alpha"].cpu().numpy(),
    }
