"""Label-free primitives for the frozen Night-8B MISAR confirmation.

This module intentionally has no evaluator and never reads the HDF5 ``Y``
dataset.  Dataset identity is used only by the data steward; training workers
receive content-addressed arrays and the assay family contract.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from .night1_pipeline import (
    PreparedData, calculate_asr_scores, calculate_weights, log_normalize_counts,
    normalize_graph_sparse, symmetric_knn_graph,
)
from .night3af_cache import load_cache, save_cache, sha256_file
from .night6c_pipeline import (
    array_sha, atomic_json, build_graph_data, canonical_json_sha, forward_model,
    sparse_sha,
)
from .night7a_consensus import (
    atomic_sparse, build_base_affinities, candidate_affinity,
    canonical_partition, partition_sha, run_spectral,
)
from .night7b_adaptive import VIEWS, graph_summary


REPO = Path(__file__).resolve().parents[1]
RAW = Path("/root/autodl-fs/night8b_raw_runs_20260820")
SOURCE = Path("/root/autodl-fs/night8a_external_data_20260820")
ANN = RAW / "annotation_carrier"
CACHE = RAW / "cache"
RUNS = RAW / "formal"
OUT = REPO / "outputs/night8b_handoff"
REGISTRY = REPO / "protocols/night8b/SpaLORA_Night8B_MISAR_Family_Policy_Registry_2026-08-20.json"

G00 = "G00_SP18_F20_CORR_UNION"
G04 = "G04_SP10_F10_EUC_UNION"
BASE_C04 = {
    "id": "C04_SHRINK25", "family": "fusion",
    "description": "Learned fusion shrunk strongly toward equal weights",
    "attention": "shrink_to_uniform", "learned_fraction_rho": 0.25,
    "corr2": False, "loss_calibration": "active_set_IGE",
}
MISAR_CFG = {
    "n_clusters": 12, "embedding_dim": 128, "epochs": 1600,
    "loss_factors": [1.5, 5.0, 1.5, 1.0],
    "locked_m_bad_expected": 2.29032296,
}
GRAPH_CONTRACTS = {
    G00: {"id": G00, "family": "reference", "spatial_k": 18,
          "feature_k": 20, "feature_metric": "correlation",
          "spatial_refinement": "none", "spatial_symmetrization": "union"},
    G04: {"id": G04, "family": "feature_scale_metric", "spatial_k": 10,
          "feature_k": 10, "feature_metric": "euclidean",
          "spatial_refinement": "none", "spatial_symmetrization": "union"},
}


def file_md5(path: Path) -> str:
    d = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def observation_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def state_sha(state: Mapping[str, torch.Tensor]) -> str:
    d = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        d.update(name.encode()); d.update(str(value.dtype).encode())
        d.update(np.asarray(value.shape, np.int64).tobytes())
        d.update(value.numpy().tobytes(order="C"))
    return d.hexdigest()


def rng_snapshot() -> dict:
    return {
        "python_repr": repr(random.getstate()),
        "numpy_repr": repr(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _selected_observations() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tissue = [x for x in pd.read_csv(SOURCE / "position_E15_5-S1.txt", header=None).iloc[0]
              if isinstance(x, str)]
    table = pd.read_csv(SOURCE / "MISAR-seq_barcode_filter.csv")
    if table["array"].duplicated().any() or len(tissue) != 1949:
        raise RuntimeError("tissue position contract mismatch")
    indexed = table.set_index("array")
    if not set(tissue) <= set(indexed.index.astype(str)):
        raise RuntimeError("tissue positions missing from barcode map")
    barcodes = indexed.loc[tissue, "barcode"].astype(str).to_numpy()
    coords_table = pd.read_csv(SOURCE / "MISAR-seq_barcode.csv").set_index("barcode")
    coords = coords_table.loc[barcodes, ["array_col", "array_row"]].to_numpy(np.float32)
    if len(set(barcodes)) != 1949 or len(set(map(tuple, coords))) != 1949:
        raise RuntimeError("observation or coordinate mapping is not one-to-one")
    return np.asarray(tissue, str), barcodes, coords


def load_official_counts():
    tissue, wanted, coords = _selected_observations()
    path = SOURCE / "E15_5-S1_raw_feature_bc_matrix.h5"
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(map(int, group["shape"][:]))
        matrix = sp.csc_matrix((group["data"][:], group["indices"][:],
                                group["indptr"][:]), shape=shape).T.tocsr()
        barcodes = group["barcodes"][:].astype("U")
        kinds = group["features/feature_type"][:].astype("U")
        names = group["features/name"][:].astype("U")
    index = {value: idx for idx, value in enumerate(barcodes)}
    if len(index) != len(barcodes) or any(x not in index for x in wanted):
        raise RuntimeError("official matrix barcode mapping mismatch")
    selected = np.asarray([index[x] for x in wanted], dtype=np.int64)
    rna_mask = kinds == "Gene Expression"; atac_mask = kinds == "Peaks"
    if int(rna_mask.sum()) != 32285 or int(atac_mask.sum()) != 141420:
        raise RuntimeError("official feature-type cardinality mismatch")
    rna = matrix[selected][:, rna_mask].tocsr()
    atac = matrix[selected][:, atac_mask].tocsr()
    if rna.shape != (1949, 32285) or atac.shape != (1949, 141420):
        raise RuntimeError("official modality shape mismatch")
    return wanted, coords, rna, atac, names[rna_mask], names[atac_mask], tissue


def _lsi50(counts: sp.csr_matrix) -> np.ndarray:
    from .preprocess import lsi
    holder = ad.AnnData(counts.astype(np.float32))
    # This is the exact repository Seurat-v3-style path used by Tutorial 2:
    # n_components=51 followed by dropping component 0 gives 50 dimensions.
    lsi(holder, use_highly_variable=False, n_components=51, random_state=0)
    value = np.asarray(holder.obsm["X_lsi"], dtype=np.float32)
    if value.shape != (1949, 50) or not np.isfinite(value).all():
        raise RuntimeError("fixed 50-dimensional ATAC LSI failed")
    return value


def _exact_seurat_v3_hvg_mask(holder: ad.AnnData, n_top: int) -> tuple[np.ndarray, dict]:
    """Resolve the Scanpy 1.9.1 NaN-rank flag bug without changing HVG ranks.

    ``scanpy.pp.highly_variable_genes(..., flavor='seurat_v3')`` can flag one
    additional feature whose ``highly_variable_rank`` is NaN.  The frozen
    family contract is exactly ``n_top`` ranked genes, so the authoritative
    mask is the finite ranks [0, n_top).  No expression or label-dependent
    parameter is introduced by this mechanical compatibility correction.
    """
    sc.pp.highly_variable_genes(holder, flavor="seurat_v3", n_top_genes=n_top)
    reported = holder.var["highly_variable"].to_numpy(dtype=bool)
    ranks = holder.var["highly_variable_rank"].to_numpy(dtype=np.float64)
    selected = np.isfinite(ranks) & (ranks < float(n_top))
    if int(selected.sum()) != int(n_top):
        raise RuntimeError("Seurat-v3 finite-rank HVG contract is not exact")
    finite = np.sort(ranks[selected])
    if not np.array_equal(finite, np.arange(n_top, dtype=np.float64)):
        raise RuntimeError("Seurat-v3 HVG ranks are not the locked 0..n_top-1 set")
    audit = {
        "scanpy_boolean_count": int(reported.sum()),
        "finite_rank_count": int(np.isfinite(ranks).sum()),
        "locked_exact_count": int(selected.sum()),
        "compatibility_rule": "finite highly_variable_rank in [0,n_top)",
        "nan_rank_boolean_true_excluded": int(np.sum(reported & ~np.isfinite(ranks))),
        "label_dependent": False,
    }
    return selected, audit


def prepare_misar() -> PreparedData:
    ids, coords, rna, atac, genes, _peaks, _tissue = load_official_counts()
    detected = np.asarray((rna > 0).sum(axis=0)).ravel()
    keep = detected >= 10
    counts = rna[:, keep].tocsr(); genes = genes[keep]
    if np.any(np.asarray((counts > 0).sum(axis=1)).ravel() < 200):
        raise RuntimeError("frozen RNA_EPIGENOME >=200-gene cell gate changed 1949 spots")
    holder = ad.AnnData(counts.copy(), dtype=counts.dtype)
    holder.var_names = pd.Index(genes.astype(str))
    selected, hvg_audit = _exact_seurat_v3_hvg_mask(holder, 2000)
    xlog = log_normalize_counts(counts, 10000.0)
    spatial16 = symmetric_knn_graph(coords, 16, "euclidean")
    scores = calculate_asr_scores(counts, xlog, spatial16, tau=20.0)
    weights = calculate_weights(scores, "corrected_unweighted", 1.0)[selected].astype(np.float32)
    scaled_holder = ad.AnnData(xlog[:, selected].copy())
    sc.pp.scale(scaled_holder)
    scaled = np.asarray(scaled_holder.X, dtype=np.float32)
    from .preprocess import pca_deterministic
    rna_pca, meta = pca_deterministic(ad.AnnData(scaled), n_comps=50,
                                      svd_solver="randomized", random_state=0,
                                      return_metadata=True)
    atac_lsi = _lsi50(atac)
    feature20_rna = symmetric_knn_graph(rna_pca, 20, "correlation")
    feature20_atac = symmetric_knn_graph(atac_lsi, 20, "correlation")
    data = {
        "features_omics1": scaled, "features_omics2": atac_lsi,
        "weight_vector_omics1": weights,
        "selected_gene_names": genes[selected].astype(str),
        "rna_pca_scores": np.asarray(rna_pca, np.float32),
        "rna_pca_explained_variance": np.asarray(meta["explained_variance"], np.float64),
        "rna_pca_explained_variance_ratio": np.asarray(meta["explained_variance_ratio"], np.float64),
        "rna_pca_metadata": {
            **{k: v for k, v in meta.items()
               if k not in {"explained_variance", "explained_variance_ratio"}},
            "hvg_selection_audit": hvg_audit,
        },
        "adj_spatial_omics1": normalize_graph_sparse(spatial16),
        "adj_spatial_omics2": normalize_graph_sparse(spatial16),
        "adj_feature_omics1": normalize_graph_sparse(feature20_rna),
        "adj_feature_omics2": normalize_graph_sparse(feature20_atac),
    }
    return PreparedData(data=data, obs_names=pd.Index(ids), coordinates=coords,
                        gene_table=pd.DataFrame(index=np.arange(len(genes))))


def build_caches() -> dict:
    base = CACHE / "base"
    if (base / "manifest.json").is_file():
        prepared = load_cache(base)
    else:
        prepared = prepare_misar()
        save_cache(base, "opaque_rna_epigenome_external", prepared,
                   {"pca_svd_solver": "randomized", "pca_random_state": 0})
    base_sha = sha256_file(base / "manifest.json")
    graph_rows = {}
    for graph_id, contract in GRAPH_CONTRACTS.items():
        target = CACHE / "graphs" / graph_id
        if not (target / "manifest.json").is_file():
            build_graph_data(prepared, contract, target, base_sha)
        graph_rows[graph_id] = {
            "directory": str(target), "manifest_sha256": sha256_file(target / "manifest.json"),
            "manifest": json.loads((target / "manifest.json").read_text()),
        }
    return {"base_dir": str(base), "base_manifest_sha256": base_sha,
            "graph_caches": graph_rows,
            "ordered_observation_sha256": observation_sha(prepared.obs_names.astype(str)),
            "observation_count": len(prepared.obs_names)}


def load_graph(graph_id: str):
    prepared = load_cache(CACHE / "base")
    from .night6c_pipeline import load_graph_data
    data, manifest = load_graph_data(prepared, CACHE / "graphs" / graph_id)
    return prepared, data, manifest


def base_affinities(view00: Mapping[str, np.ndarray], view04: Mapping[str, np.ndarray],
                    ids: Sequence[str], coords: np.ndarray):
    values = build_base_affinities(view00, view04, ids, coords)
    c06, c06_aux = candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN", values, ids)
    return values, c06, c06_aux


def adapter_endpoint(embedding: np.ndarray, c06: sp.spmatrix,
                     ids: Sequence[str]) -> sp.csr_matrix:
    from .night6c_pipeline import self_tuning_affinity
    from .night7b_adaptive import row_sparse_strict, sym_zero
    az = self_tuning_affinity(embedding, 10, ids)
    return sym_zero(row_sparse_strict(az) * 0.5 + row_sparse_strict(c06) * 0.5)


def resource_row(started: float) -> dict:
    return {
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1048576.0 if torch.cuda.is_available() else 0.0,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
