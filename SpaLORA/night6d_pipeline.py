"""Locked Night-6D D1/P22 confirmation contracts.

Graph/head algorithms are imported unchanged from the SHA-locked Night-6C
implementation.  This module only supplies the preregistered dataset contracts
and the two-by-two factorial registry.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .night5a_rnd import Night5ATrainer
from .night6c_pipeline import (  # noqa: F401
    BASE_C04, NumericalHeadFailure, VIEW_KEYS, array_sha, atomic_json,
    atomic_torch_save, build_graph_data, canonical_json_sha, file_row,
    forward_model, h00, load_graph_data, load_views, moran_scores,
    observation_sha, run_head, runtime_resources, save_views, sha256_file,
    sparse_sha,
)


DATASET_CFG = {
    "d1": {
        "n_clusters": 10,
        "embedding_dim": 64,
        "epochs": 200,
        "loss_factors": [1.9, 2.5, 1.5, 10.0],
        "locked_m_bad_expected": 2.289938091,
    },
    "p22": {
        "n_clusters": 9,
        "embedding_dim": 128,
        "epochs": 1600,
        "loss_factors": [1.5, 5.0, 1.5, 1.0],
        "locked_m_bad_expected": 2.290322960,
    },
}

GRAPHS = OrderedDict((row["id"], row) for row in (
    {
        "family": "reference",
        "feature_k": 20,
        "feature_metric": "correlation",
        "id": "G00_SP18_F20_CORR_UNION",
        "spatial_k": 18,
        "spatial_refinement": "none",
        "spatial_symmetrization": "union",
    },
    {
        "family": "feature_scale_metric",
        "feature_k": 10,
        "feature_metric": "euclidean",
        "id": "G04_SP10_F10_EUC_UNION",
        "spatial_k": 10,
        "spatial_refinement": "none",
        "spatial_symmetrization": "union",
    },
))

HEADS = OrderedDict((row["id"], row) for row in (
    {
        "algorithm": "deterministic_pca20_then_mclust_EEE",
        "family": "reference",
        "id": "H00_FUSED_PCA20_MCLUST_EEE",
        "input_views": ["SpaLORA_fused"],
    },
    {
        "algorithm": "arithmetic_mean_of_three_sparse_self_tuning_affinities_then_spectral",
        "family": "sparse_affinity",
        "id": "H05_EQUAL3_AFFINITY_SPECTRAL",
        "input_views": ["emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"],
    },
))

EXPECTED_GRAPH_SHA = {
    "G00_SP18_F20_CORR_UNION": "9e4f637c1460c7d1e7206b93df549cd4befbadeff9de220dc76297f7a47f1121",
    "G04_SP10_F10_EUC_UNION": "abd14bfc267f964927ea952359fa5cc94f9bfabe101f176805c037f1946bc639",
}
EXPECTED_HEAD_SHA = {
    "H00_FUSED_PCA20_MCLUST_EEE": "6a934dde7ed3e121937f34edf115aa3d8a6f51b242be907e27b271271ba21374",
    "H05_EQUAL3_AFFINITY_SPECTRAL": "03f542f9bd43cbbe953145dbedd3bad02a2a2c15baddbf4e05925e308ec0c6f2",
}


def validate_locked_contracts() -> None:
    if list(GRAPHS) != ["G00_SP18_F20_CORR_UNION", "G04_SP10_F10_EUC_UNION"]:
        raise RuntimeError("Night-6D graph order drift")
    if list(HEADS) != ["H00_FUSED_PCA20_MCLUST_EEE", "H05_EQUAL3_AFFINITY_SPECTRAL"]:
        raise RuntimeError("Night-6D head order drift")
    for key, value in GRAPHS.items():
        if canonical_json_sha(value) != EXPECTED_GRAPH_SHA[key]:
            raise RuntimeError(f"locked graph config SHA mismatch: {key}")
    for key, value in HEADS.items():
        if canonical_json_sha(value) != EXPECTED_HEAD_SHA[key]:
            raise RuntimeError(f"locked head config SHA mismatch: {key}")


def make_trainer(data, dataset: str, seed: int, device: torch.device) -> Night5ATrainer:
    if dataset not in DATASET_CFG:
        raise RuntimeError(f"unregistered Night-6D dataset: {dataset}")
    return Night5ATrainer(data, DATASET_CFG[dataset], BASE_C04, int(seed), device, {}, 1e-12)


validate_locked_contracts()

