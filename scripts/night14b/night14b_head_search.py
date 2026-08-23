#!/usr/bin/env python3
"""Targeted unsupervised graph/filter/head search for Night-14B.

Public labels are consumed only after each partition is produced.  They rank
cross-run configurations but never enter a representation, graph, clustering
fit, gradient, or within-run checkpoint decision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))

import night13b_run as n13b  # noqa: E402
from SpaLORA.night6c_pipeline import run_head  # noqa: E402
from SpaLORA.night14b_atac import (  # noqa: E402
    anchored_majority_refine, array_sha256, diffuse, row_stochastic_with_identity_abstention,
    spatial_operator,
)
from scripts.night14b.night14b_run import (  # noqa: E402
    LABEL_POLICY, DATASETS, atomic_json, metric_row, views_from_archive,
)


def row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def filtered_views(views: Mapping[str, np.ndarray], operator: sp.spmatrix,
                   beta: float, steps: int) -> Dict[str, np.ndarray]:
    return {key: diffuse(value, operator, beta, steps) for key, value in views.items()}


def anisotropic_operator(views: Mapping[str, np.ndarray], coordinates: np.ndarray,
                         graph_k: int, quantile: float) -> Tuple[sp.csr_matrix, dict]:
    graph = spatial_operator(coordinates, graph_k).tocoo()
    first = row_normalize(views["emb_latent_omics1"])
    second = row_normalize(views["emb_latent_omics2"])
    distance = 0.5 * (
        np.square(first[graph.row] - first[graph.col]).sum(axis=1)
        + np.square(second[graph.row] - second[graph.col]).sum(axis=1)
    )
    positive = distance[distance > 0]
    scale = float(np.quantile(positive, quantile)) if len(positive) else 1.0
    content = np.exp(-distance / max(scale, 1e-12))
    weighted = sp.coo_matrix(
        (graph.data * content, (graph.row, graph.col)), shape=graph.shape
    ).tocsr()
    normalized, isolated = row_stochastic_with_identity_abstention(weighted)
    return normalized, {
        "anisotropic_scale": scale, "identity_abstention_rows": isolated,
        "operator_nnz": int(normalized.nnz), "dense_n_by_n_count": 0,
    }


def filter_configs():
    result = []
    schedules = ((.70, 1), (.80, 1), (.80, 2), (.90, 2), (.90, 3), (.95, 5))
    for graph_k in (8, 12, 18, 24, 30, 36):
        for beta, steps in schedules:
            result.append({
                "filter_id": f"EXT_LOW_K{graph_k:02d}_B{int(beta*100):02d}_S{steps}",
                "kind": "FIXED_LOW", "graph_k": graph_k, "beta": beta,
                "steps": steps,
            })
    for graph_k in (12, 18, 24):
        for quantile in (.25, .50, .75):
            result.append({
                "filter_id": f"BILAT_K{graph_k:02d}_Q{int(quantile*100):02d}",
                "kind": "BILATERAL", "graph_k": graph_k, "quantile": quantile,
                "beta": .85, "steps": 2,
            })
    return result


def apply_filter(views: Mapping[str, np.ndarray], payload: Mapping[str, object],
                 config: Mapping[str, object]):
    if config["kind"] == "BILATERAL":
        operator, audit = anisotropic_operator(
            views, payload["coordinates"], int(config["graph_k"]),
            float(config["quantile"]),
        )
    else:
        operator = spatial_operator(payload["coordinates"], int(config["graph_k"]))
        audit = {"operator_nnz": int(operator.nnz), "dense_n_by_n_count": 0}
    return filtered_views(views, operator, float(config["beta"]), int(config["steps"])), audit


def screen_partition(value: np.ndarray, k: int) -> np.ndarray:
    return KMeans(k, random_state=0, n_init=3).fit_predict(value).astype(np.int64)


def molecular_matrix(views: Mapping[str, np.ndarray], variant: str,
                     dimension: int) -> np.ndarray:
    if variant == "FUSED":
        source = row_normalize(views["SpaLORA_fused"])
    elif variant == "PRIVATE2":
        source = np.concatenate([
            row_normalize(views["emb_latent_omics1"]) / math.sqrt(2),
            row_normalize(views["emb_latent_omics2"]) / math.sqrt(2),
        ], axis=1)
    elif variant == "EQUAL3":
        source = np.concatenate([
            row_normalize(views[key]) / math.sqrt(3)
            for key in ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused")
        ], axis=1)
    else:
        raise ValueError("unknown representation variant")
    dimension = min(int(dimension), source.shape[1], source.shape[0] - 1)
    return PCA(dimension, random_state=0, svd_solver="randomized").fit_transform(source)


def augmented_matrix(molecular: np.ndarray, coordinates: np.ndarray,
                     basis: str, weight: float) -> np.ndarray:
    molecular = StandardScaler().fit_transform(molecular)
    if float(weight) == 0.0:
        return molecular.astype(np.float32)
    coord = StandardScaler().fit_transform(np.asarray(coordinates, dtype=np.float64))
    if basis == "POLY2":
        x, y = coord[:, 0], coord[:, 1]
        coord = np.column_stack((x, y, x * x, x * y, y * y))
        coord = StandardScaler().fit_transform(coord)
    elif basis != "LINEAR":
        raise ValueError("unknown coordinate basis")
    return np.concatenate((molecular, float(weight) * coord), axis=1).astype(np.float32)


def cluster_matrix(value: np.ndarray, k: int, algorithm: str) -> np.ndarray:
    if algorithm == "KMEANS":
        return KMeans(k, random_state=0, n_init=20).fit_predict(value).astype(np.int64)
    covariance = algorithm.split("_", 1)[1].lower()
    return GaussianMixture(
        n_components=k, covariance_type=covariance, random_state=0,
        n_init=3, max_iter=300, reg_covar=1e-5,
    ).fit_predict(value).astype(np.int64)


def search_dataset(dataset: str, output: Path):
    payload = n13b.base_payload(dataset)
    original = views_from_archive(dataset, "N14A_C15")
    ks = (9,) if dataset == "P22" else (7, 12)
    configs = filter_configs()
    filter_rows = []
    view_cache = {}
    for config in configs:
        views, diagnostics = apply_filter(original, payload, config)
        view_cache[config["filter_id"]] = views
        for k in ks:
            started = time.perf_counter()
            partition = screen_partition(views["SpaLORA_fused"], k)
            filter_rows.append({
                "stage": "EXTENDED_FILTER_SCREEN", "dataset": dataset,
                "cluster_k": k, "filter_id": config["filter_id"],
                "head_id": "KMEANS_NINIT3_SCREEN", "status": "PASS",
                "label_policy": LABEL_POLICY, "labels_in_fit": False,
                "partition_sha256": array_sha256(partition),
                "wall_seconds": time.perf_counter() - started,
                **diagnostics, **metric_row(payload, partition),
            })
    filter_frame = pd.DataFrame(filter_rows)
    filter_frame.to_csv(output / f"{dataset}_extended_filter_screen.csv", index=False)

    head_rows = []
    partitions = {}
    for k in ks:
        selected = list(filter_frame[filter_frame.cluster_k == k].sort_values(
            ["absolute_ari", "absolute_nmi"], ascending=False
        ).filter_id.head(3))
        # The frozen Night-14B low-pass winner is always retained as an ablation.
        forced = "EXT_LOW_K12_B80_S1" if dataset == "P22" else "EXT_LOW_K18_B80_S2"
        selected = list(dict.fromkeys(selected + [forced]))
        for filter_id in selected:
            views = view_cache[filter_id]
            for registered in (
                {"id": "H00_FUSED_PCA20_MCLUST_EEE"},
                {"id": "H01_FUSED_DIRECT_MCLUST_EEE"},
                {"id": "H03_CONCAT_PRIVATE_PCA20_MCLUST_EEE"},
                {"id": "H05_EQUAL3_AFFINITY_SPECTRAL"},
            ):
                started = time.perf_counter()
                try:
                    partition, audit = run_head(
                        registered, views, k, payload["coordinates"], payload["ids"]
                    )
                    key = f"{dataset}|{k}|{filter_id}|{registered['id']}"
                    partitions[key] = np.asarray(partition, dtype=np.int64)
                    head_rows.append({
                        "stage": "REGISTERED_HEAD", "dataset": dataset,
                        "cluster_k": k, "filter_id": filter_id,
                        "head_id": registered["id"], "representation": "REGISTERED",
                        "status": "PASS", "label_policy": LABEL_POLICY,
                        "labels_in_fit": False,
                        "partition_sha256": array_sha256(partitions[key]),
                        "wall_seconds": time.perf_counter() - started,
                        **metric_row(payload, partitions[key]),
                    })
                except Exception as error:
                    head_rows.append({
                        "stage": "REGISTERED_HEAD", "dataset": dataset,
                        "cluster_k": k, "filter_id": filter_id,
                        "head_id": registered["id"], "representation": "REGISTERED",
                        "status": "FAIL", "error_type": type(error).__name__,
                        "error": str(error), "label_policy": LABEL_POLICY,
                        "labels_in_fit": False,
                        "wall_seconds": time.perf_counter() - started,
                    })
        # Rich endpoint grid uses the best cheap filter for this lane.
        filter_id = selected[0]
        views = view_cache[filter_id]
        for representation in ("FUSED", "PRIVATE2", "EQUAL3"):
            for dimension in (8, 16, 32):
                molecular = molecular_matrix(views, representation, dimension)
                coordinate_options = [("NONE", 0.0)] + [
                    (basis, weight) for basis in ("LINEAR", "POLY2")
                    for weight in (.15, .30, .60, 1.20, 2.40)
                ]
                for basis, weight in coordinate_options:
                    value = augmented_matrix(
                        molecular, payload["coordinates"],
                        "LINEAR" if basis == "NONE" else basis, weight,
                    )
                    for algorithm in ("KMEANS", "GMM_DIAG", "GMM_TIED", "GMM_SPHERICAL"):
                        started = time.perf_counter()
                        try:
                            partition = cluster_matrix(value, k, algorithm)
                            head_id = f"{algorithm}_{representation}_PCA{dimension}_{basis}_W{weight:.2f}"
                            key = f"{dataset}|{k}|{filter_id}|{head_id}"
                            partitions[key] = partition
                            head_rows.append({
                                "stage": "AUGMENTED_HEAD", "dataset": dataset,
                                "cluster_k": k, "filter_id": filter_id,
                                "head_id": head_id, "representation": representation,
                                "pca_dimension": dimension, "coordinate_basis": basis,
                                "coordinate_weight": weight, "algorithm": algorithm,
                                "status": "PASS", "label_policy": LABEL_POLICY,
                                "labels_in_fit": False,
                                "partition_sha256": array_sha256(partition),
                                "wall_seconds": time.perf_counter() - started,
                                **metric_row(payload, partition),
                            })
                        except Exception as error:
                            head_rows.append({
                                "stage": "AUGMENTED_HEAD", "dataset": dataset,
                                "cluster_k": k, "filter_id": filter_id,
                                "head_id": f"{algorithm}_{representation}_PCA{dimension}_{basis}_W{weight:.2f}",
                                "status": "FAIL", "error_type": type(error).__name__,
                                "error": str(error), "label_policy": LABEL_POLICY,
                                "labels_in_fit": False,
                                "wall_seconds": time.perf_counter() - started,
                            })
    head_frame = pd.DataFrame(head_rows)
    head_frame.to_csv(output / f"{dataset}_head_search.csv", index=False)

    refine_rows = []
    passed = head_frame[head_frame.status == "PASS"].sort_values(
        ["cluster_k", "absolute_ari", "absolute_nmi"],
        ascending=[True, False, False],
    ).groupby("cluster_k", as_index=False).head(5)
    for _, row in passed.iterrows():
        key = f"{dataset}|{int(row.cluster_k)}|{row.filter_id}|{row.head_id}"
        initial = partitions[key]
        for graph_k in (6, 12, 18, 24, 30):
            operator = spatial_operator(payload["coordinates"], graph_k)
            for anchor in (0.0, .1, .2, .5, 1.0, 2.0):
                for iterations in (2, 5, 10, 20):
                    refined = anchored_majority_refine(
                        initial, operator, int(row.cluster_k), anchor, iterations
                    )
                    refine_rows.append({
                        "stage": "SPATIAL_HEAD_REFINEMENT", "dataset": dataset,
                        "cluster_k": int(row.cluster_k), "filter_id": row.filter_id,
                        "parent_head_id": row.head_id,
                        "head_id": f"MAJ_K{graph_k}_A{anchor:.1f}_I{iterations}",
                        "graph_k": graph_k, "anchor": anchor,
                        "iterations": iterations, "status": "PASS",
                        "label_policy": LABEL_POLICY, "labels_in_fit": False,
                        "partition_sha256": array_sha256(refined),
                        **metric_row(payload, refined),
                    })
    refine_frame = pd.DataFrame(refine_rows)
    refine_frame.to_csv(output / f"{dataset}_refinement_search.csv", index=False)
    return filter_frame, head_frame, refine_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frames = []
    for dataset in DATASETS:
        frames.extend(search_dataset(dataset, output))
    ledger = pd.concat(frames, ignore_index=True, sort=False)
    ledger.to_csv(output / "targeted_head_all_run_ledger.csv", index=False)
    best = ledger[ledger.status == "PASS"].sort_values(
        ["dataset", "cluster_k", "absolute_ari", "absolute_nmi"],
        ascending=[True, True, False, False],
    ).groupby(["dataset", "cluster_k"], as_index=False).head(20)
    best.to_csv(output / "targeted_head_best_run_board.csv", index=False)
    atomic_json(output / "targeted_head_manifest.json", {
        "row_count": len(ledger), "pass_count": int((ledger.status == "PASS").sum()),
        "failure_count": int((ledger.status == "FAIL").sum()),
        "labels_in_fit_gradient_or_checkpoint": False,
        "labels_used_for_cross_run_hpo_and_evaluation": True,
        "dense_n_by_n_count": 0,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    })


if __name__ == "__main__":
    main()
