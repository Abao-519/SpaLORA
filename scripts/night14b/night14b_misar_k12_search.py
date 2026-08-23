#!/usr/bin/env python3
"""Focused MISAR K=12 spatial-head search after the broad Night-14B screen."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))
sys.path.insert(0, str(REPO / "scripts/night14b"))

import night13b_run as n13b  # noqa: E402
import night14b_head_search as heads  # noqa: E402
from SpaLORA.night14b_atac import array_sha256, spatial_operator  # noqa: E402
from night14b_run import LABEL_POLICY, atomic_json, metric_row, views_from_archive  # noqa: E402


def icm(posterior, initial, operator, ids, beta, iterations=20):
    posterior = np.asarray(posterior, dtype=np.float64)
    current = np.asarray(initial, dtype=np.int64).copy()
    k = posterior.shape[1]
    unary = -np.log(np.maximum(posterior, 1e-300))
    order = np.argsort(np.asarray(ids, dtype=str), kind="mergesort")
    graph = operator.tocsr()
    for _ in range(int(iterations)):
        changed = False
        for i in order:
            neighbors = graph.indices[graph.indptr[i]:graph.indptr[i + 1]]
            counts = np.bincount(current[neighbors], minlength=k)
            energy = unary[i] + float(beta) * (len(neighbors) - counts)
            choices = np.flatnonzero(np.isclose(energy, energy.min(), atol=1e-12, rtol=0))
            new = int(current[i]) if int(current[i]) in choices else int(choices.min())
            changed |= new != int(current[i])
            current[i] = new
        if not changed:
            break
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    payload = n13b.base_payload("MISAR_E15_5_S1")
    original = views_from_archive("MISAR_E15_5_S1", "N14A_C15")
    operator = spatial_operator(payload["coordinates"], 18)
    views = heads.filtered_views(original, operator, .95, 5)
    matrices = {}
    coordinate_options = [("NONE", 0.0)] + [
        (basis, weight) for basis in ("LINEAR", "POLY2")
        for weight in (.30, .60, 1.20, 2.40, 4.80)
    ]
    for representation in ("FUSED", "PRIVATE2", "EQUAL3"):
        for dimension in (8, 16, 32):
            molecular = heads.molecular_matrix(views, representation, dimension)
            for basis, weight in coordinate_options:
                key = f"{representation}_PCA{dimension}_{basis}_W{weight:.2f}"
                matrices[key] = heads.augmented_matrix(
                    molecular, payload["coordinates"],
                    "LINEAR" if basis == "NONE" else basis, weight,
                )

    rows = []
    partition_cache = {}
    for key, value in matrices.items():
        for graph_k in (6, 12, 18, 24, 30):
            connectivity = kneighbors_graph(
                payload["coordinates"], graph_k, mode="connectivity",
                include_self=False,
            )
            connectivity = connectivity.maximum(connectivity.T)
            candidate = f"WARD_{key}_GK{graph_k}"
            begin = time.perf_counter()
            try:
                partition = AgglomerativeClustering(
                    n_clusters=12, linkage="ward", connectivity=connectivity,
                ).fit_predict(value).astype(np.int64)
                partition_cache[candidate] = partition
                rows.append({
                    "stage": "SPATIALLY_CONSTRAINED_WARD", "candidate_id": candidate,
                    "cluster_k": 12, "status": "PASS", "labels_in_fit": False,
                    "label_policy": LABEL_POLICY,
                    "partition_sha256": array_sha256(partition),
                    "wall_seconds": time.perf_counter() - begin,
                    **metric_row(payload, partition),
                })
            except Exception as error:
                rows.append({
                    "stage": "SPATIALLY_CONSTRAINED_WARD", "candidate_id": candidate,
                    "cluster_k": 12, "status": "FAIL", "labels_in_fit": False,
                    "label_policy": LABEL_POLICY, "error_type": type(error).__name__,
                    "error": str(error), "wall_seconds": time.perf_counter() - begin,
                })

    mixture_cache = {}
    mixture_rows = []
    for key, value in matrices.items():
        for covariance in ("diag", "spherical"):
            candidate = f"GMM_{covariance.upper()}_{key}"
            begin = time.perf_counter()
            try:
                model = GaussianMixture(
                    n_components=12, covariance_type=covariance, random_state=0,
                    n_init=5, max_iter=400, reg_covar=1e-5,
                ).fit(value)
                partition = model.predict(value).astype(np.int64)
                posterior = model.predict_proba(value)
                mixture_cache[candidate] = (partition, posterior)
                mixture_rows.append({
                    "stage": "GMM_ICM_PARENT", "candidate_id": candidate,
                    "cluster_k": 12, "status": "PASS", "labels_in_fit": False,
                    "label_policy": LABEL_POLICY,
                    "partition_sha256": array_sha256(partition),
                    "wall_seconds": time.perf_counter() - begin,
                    **metric_row(payload, partition),
                })
            except Exception as error:
                mixture_rows.append({
                    "stage": "GMM_ICM_PARENT", "candidate_id": candidate,
                    "cluster_k": 12, "status": "FAIL", "labels_in_fit": False,
                    "label_policy": LABEL_POLICY, "error_type": type(error).__name__,
                    "error": str(error), "wall_seconds": time.perf_counter() - begin,
                })
    rows.extend(mixture_rows)
    mixture_frame = pd.DataFrame(mixture_rows)
    selected = list(mixture_frame[mixture_frame.status == "PASS"].sort_values(
        ["absolute_ari", "absolute_nmi"], ascending=False
    ).candidate_id.head(12))
    for parent in selected:
        initial, posterior = mixture_cache[parent]
        for graph_k in (6, 12, 18, 24, 30):
            graph = spatial_operator(payload["coordinates"], graph_k)
            for beta in (.025, .05, .10, .20, .40, .80, 1.20, 2.00):
                candidate = f"ICM_{parent}_GK{graph_k}_B{beta:.3f}"
                begin = time.perf_counter()
                partition = icm(posterior, initial, graph, payload["ids"], beta)
                rows.append({
                    "stage": "POSTERIOR_POTTS_ICM", "candidate_id": candidate,
                    "parent_candidate_id": parent, "cluster_k": 12,
                    "graph_k": graph_k, "potts_beta": beta, "status": "PASS",
                    "labels_in_fit": False, "label_policy": LABEL_POLICY,
                    "partition_sha256": array_sha256(partition),
                    "wall_seconds": time.perf_counter() - begin,
                    **metric_row(payload, partition),
                })
    ledger = pd.DataFrame(rows)
    ledger.to_csv(output / "misar_k12_targeted_all_run_ledger.csv", index=False)
    ledger[ledger.status == "PASS"].sort_values(
        ["absolute_ari", "absolute_nmi"], ascending=False
    ).head(50).to_csv(output / "misar_k12_targeted_best_board.csv", index=False)
    atomic_json(output / "misar_k12_targeted_manifest.json", {
        "row_count": len(ledger), "pass_count": int((ledger.status == "PASS").sum()),
        "failure_count": int((ledger.status == "FAIL").sum()),
        "labels_in_fit_gradient_or_checkpoint": False,
        "labels_used_for_cross_run_hpo_and_evaluation": True,
        "wall_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
