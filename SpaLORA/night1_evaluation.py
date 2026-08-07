"""Evaluation-only utilities for the preregistered Night 1 benchmark.

Ground-truth labels enter the workflow only through this module, after model
training and unsupervised clustering have completed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    davies_bouldin_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import kneighbors_graph


def _canonical_ids(ids: pd.Index, rule: str) -> pd.Index:
    values = ids.astype(str)
    if rule == "strip_s1_prefix":
        values = pd.Index([x[3:] if x.startswith("s1-") else x for x in values])
    elif rule != "identity":
        raise ValueError("Unknown ground-truth ID rule: %s" % rule)
    return pd.Index(values)


def load_evaluation_labels(dataset: str, cfg: dict, observation_ids: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
    """Return positions and labels; this function must be called after training."""
    if dataset == "placenta":
        source = ad.read_h5ad(cfg["rna"], backed="r")
        labels = source.obs[cfg["ground_truth_label_column"]].astype(str)
        label_map = pd.Series(labels.values, index=source.obs_names.astype(str))
        source.file.close()
        canonical = observation_ids.astype(str)
    else:
        table = pd.read_csv(cfg["ground_truth"])
        label_map = pd.Series(
            table[cfg["ground_truth_label_column"]].astype(str).values,
            index=table[cfg["ground_truth_id_column"]].astype(str).values,
        )
        if not label_map.index.is_unique:
            raise AssertionError("Ground-truth identifiers must be unique")
        canonical = _canonical_ids(observation_ids, cfg.get("ground_truth_id_rule", "identity"))

    valid = canonical.isin(label_map.index)
    positions = np.flatnonzero(valid)
    aligned = label_map.reindex(canonical[valid])
    if aligned.isna().any():
        raise AssertionError("Ground-truth alignment produced missing labels")
    if dataset in ("a1", "placenta") and positions.size != len(observation_ids):
        raise AssertionError("%s requires complete ground-truth alignment" % dataset)
    if dataset == "p22" and positions.size != len(label_map):
        raise AssertionError("P22 must evaluate the exact CSV intersection")
    return positions, aligned.to_numpy(dtype=str)


def hungarian_metrics(true_labels: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    true_levels, true_codes = np.unique(true_labels.astype(str), return_inverse=True)
    pred_levels, pred_codes = np.unique(predicted.astype(str), return_inverse=True)
    contingency = np.zeros((pred_levels.size, true_levels.size), dtype=np.int64)
    np.add.at(contingency, (pred_codes, true_codes), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {pred_levels[r]: true_levels[c] for r, c in zip(rows, cols)}
    fallback = "__unmapped__"
    mapped = np.asarray([mapping.get(str(x), fallback) for x in predicted], dtype=str)
    return {
        "hungarian_macro_f1": float(f1_score(true_labels, mapped, average="macro", zero_division=0)),
        "hungarian_weighted_f1": float(f1_score(true_labels, mapped, average="weighted", zero_division=0)),
        "hungarian_balanced_accuracy": float(balanced_accuracy_score(true_labels, mapped)),
        "hungarian_per_domain_f1": {
            str(label): float(score)
            for label, score in zip(
                true_levels,
                f1_score(true_labels, mapped, labels=true_levels, average=None, zero_division=0),
            )
        },
        "hungarian_mapping": {str(k): str(v) for k, v in mapping.items()},
    }


def _mean_cluster_moran(predicted: np.ndarray, adjacency: sparse.spmatrix) -> float:
    adjacency = adjacency.tocsr().astype(np.float64)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    s0 = float(adjacency.sum())
    if s0 == 0:
        return float("nan")
    values = []
    n = predicted.size
    for level in np.unique(predicted):
        z = (predicted == level).astype(np.float64)
        z -= z.mean()
        denom = float(np.dot(z, z))
        if denom > 0:
            values.append((n / s0) * float(z @ adjacency.dot(z)) / denom)
    return float(np.mean(values)) if values else float("nan")


def evaluate(
    true_labels: np.ndarray,
    predicted_labeled: np.ndarray,
    predicted_all: np.ndarray,
    embedding: np.ndarray,
    coordinates: np.ndarray,
    spatial_neighbors: int,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "ari": float(adjusted_rand_score(true_labels, predicted_labeled)),
        "nmi": float(normalized_mutual_info_score(true_labels, predicted_labeled)),
        "ami": float(adjusted_mutual_info_score(true_labels, predicted_labeled)),
        "homogeneity": float(homogeneity_score(true_labels, predicted_labeled)),
        "v_measure": float(v_measure_score(true_labels, predicted_labeled)),
        "fmi": float(fowlkes_mallows_score(true_labels, predicted_labeled)),
    }
    metrics.update(hungarian_metrics(true_labels, predicted_labeled))

    graph = kneighbors_graph(
        coordinates,
        n_neighbors=spatial_neighbors,
        mode="connectivity",
        include_self=False,
    ).tocsr()
    rows, cols = graph.nonzero()
    metrics["spatial_neighbor_agreement"] = float(np.mean(predicted_all[rows] == predicted_all[cols]))
    metrics["spatial_cluster_moran_mean"] = _mean_cluster_moran(predicted_all, graph.maximum(graph.T))

    unique = np.unique(predicted_all)
    if 1 < unique.size < predicted_all.size:
        sample_size = min(5000, predicted_all.size)
        metrics["embedding_silhouette"] = float(
            silhouette_score(
                embedding,
                predicted_all,
                sample_size=sample_size if sample_size < predicted_all.size else None,
                random_state=2020,
            )
        )
        metrics["embedding_davies_bouldin"] = float(davies_bouldin_score(embedding, predicted_all))
        metrics["silhouette_sample_size"] = int(sample_size)
    else:
        metrics["embedding_silhouette"] = float("nan")
        metrics["embedding_davies_bouldin"] = float("nan")
        metrics["silhouette_sample_size"] = 0
    return metrics
