"""Reliability-mass preserving sparse clustering energy for Night-15D.

The functions in this module do not accept dataset identifiers or public
labels.  Labels remain confined to the development evaluator in the runner.
All observation graph operations stay sparse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler(copy=True).fit_transform(
        np.asarray(value, dtype=np.float32)
    ).astype(np.float32)


def reduce_full(value: np.ndarray, dim: int) -> np.ndarray:
    value = standardize(value)
    dim = min(int(dim), value.shape[1], value.shape[0] - 1)
    if dim < value.shape[1]:
        value = PCA(n_components=dim, svd_solver="full").fit_transform(value)
    return standardize(value)


def csr_from_archive(archive: Mapping[str, np.ndarray], prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            archive[f"{prefix}__data"],
            archive[f"{prefix}__indices"],
            archive[f"{prefix}__indptr"],
        ),
        shape=tuple(map(int, archive[f"{prefix}__shape"])),
        dtype=np.float32,
    )


def row_stochastic(graph: sp.spmatrix, floor: float = 1e-8) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float32)
    total = np.asarray(graph.sum(axis=1)).reshape(-1)
    return (sp.diags(1.0 / np.maximum(total, float(floor))) @ graph).tocsr()


def edge_similarity(
    graph: sp.spmatrix,
    view1: np.ndarray,
    view2: np.ndarray,
    mode: str,
    tau: float = 1.0,
    dim: int = 16,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return the registered binary graph and two-modality weighted graph.

    Similarities are computed only on registered sparse edges.  ``either`` is
    the maximum modality similarity; ``both`` is the minimum; ``geomean`` is
    their geometric mean; and ``agreement`` also discounts modality conflict.
    """

    base = sp.csr_matrix(graph, dtype=np.float32).tocoo(copy=True)
    base.setdiag(0)
    base.eliminate_zeros()
    first = reduce_full(view1, min(int(dim), view1.shape[1]))
    second = reduce_full(view2, min(int(dim), view2.shape[1]))
    distance1 = np.mean((first[base.row] - first[base.col]) ** 2, axis=1)
    distance2 = np.mean((second[base.row] - second[base.col]) ** 2, axis=1)
    positive1 = distance1[distance1 > 0]
    positive2 = distance2[distance2 > 0]
    scale1 = max(float(np.median(positive1)) if positive1.size else 1.0, 1e-6)
    scale2 = max(float(np.median(positive2)) if positive2.size else 1.0, 1e-6)
    similarity1 = np.exp(-distance1 / (scale1 * max(float(tau), 1e-6)))
    similarity2 = np.exp(-distance2 / (scale2 * max(float(tau), 1e-6)))
    if mode == "spatial":
        weight = np.ones_like(similarity1)
    elif mode == "either":
        weight = np.maximum(similarity1, similarity2)
    elif mode == "both":
        weight = np.minimum(similarity1, similarity2)
    elif mode == "geomean":
        weight = np.sqrt(similarity1 * similarity2)
    elif mode == "agreement":
        weight = np.sqrt(similarity1 * similarity2) * np.maximum(
            0.0, 1.0 - np.abs(similarity1 - similarity2)
        )
    else:
        raise ValueError(mode)
    binary = sp.csr_matrix(
        (np.ones_like(weight, dtype=np.float32), (base.row, base.col)),
        shape=base.shape,
    )
    weighted = sp.csr_matrix(
        (weight.astype(np.float32), (base.row, base.col)), shape=base.shape
    )
    return binary, weighted


def reliability_transition(
    binary: sp.spmatrix,
    weighted: sp.spmatrix,
    normalization: str,
) -> Tuple[sp.csr_matrix, Dict[str, float]]:
    """Create row, sub-stochastic, or self-returning sparse support.

    Night-15C row normalization forced every row to unit neighbor mass even
    when all incident conductances were weak.  ``mass`` divides by the original
    spatial degree and therefore retains absolute accepted mass.  ``self``
    returns rejected mass to the current spot through a diagonal self-loop.
    """

    binary = sp.csr_matrix(binary, dtype=np.float32)
    weighted = sp.csr_matrix(weighted, dtype=np.float32)
    original_degree = np.asarray(binary.sum(axis=1)).reshape(-1)
    if normalization == "row":
        transition = row_stochastic(weighted)
        rejected = np.zeros(weighted.shape[0], dtype=np.float32)
    else:
        transition = (
            sp.diags(1.0 / np.maximum(original_degree, 1.0)) @ weighted
        ).tocsr()
        accepted = np.asarray(transition.sum(axis=1)).reshape(-1)
        rejected = np.maximum(0.0, 1.0 - accepted).astype(np.float32)
        if normalization == "self":
            transition = (transition + sp.diags(rejected)).tocsr()
        elif normalization != "mass":
            raise ValueError(normalization)
    row_mass = np.asarray(transition.sum(axis=1)).reshape(-1)
    diagnostics = {
        "mean_row_mass": float(np.mean(row_mass)),
        "min_row_mass": float(np.min(row_mass)),
        "max_row_mass": float(np.max(row_mass)),
        "mean_rejected_mass": float(np.mean(rejected)),
        "max_rejected_mass": float(np.max(rejected)),
    }
    return transition, diagnostics


def centroid_unary(value: np.ndarray, partition: np.ndarray, k: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    partition = np.asarray(partition, dtype=np.int32)
    centers = []
    for cluster in range(int(k)):
        members = value[partition == cluster]
        if not len(members):
            raise ValueError("empty cluster")
        centers.append(members.mean(axis=0))
    centers = np.stack(centers)
    distance = np.mean((value[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    scale = np.std(distance, axis=1, keepdims=True)
    return (
        (distance - distance.min(axis=1, keepdims=True))
        / np.maximum(scale, 1e-6)
    ).astype(np.float32)


def unary_margin(unary: np.ndarray) -> np.ndarray:
    smallest = np.partition(np.asarray(unary, dtype=np.float32), 1, axis=1)[:, :2]
    return np.maximum(0.0, smallest[:, 1] - smallest[:, 0])


def fused_dynamic_unary(
    partition: np.ndarray,
    k: int,
    mode: str,
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    margin_temperature: float = 0.5,
    retained_weight: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fuse modality-specific prototype unaries by local label-free margins."""

    retained_unary = centroid_unary(retained, partition, k)
    if mode == "retained":
        return retained_unary, {
            "mean_view1_weight": 0.0,
            "mean_view2_weight": 0.0,
            "mean_retained_weight": 1.0,
        }
    first = centroid_unary(view1, partition, k)
    second = centroid_unary(view2, partition, k)
    if mode == "dual_equal":
        weights = np.full((len(partition), 2), 0.5, dtype=np.float32)
    elif mode in ("dual_margin", "retained_dual_margin"):
        margins = np.column_stack((unary_margin(first), unary_margin(second)))
        weights = softmax(
            margins / max(float(margin_temperature), 1e-4), axis=1
        ).astype(np.float32)
    else:
        raise ValueError(mode)
    dual = weights[:, :1] * first + weights[:, 1:] * second
    if mode == "retained_dual_margin":
        retained_weight = float(np.clip(retained_weight, 0.0, 1.0))
        result = retained_weight * retained_unary + (1.0 - retained_weight) * dual
        reported_retained = retained_weight
    else:
        result = dual
        reported_retained = 0.0
    return result.astype(np.float32), {
        "mean_view1_weight": float(np.mean(weights[:, 0]) * (1.0 - reported_retained)),
        "mean_view2_weight": float(np.mean(weights[:, 1]) * (1.0 - reported_retained)),
        "mean_retained_weight": float(reported_retained),
    }


def multiscale_feature_bank(
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.spmatrix,
) -> Dict[str, np.ndarray]:
    transition = row_stochastic(graph)
    retained = reduce_full(retained, 32)
    first = reduce_full(view1, min(24, view1.shape[1]))
    second = reduce_full(view2, min(24, view2.shape[1]))
    retained_smooth = np.asarray(transition @ retained, dtype=np.float32)
    first_smooth = np.asarray(transition @ first, dtype=np.float32)
    second_smooth = np.asarray(transition @ second, dtype=np.float32)
    first_smooth2 = np.asarray(transition @ first_smooth, dtype=np.float32)
    second_smooth2 = np.asarray(transition @ second_smooth, dtype=np.float32)
    return {
        "retained": retained,
        "retained_low_high": reduce_full(
            np.column_stack(
                (retained, retained_smooth, retained - retained_smooth)
            ),
            48,
        ),
        "views_low_high": reduce_full(
            np.column_stack(
                (
                    first,
                    first_smooth,
                    first - first_smooth,
                    second,
                    second_smooth,
                    second - second_smooth,
                )
            ),
            48,
        ),
        "views_multiscale": reduce_full(
            np.column_stack(
                (
                    first,
                    first_smooth,
                    first_smooth2,
                    second,
                    second_smooth,
                    second_smooth2,
                )
            ),
            48,
        ),
        "retained_plus_views": reduce_full(
            np.column_stack(
                (
                    retained,
                    retained_smooth,
                    first,
                    first_smooth,
                    second,
                    second_smooth,
                )
            ),
            48,
        ),
    }


def reliability_energy_icm(
    initial: np.ndarray,
    transition: sp.spmatrix,
    k: int,
    beta: float,
    iterations: int,
    unary_mode: str,
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    margin_temperature: float = 0.5,
    retained_weight: float = 0.35,
    switch_penalty: float = 0.0,
) -> Tuple[np.ndarray, int, bool, Dict[str, float]]:
    """Alternating prototype energy with reliability-mass preserving support."""

    partition = np.asarray(initial, dtype=np.int32).copy()
    eye = np.eye(int(k), dtype=np.float32)
    collapse_guard = False
    completed = 0
    diagnostics: Dict[str, float] = {}
    for step in range(int(iterations)):
        unary, diagnostics = fused_dynamic_unary(
            partition,
            k,
            unary_mode,
            retained,
            view1,
            view2,
            margin_temperature=margin_temperature,
            retained_weight=retained_weight,
        )
        support = np.asarray(transition @ eye[partition], dtype=np.float32)
        energy = unary - float(beta) * support
        if float(switch_penalty) > 0:
            energy += float(switch_penalty)
            energy[np.arange(len(partition)), partition] -= float(switch_penalty)
        proposed = np.argmin(energy, axis=1).astype(np.int32)
        if len(np.unique(proposed)) != int(k):
            collapse_guard = True
            break
        completed = step + 1
        if np.array_equal(proposed, partition):
            break
        partition = proposed
    return partition, completed, collapse_guard, diagnostics


@dataclass(frozen=True)
class ReliabilityEnergyConfig:
    feature: str
    unary_mode: str
    edge_mode: str
    normalization: str
    beta: float
    steps: int
    tau: float = 1.0
    margin_temperature: float = 0.5
    retained_weight: float = 0.35
    switch_penalty: float = 0.0
