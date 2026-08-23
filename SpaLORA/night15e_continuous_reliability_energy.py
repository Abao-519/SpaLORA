"""Continuous local-evidence reliability energy for Night-15E.

The core accepts molecular arrays, a sparse observation graph, an initial
partition, and numeric controls.  It never receives dataset identifiers or
public reference labels.  All modality, edge, feature, self-return, and trust
terms are present in one continuous computation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import expit, softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return StandardScaler(copy=True).fit_transform(value).astype(np.float32)


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


def _component_bank(value: np.ndarray, transition: sp.spmatrix, dim: int) -> np.ndarray:
    base = reduce_full(value, dim)
    low1 = standardize(np.asarray(transition @ base, dtype=np.float32))
    low2 = standardize(np.asarray(transition @ low1, dtype=np.float32))
    high = standardize(base - low1)
    return np.stack((base, low1, low2, high), axis=0).astype(np.float32)


def _edge_similarity(
    graph: sp.spmatrix,
    first: np.ndarray,
    second: np.ndarray,
    dim: int,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = sp.csr_matrix(graph, dtype=np.float32).tocoo(copy=True)
    base.setdiag(0)
    base.eliminate_zeros()
    first = reduce_full(first, min(int(dim), first.shape[1]))
    second = reduce_full(second, min(int(dim), second.shape[1]))
    d1 = np.mean((first[base.row] - first[base.col]) ** 2, axis=1)
    d2 = np.mean((second[base.row] - second[base.col]) ** 2, axis=1)
    positive1 = d1[d1 > 0]
    positive2 = d2[d2 > 0]
    scale1 = max(float(np.median(positive1)) if positive1.size else 1.0, 1e-6)
    scale2 = max(float(np.median(positive2)) if positive2.size else 1.0, 1e-6)
    similarity1 = np.exp(-d1 / scale1).astype(np.float32)
    similarity2 = np.exp(-d2 / scale2).astype(np.float32)
    binary = sp.csr_matrix(
        (np.ones_like(similarity1), (base.row, base.col)), shape=base.shape
    )
    return binary, base.row.astype(np.int32), base.col.astype(np.int32), similarity1, similarity2


@dataclass(frozen=True)
class PreparedContinuousEvidence:
    binary: sp.csr_matrix
    rows: np.ndarray
    cols: np.ndarray
    similarity1: np.ndarray
    similarity2: np.ndarray
    retained_components: np.ndarray
    view1_components: np.ndarray
    view2_components: np.ndarray


@dataclass(frozen=True)
class ContinuousEnergyConfig:
    beta: float
    edge_floor: float
    conflict_center: float
    conflict_temperature: float
    conflict_union_weight: float
    conflict_penalty: float
    mass_center: float
    mass_temperature: float
    neighbor_capacity: float
    low_weight: float
    twohop_weight: float
    high_weight: float
    unary_temperature: float
    retained_bias: float
    view_balance: float
    trust_scale: float
    trust_center: float
    trust_temperature: float
    move_threshold: float
    move_fraction: float
    sweeps: int


def prepare_continuous_evidence(
    graph: sp.spmatrix,
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    retained_dim: int = 32,
    view_dim: int = 24,
    edge_dim: int = 16,
) -> PreparedContinuousEvidence:
    graph = sp.csr_matrix(graph, dtype=np.float32)
    transition = row_stochastic(graph)
    binary, rows, cols, similarity1, similarity2 = _edge_similarity(
        graph, view1, view2, edge_dim
    )
    return PreparedContinuousEvidence(
        binary=binary,
        rows=rows,
        cols=cols,
        similarity1=similarity1,
        similarity2=similarity2,
        retained_components=_component_bank(retained, transition, retained_dim),
        view1_components=_component_bank(view1, transition, min(view_dim, view1.shape[1])),
        view2_components=_component_bank(view2, transition, min(view_dim, view2.shape[1])),
    )


def _weighted_components(components: np.ndarray, config: ContinuousEnergyConfig) -> np.ndarray:
    weights = np.asarray(
        [1.0, config.low_weight, config.twohop_weight, config.high_weight],
        dtype=np.float32,
    )
    weights = np.maximum(weights, 1e-4)
    weighted = components * np.sqrt(weights)[:, None, None]
    return np.transpose(weighted, (1, 0, 2)).reshape(components.shape[1], -1)


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
    spread = np.std(distance, axis=1, keepdims=True)
    return (
        (distance - distance.min(axis=1, keepdims=True))
        / np.maximum(spread, 1e-6)
    ).astype(np.float32)


def unary_margin(unary: np.ndarray) -> np.ndarray:
    smallest = np.partition(np.asarray(unary, dtype=np.float32), 1, axis=1)[:, :2]
    return np.maximum(0.0, smallest[:, 1] - smallest[:, 0])


def continuous_prototype_unary(
    partition: np.ndarray,
    k: int,
    evidence: PreparedContinuousEvidence,
    config: ContinuousEnergyConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    retained = centroid_unary(
        _weighted_components(evidence.retained_components, config), partition, k
    )
    first = centroid_unary(
        _weighted_components(evidence.view1_components, config), partition, k
    )
    second = centroid_unary(
        _weighted_components(evidence.view2_components, config), partition, k
    )
    margins = np.column_stack(
        (unary_margin(retained), unary_margin(first), unary_margin(second))
    )
    scaled = np.log(np.maximum(margins, 1e-6)) / max(
        float(config.unary_temperature), 1e-4
    )
    scaled[:, 0] += float(config.retained_bias)
    scaled[:, 1] += float(config.view_balance)
    scaled[:, 2] -= float(config.view_balance)
    weights = softmax(scaled, axis=1).astype(np.float32)
    unary = (
        weights[:, :1] * retained
        + weights[:, 1:2] * first
        + weights[:, 2:] * second
    ).astype(np.float32)
    return unary, {
        "mean_retained_weight": float(np.mean(weights[:, 0])),
        "mean_view1_weight": float(np.mean(weights[:, 1])),
        "mean_view2_weight": float(np.mean(weights[:, 2])),
        "mean_prototype_margin": float(np.mean(np.max(margins, axis=1))),
    }


def continuous_reliability_transition(
    evidence: PreparedContinuousEvidence,
    config: ContinuousEnergyConfig,
) -> Tuple[sp.csr_matrix, Dict[str, float]]:
    first = evidence.similarity1
    second = evidence.similarity2
    conflict = np.abs(first - second)
    agreement_gate = expit(
        (float(config.conflict_center) - conflict)
        / max(float(config.conflict_temperature), 1e-4)
    ).astype(np.float32)
    intersection = np.sqrt(np.maximum(first * second, 0.0))
    union = np.maximum(first, second)
    strict = np.minimum(first, second)
    conflict_value = (
        float(config.conflict_union_weight) * union
        + (1.0 - float(config.conflict_union_weight)) * strict
    )
    content = agreement_gate * intersection + (1.0 - agreement_gate) * conflict_value
    floor = float(np.clip(config.edge_floor, 1e-4, 0.95))
    weight = floor + (1.0 - floor) * np.clip(content, 0.0, 1.0)
    weighted = sp.csr_matrix(
        (weight.astype(np.float32), (evidence.rows, evidence.cols)),
        shape=evidence.binary.shape,
    )
    degree = np.asarray(evidence.binary.sum(axis=1)).reshape(-1)
    mean_weight = np.asarray(weighted.sum(axis=1)).reshape(-1) / np.maximum(degree, 1.0)
    conflict_matrix = sp.csr_matrix(
        (conflict.astype(np.float32), (evidence.rows, evidence.cols)),
        shape=evidence.binary.shape,
    )
    mean_conflict = np.asarray(conflict_matrix.sum(axis=1)).reshape(-1) / np.maximum(
        degree, 1.0
    )
    reliability = mean_weight - float(config.conflict_penalty) * mean_conflict
    neighbor_mass = float(np.clip(config.neighbor_capacity, 1e-4, 1.0)) * expit(
        (reliability - float(config.mass_center))
        / max(float(config.mass_temperature), 1e-4)
    )
    neighbor_mass[degree <= 0] = 0.0
    normalized = row_stochastic(weighted)
    transition = (
        sp.diags(neighbor_mass.astype(np.float32)) @ normalized
        + sp.diags((1.0 - neighbor_mass).astype(np.float32))
    ).tocsr()
    row_mass = np.asarray(transition.sum(axis=1)).reshape(-1)
    return transition, {
        "mean_edge_weight": float(np.mean(weight)),
        "mean_edge_conflict": float(np.mean(conflict)),
        "mean_agreement_gate": float(np.mean(agreement_gate)),
        "mean_neighbor_mass": float(np.mean(neighbor_mass)),
        "min_neighbor_mass": float(np.min(neighbor_mass)),
        "max_neighbor_mass": float(np.max(neighbor_mass)),
        "mean_self_return": float(np.mean(1.0 - neighbor_mass)),
        "min_row_mass": float(np.min(row_mass)),
        "max_row_mass": float(np.max(row_mass)),
    }


def continuous_reliability_energy(
    initial: np.ndarray,
    k: int,
    evidence: PreparedContinuousEvidence,
    config: ContinuousEnergyConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run deterministic trust-aware sparse local conditional moves.

    Each accepted move decreases the current node's frozen-sweep local energy.
    Dynamic prototypes are recomputed between sweeps, so this is not a claim of
    globally optimal MAP inference.
    """

    partition = np.asarray(initial, dtype=np.int32).copy()
    if len(np.unique(partition)) != int(k):
        raise ValueError("initial partition cardinality mismatch")
    transition, diagnostics = continuous_reliability_transition(evidence, config)
    counts = np.bincount(partition, minlength=int(k)).astype(np.int64)
    total_moves = 0
    completed = 0
    unary_diagnostics: Dict[str, float] = {}
    max_moves = max(1, int(np.ceil(float(config.move_fraction) * len(partition))))
    for sweep in range(int(config.sweeps)):
        unary, unary_diagnostics = continuous_prototype_unary(
            partition, k, evidence, config
        )
        support = np.asarray(
            transition @ np.eye(int(k), dtype=np.float32)[partition], dtype=np.float32
        )
        current = unary[np.arange(len(partition)), partition]
        other = unary.copy()
        other[np.arange(len(partition)), partition] = np.inf
        current_advantage = np.min(other, axis=1) - current
        local_trust = expit(
            (current_advantage - float(config.trust_center))
            / max(float(config.trust_temperature), 1e-4)
        ).astype(np.float32)
        energy = unary - float(config.beta) * support
        energy += float(config.trust_scale) * local_trust[:, None]
        energy[np.arange(len(partition)), partition] -= (
            float(config.trust_scale) * local_trust
        )
        proposed = np.argmin(energy, axis=1).astype(np.int32)
        gain = energy[np.arange(len(partition)), partition] - energy[
            np.arange(len(partition)), proposed
        ]
        order = np.argsort(-gain, kind="mergesort")
        moved = 0
        for index in order:
            if moved >= max_moves or gain[index] <= float(config.move_threshold):
                break
            old = int(partition[index])
            if counts[old] <= 1:
                continue
            start, end = transition.indptr[index], transition.indptr[index + 1]
            neighbor_labels = partition[transition.indices[start:end]]
            local_support = np.bincount(
                neighbor_labels,
                weights=transition.data[start:end],
                minlength=int(k),
            ).astype(np.float32)
            local_energy = unary[index] - float(config.beta) * local_support
            local_energy += float(config.trust_scale) * local_trust[index]
            local_energy[old] -= float(config.trust_scale) * local_trust[index]
            target = int(np.argmin(local_energy))
            local_gain = float(local_energy[old] - local_energy[target])
            if target == old or local_gain <= float(config.move_threshold):
                continue
            partition[index] = target
            counts[old] -= 1
            counts[target] += 1
            moved += 1
        completed = sweep + 1
        total_moves += moved
        if moved == 0:
            break
    diagnostics = {
        **diagnostics,
        **unary_diagnostics,
        "sweeps_completed": float(completed),
        "total_moves": float(total_moves),
        "changed_observations": float(np.sum(partition != np.asarray(initial))),
        "observed_cardinality": float(len(np.unique(partition))),
        "min_cluster_size": float(np.min(np.bincount(partition, minlength=int(k)))),
    }
    return partition, diagnostics

