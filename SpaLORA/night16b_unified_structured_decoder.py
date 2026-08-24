"""Unified reliability-structured decoder used by Night-16B.

The module accepts numeric views, sparse graphs, an initial partition and K.
It deliberately has no dataset/study argument and no reference-label argument.
Public annotations are consumed only by the separate benchmark evaluator.

The decoder combines the already validated Night-15F multiscale energy with a
generic non-degeneracy repair.  The repair is label-free: tiny clusters are
merged using prototype distance plus sparse boundary support, then the most
dispersed surviving cluster is split with a deterministic proposal.  All
views, including morphology, enter through a generic weighted block list and
an explicit presence mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    PreparedExpansionEvidence,
    continuous_multiscale_expansion,
)
from SpaLORA.night15g_optional_morphology_energy import (
    OptionalMorphologyConfig,
    PreparedOptionalMorphologyEvidence,
    optional_morphology_expansion,
)


@dataclass(frozen=True)
class WeightedView:
    name: str
    value: np.ndarray
    weight: float = 1.0
    presence: np.ndarray | None = None
    reduce_dim: int | None = None


@dataclass(frozen=True)
class RepairConfig:
    enabled: bool = True
    min_cluster_fraction_of_equal: float = 0.0
    feature_mode: str = "retained"
    merge_mode: str = "pointwise"
    merge_boundary_weight: float = 0.0
    split_mode: str = "pca_quantile"
    split_cluster_score: str = "mean"
    split_quantile: float = 0.5
    boundary_refine_beta: float = 0.0
    boundary_refine_sweeps: int = 0
    seed: int = 0


@dataclass(frozen=True)
class DecoderConfig:
    use_energy: bool
    use_optional_energy: bool
    repair: RepairConfig


@dataclass(frozen=True)
class PreparedDecoderInput:
    retained: np.ndarray
    view1: np.ndarray
    view2: np.ndarray
    coordinates: np.ndarray
    graph: sp.csr_matrix
    optional_views: tuple[WeightedView, ...] = ()
    energy_evidence: PreparedExpansionEvidence | None = None
    energy_config: ExpansionEnergyConfig | None = None
    optional_evidence: PreparedOptionalMorphologyEvidence | None = None
    optional_config: OptionalMorphologyConfig | None = None


def partition_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def encode_partition(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value), return_inverse=True)[1].astype(np.int32)


def _standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("view must be a finite two-dimensional matrix")
    return StandardScaler().fit_transform(value).astype(np.float32)


def reduce_block(value: np.ndarray, dim: int | None) -> np.ndarray:
    value = _standardize(value)
    if dim is None:
        return value
    target = min(int(dim), value.shape[1], value.shape[0] - 1)
    if target <= 0:
        raise ValueError("invalid reduced dimension")
    if target < value.shape[1]:
        value = PCA(n_components=target, svd_solver="full").fit_transform(value)
    return _standardize(value)


def combine_weighted_views(
    views: Sequence[WeightedView],
    final_dim: int | None = None,
) -> np.ndarray:
    """Standardize each block, then apply its weight without cancelling it.

    Earlier Night-16A repair code multiplied a coordinate block before a
    second column-wise StandardScaler, which erased that weight.  Here each
    block is normalized first, weighted second, and the concatenation is only
    centered/PCA-projected.  No final per-column rescaling is applied.
    """

    if not views:
        raise ValueError("at least one view is required")
    n = len(np.asarray(views[0].value))
    blocks = []
    for view in views:
        value = reduce_block(view.value, view.reduce_dim)
        if len(value) != n:
            raise ValueError("view observation mismatch")
        presence = (
            np.ones(n, dtype=np.float32)
            if view.presence is None
            else np.asarray(view.presence, dtype=np.float32)
        )
        if presence.shape != (n,) or np.any((presence < 0) | (presence > 1)):
            raise ValueError("presence mask must be an n-vector in [0,1]")
        weight = float(view.weight)
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("view weight must be finite and nonnegative")
        blocks.append(value * np.sqrt(weight) * presence[:, None])
    combined = np.concatenate(blocks, axis=1).astype(np.float32)
    combined -= combined.mean(axis=0, keepdims=True)
    if final_dim is not None:
        target = min(int(final_dim), combined.shape[1], combined.shape[0] - 1)
        if target < combined.shape[1]:
            combined = PCA(n_components=target, svd_solver="full").fit_transform(combined)
    scale = np.sqrt(np.maximum(np.sum(combined * combined, axis=1, keepdims=True), 1e-8))
    return (combined / scale).astype(np.float32)


def build_feature_bank(data: PreparedDecoderInput) -> Mapping[str, np.ndarray]:
    retained = reduce_block(data.retained, min(48, data.retained.shape[1]))
    view1 = reduce_block(data.view1, min(32, data.view1.shape[1]))
    view2 = reduce_block(data.view2, min(32, data.view2.shape[1]))
    coordinates = reduce_block(data.coordinates, min(2, data.coordinates.shape[1]))
    bank: dict[str, np.ndarray] = {
        "retained": retained,
        "molecular": combine_weighted_views(
            (
                WeightedView("retained", retained, 1.0),
                WeightedView("view1", view1, 1.0),
                WeightedView("view2", view2, 1.0),
            ),
            final_dim=64,
        ),
        "molecular_coord": combine_weighted_views(
            (
                WeightedView("retained", retained, 1.0),
                WeightedView("view1", view1, 1.0),
                WeightedView("view2", view2, 1.0),
                WeightedView("coordinates", coordinates, 0.10),
            ),
            final_dim=64,
        ),
    }
    present_optional = [
        view
        for view in data.optional_views
        if float(view.weight) > 0
        and (view.presence is None or bool(np.any(np.asarray(view.presence) > 0)))
    ]
    if present_optional:
        base_views = [
            WeightedView("retained", retained, 1.0),
            WeightedView("view1", view1, 1.0),
            WeightedView("view2", view2, 1.0),
        ]
        bank["molecular_optional"] = combine_weighted_views(
            tuple(base_views + list(present_optional)), final_dim=64
        )
        bank["molecular_optional_coord"] = combine_weighted_views(
            tuple(
                base_views
                + list(present_optional)
                + [WeightedView("coordinates", coordinates, 0.10)]
            ),
            final_dim=64,
        )
    else:
        # The same optional-view configuration is valid on a lane without the
        # view.  Aliasing the already computed molecular arrays makes the
        # presence-mask fallback byte exact instead of relying on a second PCA
        # over appended all-zero columns.
        bank["molecular_optional"] = bank["molecular"]
        bank["molecular_optional_coord"] = bank["molecular_coord"]
    return bank


def _prototype(feature: np.ndarray, partition: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.stack([feature[partition == group].mean(axis=0) for group in groups])


def _neighbour_fraction(
    graph: sp.csr_matrix,
    partition: np.ndarray,
    node: int,
    groups: np.ndarray,
) -> np.ndarray:
    begin, end = graph.indptr[node : node + 2]
    neighbours = graph.indices[begin:end]
    if len(neighbours) == 0:
        return np.zeros(len(groups), dtype=np.float32)
    labels = partition[neighbours]
    return np.asarray([np.mean(labels == group) for group in groups], dtype=np.float32)


def _merge_tiny(
    partition: np.ndarray,
    feature: np.ndarray,
    graph: sp.csr_matrix,
    threshold: int,
    mode: str,
    boundary_weight: float,
) -> tuple[np.ndarray, int]:
    partition = encode_partition(partition)
    removed = 0
    while True:
        sizes = np.bincount(partition)
        tiny = np.flatnonzero(sizes < threshold)
        if not len(tiny):
            break
        surviving = np.flatnonzero(sizes >= threshold)
        if len(surviving) < 2:
            raise RuntimeError("fewer than two clusters survive the minimum-size rule")
        centers = _prototype(feature, partition, surviving)
        for group in tiny:
            indices = np.flatnonzero(partition == group)
            if mode == "clusterwise":
                center = feature[indices].mean(axis=0)
                distance = np.mean((centers - center) ** 2, axis=1)
                support = np.zeros(len(surviving), dtype=np.float64)
                for node in indices:
                    support += _neighbour_fraction(graph, partition, int(node), surviving)
                target = int(surviving[np.argmin(distance - boundary_weight * support / max(len(indices), 1))])
                partition[indices] = target
            elif mode == "pointwise":
                distance = np.mean((feature[indices, None, :] - centers[None, :, :]) ** 2, axis=2)
                for row_index, node in enumerate(indices):
                    support = _neighbour_fraction(graph, partition, int(node), surviving)
                    target_index = int(np.argmin(distance[row_index] - boundary_weight * support))
                    partition[node] = int(surviving[target_index])
            else:
                raise ValueError(f"unknown merge mode: {mode}")
            removed += 1
        partition = encode_partition(partition)
    return partition, removed


def _split_assignment(
    feature: np.ndarray,
    indices: np.ndarray,
    threshold: int,
    mode: str,
    quantile: float,
    seed: int,
) -> np.ndarray:
    centered = feature[indices] - feature[indices].mean(axis=0, keepdims=True)
    if mode == "kmeans2":
        labels = KMeans(n_clusters=2, n_init=10, random_state=int(seed)).fit_predict(centered)
        counts = np.bincount(labels, minlength=2)
        if int(counts.min()) >= threshold:
            return labels.astype(np.int32)
    if mode not in {"kmeans2", "pca_quantile"}:
        raise ValueError(f"unknown split mode: {mode}")
    direction = PCA(n_components=1, svd_solver="full").fit_transform(centered).reshape(-1)
    order = np.argsort(direction, kind="mergesort")
    cut = int(np.clip(round(float(quantile) * len(indices)), threshold, len(indices) - threshold))
    labels = np.ones(len(indices), dtype=np.int32)
    labels[order[:cut]] = 0
    return labels


def _split_to_k(
    partition: np.ndarray,
    feature: np.ndarray,
    graph: sp.csr_matrix,
    k: int,
    threshold: int,
    config: RepairConfig,
) -> tuple[np.ndarray, int]:
    partition = encode_partition(partition)
    added = 0
    while len(np.unique(partition)) < int(k):
        choices = []
        for group in np.unique(partition):
            indices = np.flatnonzero(partition == group)
            if len(indices) < 2 * threshold:
                continue
            centered = feature[indices] - feature[indices].mean(axis=0, keepdims=True)
            dispersion = float(np.sum(centered * centered))
            if config.split_cluster_score == "mean":
                dispersion /= max(len(indices), 1)
            elif config.split_cluster_score == "boundary":
                induced = graph[indices][:, indices]
                internal = float(induced.nnz) / max(len(indices), 1)
                dispersion *= 1.0 + 1.0 / max(internal, 1e-3)
            elif config.split_cluster_score != "sse":
                raise ValueError(f"unknown split score: {config.split_cluster_score}")
            choices.append((dispersion, len(indices), -int(group), indices))
        if not choices:
            raise RuntimeError("no cluster can be split while preserving the minimum size")
        indices = max(choices)[-1]
        assignment = _split_assignment(
            feature,
            indices,
            threshold,
            config.split_mode,
            config.split_quantile,
            config.seed + added,
        )
        partition[indices[assignment == 1]] = int(partition.max()) + 1
        partition = encode_partition(partition)
        added += 1
    return partition, added


def _boundary_refine(
    partition: np.ndarray,
    feature: np.ndarray,
    graph: sp.csr_matrix,
    k: int,
    threshold: int,
    beta: float,
    sweeps: int,
) -> tuple[np.ndarray, int]:
    partition = encode_partition(partition)
    moved = 0
    for _ in range(int(sweeps)):
        centers = _prototype(feature, partition, np.arange(k, dtype=np.int32))
        counts = np.bincount(partition, minlength=k).astype(np.int64)
        sweep_moves = 0
        for node in range(len(partition)):
            begin, end = graph.indptr[node : node + 2]
            neighbours = graph.indices[begin:end]
            if len(neighbours) == 0:
                continue
            candidate_groups = np.unique(np.concatenate(([partition[node]], partition[neighbours])))
            unary = np.mean((centers[candidate_groups] - feature[node]) ** 2, axis=1)
            pairwise = np.asarray(
                [np.mean(partition[neighbours] != group) for group in candidate_groups],
                dtype=np.float64,
            )
            costs = unary + float(beta) * pairwise
            old = int(partition[node])
            new = int(candidate_groups[int(np.argmin(costs))])
            if new != old and counts[old] > threshold:
                old_cost = float(costs[np.flatnonzero(candidate_groups == old)[0]])
                new_cost = float(np.min(costs))
                if new_cost < old_cost - 1e-10:
                    partition[node] = new
                    counts[old] -= 1
                    counts[new] += 1
                    sweep_moves += 1
        moved += sweep_moves
        if sweep_moves == 0:
            break
    return encode_partition(partition), moved


def generic_repair(
    initial: np.ndarray,
    feature: np.ndarray,
    graph: sp.spmatrix,
    k: int,
    config: RepairConfig,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    partition = encode_partition(initial)
    if len(np.unique(partition)) != int(k):
        raise ValueError("initial partition must have exact K")
    if not config.enabled:
        sizes = np.bincount(partition, minlength=int(k))
        return partition, {
            "repair_enabled": 0,
            "repair_threshold": 0,
            "repair_merged_clusters": 0,
            "repair_split_clusters": 0,
            "repair_boundary_moves": 0,
            "min_cluster_size": int(sizes.min()),
        }
    graph = sp.csr_matrix(graph, dtype=np.float32)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph = graph.maximum(graph.T).tocsr()
    threshold = max(
        2,
        int(np.ceil(float(config.min_cluster_fraction_of_equal) * len(partition) / int(k))),
    )
    repaired, removed = _merge_tiny(
        partition,
        np.asarray(feature, dtype=np.float32),
        graph,
        threshold,
        config.merge_mode,
        float(config.merge_boundary_weight),
    )
    repaired, added = _split_to_k(repaired, feature, graph, k, threshold, config)
    repaired, moves = _boundary_refine(
        repaired,
        feature,
        graph,
        k,
        threshold,
        config.boundary_refine_beta,
        config.boundary_refine_sweeps,
    )
    sizes = np.bincount(repaired, minlength=int(k))
    if len(np.unique(repaired)) != int(k) or int(sizes.min()) < threshold:
        raise RuntimeError("generic repair violated cardinality/minimum-size contract")
    return repaired, {
        "repair_enabled": 1,
        "repair_threshold": int(threshold),
        "repair_merged_clusters": int(removed),
        "repair_split_clusters": int(added),
        "repair_boundary_moves": int(moves),
        "min_cluster_size": int(sizes.min()),
    }


def decode(
    initial: np.ndarray,
    k: int,
    prepared: PreparedDecoderInput,
    config: DecoderConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    partition = encode_partition(initial)
    diagnostics: dict[str, object] = {
        "start_partition_sha256": partition_sha256(partition),
        "energy_enabled": int(config.use_energy),
        "optional_energy_enabled": int(config.use_optional_energy),
    }
    if config.use_energy:
        if prepared.energy_evidence is None or prepared.energy_config is None:
            raise ValueError("energy evidence/config missing")
        partition, detail = continuous_multiscale_expansion(
            partition, k, prepared.energy_evidence, prepared.energy_config
        )
        diagnostics.update({f"energy__{key}": value for key, value in detail.items()})
    if config.use_optional_energy:
        if prepared.optional_evidence is None or prepared.optional_config is None:
            raise ValueError("optional energy evidence/config missing")
        partition, detail = optional_morphology_expansion(
            partition, k, prepared.optional_evidence, prepared.optional_config
        )
        diagnostics.update({f"optional__{key}": value for key, value in detail.items()})
    features = build_feature_bank(prepared)
    if config.repair.feature_mode not in features:
        if config.repair.enabled:
            raise ValueError(f"feature mode unavailable: {config.repair.feature_mode}")
        feature = features["retained"]
    else:
        feature = features[config.repair.feature_mode]
    partition, detail = generic_repair(
        partition, feature, prepared.graph, k, config.repair
    )
    diagnostics.update(detail)
    diagnostics["final_partition_sha256"] = partition_sha256(partition)
    diagnostics["changed_observations"] = int(np.sum(partition != encode_partition(initial)))
    return partition, diagnostics


def structural_descriptors(
    partition: np.ndarray,
    graph: sp.spmatrix,
    k: int,
) -> dict[str, float | int | str]:
    partition = encode_partition(partition)
    sizes = np.bincount(partition, minlength=int(k))
    probability = sizes / max(float(sizes.sum()), 1.0)
    positive = probability[probability > 0]
    graph = sp.triu(sp.csr_matrix(graph).maximum(sp.csr_matrix(graph).T), k=1, format="coo")
    same = float(np.mean(partition[graph.row] == partition[graph.col])) if graph.nnz else 0.0
    return {
        "exact_k": int(len(np.unique(partition)) == int(k)),
        "finite": 1,
        "min_cluster_size": int(sizes.min()),
        "cluster_sizes": "[" + ",".join(str(int(x)) for x in sizes) + "]",
        "normalized_cluster_entropy": float(-np.sum(positive * np.log(positive)) / np.log(max(k, 2))),
        "graph_same_fraction": same,
        "partition_sha256": partition_sha256(partition),
    }
