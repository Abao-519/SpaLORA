"""Universal graph-feasibility gates and label-free candidate selectors.

The module consumes only locked candidate partitions, their label-free numeric
evidence, and registered sparse spatial graphs.  It has no dataset-name or
annotation input and never constructs a dense observation-by-observation array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr


FEASIBILITY_MODES = (
    "UNCONSTRAINED",
    "NO_SINGLETON",
    "ANY_SCALE_INTERNAL_EDGE",
    "SMALLEST_SCALE_INTERNAL_EDGE",
    "DEGREE_DERIVED_INTERNAL_EDGE",
)


def _encode(partition: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(np.asarray(partition), return_inverse=True)
    return encoded.astype(np.int32)


def _percentile(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("selector axis contains nonfinite values")
    ranks = rankdata(values if higher_is_better else -values, method="average")
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    return (ranks - 1.0) / (len(values) - 1.0)


def prepare_graph(graph: sp.csr_matrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def graph_degree_threshold(graph: sp.csr_matrix, prepared: bool = False) -> int:
    """Mechanically derive a minimal edge-count threshold from graph degree."""

    graph = sp.csr_matrix(graph) if prepared else prepare_graph(graph)
    degree = np.diff(graph.indptr)
    positive = degree[degree > 0]
    median_degree = float(np.median(positive)) if len(positive) else 0.0
    return max(1, int(np.ceil(median_degree / 2.0)))


def graph_support(
    graph: sp.csr_matrix, partition: np.ndarray, k: int, prepared: bool = False
) -> dict[str, object]:
    """Return per-cluster internal sparse-edge support without dense allocation."""

    graph = sp.csr_matrix(graph) if prepared else prepare_graph(graph)
    partition = _encode(partition)
    if len(np.unique(partition)) != k:
        raise ValueError("partition violates exact K")
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    cols = graph.indices
    upper = rows < cols
    same = partition[rows] == partition[cols]
    internal = upper & same
    internal_edge_counts = np.bincount(partition[rows[internal]], minlength=k).astype(np.int64)
    internal_weight = np.bincount(
        partition[rows[internal]], weights=graph.data[internal], minlength=k
    ).astype(np.float64)
    supported = np.zeros(graph.shape[0], dtype=bool)
    supported[rows[same]] = True
    supported[cols[same]] = True
    cluster_sizes = np.bincount(partition, minlength=k).astype(np.int64)
    supported_counts = np.bincount(partition[supported], minlength=k).astype(np.int64)
    supported_fraction = supported_counts / np.maximum(cluster_sizes, 1)
    return {
        "internal_edge_counts": internal_edge_counts,
        "internal_weight": internal_weight,
        "supported_node_counts": supported_counts,
        "supported_node_fraction": supported_fraction,
        "min_internal_edges": int(internal_edge_counts.min()),
        "min_internal_weight": float(internal_weight.min()),
        "min_supported_node_fraction": float(supported_fraction.min()),
        "degree_threshold": graph_degree_threshold(graph, prepared=True),
    }


def feasibility_record(
    graphs: Sequence[sp.csr_matrix],
    partition: np.ndarray,
    k: int,
    prepared_graphs: bool = False,
) -> dict[str, object]:
    partition = _encode(partition)
    sizes = np.bincount(partition, minlength=k).astype(np.int64)
    if len(sizes) != k or np.any(sizes == 0):
        raise ValueError("partition violates exact K or contains an empty cluster")
    supports = [graph_support(graph, partition, k, prepared=prepared_graphs) for graph in graphs]
    per_scale_min_edges = [int(row["min_internal_edges"]) for row in supports]
    return {
        "exact_k": True,
        "empty_cluster_count": 0,
        "min_cluster_size": int(sizes.min()),
        "cluster_sizes": sorted(int(value) for value in sizes),
        "per_scale_min_internal_edges": per_scale_min_edges,
        "per_scale_min_internal_weight": [float(row["min_internal_weight"]) for row in supports],
        "per_scale_min_supported_node_fraction": [
            float(row["min_supported_node_fraction"]) for row in supports
        ],
        "smallest_scale_degree_threshold": int(supports[0]["degree_threshold"]),
        "smallest_scale_min_internal_edges": per_scale_min_edges[0],
        "any_scale_each_cluster_has_internal_edge": bool(
            all(
                any(int(supports[scale]["internal_edge_counts"][cluster]) >= 1 for scale in range(len(supports)))
                for cluster in range(k)
            )
        ),
    }


def is_feasible(record: dict[str, object], mode: str) -> bool:
    if mode not in FEASIBILITY_MODES:
        raise ValueError(f"unknown feasibility mode {mode}")
    exact = bool(record["exact_k"]) and int(record["empty_cluster_count"]) == 0
    no_singleton = int(record["min_cluster_size"]) >= 2
    if mode == "UNCONSTRAINED":
        return exact
    if mode == "NO_SINGLETON":
        return exact and no_singleton
    if mode == "ANY_SCALE_INTERNAL_EDGE":
        return exact and no_singleton and bool(record["any_scale_each_cluster_has_internal_edge"])
    if mode == "SMALLEST_SCALE_INTERNAL_EDGE":
        return exact and no_singleton and int(record["smallest_scale_min_internal_edges"]) >= 1
    return (
        exact
        and no_singleton
        and int(record["smallest_scale_min_internal_edges"])
        >= int(record["smallest_scale_degree_threshold"])
    )


@dataclass(frozen=True)
class SelectionResult:
    selected_index: int
    feasible_count: int
    pareto_count: int
    axis_weights: tuple[float, float, float]
    score: np.ndarray
    diagnostics: Mapping[str, object]


def _axes(records: Sequence[dict[str, object]], feasible: np.ndarray) -> np.ndarray:
    raw = np.column_stack(
        [
            [float(row["molecular_joint"]) for row in records],
            [float(row["topology_joint"]) for row in records],
            [float(row["microcluster_score"]) for row in records],
        ]
    )
    axes = np.full_like(raw, np.nan, dtype=np.float64)
    for axis in range(3):
        axes[feasible, axis] = _percentile(raw[feasible, axis])
    return axes


def _pareto_mask(axes: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    result = np.zeros(len(feasible), dtype=bool)
    indices = np.flatnonzero(feasible)
    for index in indices:
        dominated = False
        for other in indices:
            if other == index:
                continue
            if np.all(axes[other] >= axes[index]) and np.any(axes[other] > axes[index]):
                dominated = True
                break
        result[index] = not dominated
    return result


def select_candidate(
    records: Sequence[dict[str, object]],
    partitions: np.ndarray,
    feasible: np.ndarray,
    similarity: np.ndarray,
    method: str,
) -> SelectionResult:
    """Select one feasible candidate using a fixed label-free rule."""

    feasible = np.asarray(feasible, dtype=bool)
    if not feasible.any():
        raise ValueError("no structurally feasible candidate")
    axes = _axes(records, feasible)
    centrality = np.full(len(records), np.nan, dtype=np.float64)
    indices = np.flatnonzero(feasible)
    sub = np.asarray(similarity)[np.ix_(indices, indices)]
    centrality[indices] = (sub.sum(axis=1) - 1.0) / max(len(indices) - 1, 1)
    centrality_rank = np.full(len(records), np.nan)
    centrality_rank[indices] = _percentile(centrality[indices])
    pareto = _pareto_mask(axes, feasible)
    score = np.full(len(records), -np.inf, dtype=np.float64)
    axis_weights = (1.0, 1.0, 1.0)

    if method == "PARETO_MAXIMIN":
        active = np.flatnonzero(pareto)
        minimum = np.min(axes[active], axis=1)
        geometric = np.prod(np.maximum(axes[active], 1e-6), axis=1) ** (1.0 / 3.0)
        mean = np.mean(axes[active], axis=1)
        score[active] = minimum + 1e-3 * geometric + 1e-6 * mean + 1e-9 * centrality_rank[active]
    elif method == "EQUAL_RANK":
        score[indices] = np.mean(axes[indices], axis=1)
    elif method == "CONTENT_ADAPTIVE":
        weights = []
        for axis in range(3):
            # SciPy versions in the supported AutoDL environments expose
            # either a tuple-like result or differently named attributes.
            correlation = spearmanr(axes[indices, axis], centrality[indices])[0]
            correlation = float(correlation) if np.isfinite(correlation) else 0.0
            weights.append(max(correlation, 0.0) + 0.05)
        weight_array = np.asarray(weights, dtype=np.float64)
        weight_array /= weight_array.sum()
        axis_weights = tuple(float(value) for value in weight_array)
        score[indices] = axes[indices] @ weight_array
    elif method == "PARETO_CENTRAL":
        active = np.flatnonzero(pareto)
        score[active] = centrality_rank[active] + 1e-3 * np.mean(axes[active], axis=1)
    elif method == "CROSS_EVIDENCE_ARBITRATION":
        # A molecularly compact solution is accepted only when it is not an
        # outlier on the independent sparse-topology axis.  Otherwise the
        # topology champion is used.  The empirical-median threshold is fixed
        # and uses only the locked candidate evidence distribution.
        molecular_champion = min(
            indices,
            key=lambda index: (-float(axes[index, 0]), str(records[index]["candidate_id"])),
        )
        topology_champion = min(
            indices,
            key=lambda index: (-float(axes[index, 1]), str(records[index]["candidate_id"])),
        )
        molecular_topology_percentile = float(axes[molecular_champion, 1])
        use_molecular = molecular_topology_percentile >= 0.5
        selected = int(molecular_champion if use_molecular else topology_champion)
        score[indices] = axes[indices, 0] if use_molecular else axes[indices, 1]
        return SelectionResult(
            selected_index=selected,
            feasible_count=int(feasible.sum()),
            pareto_count=int(pareto.sum()),
            axis_weights=(1.0, 0.0, 0.0) if use_molecular else (0.0, 1.0, 0.0),
            score=score,
            diagnostics={
                "decision": "MOLECULAR_CHAMPION" if use_molecular else "TOPOLOGY_CHAMPION",
                "molecular_champion_id": str(records[molecular_champion]["candidate_id"]),
                "topology_champion_id": str(records[topology_champion]["candidate_id"]),
                "molecular_champion_topology_percentile": molecular_topology_percentile,
                "fixed_topology_percentile_threshold": 0.5,
            },
        )
    else:
        raise ValueError(f"unknown selector method {method}")
    selected = min(
        np.flatnonzero(np.isfinite(score)),
        key=lambda index: (-float(score[index]), str(records[index]["candidate_id"])),
    )
    return SelectionResult(
        selected_index=int(selected),
        feasible_count=int(feasible.sum()),
        pareto_count=int(pareto.sum()),
        axis_weights=axis_weights,
        score=score,
        diagnostics={},
    )


def select_weighted_rank(
    records: Sequence[dict[str, object]],
    feasible: np.ndarray,
    molecular_weight: float,
    topology_weight: float,
    risk_weight: float,
) -> SelectionResult:
    """Select from the feasible set with frozen nonnegative rank weights.

    This helper is used by the strict training-study-only LOSO calibration.
    It intentionally excludes persistence: Night-16G found no formal support
    for that axis, and Night-16H does not rename ordinary consensus evidence.
    """

    feasible = np.asarray(feasible, dtype=bool)
    weights = np.asarray(
        [molecular_weight, topology_weight, risk_weight], dtype=np.float64
    )
    if not feasible.any():
        raise ValueError("no structurally feasible candidate")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("rank weights must be finite, nonnegative, and nonzero")
    axes = _axes(records, feasible)
    pareto = _pareto_mask(axes, feasible)
    score = np.full(len(records), -np.inf, dtype=np.float64)
    score[feasible] = axes[feasible] @ weights / float(weights.sum())
    selected = min(
        np.flatnonzero(feasible),
        key=lambda index: (-float(score[index]), str(records[index]["candidate_id"])),
    )
    return SelectionResult(
        selected_index=int(selected),
        feasible_count=int(feasible.sum()),
        pareto_count=int(pareto.sum()),
        axis_weights=tuple(float(value) for value in weights),
        score=score,
        diagnostics={"selector": "FROZEN_WEIGHTED_RANK"},
    )
