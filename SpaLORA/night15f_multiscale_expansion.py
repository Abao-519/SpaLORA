"""Sparse continuous multiscale Potts energy with alpha-expansion moves.

The core receives numeric arrays, sparse graphs, an initial partition and K.
It never receives study identifiers, public reference assignments or external
metric values.  The large-neighborhood solver uses integer-quantized sparse
s-t cuts and accepts a move only when the original floating-point energy
strictly decreases.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.special import expit, softmax

from SpaLORA.night15e_continuous_reliability_energy import (
    ContinuousEnergyConfig,
    PreparedContinuousEvidence,
    centroid_unary,
    prepare_continuous_evidence,
    reduce_full,
    row_stochastic,
    standardize,
    unary_margin,
)


@dataclass(frozen=True)
class ExpansionEnergyConfig:
    local: ContinuousEnergyConfig
    scale_fine: float
    scale_registered: float
    scale_broad: float
    pairwise_beta: float
    self_return_strength: float
    size_prior: float
    expansion_cycles: int
    capacity_scale: float = 100000.0
    energy_tolerance: float = 1e-9


@dataclass(frozen=True)
class ScaleEdges:
    rows: np.ndarray
    cols: np.ndarray
    similarity1: np.ndarray
    similarity2: np.ndarray
    degree: np.ndarray


@dataclass(frozen=True)
class PreparedExpansionEvidence:
    registered: PreparedContinuousEvidence
    transitions: Tuple[sp.csr_matrix, ...]
    edges: Tuple[ScaleEdges, ...]
    retained_base: np.ndarray
    view1_base: np.ndarray
    view2_base: np.ndarray
    retained_low: np.ndarray
    view1_low: np.ndarray
    view2_low: np.ndarray


def _undirected_support(graph: sp.spmatrix) -> sp.csr_matrix:
    value = sp.csr_matrix(graph, dtype=np.float32).copy()
    value.data = np.ones_like(value.data, dtype=np.float32)
    value.setdiag(0)
    value.eliminate_zeros()
    value = value.maximum(value.T).tocsr()
    value.data = np.ones_like(value.data, dtype=np.float32)
    value.sort_indices()
    return value


def _scale_edges(
    graph: sp.spmatrix,
    first: np.ndarray,
    second: np.ndarray,
) -> ScaleEdges:
    support = sp.triu(_undirected_support(graph), k=1, format="coo")
    rows = support.row.astype(np.int32)
    cols = support.col.astype(np.int32)
    distance1 = np.mean((first[rows] - first[cols]) ** 2, axis=1)
    distance2 = np.mean((second[rows] - second[cols]) ** 2, axis=1)
    positive1 = distance1[distance1 > 0]
    positive2 = distance2[distance2 > 0]
    bandwidth1 = max(float(np.median(positive1)) if positive1.size else 1.0, 1e-6)
    bandwidth2 = max(float(np.median(positive2)) if positive2.size else 1.0, 1e-6)
    similarity1 = np.exp(-distance1 / bandwidth1).astype(np.float32)
    similarity2 = np.exp(-distance2 / bandwidth2).astype(np.float32)
    degree = np.bincount(
        np.concatenate((rows, cols)), minlength=graph.shape[0]
    ).astype(np.float32)
    return ScaleEdges(rows, cols, similarity1, similarity2, degree)


def prepare_expansion_evidence(
    graphs: Sequence[sp.spmatrix],
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    retained_dim: int = 32,
    view_dim: int = 24,
    edge_dim: int = 16,
) -> PreparedExpansionEvidence:
    if len(graphs) != 3:
        raise ValueError("exactly three registered graph scales are required")
    supports = tuple(_undirected_support(graph) for graph in graphs)
    transitions = tuple(row_stochastic(graph) for graph in supports)
    registered = prepare_continuous_evidence(
        supports[1], retained, view1, view2, retained_dim, view_dim, edge_dim
    )
    retained_base = reduce_full(retained, retained_dim)
    view1_base = reduce_full(view1, min(view_dim, view1.shape[1]))
    view2_base = reduce_full(view2, min(view_dim, view2.shape[1]))
    edge1 = reduce_full(view1, min(edge_dim, view1.shape[1]))
    edge2 = reduce_full(view2, min(edge_dim, view2.shape[1]))
    edges = tuple(_scale_edges(graph, edge1, edge2) for graph in supports)

    def low_bank(base: np.ndarray) -> np.ndarray:
        return np.stack(
            [standardize(np.asarray(transition @ base, dtype=np.float32)) for transition in transitions],
            axis=0,
        ).astype(np.float32)

    return PreparedExpansionEvidence(
        registered=registered,
        transitions=transitions,
        edges=edges,
        retained_base=retained_base,
        view1_base=view1_base,
        view2_base=view2_base,
        retained_low=low_bank(retained_base),
        view1_low=low_bank(view1_base),
        view2_low=low_bank(view2_base),
    )


def normalized_scale_weights(config: ExpansionEnergyConfig) -> np.ndarray:
    value = np.asarray(
        [config.scale_fine, config.scale_registered, config.scale_broad],
        dtype=np.float64,
    )
    if np.any(value < 0) or not np.all(np.isfinite(value)) or float(value.sum()) <= 0:
        raise ValueError("graph scale weights must be finite, nonnegative and nonzero")
    return (value / value.sum()).astype(np.float32)


def _mixed_component_bank(
    base: np.ndarray,
    lows: np.ndarray,
    transitions: Sequence[sp.csr_matrix],
    scale: np.ndarray,
) -> np.ndarray:
    low1 = standardize(np.tensordot(scale, lows, axes=(0, 0)).astype(np.float32))
    low2 = np.zeros_like(low1)
    for weight, transition in zip(scale, transitions):
        low2 += float(weight) * np.asarray(transition @ low1, dtype=np.float32)
    low2 = standardize(low2)
    high = standardize(base - low1)
    return np.stack((base, low1, low2, high), axis=0).astype(np.float32)


def _weighted_components(components: np.ndarray, local: ContinuousEnergyConfig) -> np.ndarray:
    weights = np.maximum(
        np.asarray([1.0, local.low_weight, local.twohop_weight, local.high_weight], dtype=np.float32),
        1e-4,
    )
    value = components * np.sqrt(weights)[:, None, None]
    return np.transpose(value, (1, 0, 2)).reshape(components.shape[1], -1)


def continuous_multiscale_unary(
    partition: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    scale = normalized_scale_weights(config)
    banks = (
        _mixed_component_bank(evidence.retained_base, evidence.retained_low, evidence.transitions, scale),
        _mixed_component_bank(evidence.view1_base, evidence.view1_low, evidence.transitions, scale),
        _mixed_component_bank(evidence.view2_base, evidence.view2_low, evidence.transitions, scale),
    )
    costs = tuple(
        centroid_unary(_weighted_components(bank, config.local), partition, k) for bank in banks
    )
    margins = np.column_stack(tuple(unary_margin(cost) for cost in costs))
    logits = np.log(np.maximum(margins, 1e-6)) / max(float(config.local.unary_temperature), 1e-4)
    logits[:, 0] += float(config.local.retained_bias)
    logits[:, 1] += float(config.local.view_balance)
    logits[:, 2] -= float(config.local.view_balance)
    mixture = softmax(logits, axis=1).astype(np.float32)
    unary = sum(mixture[:, index : index + 1] * cost for index, cost in enumerate(costs))
    counts = np.bincount(np.asarray(partition, dtype=np.int32), minlength=int(k)).astype(np.float32)
    target = max(float(len(partition)) / float(k), 1.0)
    size_cost = float(config.size_prior) * np.log(np.maximum(counts, 1.0) / target)
    unary = np.asarray(unary + size_cost[None, :], dtype=np.float32)
    return unary, {
        "mean_retained_weight": float(np.mean(mixture[:, 0])),
        "mean_view1_weight": float(np.mean(mixture[:, 1])),
        "mean_view2_weight": float(np.mean(mixture[:, 2])),
        "mean_prototype_margin": float(np.mean(np.max(margins, axis=1))),
        "min_cluster_size_before": float(np.min(counts)),
    }


def _pairwise_scale(
    edges: ScaleEdges,
    n: int,
    local: ContinuousEnergyConfig,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict[str, float]]:
    first = edges.similarity1
    second = edges.similarity2
    conflict = np.abs(first - second)
    agreement = expit(
        (float(local.conflict_center) - conflict)
        / max(float(local.conflict_temperature), 1e-4)
    ).astype(np.float32)
    intersection = np.sqrt(np.maximum(first * second, 0.0))
    union = np.maximum(first, second)
    strict = np.minimum(first, second)
    conflict_value = float(local.conflict_union_weight) * union + (
        1.0 - float(local.conflict_union_weight)
    ) * strict
    content = agreement * intersection + (1.0 - agreement) * conflict_value
    floor = float(np.clip(local.edge_floor, 1e-4, 0.95))
    weight = floor + (1.0 - floor) * np.clip(content, 0.0, 1.0)
    incident_weight = np.bincount(
        np.concatenate((edges.rows, edges.cols)),
        weights=np.concatenate((weight, weight)),
        minlength=n,
    ).astype(np.float32)
    incident_conflict = np.bincount(
        np.concatenate((edges.rows, edges.cols)),
        weights=np.concatenate((conflict, conflict)),
        minlength=n,
    ).astype(np.float32)
    mean_weight = incident_weight / np.maximum(edges.degree, 1.0)
    mean_conflict = incident_conflict / np.maximum(edges.degree, 1.0)
    reliability = mean_weight - float(local.conflict_penalty) * mean_conflict
    mass = float(np.clip(local.neighbor_capacity, 1e-4, 1.0)) * expit(
        (reliability - float(local.mass_center)) / max(float(local.mass_temperature), 1e-4)
    )
    mass[edges.degree <= 0] = 0.0
    directed_from_row = mass[edges.rows] * weight / np.maximum(incident_weight[edges.rows], 1e-8)
    directed_from_col = mass[edges.cols] * weight / np.maximum(incident_weight[edges.cols], 1e-8)
    symmetric = 0.5 * (directed_from_row + directed_from_col)
    matrix = sp.csr_matrix(
        (
            np.concatenate((symmetric, symmetric)).astype(np.float32),
            (np.concatenate((edges.rows, edges.cols)), np.concatenate((edges.cols, edges.rows))),
        ),
        shape=(n, n),
    )
    rejected_mass = np.asarray(1.0 - mass, dtype=np.float32)
    return matrix, rejected_mass, {
        "mean_edge_weight": float(np.mean(weight)),
        "mean_edge_conflict": float(np.mean(conflict)),
        "mean_agreement_gate": float(np.mean(agreement)),
        "mean_neighbor_mass": float(np.mean(mass)),
        "mean_self_return": float(np.mean(rejected_mass)),
    }


def continuous_multiscale_pairwise(
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    scale = normalized_scale_weights(config)
    n = evidence.registered.binary.shape[0]
    matrices = []
    rejected = []
    diagnostics = []
    for edges in evidence.edges:
        matrix, rejected_mass, detail = _pairwise_scale(edges, n, config.local)
        matrices.append(matrix)
        rejected.append(rejected_mass)
        diagnostics.append(detail)
    combined = sum(float(weight) * matrix for weight, matrix in zip(scale, matrices)).tocsr()
    combined = sp.triu(combined, k=1, format="coo")
    values = np.maximum(combined.data.astype(np.float64), 0.0)
    combined_rejected = np.tensordot(
        scale, np.stack(rejected, axis=0), axes=(0, 0)
    ).astype(np.float32)
    return (
        combined.row.astype(np.int32),
        combined.col.astype(np.int32),
        values,
        combined_rejected,
        {
            "scale_fine_normalized": float(scale[0]),
            "scale_registered_normalized": float(scale[1]),
            "scale_broad_normalized": float(scale[2]),
            "pair_edge_count": float(len(values)),
            "mean_pair_weight": float(np.mean(values)) if len(values) else 0.0,
            "mean_self_return": float(sum(float(w) * d["mean_self_return"] for w, d in zip(scale, diagnostics))),
        },
    )


def potts_energy(
    partition: np.ndarray,
    unary: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
) -> float:
    partition = np.asarray(partition, dtype=np.int32)
    return float(
        np.sum(unary[np.arange(len(partition)), partition], dtype=np.float64)
        + np.sum(weights * (partition[rows] != partition[cols]), dtype=np.float64)
    )


def add_current_label_stay_cost(
    unary: np.ndarray,
    partition: np.ndarray,
    rejected_mass: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Charge rejected conductance mass only when a node leaves its current label."""

    value = np.asarray(unary, dtype=np.float32).copy()
    partition = np.asarray(partition, dtype=np.int32)
    stay_cost = float(strength) * np.asarray(rejected_mass, dtype=np.float32)
    value += stay_cost[:, None]
    value[np.arange(len(partition)), partition] -= stay_cost
    return value


def _binary_cut(
    cost0: np.ndarray,
    cost1: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    cut_weights: np.ndarray,
    capacity_scale: float,
) -> np.ndarray:
    """Return x=1 nodes for a submodular binary energy."""

    cost0 = np.asarray(cost0, dtype=np.float64).copy()
    cost1 = np.asarray(cost1, dtype=np.float64).copy()
    cut_weights = np.maximum(np.asarray(cut_weights, dtype=np.float64), 0.0)
    shift = np.minimum(cost0, cost1)
    cost0 -= shift
    cost1 -= shift
    scale = max(float(capacity_scale), 1.0)
    source_caps = np.rint(cost1 * scale).astype(np.int64)
    sink_caps = np.rint(cost0 * scale).astype(np.int64)
    edge_caps = np.rint(cut_weights * scale).astype(np.int64)
    n = len(cost0)
    source = n
    sink = n + 1
    keep_edges = edge_caps > 0
    edge_rows = rows[keep_edges]
    edge_cols = cols[keep_edges]
    edge_caps = edge_caps[keep_edges]
    graph_rows = np.concatenate(
        (
            np.full(np.count_nonzero(source_caps), source, dtype=np.int64),
            np.flatnonzero(sink_caps).astype(np.int64),
            edge_rows.astype(np.int64),
            edge_cols.astype(np.int64),
        )
    )
    graph_cols = np.concatenate(
        (
            np.flatnonzero(source_caps).astype(np.int64),
            np.full(np.count_nonzero(sink_caps), sink, dtype=np.int64),
            edge_cols.astype(np.int64),
            edge_rows.astype(np.int64),
        )
    )
    graph_data = np.concatenate(
        (source_caps[source_caps > 0], sink_caps[sink_caps > 0], edge_caps, edge_caps)
    )
    capacity = sp.csr_matrix((graph_data, (graph_rows, graph_cols)), shape=(n + 2, n + 2), dtype=np.int64)
    result = maximum_flow(capacity, source, sink, method="dinic")
    residual = (capacity - result.flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    source_side = np.zeros(n + 2, dtype=bool)
    source_side[np.asarray(reachable, dtype=np.int64)] = True
    return ~source_side[:n]


def alpha_expansion_move(
    partition: np.ndarray,
    alpha: int,
    unary: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    pair_weights: np.ndarray,
    capacity_scale: float,
    energy_tolerance: float,
) -> Tuple[np.ndarray, bool, float, float]:
    partition = np.asarray(partition, dtype=np.int32)
    before = potts_energy(partition, unary, rows, cols, pair_weights)
    left = partition[rows]
    right = partition[cols]
    e00 = pair_weights * (left != right)
    e01 = pair_weights * (left != int(alpha))
    e10 = pair_weights * (int(alpha) != right)
    cut = 0.5 * (e01 + e10 - e00)
    if np.any(cut < -1e-10):
        raise ValueError("non-submodular expansion move")
    cut = np.maximum(cut, 0.0)
    adjust1 = np.zeros(len(partition), dtype=np.float64)
    np.add.at(adjust1, rows, e10 - e00 - cut)
    np.add.at(adjust1, cols, e01 - e00 - cut)
    cost0 = unary[np.arange(len(partition)), partition].astype(np.float64)
    cost1 = unary[:, int(alpha)].astype(np.float64) + adjust1
    switch = _binary_cut(cost0, cost1, rows, cols, cut, capacity_scale)
    proposal = partition.copy()
    proposal[switch] = int(alpha)
    if len(np.unique(proposal)) != unary.shape[1]:
        return partition.copy(), False, before, before
    after = potts_energy(proposal, unary, rows, cols, pair_weights)
    accepted = after < before - float(energy_tolerance)
    return (proposal if accepted else partition.copy()), accepted, before, after


def continuous_multiscale_single_site(
    initial: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Matched asynchronous single-site descent for solver ablation.

    The unary, graph-scale mixture, pairwise weights, rejected-mass stay cost,
    cycle refresh, and cardinality guard are identical to the expansion core.
    Only the move neighborhood changes from alpha expansion to one node.
    """

    partition = np.asarray(initial, dtype=np.int32).copy()
    if len(np.unique(partition)) != int(k):
        raise ValueError("initial partition cardinality mismatch")
    rows, cols, base_weights, rejected_mass, pair_diagnostics = continuous_multiscale_pairwise(
        evidence, config
    )
    pair_weights = float(config.pairwise_beta) * base_weights
    pair_matrix = sp.csr_matrix(
        (
            np.concatenate((pair_weights, pair_weights)),
            (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
        ),
        shape=(len(partition), len(partition)),
    )
    cycle_ledger = []
    total_moves = 0
    unary_diagnostics: Dict[str, float] = {}
    for cycle in range(int(config.expansion_cycles)):
        unary, unary_diagnostics = continuous_multiscale_unary(partition, k, evidence, config)
        current = unary[np.arange(len(partition)), partition]
        alternate = unary.copy()
        alternate[np.arange(len(partition)), partition] = np.inf
        advantage = np.min(alternate, axis=1) - current
        trust = expit(
            (advantage - float(config.local.trust_center))
            / max(float(config.local.trust_temperature), 1e-4)
        ).astype(np.float32)
        solver_unary = add_current_label_stay_cost(
            unary + float(config.local.trust_scale) * trust[:, None],
            partition,
            rejected_mass,
            config.self_return_strength,
        )
        solver_unary[np.arange(len(partition)), partition] -= float(config.local.trust_scale) * trust
        start_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        counts = np.bincount(partition, minlength=int(k)).astype(np.int64)
        cycle_moves = 0
        for node in range(len(partition)):
            begin, end = pair_matrix.indptr[node : node + 2]
            neighbours = pair_matrix.indices[begin:end]
            weights = pair_matrix.data[begin:end]
            pair_cost = np.zeros(int(k), dtype=np.float64)
            for cluster_index in range(int(k)):
                pair_cost[cluster_index] = np.sum(
                    weights * (partition[neighbours] != cluster_index), dtype=np.float64
                )
            total_cost = np.asarray(solver_unary[node], dtype=np.float64) + pair_cost
            old = int(partition[node])
            new = int(np.argmin(total_cost))
            if new != old and counts[old] > 1 and total_cost[new] < total_cost[old] - float(config.energy_tolerance):
                partition[node] = new
                counts[old] -= 1
                counts[new] += 1
                cycle_moves += 1
        end_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        if end_energy > start_energy + max(float(config.energy_tolerance), 1e-10):
            raise RuntimeError("single-site descent increased frozen-cycle energy")
        total_moves += cycle_moves
        cycle_ledger.append(
            {
                "cycle_index": int(cycle),
                "start_energy": float(start_energy),
                "end_energy": float(end_energy),
                "accepted_single_site_moves": int(cycle_moves),
            }
        )
        if cycle_moves == 0:
            break
    sizes = np.bincount(partition, minlength=int(k))
    return partition, {
        **pair_diagnostics,
        **unary_diagnostics,
        "solver": "ASYNCHRONOUS_SINGLE_SITE",
        "total_move_assignments": float(total_moves),
        "changed_observations": float(np.sum(partition != np.asarray(initial))),
        "cycle_energy_ledger_json": json.dumps(cycle_ledger, separators=(",", ":")),
        "each_frozen_cycle_monotone": 1.0,
        "cross_dynamic_cycle_global_monotonicity_claimed": 0.0,
        "observed_cardinality": float(len(np.unique(partition))),
        "min_cluster_size": float(np.min(sizes)),
    }


def continuous_multiscale_expansion(
    initial: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    partition = np.asarray(initial, dtype=np.int32).copy()
    if len(np.unique(partition)) != int(k):
        raise ValueError("initial partition cardinality mismatch")
    rows, cols, base_weights, rejected_mass, pair_diagnostics = continuous_multiscale_pairwise(
        evidence, config
    )
    pair_weights = float(config.pairwise_beta) * base_weights
    total_moves = 0
    accepted_expansions = 0
    completed_cycles = 0
    initial_energy = np.nan
    final_energy = np.nan
    unary_diagnostics: Dict[str, float] = {}
    cycle_ledger = []
    for cycle in range(int(config.expansion_cycles)):
        unary, unary_diagnostics = continuous_multiscale_unary(partition, k, evidence, config)
        current = unary[np.arange(len(partition)), partition]
        alternate = unary.copy()
        alternate[np.arange(len(partition)), partition] = np.inf
        advantage = np.min(alternate, axis=1) - current
        trust = expit(
            (advantage - float(config.local.trust_center))
            / max(float(config.local.trust_temperature), 1e-4)
        ).astype(np.float32)
        # Rejected conductance mass is an explicit, continuously weighted stay
        # cost. Its zero-cost label is refreshed from the current partition at
        # every outer cycle. All alternative labels pay the same rejected-mass
        # penalty, so weak edges are not silently renormalized to full strength.
        solver_unary = add_current_label_stay_cost(
            unary + float(config.local.trust_scale) * trust[:, None],
            partition,
            rejected_mass,
            config.self_return_strength,
        )
        solver_unary[np.arange(len(partition)), partition] -= (
            float(config.local.trust_scale) * trust
        )
        cycle_start = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        if not np.isfinite(initial_energy):
            initial_energy = cycle_start
        changed_before = partition.copy()
        accepted_before = accepted_expansions
        moves_before = total_moves
        for alpha in range(int(k)):
            proposal, accepted, _, _ = alpha_expansion_move(
                partition,
                alpha,
                solver_unary,
                rows,
                cols,
                pair_weights,
                config.capacity_scale,
                config.energy_tolerance,
            )
            if accepted:
                total_moves += int(np.sum(proposal != partition))
                accepted_expansions += 1
                partition = proposal
        completed_cycles = cycle + 1
        final_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        if final_energy > cycle_start + max(float(config.energy_tolerance), 1e-10):
            raise RuntimeError("accepted alpha moves increased frozen-cycle energy")
        cycle_ledger.append(
            {
                "cycle_index": int(cycle),
                "start_energy": float(cycle_start),
                "end_energy": float(final_energy),
                "accepted_expansions": int(accepted_expansions - accepted_before),
                "move_assignments": int(total_moves - moves_before),
            }
        )
        if np.array_equal(partition, changed_before):
            break
    sizes = np.bincount(partition, minlength=int(k))
    diagnostics = {
        **pair_diagnostics,
        **unary_diagnostics,
        "expansion_cycles_completed": float(completed_cycles),
        "accepted_expansions": float(accepted_expansions),
        "total_move_assignments": float(total_moves),
        "changed_observations": float(np.sum(partition != np.asarray(initial))),
        "self_return_strength": float(config.self_return_strength),
        "mean_self_return_stay_cost": float(
            float(config.self_return_strength) * np.mean(rejected_mass)
        ),
        "first_cycle_frozen_unary_start_energy": float(initial_energy),
        "last_cycle_frozen_unary_end_energy": float(final_energy),
        "cycle_energy_ledger_json": json.dumps(cycle_ledger, separators=(",", ":")),
        "each_frozen_cycle_monotone": 1.0,
        "cross_dynamic_cycle_global_monotonicity_claimed": 0.0,
        "observed_cardinality": float(len(np.unique(partition))),
        "min_cluster_size": float(np.min(sizes)),
    }
    return partition, diagnostics
