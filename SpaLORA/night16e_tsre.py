"""Tri-State Relation Energy (TSRE) for sparse spatial clustering.

This module extends the validated Night-15F multiscale energy without changing
its producer boundary.  It receives numeric views, registered sparse graphs,
an exact-K start partition and a numeric configuration.  It never receives a
study name or a reference annotation.

For every registered spatial edge, the two modality changes form a soft
partition into support, consensus-boundary and conflict states.  The states
have deliberately different legal roles:

* support scales a non-negative Potts smoothing edge;
* consensus-boundary contributes a frozen-cycle exclusion unary;
* conflict contributes a reliability-weighted private-modality prototype
  unary;
* uncertain/rejected relation mass contributes an explicit current-state stay
  cost.

All dynamic quantities are refreshed between outer cycles.  Consequently, the
solver guarantees monotonic decrease only inside each frozen cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import rankdata

from SpaLORA.night15e_continuous_reliability_energy import centroid_unary, unary_margin
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    PreparedExpansionEvidence,
    _mixed_component_bank,
    _pairwise_scale,
    _weighted_components,
    add_current_label_stay_cost,
    alpha_expansion_move,
    continuous_multiscale_expansion,
    continuous_multiscale_unary,
    normalized_scale_weights,
    potts_energy,
)


@dataclass(frozen=True)
class TSREConfig:
    """Numeric configuration shared by both modality families.

    A family may freeze one instance of this schema.  Dataset names are not
    part of the schema and cannot alter the computation graph.
    """

    base: ExpansionEnergyConfig
    support_mix: float = 0.50
    relation_temperature: float = 1.0
    boundary_strength: float = 0.0
    private_strength: float = 0.0
    relation_stay_strength: float = 0.0

    def validate(self) -> None:
        if not np.isfinite(self.support_mix) or not 0.0 <= float(self.support_mix) <= 1.0:
            raise ValueError("support_mix must be finite and in [0,1]")
        if not np.isfinite(self.relation_temperature) or float(self.relation_temperature) <= 0:
            raise ValueError("relation_temperature must be finite and positive")
        for name in ("boundary_strength", "private_strength", "relation_stay_strength"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class TriStateScale:
    rows: np.ndarray
    cols: np.ndarray
    support: np.ndarray
    boundary: np.ndarray
    conflict: np.ndarray
    private1: np.ndarray
    private2: np.ndarray


@dataclass(frozen=True)
class PreparedTSREEvidence:
    base: PreparedExpansionEvidence
    relations: Tuple[TriStateScale, ...]


def partition_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def config_to_dict(config: TSREConfig) -> dict[str, object]:
    return asdict(config)


def _rank_change(similarity: np.ndarray) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 1 or len(similarity) == 0 or not np.isfinite(similarity).all():
        raise ValueError("edge similarity must be a finite nonempty vector")
    change = -np.log(np.clip(similarity, 1e-12, 1.0))
    return ((rankdata(change, method="average") - 0.5) / float(len(change))).astype(np.float32)


def _tri_state_from_similarities(
    rows: np.ndarray,
    cols: np.ndarray,
    similarity1: np.ndarray,
    similarity2: np.ndarray,
) -> TriStateScale:
    rank1 = _rank_change(similarity1)
    rank2 = _rank_change(similarity2)
    support = (1.0 - rank1) * (1.0 - rank2)
    boundary = rank1 * rank2
    private1 = (1.0 - rank1) * rank2
    private2 = rank1 * (1.0 - rank2)
    conflict = private1 + private2
    total = support + boundary + conflict
    support = support / np.maximum(total, 1e-12)
    boundary = boundary / np.maximum(total, 1e-12)
    private1 = private1 / np.maximum(total, 1e-12)
    private2 = private2 / np.maximum(total, 1e-12)
    conflict = private1 + private2
    return TriStateScale(
        rows=np.asarray(rows, dtype=np.int32),
        cols=np.asarray(cols, dtype=np.int32),
        support=support.astype(np.float32),
        boundary=boundary.astype(np.float32),
        conflict=conflict.astype(np.float32),
        private1=private1.astype(np.float32),
        private2=private2.astype(np.float32),
    )


def prepare_tsre_evidence(base: PreparedExpansionEvidence) -> PreparedTSREEvidence:
    relations = tuple(
        _tri_state_from_similarities(
            edges.rows,
            edges.cols,
            edges.similarity1,
            edges.similarity2,
        )
        for edges in base.edges
    )
    return PreparedTSREEvidence(base=base, relations=relations)


def _tempered_states(
    relation: TriStateScale, temperature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    power = 1.0 / max(float(temperature), 1e-4)
    support = np.power(np.maximum(relation.support, 1e-12), power)
    boundary = np.power(np.maximum(relation.boundary, 1e-12), power)
    private1 = np.power(np.maximum(relation.private1, 1e-12), power)
    private2 = np.power(np.maximum(relation.private2, 1e-12), power)
    total = support + boundary + private1 + private2
    support = support / total
    boundary = boundary / total
    private1 = private1 / total
    private2 = private2 / total
    conflict = private1 + private2
    return support, boundary, conflict, private1, private2


def _edge_vector(matrix: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[rows, cols]).reshape(-1).astype(np.float64)


def _weighted_undirected_matrix(
    n: int, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray
) -> sp.csr_matrix:
    weights = np.asarray(weights, dtype=np.float64)
    value = sp.csr_matrix(
        (
            np.concatenate((weights, weights)),
            (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
        ),
        shape=(n, n),
    )
    value.sum_duplicates()
    value.sort_indices()
    return value


def _node_incident_mean(
    n: int, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    total = np.bincount(
        np.concatenate((rows, cols)),
        weights=np.concatenate((weights, weights)),
        minlength=n,
    ).astype(np.float64)
    count = np.bincount(np.concatenate((rows, cols)), minlength=n).astype(np.float64)
    return np.divide(total, count, out=np.zeros(n, dtype=np.float64), where=count > 0)


def tsre_pairwise_and_relations(
    evidence: PreparedTSREEvidence,
    config: TSREConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    sp.csr_matrix,
    sp.csr_matrix,
    sp.csr_matrix,
    dict[str, float],
]:
    """Build legal support Potts edges and relation-specific sparse fields."""

    config.validate()
    scale = normalized_scale_weights(config.base)
    n = evidence.base.registered.binary.shape[0]
    support_matrices = []
    boundary_matrices = []
    private1_matrices = []
    private2_matrices = []
    base_rejected = []
    confidence_nodes = []
    state_means = []

    for scale_weight, edges, relation in zip(scale, evidence.base.edges, evidence.relations):
        base_matrix, rejected, _ = _pairwise_scale(edges, n, config.base.local)
        base_edge = _edge_vector(base_matrix, relation.rows, relation.cols)
        support, boundary, conflict, private1, private2 = _tempered_states(
            relation, config.relation_temperature
        )
        support_factor = (1.0 - float(config.support_mix)) + float(config.support_mix) * support
        support_matrices.append(
            float(scale_weight)
            * _weighted_undirected_matrix(n, relation.rows, relation.cols, base_edge * support_factor)
        )
        boundary_matrices.append(
            float(scale_weight)
            * _weighted_undirected_matrix(n, relation.rows, relation.cols, base_edge * boundary)
        )
        private1_matrices.append(
            float(scale_weight)
            * _weighted_undirected_matrix(n, relation.rows, relation.cols, base_edge * private1)
        )
        private2_matrices.append(
            float(scale_weight)
            * _weighted_undirected_matrix(n, relation.rows, relation.cols, base_edge * private2)
        )
        base_rejected.append(np.asarray(rejected, dtype=np.float64))
        relation_confidence = np.maximum.reduce((support, boundary, conflict))
        confidence_nodes.append(
            _node_incident_mean(n, relation.rows, relation.cols, relation_confidence)
        )
        state_means.append((float(np.mean(support)), float(np.mean(boundary)), float(np.mean(conflict))))

    support_matrix = sum(support_matrices).tocsr()
    boundary_matrix = sum(boundary_matrices).tocsr()
    private1_matrix = sum(private1_matrices).tocsr()
    private2_matrix = sum(private2_matrices).tocsr()
    upper = sp.triu(support_matrix, k=1, format="coo")
    pair_weights = np.maximum(upper.data.astype(np.float64), 0.0)
    if np.any(pair_weights < 0) or not np.isfinite(pair_weights).all():
        raise RuntimeError("support produced an invalid Potts edge")
    base_rejected_node = np.tensordot(scale, np.stack(base_rejected), axes=(0, 0))
    relation_confidence_node = np.tensordot(scale, np.stack(confidence_nodes), axes=(0, 0))
    relation_rejected_node = np.clip(1.0 - relation_confidence_node, 0.0, 1.0).astype(np.float32)
    diagnostics = {
        "support_mean": float(sum(float(w) * s[0] for w, s in zip(scale, state_means))),
        "boundary_mean": float(sum(float(w) * s[1] for w, s in zip(scale, state_means))),
        "conflict_mean": float(sum(float(w) * s[2] for w, s in zip(scale, state_means))),
        "mean_base_rejected_mass": float(np.mean(base_rejected_node)),
        "mean_relation_rejected_mass": float(np.mean(relation_rejected_node)),
        "support_pair_edge_count": float(len(pair_weights)),
        "support_pair_weight_mean": float(np.mean(pair_weights)) if len(pair_weights) else 0.0,
    }
    return (
        upper.row.astype(np.int32),
        upper.col.astype(np.int32),
        pair_weights,
        np.asarray(base_rejected_node, dtype=np.float32),
        boundary_matrix,
        private1_matrix,
        private2_matrix,
        {**diagnostics, "relation_rejected_node": relation_rejected_node},
    )


def _component_costs(
    partition: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = normalized_scale_weights(config)
    retained_bank = _mixed_component_bank(
        evidence.retained_base, evidence.retained_low, evidence.transitions, scale
    )
    view1_bank = _mixed_component_bank(
        evidence.view1_base, evidence.view1_low, evidence.transitions, scale
    )
    view2_bank = _mixed_component_bank(
        evidence.view2_base, evidence.view2_low, evidence.transitions, scale
    )
    retained = centroid_unary(_weighted_components(retained_bank, config.local), partition, k)
    view1 = centroid_unary(_weighted_components(view1_bank, config.local), partition, k)
    view2 = centroid_unary(_weighted_components(view2_bank, config.local), partition, k)
    return retained, view1, view2, np.column_stack((unary_margin(view1), unary_margin(view2)))


def boundary_exclusion_unary(
    partition: np.ndarray,
    k: int,
    boundary_matrix: sp.csr_matrix,
    strength: float,
) -> np.ndarray:
    """Penalize adopting the frozen neighbour label across boundary edges."""

    partition = np.asarray(partition, dtype=np.int32)
    coo = sp.triu(boundary_matrix, k=1, format="coo")
    unary = np.zeros((len(partition), int(k)), dtype=np.float32)
    value = float(strength) * coo.data.astype(np.float32)
    np.add.at(unary, (coo.row, partition[coo.col]), value)
    np.add.at(unary, (coo.col, partition[coo.row]), value)
    return unary


def private_modality_unary(
    view1_cost: np.ndarray,
    view2_cost: np.ndarray,
    margins: np.ndarray,
    private1_matrix: sp.csr_matrix,
    private2_matrix: sp.csr_matrix,
    strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Retain the numerically more reliable private prototype evidence."""

    node_private1 = np.asarray(private1_matrix.sum(axis=1)).reshape(-1)
    node_private2 = np.asarray(private2_matrix.sum(axis=1)).reshape(-1)
    margin1 = np.maximum(np.asarray(margins[:, 0], dtype=np.float64), 1e-8)
    margin2 = np.maximum(np.asarray(margins[:, 1], dtype=np.float64), 1e-8)
    weight1 = node_private1 * margin1
    weight2 = node_private2 * margin2
    total = weight1 + weight2
    normalized1 = np.divide(weight1, total, out=np.full_like(total, 0.5), where=total > 0)
    normalized2 = 1.0 - normalized1
    mass = np.clip(total, 0.0, np.percentile(total[total > 0], 95) if np.any(total > 0) else 1.0)
    mass_scale = float(np.median(mass[mass > 0])) if np.any(mass > 0) else 1.0
    mass = mass / max(mass_scale, 1e-8)
    unary = float(strength) * mass[:, None] * (
        normalized1[:, None] * view1_cost + normalized2[:, None] * view2_cost
    )
    return np.asarray(unary, dtype=np.float32), {
        "mean_private_view1_weight": float(np.mean(normalized1)),
        "mean_private_view2_weight": float(np.mean(normalized2)),
        "mean_private_mass": float(np.mean(mass)),
    }


def tsre_expansion(
    initial: np.ndarray,
    k: int,
    evidence: PreparedTSREEvidence,
    config: TSREConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    config.validate()
    partition = np.asarray(initial, dtype=np.int32).copy()
    if len(np.unique(partition)) != int(k):
        raise ValueError("initial partition cardinality mismatch")
    (
        rows,
        cols,
        base_pair_weights,
        base_rejected,
        boundary_matrix,
        private1_matrix,
        private2_matrix,
        relation_diagnostics,
    ) = tsre_pairwise_and_relations(evidence, config)
    relation_rejected = np.asarray(relation_diagnostics.pop("relation_rejected_node"), dtype=np.float32)
    pair_weights = float(config.base.pairwise_beta) * base_pair_weights
    cycle_ledger: list[dict[str, object]] = []
    total_moves = 0
    accepted_expansions = 0
    unary_diagnostics: Dict[str, float] = {}
    private_diagnostics: Dict[str, float] = {}

    for cycle in range(int(config.base.expansion_cycles)):
        content_unary, unary_diagnostics = continuous_multiscale_unary(
            partition, k, evidence.base, config.base
        )
        _, view1_cost, view2_cost, margins = _component_costs(
            partition, k, evidence.base, config.base
        )
        boundary_unary = boundary_exclusion_unary(
            partition, k, boundary_matrix, config.boundary_strength
        )
        private_unary, private_diagnostics = private_modality_unary(
            view1_cost,
            view2_cost,
            margins,
            private1_matrix,
            private2_matrix,
            config.private_strength,
        )
        unary = np.asarray(content_unary + boundary_unary + private_unary, dtype=np.float32)
        current = unary[np.arange(len(partition)), partition]
        alternative = unary.copy()
        alternative[np.arange(len(partition)), partition] = np.inf
        advantage = np.min(alternative, axis=1) - current
        trust = expit(
            (advantage - float(config.base.local.trust_center))
            / max(float(config.base.local.trust_temperature), 1e-4)
        ).astype(np.float32)
        solver_unary = unary + float(config.base.local.trust_scale) * trust[:, None]
        solver_unary[np.arange(len(partition)), partition] -= float(config.base.local.trust_scale) * trust
        solver_unary = add_current_label_stay_cost(
            solver_unary,
            partition,
            base_rejected,
            config.base.self_return_strength,
        )
        solver_unary = add_current_label_stay_cost(
            solver_unary,
            partition,
            relation_rejected,
            config.relation_stay_strength,
        )
        start_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        cycle_start_partition = partition.copy()
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
                config.base.capacity_scale,
                config.base.energy_tolerance,
            )
            if accepted:
                total_moves += int(np.sum(proposal != partition))
                accepted_expansions += 1
                partition = proposal
        end_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        if end_energy > start_energy + max(float(config.base.energy_tolerance), 1e-10):
            raise RuntimeError("TSRE increased a frozen-cycle energy")
        cycle_ledger.append(
            {
                "cycle_index": int(cycle),
                "start_energy": float(start_energy),
                "end_energy": float(end_energy),
                "accepted_expansions": int(accepted_expansions - accepted_before),
                "move_assignments": int(total_moves - moves_before),
                "boundary_unary_mean": float(np.mean(boundary_unary)),
                "private_unary_mean": float(np.mean(private_unary)),
            }
        )
        if np.array_equal(partition, cycle_start_partition):
            break

    sizes = np.bincount(partition, minlength=int(k))
    return partition, {
        **relation_diagnostics,
        **unary_diagnostics,
        **private_diagnostics,
        "config": config_to_dict(config),
        "initial_partition_sha256": partition_sha256(initial),
        "partition_sha256": partition_sha256(partition),
        "changed_observations": int(np.sum(partition != np.asarray(initial))),
        "accepted_expansions": int(accepted_expansions),
        "total_move_assignments": int(total_moves),
        "cluster_sizes_full": [int(x) for x in sizes],
        "min_cluster_size_full": int(sizes.min()),
        "observed_cardinality": int(len(np.unique(partition))),
        "cycle_energy_ledger": cycle_ledger,
        "each_frozen_cycle_monotone": True,
        "cross_dynamic_cycle_global_monotonicity_claimed": False,
        "support_pairwise_nonnegative": bool(np.all(pair_weights >= 0)),
        "boundary_negative_potts_count": 0,
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
    }


def direct_energy_control(
    initial: np.ndarray,
    k: int,
    evidence: PreparedTSREEvidence,
    config: TSREConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Exact Night-15F direct-energy control under the same base config."""

    partition, diagnostics = continuous_multiscale_expansion(initial, k, evidence.base, config.base)
    return partition, {
        **diagnostics,
        "initial_partition_sha256": partition_sha256(initial),
        "partition_sha256": partition_sha256(partition),
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "control": "NIGHT15F_DIRECT_ENERGY_SAME_START",
    }

