"""Optional-view morphology evidence for the Night-15F sparse energy.

The core receives only numeric views, presence masks, sparse graphs, K and an
initial partition.  No study name or reference label enters this module.
When every optional-view presence value is zero, the supplied Night-15F
authority partition is returned byte-exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from SpaLORA.night15e_continuous_reliability_energy import (
    centroid_unary,
    reduce_full,
    standardize,
    unary_margin,
)
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    PreparedExpansionEvidence,
    add_current_label_stay_cost,
    alpha_expansion_move,
    continuous_multiscale_pairwise,
    continuous_multiscale_unary,
    normalized_scale_weights,
    potts_energy,
    prepare_expansion_evidence,
)


@dataclass(frozen=True)
class OptionalMorphologyConfig:
    base: ExpansionEnergyConfig
    morphology_unary_weight: float
    morphology_edge_weight: float
    reliability_mix: float
    reliability_bias: float
    reliability_consistency_weight: float
    reliability_conflict_weight: float
    reliability_margin_weight: float
    reliability_temperature: float


@dataclass(frozen=True)
class OptionalScaleEvidence:
    rows: np.ndarray
    cols: np.ndarray
    similarity: np.ndarray
    node_consistency: np.ndarray
    node_conflict: np.ndarray


@dataclass(frozen=True)
class PreparedOptionalMorphologyEvidence:
    base: PreparedExpansionEvidence
    optional_base: np.ndarray
    optional_low: np.ndarray
    optional_scales: Tuple[OptionalScaleEvidence, ...]
    presence: np.ndarray


def _robust_zscore(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    median = float(np.median(value))
    mad = float(np.median(np.abs(value - median)))
    scale = max(1.4826 * mad, float(np.std(value)), 1e-6)
    return np.clip((value - median) / scale, -6.0, 6.0).astype(np.float32)


def _edge_similarity(feature: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    distance = np.mean((feature[rows] - feature[cols]) ** 2, axis=1)
    positive = distance[distance > 0]
    bandwidth = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    return np.exp(-distance / bandwidth).astype(np.float32)


def _node_mean(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    total = np.bincount(
        np.concatenate((rows, cols)),
        weights=np.concatenate((value, value)),
        minlength=n,
    ).astype(np.float32)
    degree = np.bincount(np.concatenate((rows, cols)), minlength=n).astype(np.float32)
    return total / np.maximum(degree, 1.0)


def prepare_optional_morphology_evidence(
    graphs: Sequence[sp.spmatrix],
    retained: np.ndarray,
    molecular_views: Sequence[np.ndarray],
    optional_views: Sequence[np.ndarray],
    optional_presence_masks: Sequence[np.ndarray],
    retained_dim: int = 32,
    molecular_view_dim: int = 24,
    optional_dim: int = 32,
    optional_edge_dim: int = 24,
) -> PreparedOptionalMorphologyEvidence:
    """Prepare one generic optional-view bank plus the registered molecular bank."""

    if len(molecular_views) != 2:
        raise ValueError("exactly two registered molecular views are required")
    if not optional_views or len(optional_views) != len(optional_presence_masks):
        raise ValueError("optional views and presence masks must be non-empty and paired")
    n = len(retained)
    for view in [*molecular_views, *optional_views]:
        if np.asarray(view).ndim != 2 or len(view) != n or not np.isfinite(view).all():
            raise ValueError("view shape/finite contract failed")
    masks = []
    reduced_optional = []
    for view, mask in zip(optional_views, optional_presence_masks):
        numeric_mask = np.asarray(mask, dtype=np.float32)
        if numeric_mask.shape != (n,) or np.any((numeric_mask < 0) | (numeric_mask > 1)):
            raise ValueError("presence mask must be an n-vector in [0,1]")
        masks.append(numeric_mask)
        reduced = reduce_full(np.asarray(view, dtype=np.float32), min(optional_dim, view.shape[1]))
        reduced_optional.append(reduced * numeric_mask[:, None])
    presence = np.maximum.reduce(masks).astype(np.float32)
    optional_base = standardize(np.concatenate(reduced_optional, axis=1)).astype(np.float32)
    optional_base[presence <= 0] = 0.0
    base = prepare_expansion_evidence(
        graphs,
        retained,
        molecular_views[0],
        molecular_views[1],
        retained_dim=retained_dim,
        view_dim=molecular_view_dim,
    )
    optional_low = np.stack(
        [standardize(np.asarray(transition @ optional_base, dtype=np.float32)) for transition in base.transitions],
        axis=0,
    ).astype(np.float32)
    optional_edge = reduce_full(optional_base, min(optional_edge_dim, optional_base.shape[1]))
    scales = []
    for molecular_edges in base.edges:
        rows, cols = molecular_edges.rows, molecular_edges.cols
        similarity = _edge_similarity(optional_edge, rows, cols)
        edge_present = presence[rows] * presence[cols]
        similarity *= edge_present
        molecular_similarity = np.sqrt(
            np.maximum(molecular_edges.similarity1 * molecular_edges.similarity2, 0.0)
        )
        scales.append(
            OptionalScaleEvidence(
                rows=rows,
                cols=cols,
                similarity=similarity,
                node_consistency=_node_mean(n, rows, cols, similarity),
                node_conflict=_node_mean(
                    n,
                    rows,
                    cols,
                    np.abs(similarity - molecular_similarity) * edge_present,
                ),
            )
        )
    return PreparedOptionalMorphologyEvidence(
        base=base,
        optional_base=optional_base,
        optional_low=optional_low,
        optional_scales=tuple(scales),
        presence=presence,
    )


def _optional_component_bank(
    evidence: PreparedOptionalMorphologyEvidence,
    scale: np.ndarray,
    config: ExpansionEnergyConfig,
) -> np.ndarray:
    low1 = standardize(np.tensordot(scale, evidence.optional_low, axes=(0, 0)).astype(np.float32))
    low2 = np.zeros_like(low1)
    for weight, transition in zip(scale, evidence.base.transitions):
        low2 += float(weight) * np.asarray(transition @ low1, dtype=np.float32)
    low2 = standardize(low2)
    high = standardize(evidence.optional_base - low1)
    components = np.stack((evidence.optional_base, low1, low2, high), axis=0)
    weights = np.maximum(
        np.asarray(
            [
                1.0,
                config.local.low_weight,
                config.local.twohop_weight,
                config.local.high_weight,
            ],
            dtype=np.float32,
        ),
        1e-4,
    )
    weighted = components * np.sqrt(weights)[:, None, None]
    return np.transpose(weighted, (1, 0, 2)).reshape(len(evidence.presence), -1).astype(np.float32)


def _static_optional_evidence(
    evidence: PreparedOptionalMorphologyEvidence,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    consistency = np.tensordot(
        scale,
        np.stack([item.node_consistency for item in evidence.optional_scales], axis=0),
        axes=(0, 0),
    ).astype(np.float32)
    conflict = np.tensordot(
        scale,
        np.stack([item.node_conflict for item in evidence.optional_scales], axis=0),
        axes=(0, 0),
    ).astype(np.float32)
    n = len(evidence.presence)
    matrices = []
    for item in evidence.optional_scales:
        matrices.append(
            sp.csr_matrix(
                (
                    np.concatenate((item.similarity, item.similarity)),
                    (np.concatenate((item.rows, item.cols)), np.concatenate((item.cols, item.rows))),
                ),
                shape=(n, n),
            )
        )
    combined = sum(float(weight) * matrix for weight, matrix in zip(scale, matrices)).tocsr()
    return consistency, conflict, combined


def _reliability(
    consistency: np.ndarray,
    conflict: np.ndarray,
    margin: np.ndarray,
    presence: np.ndarray,
    config: OptionalMorphologyConfig,
) -> np.ndarray:
    logit = (
        float(config.reliability_bias)
        + float(config.reliability_consistency_weight) * _robust_zscore(consistency)
        - float(config.reliability_conflict_weight) * _robust_zscore(conflict)
        + float(config.reliability_margin_weight) * _robust_zscore(margin)
    )
    content = expit(logit / max(float(config.reliability_temperature), 1e-4)).astype(np.float32)
    mix = float(np.clip(config.reliability_mix, 0.0, 1.0))
    return (presence * ((1.0 - mix) + mix * content)).astype(np.float32)


def _normalize_cost(cost: np.ndarray) -> np.ndarray:
    value = np.asarray(cost, dtype=np.float32)
    positive = value[value > 0]
    scale = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    return value / scale


def optional_morphology_expansion(
    initial: np.ndarray,
    k: int,
    evidence: PreparedOptionalMorphologyEvidence,
    config: OptionalMorphologyConfig,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Run the optional-view incremental energy from a Night-15F authority."""

    partition = np.asarray(initial, dtype=np.int32).copy()
    if len(np.unique(partition)) != int(k):
        raise ValueError("initial partition cardinality mismatch")
    if float(np.max(evidence.presence)) <= 0.0:
        return partition, {
            "optional_view_present": 0.0,
            "missing_view_exact_fallback": 1.0,
            "changed_observations": 0.0,
            "observed_cardinality": float(len(np.unique(partition))),
            "cycle_energy_ledger_json": "[]",
        }
    if float(config.morphology_unary_weight) == 0.0 and float(config.morphology_edge_weight) == 0.0:
        return partition, {
            "optional_view_present": 1.0,
            "missing_view_exact_fallback": 0.0,
            "optional_weight_noop": 1.0,
            "changed_observations": 0.0,
            "observed_cardinality": float(len(np.unique(partition))),
            "cycle_energy_ledger_json": "[]",
        }
    scale = normalized_scale_weights(config.base)
    optional_bank = _optional_component_bank(evidence, scale, config.base)
    consistency, conflict, optional_matrix = _static_optional_evidence(evidence, scale)
    rows, cols, base_weights, rejected_mass, pair_diagnostics = continuous_multiscale_pairwise(
        evidence.base, config.base
    )
    optional_values = np.asarray(optional_matrix[rows, cols]).reshape(-1).astype(np.float64)
    base_positive = base_weights[base_weights > 0]
    optional_positive = optional_values[optional_values > 0]
    if optional_positive.size:
        optional_values *= max(float(np.median(base_positive)) if base_positive.size else 1.0, 1e-8) / max(
            float(np.median(optional_positive)), 1e-8
        )
    total_moves = 0
    accepted_expansions = 0
    cycle_ledger = []
    reliability = evidence.presence.copy()
    unary_diagnostics: Dict[str, float] = {}
    for cycle in range(int(config.base.expansion_cycles)):
        unary, unary_diagnostics = continuous_multiscale_unary(partition, k, evidence.base, config.base)
        optional_cost = centroid_unary(optional_bank, partition, k)
        optional_margin = unary_margin(optional_cost)
        reliability = _reliability(
            consistency, conflict, optional_margin, evidence.presence, config
        )
        unary = np.asarray(
            unary
            + float(config.morphology_unary_weight)
            * reliability[:, None]
            * _normalize_cost(optional_cost),
            dtype=np.float32,
        )
        edge_reliability = np.sqrt(np.maximum(reliability[rows] * reliability[cols], 0.0))
        pair_weights = float(config.base.pairwise_beta) * (
            base_weights
            + float(config.morphology_edge_weight) * edge_reliability * optional_values
        )
        current = unary[np.arange(len(partition)), partition]
        alternate = unary.copy()
        alternate[np.arange(len(partition)), partition] = np.inf
        advantage = np.min(alternate, axis=1) - current
        trust = expit(
            (advantage - float(config.base.local.trust_center))
            / max(float(config.base.local.trust_temperature), 1e-4)
        ).astype(np.float32)
        solver_unary = add_current_label_stay_cost(
            unary + float(config.base.local.trust_scale) * trust[:, None],
            partition,
            rejected_mass,
            config.base.self_return_strength,
        )
        solver_unary[np.arange(len(partition)), partition] -= (
            float(config.base.local.trust_scale) * trust
        )
        start_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
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
                config.base.capacity_scale,
                config.base.energy_tolerance,
            )
            if accepted:
                total_moves += int(np.sum(proposal != partition))
                accepted_expansions += 1
                partition = proposal
        end_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        if end_energy > start_energy + max(float(config.base.energy_tolerance), 1e-10):
            raise RuntimeError("optional-view alpha moves increased frozen-cycle energy")
        cycle_ledger.append(
            {
                "cycle_index": int(cycle),
                "start_energy": float(start_energy),
                "end_energy": float(end_energy),
                "accepted_expansions": int(accepted_expansions - accepted_before),
                "move_assignments": int(total_moves - moves_before),
            }
        )
        if np.array_equal(changed_before, partition):
            break
    sizes = np.bincount(partition, minlength=int(k))
    return partition, {
        **pair_diagnostics,
        **unary_diagnostics,
        "optional_view_present": 1.0,
        "missing_view_exact_fallback": 0.0,
        "mean_optional_consistency": float(np.mean(consistency)),
        "mean_optional_conflict": float(np.mean(conflict)),
        "mean_optional_reliability": float(np.mean(reliability)),
        "mean_optional_presence": float(np.mean(evidence.presence)),
        "accepted_expansions": float(accepted_expansions),
        "total_move_assignments": float(total_moves),
        "changed_observations": float(np.sum(partition != np.asarray(initial))),
        "observed_cardinality": float(len(np.unique(partition))),
        "min_cluster_size": float(np.min(sizes)),
        "cycle_energy_ledger_json": json.dumps(cycle_ledger, separators=(",", ":")),
        "each_frozen_cycle_monotone": 1.0,
    }
