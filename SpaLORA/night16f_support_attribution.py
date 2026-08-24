"""Matched support-attribution controls for the Night-16F RNA+chromatin study.

The module receives numeric views, sparse registered graphs, ordered observation
identifiers, an exact-K start and the frozen Night-16E chromatin configuration.
It never receives a dataset name or an annotation.  The primary controls use
the same start, base unary, Potts coefficient, base rejected-mass stay cost and
alpha-expansion endpoint; only the spatial-edge support factor changes.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from SpaLORA.night15f_multiscale_expansion import (
    _pairwise_scale,
    add_current_label_stay_cost,
    alpha_expansion_move,
    continuous_multiscale_unary,
    normalized_scale_weights,
    potts_energy,
)
from SpaLORA.night16e_tsre import (
    PreparedTSREEvidence,
    TSREConfig,
    _edge_vector,
    _rank_change,
    _tempered_states,
    _weighted_undirected_matrix,
    direct_energy_control,
    partition_sha256,
    tsre_expansion,
)


PRIMARY_ATTRIBUTION_ARMS = (
    "DIRECT_BASE",
    "BIMODAL_SUPPORT",
    "UNIFORM_MASS_MATCHED",
    "PERMUTED_SUPPORT",
    "RNA_ONLY_SUPPORT",
    "ATAC_ONLY_SUPPORT",
)

SECONDARY_ARMS = (
    "FULL_TSRE",
    "RELATION_STAY_OFF_BASE_STAY_ON",
    "BOUNDARY_OFF",
    "PRIVATE_CONFLICT_OFF",
    "PURE_SUPPORT",
)


def _identifier_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _mass_match(
    base_edge: np.ndarray,
    factor: np.ndarray,
    target_mass: float,
) -> tuple[np.ndarray, float]:
    """Scale a factor vector so weighted edge mass exactly matches the target."""

    base_edge = np.asarray(base_edge, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.float64)
    denominator = float(np.dot(base_edge, factor))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("support factor has no finite positive weighted mass")
    multiplier = float(target_mass) / denominator
    matched = factor * multiplier
    observed = float(np.dot(base_edge, matched))
    tolerance = max(1e-10, abs(float(target_mass)) * 1e-12)
    if not np.isfinite(matched).all() or abs(observed - float(target_mass)) > tolerance:
        raise RuntimeError("weighted support mass matching failed")
    return matched, multiplier


def _stable_edge_order(
    ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    scale_index: int,
) -> tuple[np.ndarray, list[bytes]]:
    ids = np.asarray(ids)
    keys: list[bytes] = []
    for row, col in zip(rows, cols):
        left = _identifier_text(ids[int(row)])
        right = _identifier_text(ids[int(col)])
        if right < left:
            left, right = right, left
        keys.append(
            hashlib.sha256(
                f"night16f-edge\0{int(scale_index)}\0{left}\0{right}".encode("utf-8")
            ).digest()
        )
    order = np.asarray(sorted(range(len(keys)), key=lambda index: keys[index]), dtype=np.int64)
    return order, keys


def deterministic_permuted_support(
    ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    scale_index: int,
    factor: np.ndarray,
) -> np.ndarray:
    """Permute edge factors independently of the incoming sparse-edge order."""

    factor = np.asarray(factor, dtype=np.float64)
    order, keys = _stable_edge_order(ids, rows, cols, scale_index)
    canonical = factor[order]
    seed_payload = b"night16f-support-permutation\0" + b"".join(keys[index] for index in order)
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little", signed=False)
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(canonical))
    result = np.empty_like(factor)
    result[order] = canonical[permutation]
    return result


def _single_view_support(similarity: np.ndarray, temperature: float) -> np.ndarray:
    change = _rank_change(similarity).astype(np.float64)
    power = 1.0 / max(float(temperature), 1e-4)
    inside = np.power(np.maximum(1.0 - change, 1e-12), power)
    boundary = np.power(np.maximum(change, 1e-12), power)
    return inside / np.maximum(inside + boundary, 1e-12)


def attribution_pairwise(
    evidence: PreparedTSREEvidence,
    config: TSREConfig,
    ordered_ids: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build an arm-specific pairwise field with per-scale mass accounting."""

    if arm not in PRIMARY_ATTRIBUTION_ARMS[1:]:
        raise ValueError(f"not a support-attribution pairwise arm: {arm}")
    config.validate()
    scale_weights = normalized_scale_weights(config.base)
    n = evidence.base.registered.binary.shape[0]
    if len(ordered_ids) != n:
        raise ValueError("ordered identifier length mismatch")
    matrices: list[sp.csr_matrix] = []
    rejected: list[np.ndarray] = []
    ledger: list[dict[str, object]] = []

    for scale_index, (scale_weight, edges, relation) in enumerate(
        zip(scale_weights, evidence.base.edges, evidence.relations)
    ):
        base_matrix, base_rejected, _ = _pairwise_scale(edges, n, config.base.local)
        base_edge = _edge_vector(base_matrix, relation.rows, relation.cols)
        support, _, _, _, _ = _tempered_states(relation, config.relation_temperature)
        bimodal = (1.0 - float(config.support_mix)) + float(config.support_mix) * support
        target_mass = float(np.dot(base_edge, bimodal))

        if arm == "BIMODAL_SUPPORT":
            factor = np.asarray(bimodal, dtype=np.float64)
            multiplier = 1.0
        elif arm == "UNIFORM_MASS_MATCHED":
            denominator = float(np.sum(base_edge))
            if denominator <= 0:
                raise ValueError("base graph has no positive capacity")
            factor = np.full(len(base_edge), target_mass / denominator, dtype=np.float64)
            multiplier = 1.0
        elif arm == "PERMUTED_SUPPORT":
            permuted = deterministic_permuted_support(
                ordered_ids, relation.rows, relation.cols, scale_index, bimodal
            )
            factor, multiplier = _mass_match(base_edge, permuted, target_mass)
        elif arm == "RNA_ONLY_SUPPORT":
            one = _single_view_support(edges.similarity1, config.relation_temperature)
            raw = (1.0 - float(config.support_mix)) + float(config.support_mix) * one
            factor, multiplier = _mass_match(base_edge, raw, target_mass)
        elif arm == "ATAC_ONLY_SUPPORT":
            two = _single_view_support(edges.similarity2, config.relation_temperature)
            raw = (1.0 - float(config.support_mix)) + float(config.support_mix) * two
            factor, multiplier = _mass_match(base_edge, raw, target_mass)
        else:  # pragma: no cover - guarded above
            raise AssertionError(arm)

        observed_mass = float(np.dot(base_edge, factor))
        tolerance = max(1e-10, abs(target_mass) * 1e-12)
        if abs(observed_mass - target_mass) > tolerance:
            raise RuntimeError("arm does not match bimodal weighted mass")
        matrices.append(
            float(scale_weight)
            * _weighted_undirected_matrix(
                n, relation.rows, relation.cols, base_edge * np.asarray(factor, dtype=np.float64)
            )
        )
        rejected.append(np.asarray(base_rejected, dtype=np.float64))
        ledger.append(
            {
                "scale_index": int(scale_index),
                "scale_weight": float(scale_weight),
                "base_mass": float(np.sum(base_edge)),
                "bimodal_target_mass": target_mass,
                "arm_observed_mass": observed_mass,
                "absolute_mass_error": abs(observed_mass - target_mass),
                "factor_mean": float(np.mean(factor)),
                "factor_min": float(np.min(factor)),
                "factor_max": float(np.max(factor)),
                "mass_match_multiplier": float(multiplier),
            }
        )

    matrix = sum(matrices).tocsr()
    upper = sp.triu(matrix, k=1, format="coo")
    pair_weights = np.maximum(upper.data.astype(np.float64), 0.0)
    base_rejected_node = np.tensordot(
        scale_weights, np.stack(rejected), axes=(0, 0)
    ).astype(np.float32)
    return (
        upper.row.astype(np.int32),
        upper.col.astype(np.int32),
        pair_weights,
        base_rejected_node,
        {
            "arm": arm,
            "per_scale_mass_ledger": ledger,
            "pairwise_edge_count": int(len(pair_weights)),
            "pairwise_weight_sum_before_beta": float(np.sum(pair_weights)),
            "base_self_return_preserved": True,
            "relation_stay_applied": False,
        },
    )


def support_attribution_expansion(
    initial: np.ndarray,
    k: int,
    evidence: PreparedTSREEvidence,
    config: TSREConfig,
    ordered_ids: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, dict[str, object]]:
    rows, cols, base_pair_weights, base_rejected, diagnostics = attribution_pairwise(
        evidence, config, ordered_ids, arm
    )
    partition = np.asarray(initial, dtype=np.int32).copy()
    if len(np.unique(partition)) != int(k):
        raise ValueError("initial partition cardinality mismatch")
    pair_weights = float(config.base.pairwise_beta) * base_pair_weights
    cycle_ledger: list[dict[str, object]] = []
    total_moves = 0
    accepted_expansions = 0

    for cycle in range(int(config.base.expansion_cycles)):
        unary, unary_diagnostics = continuous_multiscale_unary(
            partition, k, evidence.base, config.base
        )
        current = unary[np.arange(len(partition)), partition]
        alternative = unary.copy()
        alternative[np.arange(len(partition)), partition] = np.inf
        advantage = np.min(alternative, axis=1) - current
        trust = expit(
            (advantage - float(config.base.local.trust_center))
            / max(float(config.base.local.trust_temperature), 1e-4)
        ).astype(np.float32)
        solver_unary = unary + float(config.base.local.trust_scale) * trust[:, None]
        solver_unary[np.arange(len(partition)), partition] -= (
            float(config.base.local.trust_scale) * trust
        )
        solver_unary = add_current_label_stay_cost(
            solver_unary,
            partition,
            base_rejected,
            config.base.self_return_strength,
        )
        start_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        before = partition.copy()
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
            raise RuntimeError("attribution arm increased frozen-cycle energy")
        cycle_ledger.append(
            {
                "cycle_index": int(cycle),
                "start_energy": float(start_energy),
                "end_energy": float(end_energy),
                "accepted_expansions": int(accepted_expansions - accepted_before),
                "move_assignments": int(total_moves - moves_before),
            }
        )
        if np.array_equal(partition, before):
            break

    sizes = np.bincount(partition, minlength=int(k))
    if len(np.unique(partition)) != int(k) or np.any(sizes <= 0):
        raise RuntimeError("attribution arm lost exact K or created an empty cluster")
    return partition, {
        **diagnostics,
        **unary_diagnostics,
        "initial_partition_sha256": partition_sha256(initial),
        "partition_sha256": partition_sha256(partition),
        "changed_observations": int(np.sum(partition != np.asarray(initial))),
        "accepted_expansions": int(accepted_expansions),
        "total_move_assignments": int(total_moves),
        "cluster_sizes_full": [int(value) for value in sizes],
        "min_cluster_size_full": int(sizes.min()),
        "observed_cardinality": int(len(np.unique(partition))),
        "cycle_energy_ledger": cycle_ledger,
        "each_frozen_cycle_monotone": True,
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
    }


def run_attribution_arm(
    initial: np.ndarray,
    k: int,
    evidence: PreparedTSREEvidence,
    config: TSREConfig,
    ordered_ids: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run one registered arm through the shared producer computation."""

    if arm == "INPUT_START":
        partition = np.asarray(initial, dtype=np.int32).copy()
        return partition, {
            "control": "BYTE_EXACT_INPUT_START",
            "partition_sha256": partition_sha256(partition),
            "producer_label_reads": 0,
            "dense_n_by_n_count": 0,
        }
    if arm == "DIRECT_BASE":
        return direct_energy_control(initial, k, evidence, config)
    if arm in PRIMARY_ATTRIBUTION_ARMS[1:]:
        return support_attribution_expansion(initial, k, evidence, config, ordered_ids, arm)
    if arm == "FULL_TSRE":
        return tsre_expansion(initial, k, evidence, config)
    if arm == "RELATION_STAY_OFF_BASE_STAY_ON":
        return tsre_expansion(initial, k, evidence, replace(config, relation_stay_strength=0.0))
    if arm == "BOUNDARY_OFF":
        return tsre_expansion(initial, k, evidence, replace(config, boundary_strength=0.0))
    if arm == "PRIVATE_CONFLICT_OFF":
        return tsre_expansion(initial, k, evidence, replace(config, private_strength=0.0))
    if arm == "PURE_SUPPORT":
        return tsre_expansion(initial, k, evidence, replace(config, support_mix=1.0))
    raise ValueError(f"unknown Night-16F arm: {arm}")


def registered_arms() -> tuple[str, ...]:
    return ("INPUT_START",) + PRIMARY_ATTRIBUTION_ARMS + SECONDARY_ARMS

