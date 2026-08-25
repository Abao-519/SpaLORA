"""Learned relation-conditioned nonnegative multiscale Potts energy.

The module consumes frozen Night-17C representations and a Night-15F
multiscale energy.  Learned relations can only rescale nonnegative Potts
capacities.  The public annotation is intentionally absent from every API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import rankdata

from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    PreparedExpansionEvidence,
    _pairwise_scale,
    add_current_label_stay_cost,
    alpha_expansion_move,
    continuous_multiscale_unary,
    normalized_scale_weights,
    potts_energy,
)


@dataclass(frozen=True)
class LRCCConfig:
    base: ExpansionEnergyConfig
    relation_mix: float
    relation_floor: float
    relation_power: float
    uncertainty_scale: float = 1.0


@dataclass(frozen=True)
class RelationScaleEvidence:
    rows: np.ndarray
    cols: np.ndarray
    learned_factor: np.ndarray
    zero_factor: np.ndarray
    learned_support: np.ndarray
    learned_uncertainty: np.ndarray


def _row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return np.asarray(value / np.maximum(norm, 1e-12), dtype=np.float32)


def _rank_support(
    representations: Sequence[np.ndarray], rows: np.ndarray, cols: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if not representations:
        raise ValueError("at least one representation seed is required")
    ranked = []
    for representation in representations:
        normalized = _row_normalize(representation)
        similarity = np.sum(normalized[rows] * normalized[cols], axis=1, dtype=np.float64)
        if not np.all(np.isfinite(similarity)):
            raise FloatingPointError("nonfinite learned edge similarity")
        # Ranks make the relation scale comparable across lanes and graph scales.
        ranked.append((rankdata(similarity, method="average") - 0.5) / max(len(similarity), 1))
    bank = np.stack(ranked, axis=0).astype(np.float32)
    mean = np.mean(bank, axis=0, dtype=np.float64).astype(np.float32)
    # sqrt(12) maps the standard deviation of a uniform [0,1] variable to one.
    uncertainty = np.clip(np.std(bank, axis=0) * np.sqrt(12.0), 0.0, 1.0).astype(np.float32)
    return mean, uncertainty


def _factor(
    support: np.ndarray, uncertainty: np.ndarray, config: LRCCConfig
) -> np.ndarray:
    mix = float(np.clip(config.relation_mix, 0.0, 1.0))
    floor = float(np.clip(config.relation_floor, 1e-6, 1.0))
    confidence = np.clip(1.0 - float(config.uncertainty_scale) * uncertainty, 0.0, 1.0)
    evidence = np.clip(support * confidence, 0.0, 1.0) ** max(float(config.relation_power), 1e-4)
    return np.asarray((1.0 - mix) + mix * (floor + (1.0 - floor) * evidence), dtype=np.float64)


def prepare_relation_evidence(
    evidence: PreparedExpansionEvidence,
    learned_representations: Sequence[np.ndarray],
    zero_representations: Sequence[np.ndarray],
    config: LRCCConfig,
) -> Tuple[RelationScaleEvidence, ...]:
    n = evidence.registered.binary.shape[0]
    for bank in (learned_representations, zero_representations):
        for representation in bank:
            if np.asarray(representation).shape[0] != n:
                raise ValueError("relation representation observation mismatch")
    output = []
    for edges in evidence.edges:
        learned_support, learned_uncertainty = _rank_support(
            learned_representations, edges.rows, edges.cols
        )
        zero_support, zero_uncertainty = _rank_support(
            zero_representations, edges.rows, edges.cols
        )
        output.append(
            RelationScaleEvidence(
                rows=np.asarray(edges.rows, dtype=np.int32),
                cols=np.asarray(edges.cols, dtype=np.int32),
                learned_factor=_factor(learned_support, learned_uncertainty, config),
                zero_factor=_factor(zero_support, zero_uncertainty, config),
                learned_support=learned_support,
                learned_uncertainty=learned_uncertainty,
            )
        )
    return tuple(output)


def _stable_stratified_permutation(
    ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    base_weights: np.ndarray,
    scale_index: int,
) -> np.ndarray:
    """Return a deterministic nonidentity bijection within base-weight quartiles."""

    ids = np.asarray(ids).astype(str)
    keys = np.empty(len(rows), dtype="S16")
    for index, (left, right) in enumerate(zip(rows, cols)):
        first, second = sorted((ids[int(left)], ids[int(right)]))
        keys[index] = hashlib.blake2b(
            f"night17e|{scale_index}|{first}|{second}".encode(), digest_size=16
        ).digest()
    boundaries = np.quantile(np.asarray(base_weights, dtype=np.float64), [0.25, 0.5, 0.75])
    strata = np.digitize(base_weights, boundaries, right=True)
    mapping = np.arange(len(rows), dtype=np.int64)
    for stratum in range(4):
        indices = np.flatnonzero(strata == stratum)
        if len(indices) <= 1:
            continue
        keyed = indices[np.argsort(keys[indices], kind="mergesort")]
        mapping[keyed] = np.roll(keyed, 1)
    return mapping


def _symmetric_factor_matrix(
    n: int, rows: np.ndarray, cols: np.ndarray, factor: np.ndarray
) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.concatenate((factor, factor)),
            (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
        ),
        shape=(n, n),
        dtype=np.float64,
    )


def relation_conditioned_pairwise(
    evidence: PreparedExpansionEvidence,
    relation: Sequence[RelationScaleEvidence],
    config: LRCCConfig,
    arm: str,
    ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    if arm not in {
        "LEARNED_RELATION",
        "ZERO_RELATION",
        "PERMUTED_RELATION",
        "UNIFORM_MASS_MATCHED",
        "RELATION_DISABLED",
    }:
        raise ValueError(f"unknown relation arm: {arm}")
    if len(relation) != len(evidence.edges):
        raise ValueError("relation/evidence graph-scale count mismatch")
    scale_weights = normalized_scale_weights(config.base)
    n = evidence.registered.binary.shape[0]
    matrices = []
    rejected = []
    diagnostics = []
    for scale_index, (edges, rel) in enumerate(zip(evidence.edges, relation)):
        lengths = {
            len(rel.rows),
            len(rel.cols),
            len(rel.learned_factor),
            len(rel.zero_factor),
            len(rel.learned_support),
            len(rel.learned_uncertainty),
        }
        if len(lengths) != 1:
            raise ValueError("relation scale array length mismatch")
        relation_pairs = list(zip(rel.rows.tolist(), rel.cols.tolist()))
        if len(set(relation_pairs)) != len(relation_pairs):
            raise ValueError("duplicate relation-scale edge")
        base_matrix, base_rejected, base_detail = _pairwise_scale(edges, n, config.base.local)
        base_upper = sp.triu(base_matrix, k=1, format="coo")
        # ScaleEdges and the upper-triangle matrix contain the same registered edges.
        if len(base_upper.data) != len(rel.rows) or set(relation_pairs) != set(
            zip(base_upper.row.tolist(), base_upper.col.tolist())
        ):
            raise RuntimeError("ScaleEdges/base-upper undirected edge set mismatch")
        edge_to_factor = {
            (int(left), int(right)): float(value)
            for left, right, value in zip(rel.rows, rel.cols, rel.learned_factor)
        }
        learned_factor = np.asarray(
            [edge_to_factor[(int(left), int(right))] for left, right in zip(base_upper.row, base_upper.col)],
            dtype=np.float64,
        )
        weighted_mass = float(np.sum(base_upper.data.astype(np.float64) * learned_factor))
        base_mass = float(np.sum(base_upper.data.astype(np.float64)))
        if not np.isfinite(base_mass) or not np.isfinite(weighted_mass) or base_mass <= 0 or weighted_mass <= 0:
            raise FloatingPointError("nonpositive/nonfinite pairwise reference mass")
        uniform = weighted_mass / max(base_mass, 1e-18)
        if arm == "LEARNED_RELATION":
            factor = learned_factor
        elif arm == "ZERO_RELATION":
            zero_map = {
                (int(left), int(right)): float(value)
                for left, right, value in zip(rel.rows, rel.cols, rel.zero_factor)
            }
            factor = np.asarray(
                [zero_map[(int(left), int(right))] for left, right in zip(base_upper.row, base_upper.col)],
                dtype=np.float64,
            )
            current_mass = float(np.sum(base_upper.data.astype(np.float64) * factor))
            if not np.isfinite(current_mass) or current_mass <= 0:
                raise FloatingPointError("nonpositive/nonfinite zero-relation mass")
            factor *= weighted_mass / max(current_mass, 1e-18)
        elif arm == "UNIFORM_MASS_MATCHED":
            factor = np.full(len(base_upper.data), uniform, dtype=np.float64)
        elif arm == "PERMUTED_RELATION":
            order = _stable_stratified_permutation(
                ids,
                base_upper.row,
                base_upper.col,
                base_upper.data.astype(np.float64),
                scale_index,
            )
            factor = learned_factor[order]
            current_mass = float(np.sum(base_upper.data.astype(np.float64) * factor))
            if not np.isfinite(current_mass) or current_mass <= 0:
                raise FloatingPointError("nonpositive/nonfinite permuted-relation mass")
            factor *= weighted_mass / max(current_mass, 1e-18)
        else:
            factor = np.ones(len(base_upper.data), dtype=np.float64)
        if np.any(factor < 0) or not np.all(np.isfinite(factor)):
            raise ValueError("relation factor must be finite and nonnegative")
        arm_mass = float(np.sum(base_upper.data.astype(np.float64) * factor))
        mass_error = abs(arm_mass - weighted_mass)
        if arm in {"ZERO_RELATION", "UNIFORM_MASS_MATCHED", "PERMUTED_RELATION"} and not np.isclose(
            arm_mass, weighted_mass, rtol=1e-11, atol=1e-12
        ):
            raise RuntimeError("matched arm failed exact base-weighted pairwise mass tolerance")
        factor_matrix = _symmetric_factor_matrix(n, base_upper.row, base_upper.col, factor)
        matrices.append(base_matrix.astype(np.float64).multiply(factor_matrix).tocsr())
        rejected.append(base_rejected)
        diagnostics.append(
            {
                **base_detail,
                "factor_mean": float(np.mean(factor)),
                "factor_min": float(np.min(factor)),
                "factor_max": float(np.max(factor)),
                "learned_weighted_mass": weighted_mass,
                "arm_weighted_mass": arm_mass,
                "mass_match_absolute_error": float(mass_error if arm in {"ZERO_RELATION", "UNIFORM_MASS_MATCHED", "PERMUTED_RELATION"} else 0.0),
                "mass_match_relative_error": float(mass_error / weighted_mass if arm in {"ZERO_RELATION", "UNIFORM_MASS_MATCHED", "PERMUTED_RELATION"} else 0.0),
                "uniform_mass_factor": uniform,
            }
        )
    combined = sum(
        float(weight) * matrix for weight, matrix in zip(scale_weights, matrices)
    ).tocsr()
    upper = sp.triu(combined, k=1, format="coo")
    values = np.maximum(upper.data.astype(np.float64), 0.0)
    combined_rejected = np.tensordot(
        scale_weights, np.stack(rejected, axis=0), axes=(0, 0)
    ).astype(np.float32)
    return (
        upper.row.astype(np.int32),
        upper.col.astype(np.int32),
        values,
        combined_rejected,
        {
            "arm": arm,
            "pair_edge_count": int(len(values)),
            "mean_pair_weight": float(np.mean(values)) if len(values) else 0.0,
            "all_pairwise_nonnegative": bool(np.all(values >= 0)),
            "scale_diagnostics": diagnostics,
            "mean_learned_support": float(
                np.mean(np.concatenate([item.learned_support for item in relation]))
            ),
            "mean_learned_uncertainty": float(
                np.mean(np.concatenate([item.learned_uncertainty for item in relation]))
            ),
        },
    )


def lrcc_expansion(
    initial: np.ndarray,
    k: int,
    ids: np.ndarray,
    evidence: PreparedExpansionEvidence,
    relation: Sequence[RelationScaleEvidence],
    config: LRCCConfig,
    arm: str,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    partition = np.asarray(initial, dtype=np.int32).copy()
    if np.unique(partition).size != int(k):
        raise ValueError("initial partition cardinality mismatch")
    rows, cols, base_weights, rejected_mass, pair_diagnostics = relation_conditioned_pairwise(
        evidence, relation, config, arm, ids
    )
    pair_weights = float(config.base.pairwise_beta) * base_weights
    cycle_ledger = []
    total_moves = 0
    accepted_expansions = 0
    unary_diagnostics: Dict[str, float] = {}
    for cycle in range(int(config.base.expansion_cycles)):
        unary, unary_diagnostics = continuous_multiscale_unary(partition, k, evidence, config.base)
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
        before_partition = partition.copy()
        cycle_moves = 0
        cycle_accepts = 0
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
                moved = int(np.sum(proposal != partition))
                partition = proposal
                cycle_moves += moved
                cycle_accepts += 1
        end_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        if end_energy > start_energy + max(float(config.base.energy_tolerance), 1e-10):
            raise RuntimeError("accepted alpha moves increased frozen-cycle energy")
        total_moves += cycle_moves
        accepted_expansions += cycle_accepts
        cycle_ledger.append(
            {
                "cycle_index": cycle,
                "start_energy": float(start_energy),
                "end_energy": float(end_energy),
                "accepted_expansions": cycle_accepts,
                "move_assignments": cycle_moves,
            }
        )
        if np.array_equal(before_partition, partition):
            break
    sizes = np.bincount(partition, minlength=int(k))
    if np.unique(partition).size != int(k) or np.any(sizes <= 0):
        raise RuntimeError("LRCC violated exact K/no-empty constraint")
    return partition, {
        **pair_diagnostics,
        **unary_diagnostics,
        "changed_observations": int(np.sum(partition != np.asarray(initial))),
        "accepted_expansions": accepted_expansions,
        "total_move_assignments": total_moves,
        "cluster_sizes": sizes.astype(int).tolist(),
        "min_cluster_size": int(np.min(sizes)),
        "cycle_energy_ledger_json": json.dumps(cycle_ledger, separators=(",", ":")),
        "each_frozen_cycle_monotone": True,
        "cross_dynamic_cycle_global_monotonicity_claimed": False,
    }
