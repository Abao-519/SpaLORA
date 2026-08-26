"""Confidence-certified self-return for sparse multiscale Potts energies.

The module adds a sufficient node-persistency leave penalty to the frozen
Night-15F energy.  A fixed input partition supplies the protected label.  A
node is eligible only when label-free three-view prototype support and a
robust unary-margin rank agree.  The certificate is recomputed in the final
solver units at every frozen-unary outer cycle.

Classic MRF persistency/partial optimality is prior art.  The implemented
research object is the cross-modal construction of the trusted set and its
connection to the existing sparse multiscale energy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.special import expit
from scipy.stats import rankdata

from SpaLORA.night15e_continuous_reliability_energy import centroid_unary
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    PreparedExpansionEvidence,
    _mixed_component_bank,
    _weighted_components,
    continuous_multiscale_pairwise,
    continuous_multiscale_unary,
    normalized_scale_weights,
    potts_energy,
)


@dataclass(frozen=True)
class CertificateConfig:
    margin_rank_quantile: float
    minimum_view_support: int
    keep_untrusted_original_self_return: bool
    epsilon_relative: float = 1.0e-6


CCSR_ARMS = (
    "CCSR_FULL",
    "CCSR_CERTIFICATE_DISABLED",
    "CCSR_RANDOM_MASK_COUNT_MATCHED",
    "CCSR_UNARY_MARGIN_ONLY",
    "CCSR_VIEW_SUPPORT_ONLY",
    "CCSR_PROTECT_ALL",
)


def _robust_scale(value: np.ndarray, floor: float = 1.0e-8) -> float:
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    if value.size == 0:
        return float(floor)
    median = float(np.median(value))
    mad = 1.4826 * float(np.median(np.abs(value - median)))
    return max(mad, 0.25 * float(np.std(value)), float(floor))


def _view_unaries(
    partition: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three Night-15F prototype costs before local mixing."""

    scale = normalized_scale_weights(config)
    banks = (
        _mixed_component_bank(
            evidence.retained_base, evidence.retained_low, evidence.transitions, scale
        ),
        _mixed_component_bank(
            evidence.view1_base, evidence.view1_low, evidence.transitions, scale
        ),
        _mixed_component_bank(
            evidence.view2_base, evidence.view2_low, evidence.transitions, scale
        ),
    )
    return tuple(
        np.asarray(
            centroid_unary(_weighted_components(bank, config.local), partition, k),
            dtype=np.float64,
        )
        for bank in banks
    )


def margin_against_target(unary: np.ndarray, target: np.ndarray) -> np.ndarray:
    unary = np.asarray(unary, dtype=np.float64)
    target = np.asarray(target, dtype=np.int32)
    alternate = unary.copy()
    alternate[np.arange(len(target)), target] = np.inf
    return np.min(alternate, axis=1) - unary[np.arange(len(target)), target]


def _percentile_rank(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("nonfinite confidence margin")
    if len(value) == 1:
        return np.ones(1, dtype=np.float64)
    return (rankdata(value, method="average") - 1.0) / (len(value) - 1.0)


def confidence_evidence(
    initial: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
) -> dict[str, np.ndarray | float]:
    initial = np.asarray(initial, dtype=np.int32)
    unary, _ = continuous_multiscale_unary(initial, k, evidence, config)
    unary = np.asarray(unary, dtype=np.float64)
    costs = _view_unaries(initial, k, evidence, config)
    support = np.column_stack(
        [np.argmin(cost, axis=1) == initial for cost in costs]
    )
    margin = margin_against_target(unary, initial)
    scale = _robust_scale(margin)
    return {
        "support_matrix": support,
        "support_count": support.sum(axis=1).astype(np.int8),
        "margin": margin,
        # This is deliberately a rank-only, dataset-scale-invariant rule.
        # Dividing by one positive global scale would leave ranks unchanged,
        # so the robust scale is retained only as a diagnostic, not described
        # as amplitude calibration.
        "margin_rank": _percentile_rank(margin),
        "margin_robust_scale": float(scale),
    }


def stable_random_mask(
    ordered_ids: Sequence[str], count: int, salt: str = "night18e-random-mask-v1"
) -> np.ndarray:
    keys = []
    for index, identifier in enumerate(np.asarray(ordered_ids).astype("U")):
        digest = hashlib.sha256(f"{salt}\0{identifier}".encode("utf-8")).digest()
        keys.append((digest, index))
    chosen = {index for _, index in sorted(keys)[: int(count)]}
    return np.asarray([index in chosen for index in range(len(keys))], dtype=bool)


def trusted_mask(
    confidence: Mapping[str, np.ndarray | float],
    config: CertificateConfig,
    arm: str,
    ordered_ids: Sequence[str],
) -> np.ndarray:
    support = np.asarray(confidence["support_count"], dtype=np.int8)
    rank = np.asarray(confidence["margin_rank"], dtype=np.float64)
    margin_ok = rank >= float(config.margin_rank_quantile)
    support_ok = support >= int(config.minimum_view_support)
    primary = margin_ok & support_ok
    if arm in {"CCSR_FULL", "CCSR_CERTIFICATE_DISABLED"}:
        return primary
    if arm == "CCSR_RANDOM_MASK_COUNT_MATCHED":
        return stable_random_mask(ordered_ids, int(primary.sum()))
    if arm == "CCSR_UNARY_MARGIN_ONLY":
        return margin_ok
    if arm == "CCSR_VIEW_SUPPORT_ONLY":
        return support_ok
    if arm == "CCSR_PROTECT_ALL":
        return np.ones(len(rank), dtype=bool)
    raise ValueError(f"unknown CCSR arm {arm}")


def incident_capacity(
    n: int, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if not (len(rows) == len(cols) == len(weights)):
        raise ValueError("edge arrays have inconsistent length")
    if np.any(rows == cols) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("invalid nonnegative Potts edge capacity")
    degree = np.bincount(np.concatenate((rows, cols)), minlength=n).astype(np.int64)
    capacity = np.bincount(
        np.concatenate((rows, cols)),
        weights=np.concatenate((weights, weights)),
        minlength=n,
    ).astype(np.float64)
    return capacity, degree


def certified_leave_penalty(
    unary: np.ndarray,
    target: np.ndarray,
    trusted: np.ndarray,
    incident: np.ndarray,
    edge_degree: np.ndarray,
    capacity_scale: float,
    epsilon_relative: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    unary = np.asarray(unary, dtype=np.float64)
    target = np.asarray(target, dtype=np.int32)
    trusted = np.asarray(trusted, dtype=bool)
    margin = margin_against_target(unary, target)
    scale = _robust_scale(
        unary - unary[np.arange(len(target)), target][:, None], floor=1.0e-8
    )
    # Each binary move rounds two terminal capacities and at most degree_i
    # incident cut terms.  A factor two is a conservative guard for source/
    # sink decomposition and makes the sufficient inequality strict after
    # integer quantization.
    rounding = 2.0 * (np.asarray(edge_degree, dtype=np.float64) + 4.0) / max(
        float(capacity_scale), 1.0
    )
    epsilon = max(float(epsilon_relative), 0.0) * scale + rounding + 1.0e-12
    gap = np.maximum(0.0, np.asarray(incident) - margin + epsilon)
    gap[~trusted] = 0.0
    output = unary.copy()
    output += gap[:, None]
    output[np.arange(len(target)), target] -= gap
    slack = margin + gap - np.asarray(incident)
    if np.any(slack[trusted] <= 0):
        raise RuntimeError("certificate strict slack is not positive")
    return output, {
        "dynamic_margin": margin,
        "incident_capacity": np.asarray(incident, dtype=np.float64),
        "certificate_gap": gap,
        "epsilon": epsilon,
        "certificate_slack": slack,
        "unary_robust_scale": float(scale),
    }


def quantize_nonnegative_capacity(value: np.ndarray, capacity_scale: float) -> np.ndarray:
    """The single integer-capacity map used by the formal CCSR cut and tests."""

    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)) or np.any(value < -1.0e-12):
        raise ValueError("capacities must be finite and nonnegative")
    scale = max(float(capacity_scale), 1.0)
    return np.rint(np.maximum(value, 0.0) * scale).astype(np.int64)


def quantized_binary_terms(
    cost0: np.ndarray,
    cost1: np.ndarray,
    cut_weights: np.ndarray,
    capacity_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact integer terms minimized by the formal binary cut."""

    cost0 = np.asarray(cost0, dtype=np.float64).copy()
    cost1 = np.asarray(cost1, dtype=np.float64).copy()
    shift = np.minimum(cost0, cost1)
    cost0 -= shift
    cost1 -= shift
    return (
        quantize_nonnegative_capacity(cost0, capacity_scale),
        quantize_nonnegative_capacity(cost1, capacity_scale),
        quantize_nonnegative_capacity(cut_weights, capacity_scale),
    )


def ccsr_binary_cut(
    cost0: np.ndarray,
    cost1: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    cut_weights: np.ndarray,
    capacity_scale: float,
) -> np.ndarray:
    """Return x=1 nodes using the exact quantizer exposed to tiny tests."""

    sink_caps, source_caps, edge_caps = quantized_binary_terms(
        cost0, cost1, cut_weights, capacity_scale
    )
    n = len(sink_caps)
    source = n
    sink = n + 1
    keep_edges = edge_caps > 0
    edge_rows = np.asarray(rows, dtype=np.int64)[keep_edges]
    edge_cols = np.asarray(cols, dtype=np.int64)[keep_edges]
    edge_caps = edge_caps[keep_edges]
    graph_rows = np.concatenate(
        (
            np.full(np.count_nonzero(source_caps), source, dtype=np.int64),
            np.flatnonzero(sink_caps).astype(np.int64),
            edge_rows,
            edge_cols,
        )
    )
    graph_cols = np.concatenate(
        (
            np.flatnonzero(source_caps).astype(np.int64),
            np.full(np.count_nonzero(sink_caps), sink, dtype=np.int64),
            edge_cols,
            edge_rows,
        )
    )
    graph_data = np.concatenate(
        (source_caps[source_caps > 0], sink_caps[sink_caps > 0], edge_caps, edge_caps)
    )
    capacity = sp.csr_matrix(
        (graph_data, (graph_rows, graph_cols)), shape=(n + 2, n + 2), dtype=np.int64
    )
    result = maximum_flow(capacity, source, sink, method="dinic")
    flow = result.flow if hasattr(result, "flow") else result.residual
    residual = (capacity - flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = breadth_first_order(
        residual, source, directed=True, return_predecessors=False
    )
    source_side = np.zeros(n + 2, dtype=bool)
    source_side[np.asarray(reachable, dtype=np.int64)] = True
    return ~source_side[:n]


def ccsr_alpha_expansion_move(
    partition: np.ndarray,
    alpha: int,
    unary: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    pair_weights: np.ndarray,
    capacity_scale: float,
    energy_tolerance: float,
) -> tuple[np.ndarray, bool, float, float]:
    """Night-15F alpha move with an exposed, test-identical quantizer."""

    partition = np.asarray(partition, dtype=np.int32)
    before = potts_energy(partition, unary, rows, cols, pair_weights)
    left = partition[rows]
    right = partition[cols]
    e00 = pair_weights * (left != right)
    e01 = pair_weights * (left != int(alpha))
    e10 = pair_weights * (int(alpha) != right)
    cut = 0.5 * (e01 + e10 - e00)
    if np.any(cut < -1.0e-10):
        raise ValueError("non-submodular expansion move")
    cut = np.maximum(cut, 0.0)
    adjust1 = np.zeros(len(partition), dtype=np.float64)
    np.add.at(adjust1, rows, e10 - e00 - cut)
    np.add.at(adjust1, cols, e01 - e00 - cut)
    cost0 = np.asarray(unary[np.arange(len(partition)), partition], dtype=np.float64)
    cost1 = np.asarray(unary[:, int(alpha)], dtype=np.float64) + adjust1
    switch = ccsr_binary_cut(cost0, cost1, rows, cols, cut, capacity_scale)
    proposal = partition.copy()
    proposal[switch] = int(alpha)
    if len(np.unique(proposal)) != unary.shape[1]:
        return partition.copy(), False, before, before
    after = potts_energy(proposal, unary, rows, cols, pair_weights)
    accepted = after < before - float(energy_tolerance)
    return (proposal if accepted else partition.copy()), accepted, before, after


def exhaustive_quantized_alpha_check(
    current: np.ndarray,
    target: np.ndarray,
    unary: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    pair_weights: np.ndarray,
    trusted: np.ndarray,
    capacity_scale: float,
) -> dict[str, object]:
    """Enumerate every quantized binary alpha subproblem and test persistency."""

    current = np.asarray(current, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)
    trusted = np.asarray(trusted, dtype=bool)
    if np.any(current[trusted] != target[trusted]):
        raise ValueError("induction premise failed before alpha cycle")
    k = unary.shape[1]
    enumerated = 0
    for alpha in range(k):
        left = current[rows]
        right = current[cols]
        e00 = pair_weights * (left != right)
        e01 = pair_weights * (left != alpha)
        e10 = pair_weights * (alpha != right)
        cut = np.maximum(0.5 * (e01 + e10 - e00), 0.0)
        adjust1 = np.zeros(len(current), dtype=np.float64)
        np.add.at(adjust1, rows, e10 - e00 - cut)
        np.add.at(adjust1, cols, e01 - e00 - cut)
        cost0 = unary[np.arange(len(current)), current].astype(np.float64)
        cost1 = unary[:, alpha].astype(np.float64) + adjust1
        sink_caps, source_caps, edge_caps = quantized_binary_terms(
            cost0, cost1, cut, capacity_scale
        )
        energies = []
        assignments = []
        for bits in itertools.product((0, 1), repeat=len(current)):
            bits = np.asarray(bits, dtype=np.int8)
            value = int(np.sum(np.where(bits == 0, sink_caps, source_caps)))
            value += int(np.sum(edge_caps * (bits[rows] != bits[cols])))
            energies.append(value)
            assignments.append(bits)
        optimum = min(energies)
        for bits, value in zip(assignments, energies):
            if value != optimum:
                continue
            proposal = current.copy()
            proposal[bits.astype(bool)] = alpha
            if np.any(proposal[trusted] != target[trusted]):
                raise AssertionError("quantized alpha minimizer changes a trusted node")
        cut_solution = ccsr_binary_cut(cost0, cost1, rows, cols, cut, capacity_scale)
        cut_energy = int(
            np.sum(np.where(cut_solution == 0, sink_caps, source_caps))
            + np.sum(edge_caps * (cut_solution[rows] != cut_solution[cols]))
        )
        if cut_energy != optimum:
            raise AssertionError("formal min-cut integer energy differs from exhaustive optimum")
        proposal = current.copy()
        proposal[cut_solution] = alpha
        if np.any(proposal[trusted] != target[trusted]):
            raise AssertionError("formal min-cut changes a trusted node")
        enumerated += 2 ** len(current)
    return {
        "n": int(len(current)),
        "k": int(k),
        "enumerated_binary_states": int(enumerated),
        "formal_cut_integer_energy_matches_exhaustive": True,
        "status": "PASS",
    }


def _base_solver_unary(
    current: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    config: ExpansionEnergyConfig,
    rejected_mass: np.ndarray,
    trusted: np.ndarray,
    keep_untrusted_self_return: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    raw, diagnostics = continuous_multiscale_unary(current, k, evidence, config)
    raw = np.asarray(raw, dtype=np.float64)
    current = np.asarray(current, dtype=np.int32)
    margin = margin_against_target(raw, current)
    trust = expit(
        (margin - float(config.local.trust_center))
        / max(float(config.local.trust_temperature), 1.0e-4)
    ).astype(np.float64)
    trust_cost = float(config.local.trust_scale) * trust
    value = raw + trust_cost[:, None]
    value[np.arange(len(current)), current] -= trust_cost
    old_stay = np.zeros(len(current), dtype=np.float64)
    if keep_untrusted_self_return:
        old_stay = (
            float(config.self_return_strength)
            * np.asarray(rejected_mass, dtype=np.float64)
            * (~np.asarray(trusted, dtype=bool))
        )
        value += old_stay[:, None]
        value[np.arange(len(current)), current] -= old_stay
    return value, {
        **diagnostics,
        "mean_base_trust_cost": float(np.mean(trust_cost)),
        "mean_untrusted_original_stay": float(np.mean(old_stay)),
    }


def ccsr_expansion(
    initial: np.ndarray,
    ordered_ids: Sequence[str],
    k: int,
    evidence: PreparedExpansionEvidence,
    energy_config: ExpansionEnergyConfig,
    certificate_config: CertificateConfig,
    arm: str = "CCSR_FULL",
) -> tuple[np.ndarray, dict[str, object]]:
    if arm not in CCSR_ARMS:
        raise ValueError(f"unknown arm {arm}")
    initial = np.asarray(initial, dtype=np.int32)
    ordered_ids = np.asarray(ordered_ids).astype("U")
    if len(initial) != len(ordered_ids) or len(np.unique(initial)) != int(k):
        raise ValueError("initial partition/ID/exact-K contract invalid")
    evidence_confidence = confidence_evidence(initial, k, evidence, energy_config)
    trusted = trusted_mask(evidence_confidence, certificate_config, arm, ordered_ids)
    rows, cols, base_weights, rejected_mass, pair_diagnostics = (
        continuous_multiscale_pairwise(evidence, energy_config)
    )
    pair_weights = float(energy_config.pairwise_beta) * np.asarray(
        base_weights, dtype=np.float64
    )
    incident, edge_degree = incident_capacity(len(initial), rows, cols, pair_weights)
    partition = initial.copy()
    cycle_ledger: list[dict[str, object]] = []
    total_moves = 0
    accepted_expansions = 0
    final_certificate: dict[str, np.ndarray | float] = {}
    last_unary_diagnostics: dict[str, float] = {}
    certificate_applied = arm != "CCSR_CERTIFICATE_DISABLED"

    for cycle in range(int(energy_config.expansion_cycles)):
        base_unary, last_unary_diagnostics = _base_solver_unary(
            partition,
            k,
            evidence,
            energy_config,
            rejected_mass,
            trusted,
            certificate_config.keep_untrusted_original_self_return,
        )
        if certificate_applied:
            solver_unary, final_certificate = certified_leave_penalty(
                base_unary,
                initial,
                trusted,
                incident,
                edge_degree,
                energy_config.capacity_scale,
                certificate_config.epsilon_relative,
            )
        else:
            solver_unary = base_unary
            final_certificate = {
                "dynamic_margin": margin_against_target(base_unary, initial),
                "incident_capacity": incident,
                "certificate_gap": np.zeros(len(initial), dtype=np.float64),
                "epsilon": np.zeros(len(initial), dtype=np.float64),
                "certificate_slack": margin_against_target(base_unary, initial) - incident,
                "unary_robust_scale": _robust_scale(base_unary),
            }
        start_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        cycle_moves = 0
        cycle_accepted = 0
        for alpha in range(int(k)):
            proposal, accepted, _, _ = ccsr_alpha_expansion_move(
                partition,
                alpha,
                solver_unary,
                rows,
                cols,
                pair_weights,
                energy_config.capacity_scale,
                energy_config.energy_tolerance,
            )
            if accepted:
                changed = int(np.sum(proposal != partition))
                partition = proposal
                cycle_moves += changed
                cycle_accepted += 1
                if certificate_applied and np.any(partition[trusted] != initial[trusted]):
                    raise RuntimeError("a certified node changed in an accepted alpha move")
        end_energy = potts_energy(partition, solver_unary, rows, cols, pair_weights)
        if end_energy > start_energy + max(float(energy_config.energy_tolerance), 1.0e-10):
            raise RuntimeError("frozen-cycle energy increased")
        total_moves += cycle_moves
        accepted_expansions += cycle_accepted
        cycle_ledger.append(
            {
                "cycle_index": int(cycle),
                "start_energy": float(start_energy),
                "end_energy": float(end_energy),
                "accepted_expansions": int(cycle_accepted),
                "move_assignments": int(cycle_moves),
                "certified_changed_count": int(np.sum(partition[trusted] != initial[trusted])),
            }
        )
        if cycle_moves == 0:
            break

    sizes = np.bincount(partition, minlength=int(k))
    if len(np.unique(partition)) != int(k) or np.any(sizes <= 0):
        raise RuntimeError("CCSR endpoint violated exact K/no-empty")
    certified_changed = int(np.sum(partition[trusted] != initial[trusted]))
    if certificate_applied and certified_changed:
        raise RuntimeError("certified_changed_count must be zero")

    def summary(prefix: str, value: np.ndarray) -> dict[str, float]:
        value = np.asarray(value, dtype=np.float64)
        return {
            f"{prefix}_min": float(np.min(value)),
            f"{prefix}_median": float(np.median(value)),
            f"{prefix}_mean": float(np.mean(value)),
            f"{prefix}_max": float(np.max(value)),
        }

    diagnostics: dict[str, object] = {
        **pair_diagnostics,
        **last_unary_diagnostics,
        **summary("static_margin", np.asarray(evidence_confidence["margin"])),
        **summary("incident_capacity", np.asarray(final_certificate["incident_capacity"])),
        **summary("certificate_gap", np.asarray(final_certificate["certificate_gap"])),
        **summary("certificate_epsilon", np.asarray(final_certificate["epsilon"])),
        "arm": arm,
        "certificate_applied": int(certificate_applied),
        "margin_rank_quantile": float(certificate_config.margin_rank_quantile),
        "minimum_view_support": int(certificate_config.minimum_view_support),
        "keep_untrusted_original_self_return": int(
            certificate_config.keep_untrusted_original_self_return
        ),
        "trusted_count": int(trusted.sum()),
        "trusted_fraction": float(np.mean(trusted)),
        "view_support_3_count": int(
            np.sum(np.asarray(evidence_confidence["support_count"]) == 3)
        ),
        "view_support_at_least_2_count": int(
            np.sum(np.asarray(evidence_confidence["support_count"]) >= 2)
        ),
        "certified_changed_count": certified_changed,
        "noncertified_changed_count": int(
            np.sum(partition[~trusted] != initial[~trusted])
        ),
        "changed_observations": int(np.sum(partition != initial)),
        "accepted_expansions": int(accepted_expansions),
        "total_move_assignments": int(total_moves),
        "observed_cardinality": int(len(np.unique(partition))),
        "min_cluster_size": int(np.min(sizes)),
        "cluster_sizes": [int(x) for x in sizes],
        "cycle_energy_ledger_json": json.dumps(cycle_ledger, separators=(",", ":")),
        "each_frozen_cycle_monotone": 1,
        "cross_dynamic_cycle_global_monotonicity_claimed": 0,
        "final_potts_capacity_includes_pairwise_beta": 1,
    }
    return partition, diagnostics


def exhaustive_certificate_check(
    initial: np.ndarray,
    unary: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    trusted: np.ndarray,
    capacity_scale: float = 1.0e6,
) -> dict[str, object]:
    """Exhaust all K**N states and verify the sufficient persistency claim."""

    initial = np.asarray(initial, dtype=np.int32)
    unary = np.asarray(unary, dtype=np.float64)
    trusted = np.asarray(trusted, dtype=bool)
    incident, degree = incident_capacity(len(initial), rows, cols, weights)
    certified, detail = certified_leave_penalty(
        unary, initial, trusted, incident, degree, capacity_scale, 1.0e-6
    )
    k = unary.shape[1]
    energies = []
    states = []
    for labels in itertools.product(range(k), repeat=len(initial)):
        labels = np.asarray(labels, dtype=np.int32)
        states.append(labels)
        energies.append(potts_energy(labels, certified, rows, cols, weights))
    energies = np.asarray(energies, dtype=np.float64)
    optimum = float(np.min(energies))
    # ``rtol`` must be zero: the deliberately tiny strict certificate slack
    # is meaningful in solver units and must not cause near-optima to be
    # misclassified as exact minimizers merely because the total energy is
    # large.
    minimizers = [
        states[i]
        for i in np.flatnonzero(np.isclose(energies, optimum, atol=1e-10, rtol=0.0))
    ]
    if any(np.any(labels[trusted] != initial[trusted]) for labels in minimizers):
        raise AssertionError("exhaustive optimum violates certified persistency")
    return {
        "n": int(len(initial)),
        "k": int(k),
        "enumerated_states": int(k ** len(initial)),
        "minimizer_count": int(len(minimizers)),
        "trusted_count": int(trusted.sum()),
        "minimum_certificate_slack": float(
            np.min(np.asarray(detail["certificate_slack"])[trusted])
            if np.any(trusted)
            else np.nan
        ),
        "status": "PASS",
    }
