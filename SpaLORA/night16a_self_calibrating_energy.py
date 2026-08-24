"""Score-driven self-calibrating sparse spatial energy.

The module consumes numeric views, sparse graph scales, a start bank and K.
It computes all energy controls from observable statistics plus a compact set
of family-level calibration constants.  Dataset identifiers and reference
assignments are not accepted by the core API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.metrics import adjusted_rand_score

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    PreparedExpansionEvidence,
    continuous_multiscale_expansion,
    prepare_expansion_evidence,
)
from SpaLORA.night15g_optional_morphology_energy import (
    OptionalMorphologyConfig,
    optional_morphology_expansion,
    prepare_optional_morphology_evidence,
)


@dataclass(frozen=True)
class CalibrationConstants:
    """Seven interpretable controls shared at global or family level."""

    pairwise_factor: float = 8.0
    self_return_factor: float = 4.0
    size_factor: float = 0.15
    retained_prior: float = 1.0
    optional_factor: float = 0.6
    scale_temperature: float = 0.25
    expansion_cycles: int = 2


@dataclass(frozen=True)
class CalibratedEnergy:
    config: ExpansionEnergyConfig
    optional_config: OptionalMorphologyConfig
    statistics: Mapping[str, float]
    parameter_source: str = "observable_statistics_plus_frozen_family_constants"


def _robust_scale(value: np.ndarray, floor: float = 1e-3) -> float:
    value = np.asarray(value, dtype=np.float64)
    median = float(np.median(value))
    mad = 1.4826 * float(np.median(np.abs(value - median)))
    return max(mad, float(np.std(value)) * 0.25, float(floor))


def _effective_rank(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    value = value - value.mean(axis=0, keepdims=True)
    covariance = value.T @ value / max(len(value) - 1, 1)
    eigen = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    if float(eigen.sum()) <= 0:
        return 1.0
    probability = eigen / eigen.sum()
    positive = probability[probability > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.std(first) <= 1e-10 or np.std(second) <= 1e-10:
        return 0.0
    return float(np.clip(np.corrcoef(first, second)[0, 1], -1.0, 1.0))


def _start_stability(starts: Sequence[np.ndarray]) -> float:
    values = []
    for left in range(len(starts)):
        for right in range(left + 1, len(starts)):
            values.append(adjusted_rand_score(starts[left], starts[right]))
    return float(np.median(values)) if values else 1.0


def _optional_edge_reliability(
    evidence: PreparedExpansionEvidence,
    optional_view: np.ndarray | None,
) -> tuple[float, float]:
    if optional_view is None:
        return 0.0, 0.0
    value = np.asarray(optional_view, dtype=np.float32)
    edges = evidence.edges[1]
    rows, cols = edges.rows, edges.cols
    distance = np.mean((value[rows] - value[cols]) ** 2, axis=1)
    positive = distance[distance > 0]
    bandwidth = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    similarity = np.exp(-distance / bandwidth)
    molecular = np.sqrt(np.maximum(edges.similarity1 * edges.similarity2, 0.0))
    correlation = _safe_corr(similarity, molecular)
    # ``similarity`` is scaled by its own median distance.  Its median is
    # therefore approximately exp(-1) for every non-degenerate lane and is
    # not an informative calibration statistic.  The mean and upper-tail
    # persistence retain distributional information after the robust
    # bandwidth normalization.
    coherence = float(np.mean(similarity))
    persistence = float(np.quantile(similarity, 0.75) - np.quantile(similarity, 0.25))
    reliability = float(
        np.clip(0.5 * (correlation + 1.0) * (coherence + 0.25 * persistence), 0.0, 1.0)
    )
    return reliability, correlation


def observable_statistics(
    evidence: PreparedExpansionEvidence,
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    starts: Sequence[np.ndarray],
    k: int,
    optional_view: np.ndarray | None = None,
) -> dict[str, float | list[float]]:
    conflicts = []
    scale_scores = []
    correlations = []
    overlaps = []
    coherence1 = []
    coherence2 = []
    persistence1 = []
    persistence2 = []
    for edges in evidence.edges:
        first = np.asarray(edges.similarity1, dtype=np.float64)
        second = np.asarray(edges.similarity2, dtype=np.float64)
        conflict = np.abs(first - second)
        conflicts.append(conflict)
        correlations.append(_safe_corr(first, second))
        coherence1.append(float(np.mean(first)))
        coherence2.append(float(np.mean(second)))
        persistence1.append(float(np.quantile(first, 0.75) - np.quantile(first, 0.25)))
        persistence2.append(float(np.quantile(second, 0.75) - np.quantile(second, 0.25)))
        threshold1, threshold2 = np.median(first), np.median(second)
        high1, high2 = first >= threshold1, second >= threshold2
        overlap = float(np.sum(high1 & high2) / max(np.sum(high1 | high2), 1))
        overlaps.append(overlap)
        scale_scores.append(
            float(np.sqrt(max(np.mean(first) * np.mean(second), 0.0)) * (1.0 - np.median(conflict)))
        )
    registered_conflict = conflicts[1]
    optional_reliability, optional_correlation = _optional_edge_reliability(evidence, optional_view)
    return {
        "n": float(len(retained)),
        "k": float(k),
        "observations_per_cluster": float(len(retained) / max(k, 1)),
        "retained_effective_rank": _effective_rank(retained),
        "view1_effective_rank": _effective_rank(view1),
        "view2_effective_rank": _effective_rank(view2),
        "start_stability_median_ari": _start_stability(starts),
        "registered_conflict_median": float(np.median(registered_conflict)),
        "registered_conflict_scale": _robust_scale(registered_conflict, 0.025),
        "registered_similarity_correlation": correlations[1],
        "registered_neighbor_overlap": overlaps[1],
        "view1_edge_coherence": coherence1[1],
        "view2_edge_coherence": coherence2[1],
        "view1_edge_persistence": persistence1[1],
        "view2_edge_persistence": persistence2[1],
        "view1_zero_fraction": float(np.mean(np.asarray(view1) == 0)),
        "view2_zero_fraction": float(np.mean(np.asarray(view2) == 0)),
        "scale_scores": scale_scores,
        "scale_correlations": correlations,
        "scale_neighbor_overlaps": overlaps,
        "optional_reliability": optional_reliability,
        "optional_molecular_edge_correlation": optional_correlation,
    }


def calibrate_from_statistics(
    statistics: Mapping[str, float | list[float]],
    constants: CalibrationConstants,
) -> CalibratedEnergy:
    conflict = float(statistics["registered_conflict_median"])
    conflict_scale = float(statistics["registered_conflict_scale"])
    overlap = float(statistics["registered_neighbor_overlap"])
    correlation = float(statistics["registered_similarity_correlation"])
    stability = float(statistics["start_stability_median_ari"])
    coherence1 = float(statistics["view1_edge_coherence"])
    coherence2 = float(statistics["view2_edge_coherence"])
    observations_per_cluster = float(statistics["observations_per_cluster"])
    scale_scores = np.asarray(statistics["scale_scores"], dtype=np.float64)
    scale = softmax(scale_scores / max(float(constants.scale_temperature), 1e-4))

    complementarity = float(np.clip(0.5 * (1.0 - correlation), 0.0, 1.0))
    spatial_coherence = float(np.clip(0.5 * (coherence1 + coherence2), 0.0, 1.0))
    edge_floor = float(np.clip(0.02 + 0.10 * (1.0 - overlap), 0.002, 0.15))
    capacity = float(np.clip(0.30 + 0.60 * overlap, 0.25, 0.90))
    mass_center = float(np.clip(spatial_coherence - 0.25 * conflict, -0.25, 0.75))
    mass_temperature = float(np.clip(conflict_scale + 0.05, 0.05, 0.40))
    start_uncertainty = float(np.clip(1.0 - stability, 0.0, 1.5))
    unary_temperature = float(np.clip(0.20 + 0.35 * start_uncertainty, 0.15, 0.80))
    rank_ratio = float(statistics["retained_effective_rank"]) / max(
        np.sqrt(float(statistics["view1_effective_rank"]) * float(statistics["view2_effective_rank"])), 1e-6
    )
    retained_bias = float(np.log(max(float(constants.retained_prior) * rank_ratio, 1e-4)))
    view_balance = 0.5 * float(np.log(max(coherence1, 1e-4) / max(coherence2, 1e-4)))
    graph_factor = 0.75 + overlap + 0.15 * np.log1p(observations_per_cluster)
    pairwise_beta = float(max(0.0, constants.pairwise_factor) * graph_factor)
    self_return = float(max(0.0, constants.self_return_factor) * (0.5 + 1.0 - overlap))
    size_prior = float(max(0.0, constants.size_factor) * start_uncertainty)
    local = ContinuousEnergyConfig(
        beta=1.0,
        edge_floor=edge_floor,
        conflict_center=conflict,
        conflict_temperature=conflict_scale,
        conflict_union_weight=complementarity,
        conflict_penalty=conflict,
        mass_center=mass_center,
        mass_temperature=mass_temperature,
        neighbor_capacity=capacity,
        low_weight=0.20 + 1.20 * spatial_coherence,
        twohop_weight=0.10 + 0.70 * spatial_coherence,
        high_weight=0.10 + 0.90 * (1.0 - spatial_coherence),
        unary_temperature=unary_temperature,
        retained_bias=retained_bias,
        view_balance=view_balance,
        trust_scale=0.50 + 1.50 * float(np.clip(stability, 0.0, 1.0)),
        trust_center=0.20 + 0.20 * start_uncertainty,
        trust_temperature=0.15 + 0.10 * start_uncertainty,
        move_threshold=0.0,
        move_fraction=1.0,
        sweeps=1,
    )
    config = ExpansionEnergyConfig(
        local=local,
        scale_fine=float(scale[0]),
        scale_registered=float(scale[1]),
        scale_broad=float(scale[2]),
        pairwise_beta=pairwise_beta,
        self_return_strength=self_return,
        size_prior=size_prior,
        expansion_cycles=int(constants.expansion_cycles),
        capacity_scale=1_000_000.0,
        energy_tolerance=1e-9,
    )
    optional_reliability = float(statistics["optional_reliability"])
    optional_weight = float(max(0.0, constants.optional_factor) * optional_reliability)
    optional = OptionalMorphologyConfig(
        base=config,
        morphology_unary_weight=optional_weight,
        morphology_edge_weight=0.5 * optional_weight,
        reliability_mix=1.0,
        reliability_bias=0.0,
        reliability_consistency_weight=1.0,
        reliability_conflict_weight=1.0,
        reliability_margin_weight=0.5,
        reliability_temperature=max(0.10, conflict_scale),
    )
    parameters = {
        **{key: value for key, value in statistics.items()},
        "calibration_constants": asdict(constants),
        "calibrated_scale_weights": scale.tolist(),
        "calibrated_pairwise_beta": pairwise_beta,
        "calibrated_self_return_strength": self_return,
        "calibrated_size_prior": size_prior,
        "calibrated_optional_weight": optional_weight,
    }
    return CalibratedEnergy(config=config, optional_config=optional, statistics=parameters)


def prepare_and_calibrate(
    graphs: Sequence[sp.spmatrix],
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    starts: Sequence[np.ndarray],
    k: int,
    constants: CalibrationConstants,
    optional_view: np.ndarray | None = None,
) -> tuple[PreparedExpansionEvidence, CalibratedEnergy]:
    evidence = prepare_expansion_evidence(graphs, retained, view1, view2)
    statistics = observable_statistics(evidence, retained, view1, view2, starts, k, optional_view)
    return evidence, calibrate_from_statistics(statistics, constants)


def run_calibrated_energy(
    initial: np.ndarray,
    k: int,
    evidence: PreparedExpansionEvidence,
    calibration: CalibratedEnergy,
    graphs: Sequence[sp.spmatrix],
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    optional_view: np.ndarray | None = None,
    optional_presence: np.ndarray | None = None,
    optional_evidence=None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    molecular, molecular_diagnostics = continuous_multiscale_expansion(
        initial, k, evidence, calibration.config
    )
    if optional_view is None:
        return molecular, {
            **molecular_diagnostics,
            "optional_view_present": 0.0,
            "optional_missing_exact_molecular_fallback": 1.0,
        }
    presence = (
        np.ones(len(molecular), dtype=np.float32)
        if optional_presence is None
        else np.asarray(optional_presence, dtype=np.float32)
    )
    if optional_evidence is None:
        optional_evidence = prepare_optional_morphology_evidence(
            graphs,
            retained,
            (view1, view2),
            (optional_view,),
            (presence,),
        )
    final, optional_diagnostics = optional_morphology_expansion(
        molecular, k, optional_evidence, calibration.optional_config
    )
    return final, {
        **{f"molecular_{key}": value for key, value in molecular_diagnostics.items()},
        **optional_diagnostics,
    }
