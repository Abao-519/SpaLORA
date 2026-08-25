"""Direct feasible-candidate relation posterior for a sparse Potts cut.

The producer-facing APIs in this file never accept public annotations.  They
consume the locked Night-16H candidate partitions plus pre-evaluation evidence
and place their soft co-clustering posterior directly on registered graph
edges.  No node embedding is trained or used as an information bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from SpaLORA.night17e_lrcc import (
    LRCCConfig,
    RelationScaleEvidence,
    lrcc_expansion,
)
from SpaLORA.night15f_multiscale_expansion import PreparedExpansionEvidence


PRIMARY_ARM = "DIRECT_WEIGHTED_POSTERIOR"
ANALYTIC_ARM = "ANALYTIC_UNWEIGHTED_POSTERIOR"
PERMUTED_ARM = "PERMUTED_POSTERIOR"
UNIFORM_ARM = "UNIFORM_MASS_MATCHED"
DISABLED_ARM = "RELATION_DISABLED"


@dataclass(frozen=True)
class PosteriorDiagnostic:
    probability_same: np.ndarray
    uncertainty: np.ndarray
    candidate_weights: np.ndarray
    selected_candidate_count: int


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1.0)


def candidate_mask(records: Sequence[Mapping[str, object]], bank_mode: str) -> np.ndarray:
    feasible = np.asarray(
        [str(row["feasible_SMALLEST_SCALE_INTERNAL_EDGE"]).lower() == "true" for row in records],
        dtype=bool,
    )
    if bank_mode == "FULL_BANK":
        return feasible
    if bank_mode != "UNBIASED_BANK":
        raise ValueError(f"unknown bank mode: {bank_mode}")
    unbiased = np.asarray(
        [
            str(row["candidate_id"]).startswith("KMEANS_RETAINED_")
            or str(row["candidate_id"]).startswith("PATH_UNIFORM__KMEANS_RETAINED_")
            for row in records
        ],
        dtype=bool,
    )
    return feasible & unbiased


def direct_relation_posterior(
    partitions: np.ndarray,
    records: Sequence[Mapping[str, object]],
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    bank_mode: str,
    weighted: bool,
) -> PosteriorDiagnostic:
    partitions = np.asarray(partitions, dtype=np.int32)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    if partitions.ndim != 2 or partitions.shape[0] != len(records):
        raise ValueError("candidate partition/record count mismatch")
    if rows.shape != cols.shape or np.any(rows >= partitions.shape[1]) or np.any(cols >= partitions.shape[1]):
        raise ValueError("relation edge index mismatch")
    indices = np.flatnonzero(candidate_mask(records, bank_mode))
    if indices.size < 2:
        raise ValueError(f"{bank_mode} has fewer than two feasible candidates")
    if weighted:
        axes = np.column_stack(
            [
                [float(records[index]["molecular_joint"]) for index in indices],
                [float(records[index]["topology_joint"]) for index in indices],
                [float(records[index]["persistence"]) for index in indices],
            ]
        )
        evidence = np.mean(
            np.column_stack([_percentile(axes[:, column]) for column in range(3)]), axis=1
        )
        logits = (evidence - evidence.max()) / 0.35
        weights = np.exp(logits)
        weights /= weights.sum()
    else:
        weights = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
    same = partitions[indices][:, rows] == partitions[indices][:, cols]
    probability = np.asarray(weights @ same.astype(np.float64), dtype=np.float64)
    eps = 1e-12
    entropy = -(
        probability * np.log(probability + eps)
        + (1.0 - probability) * np.log(1.0 - probability + eps)
    )
    uncertainty = np.clip(entropy / np.log(2.0), 0.0, 1.0)
    all_weights = np.zeros(len(records), dtype=np.float64)
    all_weights[indices] = weights
    return PosteriorDiagnostic(
        probability_same=probability.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        candidate_weights=all_weights,
        selected_candidate_count=int(indices.size),
    )


def posterior_factor(
    posterior: PosteriorDiagnostic, config: LRCCConfig
) -> np.ndarray:
    mix = float(np.clip(config.relation_mix, 0.0, 1.0))
    floor = float(np.clip(config.relation_floor, 1e-6, 1.0))
    confidence = np.clip(
        1.0 - float(config.uncertainty_scale) * posterior.uncertainty, 0.0, 1.0
    )
    evidence = np.clip(posterior.probability_same * confidence, 0.0, 1.0) ** max(
        float(config.relation_power), 1e-4
    )
    return np.asarray(
        (1.0 - mix) + mix * (floor + (1.0 - floor) * evidence), dtype=np.float64
    )


def prepare_direct_relation_evidence(
    evidence: PreparedExpansionEvidence,
    candidate_partitions: np.ndarray,
    records: Sequence[Mapping[str, object]],
    config: LRCCConfig,
    *,
    bank_mode: str = "UNBIASED_BANK",
) -> Tuple[Tuple[RelationScaleEvidence, ...], Tuple[Mapping[str, object], ...]]:
    if candidate_partitions.shape[1] != evidence.registered.binary.shape[0]:
        raise ValueError("candidate/evidence observation mismatch")
    output = []
    diagnostics = []
    for scale_index, edges in enumerate(evidence.edges):
        weighted = direct_relation_posterior(
            candidate_partitions,
            records,
            edges.rows,
            edges.cols,
            bank_mode=bank_mode,
            weighted=True,
        )
        analytic = direct_relation_posterior(
            candidate_partitions,
            records,
            edges.rows,
            edges.cols,
            bank_mode=bank_mode,
            weighted=False,
        )
        output.append(
            RelationScaleEvidence(
                rows=np.asarray(edges.rows, dtype=np.int32),
                cols=np.asarray(edges.cols, dtype=np.int32),
                learned_factor=posterior_factor(weighted, config),
                zero_factor=posterior_factor(analytic, config),
                learned_support=np.asarray(weighted.probability_same, dtype=np.float32),
                learned_uncertainty=np.asarray(weighted.uncertainty, dtype=np.float32),
            )
        )
        diagnostics.append(
            {
                "scale_index": scale_index,
                "edge_count": int(len(edges.rows)),
                "weighted_selected_candidate_count": weighted.selected_candidate_count,
                "analytic_selected_candidate_count": analytic.selected_candidate_count,
                "weighted_probability_mean": float(np.mean(weighted.probability_same)),
                "weighted_uncertainty_mean": float(np.mean(weighted.uncertainty)),
                "weighted_candidate_weight_sha_input": weighted.candidate_weights.tolist(),
            }
        )
    return tuple(output), tuple(diagnostics)


def direct_relation_cut(
    initial: np.ndarray,
    k: int,
    ids: np.ndarray,
    evidence: PreparedExpansionEvidence,
    relation: Sequence[RelationScaleEvidence],
    config: LRCCConfig,
    arm: str,
):
    arm_map = {
        PRIMARY_ARM: "LEARNED_RELATION",
        ANALYTIC_ARM: "ZERO_RELATION",
        PERMUTED_ARM: "PERMUTED_RELATION",
        UNIFORM_ARM: "UNIFORM_MASS_MATCHED",
        DISABLED_ARM: "RELATION_DISABLED",
    }
    if arm not in arm_map:
        raise ValueError(f"unknown direct-relation arm: {arm}")
    return lrcc_expansion(initial, k, ids, evidence, relation, config, arm_map[arm])

