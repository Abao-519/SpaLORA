"""Label-free learned evidence for selecting locked feasible partitions.

Night-17D never changes the locked candidate partitions.  It scores each
partition against three frozen Night-17C representations and a relation
posterior.  Candidate-specific relation alignment is leave-one-candidate-out
when that candidate contributed to the posterior, so a partition cannot reward
itself through the transductive ensemble.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from SpaLORA.night16g_basin_selector import molecular_separation, robust_standardize
from SpaLORA.night17b_sfrd import RelationPosterior


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("percentile input must be a nonempty finite vector")
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1.0)


def centroid_margin(view: np.ndarray, partition: np.ndarray) -> Mapping[str, float]:
    view = np.asarray(view, dtype=np.float64)
    _, labels = np.unique(np.asarray(partition), return_inverse=True)
    k = int(labels.max()) + 1
    centroids = np.stack([view[labels == group].mean(axis=0) for group in range(k)])
    squared = np.maximum(
        np.sum(view**2, axis=1)[:, None]
        + np.sum(centroids**2, axis=1)[None, :]
        - 2.0 * view @ centroids.T,
        0.0,
    )
    distance = np.sqrt(squared)
    own = distance[np.arange(view.shape[0]), labels]
    distance[np.arange(view.shape[0]), labels] = np.inf
    other = distance.min(axis=1)
    margin = (other - own) / np.maximum(other + own, 1e-12)
    return {
        "margin_min": float(np.min(margin)),
        "margin_q10": float(np.quantile(margin, 0.10)),
        "margin_mean": float(np.mean(margin)),
    }


def permutation_order(is_spatial: np.ndarray, seed: int = 20260826) -> np.ndarray:
    """Exact order used by Night-17C stratified relation permutation."""
    strata = np.asarray(is_spatial, dtype=bool)
    order = np.arange(strata.size, dtype=np.int64)
    for offset, value in enumerate((False, True)):
        indices = np.flatnonzero(strata == value)
        rng = np.random.RandomState(int(seed) + offset)
        order[indices] = indices[rng.permutation(indices.size)]
    return order


def posterior_without_candidate(
    probability: np.ndarray,
    candidate_weight: float,
    candidate_same_contribution: np.ndarray,
) -> tuple[np.ndarray, bool]:
    probability = np.asarray(probability, dtype=np.float64)
    same = np.asarray(candidate_same_contribution, dtype=np.float64)
    weight = float(candidate_weight)
    if weight <= 0.0:
        return probability.copy(), False
    if weight >= 1.0 - 1e-12:
        raise ValueError("cannot leave out the only posterior candidate")
    result = (probability - weight * same) / (1.0 - weight)
    return np.clip(result, 0.0, 1.0), True


def relation_alignment(
    partition: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    posterior: RelationPosterior,
    candidate_index: int,
    candidate_same_contribution: np.ndarray | None = None,
) -> Mapping[str, float | bool]:
    labels = np.asarray(partition)
    same = labels[pair_i] == labels[pair_j]
    if candidate_same_contribution is None:
        candidate_same_contribution = same
    full_probability = posterior.probability_same.astype(np.float64)
    loo_probability, removed = posterior_without_candidate(
        full_probability,
        float(posterior.candidate_weights[int(candidate_index)]),
        candidate_same_contribution,
    )

    def score(probability: np.ndarray) -> float:
        eps = 1e-12
        entropy = -(
            probability * np.log(probability + eps)
            + (1.0 - probability) * np.log(1.0 - probability + eps)
        ) / np.log(2.0)
        confidence = np.clip(1.0 - entropy, 0.0, 1.0)
        edge_scale = np.where(np.asarray(is_spatial, dtype=bool), 1.0, 0.5)
        expected = np.where(same, probability, 1.0 - probability)
        weight = confidence * edge_scale
        return float(np.sum(weight * expected) / max(np.sum(weight), 1e-12))

    return {
        "relation_alignment_loo": score(loo_probability),
        "relation_alignment_full": score(full_probability),
        "relation_self_removed": bool(removed),
        "relation_candidate_weight": float(posterior.candidate_weights[int(candidate_index)]),
    }


def raw_candidate_features(
    representation: np.ndarray,
    partition: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    posterior: RelationPosterior,
    candidate_index: int,
    candidate_same_contribution: np.ndarray | None = None,
) -> dict[str, float | bool]:
    view = robust_standardize(np.asarray(representation, dtype=np.float64))
    separation = molecular_separation(view, partition)
    margin = centroid_margin(view, partition)
    alignment = relation_alignment(
        partition,
        pair_i,
        pair_j,
        is_spatial,
        posterior,
        candidate_index,
        candidate_same_contribution,
    )
    return {**separation, **margin, **alignment}


EVIDENCE_AXES = ("explained", "ch", "margin_q10", "relation_alignment_loo")


def compress_seed_features(rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    """Return candidate evidence percentiles for one representation seed."""
    if not rows:
        raise ValueError("no feature rows")
    axes = np.column_stack(
        [percentile(np.asarray([float(row[name]) for row in rows])) for name in EVIDENCE_AXES]
    )
    return np.mean(axes, axis=1)


@dataclass(frozen=True)
class SelectorWeights:
    molecular: float
    topology: float
    learned: float
    uncertainty: float

    @property
    def config_id(self) -> str:
        return (
            f"M{self.molecular:g}_T{self.topology:g}_L{self.learned:g}_U{self.uncertainty:g}"
        )


def select_candidate(
    records: Sequence[Mapping[str, object]],
    weights: SelectorWeights,
    evidence_prefix: str = "LEARNED",
) -> Mapping[str, object]:
    feasible = [
        row
        for row in records
        if str(row["feasible_SMALLEST_SCALE_INTERNAL_EDGE"]).lower() == "true"
    ]
    if not feasible:
        raise ValueError("no structurally feasible candidates")
    evidence_key = f"{evidence_prefix}_evidence"
    uncertainty_key = f"{evidence_prefix}_uncertainty_rank"
    scored = []
    for row in feasible:
        score = (
            weights.molecular * float(row["molecular_rank"])
            + weights.topology * float(row["topology_rank"])
            + weights.learned * float(row[evidence_key])
            - weights.uncertainty * float(row[uncertainty_key])
        )
        scored.append((float(score), str(row["candidate_id"]), row))
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def weight_grid(specification: Mapping[str, Sequence[float]]) -> list[SelectorWeights]:
    return [
        SelectorWeights(*map(float, values))
        for values in product(
            specification["molecular"],
            specification["topology"],
            specification["learned"],
            specification["uncertainty"],
        )
    ]


def fit_weight_config(
    lane_records: Mapping[str, Sequence[Mapping[str, object]]],
    evaluation: Mapping[str, Mapping[str, Mapping[str, object]]],
    authority: Mapping[str, Mapping[str, float]],
    grid: Sequence[SelectorWeights],
) -> tuple[SelectorWeights, list[dict[str, object]], list[dict[str, object]]]:
    """Fit on explicitly supplied training lanes; held-out data cannot enter."""
    summaries: list[dict[str, object]] = []
    detailed: list[dict[str, object]] = []
    for weights in grid:
        deltas_ari, deltas_nmi, rows = [], [], []
        for lane in sorted(lane_records):
            selected = select_candidate(lane_records[lane], weights, "LEARNED")
            candidate_id = str(selected["candidate_id"])
            metric = evaluation[lane][candidate_id]
            delta_ari = float(metric["absolute_ari"]) - float(authority[lane]["absolute_ari"])
            delta_nmi = float(metric["absolute_nmi"]) - float(authority[lane]["absolute_nmi"])
            deltas_ari.append(delta_ari)
            deltas_nmi.append(delta_nmi)
            rows.append(
                {
                    "config_id": weights.config_id,
                    "lane": lane,
                    "candidate_id": candidate_id,
                    "absolute_ari": float(metric["absolute_ari"]),
                    "absolute_nmi": float(metric["absolute_nmi"]),
                    "delta_vs_night16h_ari": delta_ari,
                    "delta_vs_night16h_nmi": delta_nmi,
                }
            )
        summary = {
            "config_id": weights.config_id,
            "molecular": weights.molecular,
            "topology": weights.topology,
            "learned": weights.learned,
            "uncertainty": weights.uncertainty,
            "dual_improvement_count": int(sum(a > 1e-12 and n > 1e-12 for a, n in zip(deltas_ari, deltas_nmi))),
            "worst_min_delta": float(min(min(a, n) for a, n in zip(deltas_ari, deltas_nmi))),
            "mean_delta_ari": float(np.mean(deltas_ari)),
            "mean_delta_nmi": float(np.mean(deltas_nmi)),
            "weight_sum": float(weights.molecular + weights.topology + weights.learned + weights.uncertainty),
        }
        summaries.append(summary)
        detailed.extend(rows)
    ordered = sorted(
        summaries,
        key=lambda row: (
            -int(row["dual_improvement_count"]),
            -float(row["worst_min_delta"]),
            -float(row["mean_delta_ari"]),
            -float(row["mean_delta_nmi"]),
            float(row["weight_sum"]),
            str(row["config_id"]),
        ),
    )
    winner_id = str(ordered[0]["config_id"])
    return next(item for item in grid if item.config_id == winner_id), summaries, detailed
