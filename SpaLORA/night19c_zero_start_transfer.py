"""Night-19C authority adapter for equal-weight placenta relation transfer.

This module does not alter the frozen Night-17C Z01 model or loss.  It only
constructs the explicitly registered 16-candidate relation posterior required
for a new physical study whose candidate IDs do not use Night-17B prefixes.
Annotations are absent from this module.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from SpaLORA.night17b_sfrd import RelationPosterior, sha256_array


Z01_CONSERVATIVE = {
    "config_id": "Z01_CONSERVATIVE",
    "hidden_dim": 32,
    "residual_scale": 0.05,
    "learning_rate": 0.0007,
    "steps": 40,
    "relation_weight": 1.0,
    "anchor_weight": 4.0,
    "self_return_weight": 4.0,
    "consistency_weight": 0.5,
    "variance_weight": 0.1,
}

PLACENTA_INCLUDED_ARMS = (
    "RNA_ONLY_DIFFUSION",
    "ATAC_ONLY_DIFFUSION",
    "SPATIAL_ONLY_DIFFUSION",
    "CONCATENATED_FEATURE_KNN",
    "SIMPLE_OPERATOR_AVERAGE",
    "CLASSICAL_ALTERNATING_RA",
    "SPATIALLY_ANCHORED_ALTERNATING",
    "CSAD_FULL",
)
PLACENTA_PROFILES = ("S01_LOCAL", "S02_MESOSCALE")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_placenta_bank(candidate_ids: Sequence[str]) -> np.ndarray:
    """Select the preregistered 2 profiles x 8 semantic arms without renaming."""

    expected = [f"{profile}__{arm}__E0" for profile in PLACENTA_PROFILES for arm in PLACENTA_INCLUDED_ARMS]
    observed = [str(value) for value in candidate_ids]
    if len(observed) != len(set(observed)):
        raise ValueError("source candidate IDs are not unique")
    missing = [value for value in expected if value not in observed]
    if missing:
        raise ValueError(f"registered placenta relation candidates missing: {missing}")
    indices = np.asarray([observed.index(value) for value in expected], dtype=np.int64)
    if indices.size != 16:
        raise RuntimeError("placenta bank must contain exactly 16 candidates")
    return indices


def equal_weight_relation_posterior(
    partitions: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> RelationPosterior:
    """Explicit equal-weight posterior over the already selected authority bank."""

    partitions = np.asarray(partitions)
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    if partitions.ndim != 2 or partitions.shape[0] != 16:
        raise ValueError("explicit placenta posterior requires a 16 x N partition bank")
    if pair_i.shape != pair_j.shape or pair_i.ndim != 1:
        raise ValueError("pair authority shape mismatch")
    if pair_i.size == 0 or pair_i.min() < 0 or pair_j.max() >= partitions.shape[1]:
        raise ValueError("pair authority is empty or outside partition bounds")
    same = partitions[:, pair_i] == partitions[:, pair_j]
    weights = np.full(16, 1.0 / 16.0, dtype=np.float64)
    probability = np.asarray(weights @ same.astype(np.float64), dtype=np.float64)
    eps = 1e-12
    entropy = -(probability * np.log(probability + eps) + (1.0 - probability) * np.log(1.0 - probability + eps))
    uncertainty = np.clip(entropy / np.log(2.0), 0.0, 1.0)
    confidence = np.abs(2.0 * probability - 1.0) * (1.0 - uncertainty)
    return RelationPosterior(
        probability_same=probability.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        positive_weight=(confidence * (probability > 0.5)).astype(np.float32),
        negative_weight=(confidence * (probability < 0.5)).astype(np.float32),
        candidate_weights=weights,
        selected_candidate_count=16,
    )


def bank_authority_sha(candidate_ids: Sequence[str], partitions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for candidate_id, partition in zip(candidate_ids, np.asarray(partitions)):
        digest.update(str(candidate_id).encode("utf-8"))
        digest.update(sha256_array(np.asarray(partition)).encode("ascii"))
        digest.update(np.float64(1.0 / 16.0).tobytes())
    return digest.hexdigest()
