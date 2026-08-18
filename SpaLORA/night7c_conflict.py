"""Identity-blind Night-7C conflict routing and weighted-MNN primitives.

The public numerical functions accept only locked arrays, sparse affinities,
registered candidate identifiers, and scalar hyperparameters.  They do not
accept unit identity or evaluation values.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .night6c_pipeline import NumericalHeadFailure
from .night7a_consensus import canonical_csr
from .night7b_adaptive import row_l2


ROUTING_ORDER = tuple("T%02d" % i for i in range(10))
WEIGHTED_ORDER = tuple("W%02d" % i for i in range(6))
FORBIDDEN_METADATA = frozenset({
    "dataset", "dataset_id", "tissue", "organism", "platform", "technology",
    "modality", "file", "file_name", "path", "label", "ground_truth",
    "ari", "nmi", "q", "metric",
})


def reject_identity_metadata(value: Mapping[str, object]) -> None:
    """Fail closed if a caller tries to provide identity or evaluation data."""
    bad = sorted(FORBIDDEN_METADATA.intersection(str(key).lower() for key in value))
    if bad:
        raise RuntimeError("identity-blind contract rejected keys: %r" % bad)


def tied_average_percentile(values: np.ndarray) -> np.ndarray:
    """Deterministic tied-average ranks scaled to [0, 1]."""
    work = np.asarray(values, dtype=np.float64)
    if work.ndim != 1 or not np.isfinite(work).all():
        raise NumericalHeadFailure("rank input must be one finite vector")
    n = len(work)
    if n == 0:
        raise NumericalHeadFailure("rank input is empty")
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(work, kind="mergesort")
    ranked = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and work[order[stop]] == work[order[start]]:
            stop += 1
        average_zero_rank = 0.5 * (start + stop - 1)
        ranked[order[start:stop]] = average_zero_rank / float(n - 1)
        start = stop
    return ranked


def _validate_unit_vectors(u00: np.ndarray, u04: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = row_l2(u00), row_l2(u04)
    if a.ndim != 2 or b.shape != a.shape or len(a) < 2:
        raise NumericalHeadFailure("paired summaries must be matching 2D arrays with n>=2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise NumericalHeadFailure("paired summaries contain non-finite values")
    return a, b


def conflict_rank(u00: np.ndarray, u04: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = _validate_unit_vectors(u00, u04)
    conflict = np.clip(1.0 - np.sum(a * b, axis=1), 0.0, 2.0)
    return conflict, tied_average_percentile(conflict)


def matching_quality(u00: np.ndarray, u04: np.ndarray,
                     positives: np.ndarray, observation_ids: Sequence[str],
                     block: int = 256) -> np.ndarray:
    """Relative squared-Euclidean margin to the nearest non-positive match."""
    a, b = _validate_unit_vectors(u00, u04)
    positive = np.asarray(positives, dtype=np.int64)
    names = np.asarray(observation_ids, dtype=str)
    n = len(a)
    if positive.shape != (n,) or names.shape != (n,):
        raise NumericalHeadFailure("matching-quality input shape mismatch")
    if np.any(positive < 0) or np.any(positive >= n):
        raise NumericalHeadFailure("matching-quality positive index out of range")
    quality = np.empty(n, dtype=np.float64)
    for start in range(0, n, int(block)):
        stop = min(n, start + int(block))
        distance = np.maximum(0.0, 2.0 - 2.0 * (a[start:stop] @ b.T))
        if not np.isfinite(distance).all():
            raise NumericalHeadFailure("matching distance is non-finite")
        for offset, row in enumerate(range(start, stop)):
            pos = int(positive[row])
            d_positive = float(distance[offset, pos])
            candidates = np.flatnonzero(np.arange(n) != pos)
            if len(candidates) == 0:
                raise NumericalHeadFailure("no second matching alternative")
            ordered = np.lexsort((names[candidates], distance[offset, candidates]))
            d_second = float(distance[offset, candidates[ordered[0]]])
            quality[row] = np.clip(
                (d_second - d_positive) / (d_second + 1e-12), 0.0, 1.0)
    return quality


def _top_neighbors(values: np.ndarray, observation_ids: Sequence[str], k: int) -> np.ndarray:
    work = row_l2(values)
    names = np.asarray(observation_ids, dtype=str)
    n = len(work)
    if names.shape != (n,) or n <= int(k):
        raise NumericalHeadFailure("shared-support input has too few rows")
    out = np.empty((n, int(k)), dtype=np.int64)
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        distance = np.maximum(0.0, 2.0 - 2.0 * (work[start:stop] @ work.T))
        for offset, row in enumerate(range(start, stop)):
            distance[offset, row] = np.inf
            order = np.lexsort((names, distance[offset]))[:int(k)]
            if not np.isfinite(distance[offset, order]).all():
                raise NumericalHeadFailure("shared-support neighbor is non-finite")
            out[row] = order
    return out


def shared_neighbor_support(u00: np.ndarray, u04: np.ndarray,
                            observation_ids: Sequence[str], k: int = 10) -> np.ndarray:
    a, b = _validate_unit_vectors(u00, u04)
    n00, n04 = _top_neighbors(a, observation_ids, k), _top_neighbors(b, observation_ids, k)
    result = np.empty(len(a), dtype=np.float64)
    for row in range(len(a)):
        left, right = set(map(int, n00[row])), set(map(int, n04[row]))
        union = left | right
        if not union:
            raise NumericalHeadFailure("empty shared-neighbor union")
        result[row] = len(left & right) / float(len(union))
    return result


def _validate_features(rank_c: np.ndarray, quality: np.ndarray,
                       support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(x, dtype=np.float64) for x in (rank_c, quality, support))
    if any(x.ndim != 1 for x in arrays) or len({len(x) for x in arrays}) != 1:
        raise NumericalHeadFailure("router feature shape mismatch")
    if any(not np.isfinite(x).all() for x in arrays):
        raise NumericalHeadFailure("router feature is non-finite")
    if any(np.any((x < 0) | (x > 1)) for x in arrays):
        raise NumericalHeadFailure("router feature outside [0,1]")
    return arrays


def routing_weights(candidate_id: str, m_initial: float, rank_c: np.ndarray,
                    quality: np.ndarray, support: np.ndarray) -> np.ndarray:
    rank_c, quality, support = _validate_features(rank_c, quality, support)
    if not np.isfinite(float(m_initial)):
        raise NumericalHeadFailure("initial conflict is non-finite")
    n = len(rank_c)
    if candidate_id == "T00_C00_REFERENCE":
        return np.column_stack((np.ones(n), np.zeros(n)))
    if candidate_id == "T01_R02_REFERENCE":
        return np.column_stack((np.zeros(n), np.ones(n)))
    if candidate_id == "T02_GLOBAL_WIDE":
        alpha = np.full(n, np.clip((m_initial - .20) / .15, 0.0, 1.0))
    elif candidate_id == "T03_GLOBAL_CONSERVATIVE":
        alpha = np.full(n, np.clip((m_initial - .25) / .10, 0.0, 1.0))
    elif candidate_id == "T04_HARD_CONFLICT_030":
        alpha = np.full(n, 1.0 if m_initial >= .30 else 0.0)
    elif candidate_id == "T05_LOCAL_CONFLICT":
        alpha = np.clip((m_initial - .20) / .15, 0.0, 1.0) * rank_c
    elif candidate_id == "T06_LOCAL_CONFLICT_SQUARED":
        alpha = np.clip((m_initial - .25) / .10, 0.0, 1.0) * (rank_c ** 2)
    elif candidate_id == "T07_LOCAL_QUALITY_CONFLICT":
        alpha = np.clip((m_initial - .20) / .15, 0.0, 1.0) * rank_c * quality
    elif candidate_id == "T08_LOCAL_SHARED_SUPPORT":
        alpha = np.clip((m_initial - .20) / .15, 0.0, 1.0) * np.sqrt(rank_c * quality * support)
    elif candidate_id == "T09_THREE_SPECIALIST_EXPLORATORY":
        w2 = np.full(n, np.clip((m_initial - .30) / .05, 0.0, 1.0))
        w8 = (1.0 - w2) * np.clip((m_initial - .22) / .08, 0.0, 1.0)
        result = np.column_stack((1.0 - w2 - w8, w8, w2))
        if np.any(result < -1e-15) or not np.allclose(result.sum(1), 1.0, atol=1e-15):
            raise NumericalHeadFailure("three-specialist weights violate simplex")
        return result
    else:
        raise KeyError(candidate_id)
    alpha = np.clip(alpha, 0.0, 1.0)
    return np.column_stack((1.0 - alpha, alpha))


def _row_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    value = canonical_csr(matrix)
    if np.any(~np.isfinite(value.data)) or np.any(value.data < 0):
        raise NumericalHeadFailure("specialist affinity must be finite and nonnegative")
    degree = np.asarray(value.sum(axis=1)).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0):
        raise NumericalHeadFailure("specialist affinity has zero degree")
    return canonical_csr(sp.diags(1.0 / degree) @ value)


def mix_affinities(weights: np.ndarray, s0: sp.spmatrix, s2: sp.spmatrix,
                   s8: sp.spmatrix | None = None) -> sp.csr_matrix:
    value = np.asarray(weights, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != s0.shape[0] or s0.shape != s2.shape:
        raise NumericalHeadFailure("affinity mixing shape mismatch")
    if not np.isfinite(value).all() or np.any(value < -1e-15):
        raise NumericalHeadFailure("affinity mixing weights invalid")
    if not np.allclose(value.sum(1), 1.0, atol=1e-15):
        raise NumericalHeadFailure("affinity mixing weights do not sum to one")
    if value.shape[1] == 2 and s8 is None:
        directed = sp.diags(value[:, 0]) @ _row_normalize(s0)
        directed += sp.diags(value[:, 1]) @ _row_normalize(s2)
    elif value.shape[1] == 3 and s8 is not None and s8.shape == s0.shape:
        directed = sp.diags(value[:, 0]) @ _row_normalize(s0)
        directed += sp.diags(value[:, 1]) @ _row_normalize(s8)
        directed += sp.diags(value[:, 2]) @ _row_normalize(s2)
    else:
        raise NumericalHeadFailure("specialist count does not match registered weights")
    result = canonical_csr((directed + directed.T) * .5)
    result.setdiag(0.0)
    result.eliminate_zeros()
    result.sort_indices()
    degree = np.asarray(result.sum(axis=1)).ravel()
    if np.any(degree <= 0) or not np.isfinite(result.data).all():
        raise NumericalHeadFailure("mixed affinity is invalid")
    return canonical_csr(result)


def weighted_mnn_weights(candidate_id: str, rank_c: np.ndarray,
                         quality: np.ndarray, support: np.ndarray,
                         observation_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rank_c, quality, support = _validate_features(rank_c, quality, support)
    names = np.asarray(observation_ids, dtype=str)
    if names.shape != rank_c.shape:
        raise NumericalHeadFailure("weighted-MNN observation order mismatch")
    if candidate_id == "W00_FILTER75":
        raw = np.zeros(len(rank_c), dtype=np.float64)
        count = int(math.ceil(.75 * len(raw)))
        order = np.lexsort((names, -quality))
        raw[order[:count]] = 1.0
    elif candidate_id == "W01_QUALITY_SOFT":
        raw = quality.copy()
    elif candidate_id == "W02_CONFLICT_RANK":
        raw = rank_c.copy()
    elif candidate_id == "W03_QUALITY_CONFLICT":
        raw = quality * rank_c
    elif candidate_id == "W04_QUALITY_SHARED":
        raw = quality * np.sqrt(support)
    elif candidate_id == "W05_QUALITY_CONFLICT_SHARED":
        raw = quality * rank_c * np.sqrt(support)
    else:
        raise KeyError(candidate_id)
    nonzero = raw > 0
    if not np.any(nonzero):
        raise NumericalHeadFailure("all weighted-MNN weights are zero")
    normalized = raw.copy()
    normalized[nonzero] /= float(raw[nonzero].mean())
    return raw, normalized


def weighted_triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor,
                                 negative: torch.Tensor, weights: torch.Tensor,
                                 margin: float = .5) -> torch.Tensor:
    if weights.ndim != 1 or weights.shape[0] != anchor.shape[0]:
        raise RuntimeError("weighted-MNN tensor shape mismatch")
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
        raise RuntimeError("weighted-MNN tensor is invalid")
    if not bool(torch.any(weights > 0)):
        raise RuntimeError("all weighted-MNN tensor weights are zero")
    per_spot = F.triplet_margin_loss(anchor, positive, negative,
                                     margin=float(margin), reduction="none")
    return torch.mean(per_spot * weights)
