import numpy as np
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night6c_pipeline import NumericalHeadFailure
from SpaLORA.night7c_conflict import (
    conflict_rank,
    matching_quality,
    mix_affinities,
    reject_identity_metadata,
    routing_weights,
    shared_neighbor_support,
    tied_average_percentile,
    weighted_mnn_weights,
    weighted_triplet_margin_loss,
)


def test_tied_average_percentile_and_singleton():
    assert np.array_equal(tied_average_percentile(np.array([7.0])), np.array([0.0]))
    got = tied_average_percentile(np.array([3.0, 1.0, 1.0, 5.0]))
    assert np.allclose(got, np.array([2 / 3, 1 / 6, 1 / 6, 1.0]))


def test_conflict_quality_and_shared_support_are_bounded():
    theta = np.linspace(0.0, 1.0, 12)
    a = np.column_stack((np.cos(theta), np.sin(theta)))
    b = np.roll(a, 1, axis=0)
    ids = ["x%02d" % i for i in range(len(a))]
    conflict, rank = conflict_rank(a, b)
    positive = np.arange(len(a), dtype=np.int64)
    quality = matching_quality(a, b, positive, ids)
    support = shared_neighbor_support(a, b, ids, k=10)
    for value in (conflict, rank, quality, support):
        assert np.isfinite(value).all()
    for value in (rank, quality, support):
        assert ((value >= 0) & (value <= 1)).all()


def test_router_hand_calculations():
    rank = np.array([0.0, 0.5, 1.0])
    quality = np.array([0.25, 0.5, 1.0])
    support = np.array([1.0, 0.25, 0.0])
    t02 = routing_weights("T02_GLOBAL_WIDE", .275, rank, quality, support)
    assert np.allclose(t02[:, 1], .5)
    t03 = routing_weights("T03_GLOBAL_CONSERVATIVE", .30, rank, quality, support)
    assert np.allclose(t03[:, 1], .5)
    t04 = routing_weights("T04_HARD_CONFLICT_030", .30, rank, quality, support)
    assert np.array_equal(t04[:, 1], np.ones(3))
    t05 = routing_weights("T05_LOCAL_CONFLICT", .275, rank, quality, support)
    assert np.allclose(t05[:, 1], .5 * rank)
    t06 = routing_weights("T06_LOCAL_CONFLICT_SQUARED", .30, rank, quality, support)
    assert np.allclose(t06[:, 1], .5 * rank ** 2)
    t07 = routing_weights("T07_LOCAL_QUALITY_CONFLICT", .275, rank, quality, support)
    assert np.allclose(t07[:, 1], .5 * rank * quality)
    t08 = routing_weights("T08_LOCAL_SHARED_SUPPORT", .275, rank, quality, support)
    assert np.allclose(t08[:, 1], .5 * np.sqrt(rank * quality * support))
    t09 = routing_weights("T09_THREE_SPECIALIST_EXPLORATORY", .325,
                          rank, quality, support)
    assert np.allclose(t09[:, 2], .5)
    assert np.allclose(t09[:, 1], .5)
    assert np.allclose(t09[:, 0], 0.0)
    for value in (t02, t03, t04, t05, t06, t07, t08, t09):
        assert np.all(value >= 0)
        assert np.allclose(value.sum(1), 1.0)


def test_sparse_mix_and_zero_degree_fail_closed():
    a = sp.csr_matrix(np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0.0]]))
    b = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0.0]]))
    weights = np.array([[.25, .75], [.5, .5], [1.0, 0.0]])
    mixed = mix_affinities(weights, a, b)
    assert (mixed - mixed.T).nnz == 0
    assert np.array_equal(mixed.diagonal(), np.zeros(3))
    assert mixed.has_sorted_indices
    bad = sp.csr_matrix(np.array([[0, 0], [0, 1.0]]))
    with pytest.raises(NumericalHeadFailure):
        mix_affinities(np.full((2, 2), .5), bad, bad)


def test_weight_formulas_normalization_and_all_zero_guard():
    rank = np.array([0.0, .25, .5, 1.0])
    quality = np.array([0.0, .4, .8, 1.0])
    support = np.array([0.0, .25, .5, 1.0])
    ids = ["d", "c", "b", "a"]
    for candidate in ("W00_FILTER75", "W01_QUALITY_SOFT", "W02_CONFLICT_RANK",
                      "W03_QUALITY_CONFLICT", "W04_QUALITY_SHARED",
                      "W05_QUALITY_CONFLICT_SHARED"):
        raw, normalized = weighted_mnn_weights(candidate, rank, quality, support, ids)
        assert np.array_equal(raw == 0, normalized == 0)
        assert np.isclose(normalized[normalized > 0].mean(), 1.0)
    with pytest.raises(NumericalHeadFailure):
        weighted_mnn_weights("W01_QUALITY_SOFT", rank, np.zeros(4), support, ids)


def test_weighted_triplet_and_identity_guard():
    anchor = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    positive = torch.tensor([[0.1, 0.0], [1.1, 0.0]])
    negative = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    weights = torch.tensor([0.0, 1.0])
    value = weighted_triplet_margin_loss(anchor, positive, negative, weights)
    assert torch.isfinite(value)
    with pytest.raises(RuntimeError):
        weighted_triplet_margin_loss(anchor, positive, negative, torch.zeros(2))
    reject_identity_metadata({"feature_sha": "x"})
    with pytest.raises(RuntimeError):
        reject_identity_metadata({"dataset": "forbidden"})
