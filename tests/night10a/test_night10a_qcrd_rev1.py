from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from SpaLORA.night10a_qcrd import (
    ALL_TRAINABLE_CANDIDATES, EPS, GLOBAL_CANDIDATES, LOSS_WEIGHTS,
    MASKED_CANDIDATES, QCRDAdapter, binary_spatial_graph,
    candidate_weights_and_gate, corrected_views, deterministic_mask,
    fourier_coordinates, frozen_quality, mask_dimension_count,
    modality_global_features, neighbor_entropy, qcrd_forward,
    qcrd_loss_components, row_normalize, sparse_neighbor_mean, standardize,
)


def fixture():
    first = np.asarray([[1, 0, .1], [.9, .1, 0], [0, 1, .1], [.1, .9, 0], [1, 1, .2], [.2, 1, 1]], dtype=np.float32)
    second = np.asarray([[1, .1, 0], [.8, .2, .1], [.1, 1, 0], [.2, .7, .2], [.9, 1, .1], [.1, .9, 1]], dtype=np.float32)
    fused = (first + second) / 2
    p1 = np.asarray([0, 0, 1, 1, 0, 2]); p2 = np.asarray([0, 0, 1, 1, 2, 2])
    # Node 5 is deliberately zero degree.
    graph = sp.csr_matrix((np.ones(8), ([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3])), shape=(6, 6))
    return first, second, fused, p1, p2, graph


def quality():
    a, b, f, p1, p2, g = fixture()
    return frozen_quality(a, b, f, p1, p2, g, 3, mnn_k=3)


def test_global_five_features_and_hand_contrast():
    a, b, _, p1, p2, g = fixture()
    q = quality()
    for z, p, observed in ((a, p1, q.global_features_1), (b, p2, q.global_features_2)):
        support = binary_spatial_graph(g, len(z)); degree = np.asarray(support.sum(1)).ravel(); rows, cols = support.nonzero()
        same = np.zeros(len(z)); np.add.at(same, rows, p[rows] == p[cols])
        consistency = np.mean(same[degree > 0] / degree[degree > 0])
        residual = -math.log1p(np.linalg.norm(row_normalize(z) - sparse_neighbor_mean(support, row_normalize(z)), axis=1).mean())
        assert np.allclose(observed[-2:], [consistency, residual], atol=1e-12)
        assert len(observed) == 5 and np.isfinite(observed).all()
    manual = np.clip((q.global_features_1 - q.global_features_2) /
                     (np.abs(q.global_features_1) + np.abs(q.global_features_2) + EPS), -1, 1)
    weight = 1 / (1 + math.exp(-manual.mean()))
    assert np.allclose(q.global_contrasts, manual, atol=1e-12)
    assert np.allclose(q.global_weights, [weight, 1 - weight], atol=1e-12)


def test_entropy_zero_degree_spot_logits_boundary_and_gate_hand():
    a, b, f, p1, p2, g = fixture(); q = quality()
    # Node 1 sees labels [self=0, neighbor0=0, neighbor2=1].
    expected_h1 = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)) / math.log(3)
    assert abs(q.entropy_1[1] - expected_h1) < 1e-12
    assert q.entropy_1[5] == q.entropy_2[5] == 0.0
    assert q.zero_degree_count == 1
    z1, z2 = row_normalize(a), row_normalize(b)
    r1 = np.linalg.norm(z1 - sparse_neighbor_mean(g, z1), axis=1)
    r2 = np.linalg.norm(z2 - sparse_neighbor_mean(g, z2), axis=1)
    scale = max(float(np.median(np.r_[r1, r2])), EPS)
    l1 = math.log(q.global_weights[0] + EPS) - r1[1] / scale - .5 * q.entropy_1[1] + .5 * q.support_1[1]
    l2 = math.log(q.global_weights[1] + EPS) - r2[1] / scale - .5 * q.entropy_2[1] + .5 * q.support_2[1]
    expected_weight = np.exp([l1, l2] - np.max([l1, l2])); expected_weight /= expected_weight.sum()
    assert np.allclose(q.spot_logits[1], [l1, l2], atol=1e-12)
    assert np.allclose(q.spot_weights[1], expected_weight, atol=1e-12)
    expected_gate = np.clip(.5 * abs(expected_weight[0] - expected_weight[1]) + .25 * q.disagreement[1]
                            + .25 * max(q.support_1[1], q.support_2[1]), 0, 1) * (1 - .25 * (q.entropy_1[1] + q.entropy_2[1]))
    assert abs(q.base_gate[1] - expected_gate) < 1e-12
    assert np.all((q.boundary_risk >= 0) & (q.boundary_risk <= 1))


def test_mask_cardinality_determinism_epoch_and_artifact_changes():
    candidate = "Q04_SPOT_GATED_MASKED_RESIDUAL"; n, d = 9, 17
    first = deterministic_mask(candidate, "a" * 64, 2, 4, n, d)
    assert np.array_equal(first, deterministic_mask(candidate, "a" * 64, 2, 4, n, d))
    assert np.all(first.sum(1) == mask_dimension_count(d))
    assert mask_dimension_count(d) == 3
    assert not np.array_equal(first, deterministic_mask(candidate, "a" * 64, 2, 5, n, d))
    assert not np.array_equal(first, deterministic_mask(candidate, "b" * 64, 2, 4, n, d))
    assert not deterministic_mask("Q02_SPOT_QUALITY_BLEND", "a" * 64, 2, 4, n, d).any()


def test_zero_up_projection_is_exact_epoch0_noop_and_teacher_frozen():
    a, b, f, *_ = fixture(); q = quality(); model = QCRDAdapter(3)
    assert torch.count_nonzero(model.up.weight) == 0
    ta, tb, tf = map(torch.tensor, (row_normalize(a), row_normalize(b), row_normalize(f)))
    out = qcrd_forward(model, ta, tb, tf, q, "Q02_SPOT_QUALITY_BLEND")
    assert torch.count_nonzero(out["correction"]) == 0
    assert not out["teacher"].requires_grad
    assert all(not isinstance(value, torch.Tensor) for value in (q.global_weights, q.spot_weights, q.base_gate))


def test_candidate_truth_table():
    q = quality(); dtype, device = torch.float32, torch.device("cpu")
    for candidate in sorted(ALL_TRAINABLE_CANDIDATES):
        weights, gate = candidate_weights_and_gate(q, candidate, dtype, device)
        if candidate in GLOBAL_CANDIDATES:
            assert torch.allclose(weights, weights[:1].expand_as(weights))
            assert torch.allclose(gate, gate[:1].expand_as(gate))
        else:
            assert np.allclose(weights.numpy(), q.spot_weights)
        if candidate == "Q05_BOUNDARY_GATED_RESIDUAL":
            assert np.allclose(gate.numpy().ravel(), q.base_gate * (1 - q.boundary_risk))
        assert (candidate in MASKED_CANDIDATES) == (candidate not in {"Q01_GLOBAL_QUALITY_BLEND", "Q02_SPOT_QUALITY_BLEND"})


def test_all_six_losses_and_fixed_weight_sum_against_independent_expression():
    a, b, f, _, _, g = fixture(); q = quality(); candidate = "Q07_CONFIDENCE_MNN_RESIDUAL"
    ta, tb, tf = map(torch.tensor, (row_normalize(a), row_normalize(b), row_normalize(f)))
    model = QCRDAdapter(3); torch.manual_seed(11)
    with torch.no_grad(): model.up.weight.normal_(0, .02)
    mask = torch.as_tensor(deterministic_mask(candidate, "c" * 64, 0, 0, len(a), 3))
    forward = qcrd_forward(model, ta, tb, tf, q, candidate, mask=mask)
    losses = qcrd_loss_components(forward, ta, tb, tf, q, candidate, g, np.asarray([0, 0, 1, 1, 0, 2]), mask)
    expected = (losses["align"] + losses["mask"] + .25 * losses["anchor"]
                + .05 * losses["correction"] + .25 * losses["boundary"] + .10 * losses["mnn"])
    assert torch.allclose(losses["total"], expected, atol=0, rtol=0)
    assert set(losses) == {"align", "mask", "anchor", "correction", "boundary", "mnn", "total"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert LOSS_WEIGHTS == {"align": 1.0, "mask": 1.0, "anchor": .25, "correction": .05, "boundary": .25, "mnn": .10}


def test_negative_contract_coverage_rejects_omission():
    required_features = {"silhouette", "davies_bouldin", "calinski_harabasz", "spatial_local_consistency", "graph_local_residual"}
    required_losses = {"align", "mask", "anchor", "correction", "boundary", "mnn"}
    def validate(features, losses):
        if set(features) != required_features or set(losses) != required_losses: raise RuntimeError("semantic omission")
    validate(required_features, required_losses)
    for missing in required_features:
        try: validate(required_features - {missing}, required_losses)
        except RuntimeError: pass
        else: raise AssertionError("omitted feature was accepted")
    for missing in required_losses:
        try: validate(required_features, required_losses - {missing})
        except RuntimeError: pass
        else: raise AssertionError("omitted loss was accepted")


def test_no_dataset_routing_no_dense_pairwise_and_same_module_class():
    import SpaLORA.night10a_qcrd as module
    source = inspect.getsource(module)
    assert "dataset" not in inspect.signature(QCRDAdapter).parameters
    assert "dataset" not in inspect.signature(frozen_quality).parameters
    assert "pairwise_distances(" not in source and "cdist(" not in source
    assert QCRDAdapter(64).__class__ is QCRDAdapter(128).__class__


def test_coordinate_features_and_checkpoint_round_trip():
    a, b, f, *_ = fixture(); q = quality(); coords = fourier_coordinates(np.c_[np.arange(len(a)), np.zeros(len(a))])
    torch.manual_seed(3); first = QCRDAdapter(3, coord_features=coords.shape[1]).eval()
    ta, tb, tf, tc = map(torch.tensor, (row_normalize(a), row_normalize(b), row_normalize(f), coords))
    with torch.no_grad(): expected = corrected_views(first, ta, tb, tf, q, "Q06_COORDINATE_PRIOR_RESIDUAL", tc)[2]
    second = QCRDAdapter(3, coord_features=coords.shape[1]); second.load_state_dict(first.state_dict()); second.eval()
    with torch.no_grad(): observed = corrected_views(second, ta, tb, tf, q, "Q06_COORDINATE_PRIOR_RESIDUAL", tc)[2]
    assert torch.equal(expected, observed)


def test_sparse_zero_degree_and_no_silent_self_edge():
    a, *_rest, g = fixture(); support = binary_spatial_graph(g, len(a))
    assert support.diagonal().sum() == 0
    mean = sparse_neighbor_mean(support, row_normalize(a))
    assert np.allclose(mean[5], row_normalize(a)[5])
