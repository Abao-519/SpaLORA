from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night9b_racf import (
    RACFModel, fixed_permutation, racf_loss, rna_common_graph,
    robust_reliability, row_l2_np, spatial_operator, torch_sparse,
)


def candidate(**updates):
    value = {"id": "T", "common_graph_k": 2, "hierarchical_fusion": True,
             "reliability_gate": True, "dgi": True}
    value.update(updates); return value


def toy():
    rng = np.random.default_rng(4)
    ids = [f"o{i:02d}" for i in range(12)]
    x = row_l2_np(rng.normal(size=(12, 5)))
    coords = np.column_stack((np.arange(12), np.zeros(12)))
    return ids, x, coords


def test_common_graph_is_intersection_and_has_loops():
    ids, x, coords = toy()
    graph, audit = rna_common_graph(x, coords, ids, 2)
    assert audit["combine"] == "edge_intersection_then_symmetrize"
    assert audit["union_fallback"] is False
    assert np.all(graph.diagonal() > 0)
    assert np.all(np.asarray(graph.sum(1)).ravel() > 0)
    # A deliberately constructed union contains support outside the intersection.
    from SpaLORA.night9b_racf import symmetric_knn, canonical_csr
    feature = symmetric_knn(x, ids, 2, "cosine")
    spatial = symmetric_knn(coords, ids, 2, "euclidean")
    union = canonical_csr(feature.maximum(spatial))
    inter = canonical_csr(feature.multiply(spatial))
    assert canonical_csr(union - union.multiply((inter != 0))).nnz > 0


def test_auxiliary_features_cannot_replace_rna_feature_support():
    ids, x, coords = toy(); aux = x[::-1].copy()
    rna, _ = rna_common_graph(x, coords, ids, 2)
    wrong, _ = rna_common_graph(aux, coords, ids, 2)
    assert (rna != wrong).nnz > 0


def test_reliability_per_spot_simplex_and_stop_gradient():
    a = torch.tensor([1., 2., 9.], requires_grad=True)
    b = torch.tensor([3., 2., 1.], requires_grad=True)
    weights = robust_reliability(a, b)
    assert torch.allclose(weights.sum(1), torch.ones(3))
    assert float(weights.min()) >= .1 - 1e-7
    assert float(weights.max()) <= .9 + 1e-7
    assert not weights.requires_grad


def test_dgi_permutation_seed_contract():
    assert np.array_equal(fixed_permutation(20, 3), fixed_permutation(20, 3))
    assert not np.array_equal(fixed_permutation(20, 3), fixed_permutation(20, 4))


def test_hierarchy_gradient_reaches_all_three_branches():
    ids, x, coords = toy(); device = torch.device("cpu")
    spatial = torch_sparse(spatial_operator(coords, ids, 2), device)
    common, _ = rna_common_graph(x, coords, ids, 2)
    common_t = torch_sparse(common, device)
    model = RACFModel(5, candidate())
    tx = torch.as_tensor(x); aux = torch.as_tensor(np.roll(x, 1, axis=0).copy())
    ref = torch.as_tensor(row_l2_np((x + np.roll(x, 1, axis=0)) / 2))
    out = model(tx, aux, spatial, common_t, ref)
    loss, _ = racf_loss(model, out, tx, aux, ref,
                        torch.as_tensor(fixed_permutation(len(x), 2)))
    loss.backward()
    for prefix in ("rna_feature", "rna_spatial", "aux_common"):
        params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
        assert params and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_candidate_rejects_identity_and_label_fields():
    import pytest
    with pytest.raises(ValueError): RACFModel(5, {**candidate(), "dataset": "x"})
    with pytest.raises(ValueError): RACFModel(5, {**candidate(), "labels": [1]})
