import inspect

import numpy as np
import pytest
from scipy import sparse

from SpaLORA.night11a_corruption import conflict_permutation, sparse_patch
from SpaLORA.selective_transfer import (ARMS, apply_gate, feature_folds,
                                        fit_ridge, predict_ridge, spot_folds)


def test_four_arms_and_exact_null_boundaries():
    assert ARMS == ("B0_SELF_ONLY", "B1_ALWAYS_TRANSFER",
                    "B2_UNCERTAINTY_ONLY", "B3_SELECTIVE_NULL")
    a = np.arange(24, dtype=np.float64).reshape(6, 4)
    b = a + 7
    z = apply_gate(a, b, np.zeros(6))
    assert z.tobytes() == a.tobytes()
    assert np.array_equal(apply_gate(a, b, np.ones(6)), b)


def test_spot_and_feature_crossfit_deterministic_and_separate():
    ids = ["o%03d" % i for i in range(200)]
    s1 = spot_folds("u", ids); s2 = spot_folds("u", ids)
    assert np.array_equal(s1, s2) and set(s1) == {0, 1}
    f = feature_folds("u", "M2_TO_M1", 1000)
    assert set(f) == set(range(5))
    for e in range(5):
        evidence = {(e + 1) % 5, (e + 2) % 5}
        observable = set(range(5)) - evidence - {e}
        assert e not in evidence and e not in observable and len(observable) == 2


def test_ridge_fit_has_no_heldout_row_leakage():
    rng = np.random.RandomState(3); x = rng.randn(40, 8); y = rng.randn(40, 5)
    fit = fit_ridge(x[:20], y[:20], 1.0)
    p1 = predict_ridge(fit, x[20:])
    x2 = x.copy(); y2 = y.copy(); x2[20:] += 1e6; y2[20:] -= 1e6
    fit2 = fit_ridge(x2[:20], y2[:20], 1.0)
    assert np.array_equal(fit.coefficient, fit2.coefficient)
    assert not np.allclose(p1, predict_ridge(fit2, x2[20:]))


def test_patch_and_permutation_sparse_deterministic_no_fixed_neighbor():
    n = 200
    rows = np.arange(n); graph = sparse.csr_matrix((np.ones(2*n),
        (np.r_[rows, rows], np.r_[(rows+1)%n, (rows-1)%n])), shape=(n,n))
    ids = ["x%04d" % i for i in range(n)]
    p1 = sparse_patch(graph, ids, "u", "d", "c", 17)
    p2 = sparse_patch(graph, ids, "u", "d", "c", 17)
    assert np.array_equal(p1, p2) and len(p1) == 40
    q1 = conflict_permutation(graph, ids, p1, "u", "d", "c", 17)
    q2 = conflict_permutation(graph, ids, p1, "u", "d", "c", 17)
    assert np.array_equal(q1, q2) and not np.any(p1 == q1)
    assert all(graph[a, b] == 0 for a, b in zip(p1, q1))


def test_public_pilot_api_is_identity_blind():
    from SpaLORA.selective_transfer import run_direction
    forbidden = {"dataset", "tissue", "labels", "ari", "nmi", "q", "metric"}
    assert not (forbidden & set(inspect.signature(run_direction).parameters))


def test_sparse_path_does_not_materialize_dense_nxn(monkeypatch):
    graph = sparse.eye(50, format="csr")
    monkeypatch.setattr(sparse.csr_matrix, "toarray",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense")))
    patch = sparse_patch(graph, [str(i) for i in range(50)], "u", "d", "c", 17)
    assert patch.shape == (10,)
