import numpy as np
import scipy.sparse as sp
import pytest

from SpaLORA.night7b_adaptive import (
    HEAD_ORDER, RECIPE_ORDER, _top_weight_neighbors, row_sparse_strict,
    sym_zero,
)


def test_locked_candidate_counts_and_order():
    assert HEAD_ORDER == tuple("H%02d" % i for i in range(18))
    assert RECIPE_ORDER == tuple("R%02d" % i for i in range(10))


def test_row_weight_arithmetic_and_symmetry():
    a = sp.csr_matrix([[0., 2., 1.], [2., 0., 1.], [1., 1., 0.]])
    b = sp.csr_matrix([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
    mixed = sym_zero(.3 * row_sparse_strict(a) + .7 * row_sparse_strict(b))
    assert np.allclose(mixed.toarray(), mixed.toarray().T, atol=0, rtol=0)
    assert np.allclose(mixed.diagonal(), 0, atol=0, rtol=0)


def test_zero_degree_fails_closed():
    with pytest.raises(Exception):
        row_sparse_strict(sp.csr_matrix((3, 3)))


def test_lexical_tie_break():
    value = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    assert _top_weight_neighbors(value, 1, ["c", "a", "b"])[0, 0] == 1
