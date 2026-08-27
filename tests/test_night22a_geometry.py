import numpy as np
import scipy.sparse as sp

from SpaLORA.night22a_geometry import (
    combine_graphs,
    density_correct,
    exact_k,
    generate_geometry_bank,
    self_tuning_knn,
)


def toy_graph(n=48):
    rows, cols = [], []
    for i in range(n):
        for delta in (1, 2):
            j = (i + delta) % n
            rows.extend([i, j])
            cols.extend([j, i])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_self_tuning_graph_is_sparse_symmetric_and_finite():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(48, 7))
    graph = self_tuning_knn(x, 8)
    assert graph.shape == (48, 48)
    assert graph.nnz < 48 * 48
    assert (graph - graph.T).nnz == 0
    assert np.isfinite(graph.data).all()
    assert np.all(graph.data > 0)


def test_graph_combination_and_density_correction_are_nonnegative():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(48, 5))
    combined = combine_graphs(self_tuning_knn(x, 7), toy_graph(), 0.5)
    corrected = density_correct(combined)
    assert np.isfinite(corrected.data).all()
    assert np.all(corrected.data >= 0)
    assert (corrected - corrected.T).nnz == 0


def test_geometry_bank_is_exact_k_and_deterministic():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(-2, 0.5, size=(24, 6)), rng.normal(2, 0.5, size=(24, 6))]
    first = generate_geometry_bank(x, toy_graph(), 2)
    second = generate_geometry_bank(x, toy_graph(), 2)
    assert np.array_equal(first.candidate_ids, second.candidate_ids)
    assert np.array_equal(first.partitions, second.partitions)
    assert len(first.candidate_ids) == len(set(first.candidate_ids.tolist()))
    for partition in first.partitions:
        assert np.array_equal(partition, exact_k(partition, 2))
