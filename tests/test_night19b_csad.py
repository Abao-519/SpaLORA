import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from SpaLORA.night19b_csad import (
    apply_conflict_self_return,
    arm_operator,
    build_operator_bank,
    deterministic_topk,
    mutual_knn_operator,
    permute_operator,
    row_normalize,
    sparse_chain,
    stable_permutation,
    standardize,
    spectral_partition,
)


def dense_topk_reference(value, k):
    out = np.zeros_like(value, dtype=float)
    for row in range(value.shape[0]):
        columns = np.flatnonzero(value[row] > 0)
        order = sorted(columns, key=lambda col: (-value[row, col], col))[:k]
        out[row, order] = value[row, order]
    return out


def test_deterministic_topk_matches_dense_reference_with_ties():
    value = np.array([[0, 2, 2, 1], [3, 0, 4, 4], [1, 2, 0, 3], [2, 1, 4, 0.0]], dtype=float)
    actual = deterministic_topk(sp.csr_matrix(value), 2).toarray()
    assert np.array_equal(actual, dense_topk_reference(value, 2))


def test_sparse_chain_matches_dense_reference():
    a = row_normalize(sp.csr_matrix([[1, 2, 0], [2, 1, 1], [0, 1, 1]], dtype=float))
    b = row_normalize(sp.csr_matrix([[1, 0, 1], [0, 2, 1], [1, 1, 1]], dtype=float))
    actual, ledger = sparse_chain((a, b), topk=3)
    expected = a.toarray() @ b.toarray()
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(actual.toarray(), expected, atol=1e-12)
    assert ledger[0]["nnz_after_prune"] == actual.nnz


def test_local_scale_kernel_uses_per_node_kth_radius_and_handles_duplicate():
    value = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 3.0]], dtype=float)
    operator = mutual_knn_operator(value, neighbors=3, self_loop=0.25)
    assert operator.shape == (4, 4)
    assert np.all(np.isfinite(operator.data))
    assert np.allclose(np.asarray(operator.sum(axis=1)).ravel(), 1.0)
    # Independent dense reference with per-node kth-neighbour radii.
    x = standardize(value).astype(float)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    distances, indices = NearestNeighbors(n_neighbors=4, metric="cosine", algorithm="brute").fit(x).kneighbors(x)
    positive = distances[distances > 1e-10]
    fallback = np.median(positive)
    scales = distances[:, -1].copy()
    scales[scales <= 1e-10] = fallback
    directed = np.zeros((4, 4), dtype=float)
    for row in range(4):
        for distance, column in zip(distances[row], indices[row]):
            if row != column:
                directed[row, column] = np.exp(-(distance ** 2) / max(scales[row] * scales[column], 1e-12))
    expected = np.minimum(directed, directed.T) + 0.25 * np.eye(4)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(operator.toarray(), expected, atol=1e-12)


def test_all_rejected_conflict_mass_returns_exactly_to_self():
    base = row_normalize(sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    output, diagnostics = apply_conflict_self_return(base, np.zeros(3), strength=1.0, floor=0.0)
    assert np.allclose(output.toarray(), np.eye(3), atol=1e-12)
    assert diagnostics["row_mass_error"] <= 1e-12


def test_node_conjugation_is_deterministic_nonidentity_and_globally_matched():
    ids = np.asarray(["a", "b", "c", "d"])
    graph = sp.csr_matrix([[0, 1, 2, 0], [1, 0, 0, 3], [2, 0, 0, 4], [0, 3, 4, 0]], dtype=float)
    order1 = stable_permutation(ids, "salt")
    order2 = stable_permutation(ids, "salt")
    assert np.array_equal(order1, order2)
    assert not np.array_equal(order1, np.arange(ids.size))
    permuted = permute_operator(graph, order1)
    assert np.array_equal(np.sort(graph.getnnz(axis=1)), np.sort(permuted.getnnz(axis=1)))
    assert np.allclose(np.sort(graph.data), np.sort(permuted.data))
    assert np.allclose(np.sort(np.linalg.eigvalsh(graph.toarray())), np.sort(np.linalg.eigvalsh(permuted.toarray())))


def test_all_arms_are_sparse_symmetric_nonnegative_and_full_changes_capacity():
    rng = np.random.RandomState(7)
    n = 30
    view1 = rng.normal(size=(n, 5))
    view2 = 0.4 * view1[:, :3] @ rng.normal(size=(3, 6)) + rng.normal(scale=0.8, size=(n, 6))
    rows = np.arange(n)
    spatial = sp.csr_matrix((np.ones(2 * n), (np.r_[rows, rows], np.r_[np.roll(rows, 1), np.roll(rows, -1)])), shape=(n, n))
    bank = build_operator_bank(view1, view2, spatial, feature_neighbors=5, spatial_topk=4, self_loop=0.25)
    ids = np.asarray([f"x{i}" for i in range(n)])
    arms = (
        "RNA_ONLY_DIFFUSION", "ATAC_ONLY_DIFFUSION", "SPATIAL_ONLY_DIFFUSION",
        "CONCATENATED_FEATURE_KNN", "SIMPLE_OPERATOR_AVERAGE", "CLASSICAL_ALTERNATING_RA",
        "SPATIALLY_ANCHORED_ALTERNATING", "CSAD_FULL", "CSAD_CONFLICT_DISABLED",
        "CSAD_MODALITY_EDGE_PERMUTED",
    )
    operators = {}
    for arm in arms:
        operator, diagnostics = arm_operator(arm, bank, ids, topk=8, conflict_strength=0.8, conflict_floor=0.1)
        operators[arm] = operator
        assert diagnostics["finite"] and diagnostics["nonnegative"]
        assert diagnostics["symmetry_max_abs"] <= 1e-12
        assert operator.nnz <= n * 16
    assert not np.allclose(operators["CSAD_FULL"].toarray(), operators["CSAD_CONFLICT_DISABLED"].toarray())


def test_spectral_endpoint_is_exact_k_and_reproducible():
    n = 24
    rows = np.arange(n)
    graph = sp.csr_matrix((np.ones(2 * n), (np.r_[rows, rows], np.r_[np.roll(rows, 1), np.roll(rows, -1)])), shape=(n, n))
    p1, e1, v1, _ = spectral_partition(graph, k=4, spectral_dim=6, endpoint_seed=0)
    p2, e2, v2, _ = spectral_partition(graph, k=4, spectral_dim=6, endpoint_seed=0)
    assert np.unique(p1).size == 4
    assert np.array_equal(p1, p2)
    assert np.array_equal(e1, e2)
    assert np.array_equal(v1, v2)
