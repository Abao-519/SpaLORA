import numpy as np
import scipy.sparse as sp

from SpaLORA.night18c_rsp_gtd import (
    TrendConfig,
    common_kmeans_endpoint,
    compose_representation,
    deterministic_permutation,
    orthogonal_procrustes_align,
    row_normalize,
    robust_standardize,
    shared_view_basis,
    solve_permuted_private,
    solve_shared_private,
)


def toy():
    rng = np.random.RandomState(3)
    n = 24
    rows = np.arange(n - 1); cols = rows + 1
    graph = sp.csr_matrix((np.ones(n - 1), (rows, cols)), shape=(n, n)); graph = graph + graph.T
    latent = np.repeat(np.asarray([[0.0, 0.0], [2.0, -1.0], [-1.0, 2.0]]), 8, axis=0)
    h1 = latent + 0.08 * rng.randn(n, 2); h2 = latent + 0.08 * rng.randn(n, 2)
    h1[5] += 3.0; h2[17] -= 2.5
    return h1, h2, graph, np.asarray([f"n{x}" for x in range(n)])


def config():
    return TrendConfig("TEST", 0.08, 0.05, iterations=12, trend_dimension=2)


def test_solver_is_finite_monotone_and_private_active():
    h1, h2, graph, _ = toy()
    z, r1, r2, diagnostics = solve_shared_private(h1, h2, graph, config(), "FULL")
    assert np.isfinite(z).all() and np.isfinite(r1).all() and np.isfinite(r2).all()
    objective = [row["objective"] for row in diagnostics["trace"]]
    assert all(right <= left + 1e-8 for left, right in zip(objective, objective[1:]))
    assert diagnostics["private1_norm"] > 0 and diagnostics["private2_norm"] > 0


def test_graph_tv_and_private_only_are_distinct():
    h1, h2, graph, _ = toy()
    tv, _, _, _ = solve_shared_private(h1, h2, graph, config(), "GRAPH_TV_ONLY")
    private, _, _, _ = solve_shared_private(h1, h2, graph, config(), "PRIVATE_ONLY")
    assert not np.allclose(tv, private)


def test_permuted_private_is_deterministic_nonidentity():
    h1, h2, graph, ids = toy()
    order1 = deterministic_permutation(ids, "x"); order2 = deterministic_permutation(ids, "x")
    assert np.array_equal(order1, order2) and not np.array_equal(order1, np.arange(len(ids)))
    z1, d1 = solve_permuted_private(h1, h2, graph, config(), ids)
    z2, d2 = solve_permuted_private(h1, h2, graph, config(), ids)
    assert np.array_equal(z1, z2) and d1["permutation1_sha256"] == d2["permutation1_sha256"]


def test_fixed_adapters_and_endpoint_exact_k():
    rng = np.random.RandomState(7)
    a = rng.randn(40, 8); b = rng.randn(40, 11); retained = rng.randn(40, 6)
    h1, h2, diagnostics = shared_view_basis(a, b, 5)
    assert h1.shape == h2.shape == (40, 5)
    assert diagnostics["rotation_orthogonality_error"] < 1e-8
    representation = compose_representation(retained, (h1 + h2) / 2, 0.3)
    partition = common_kmeans_endpoint(representation, 4)
    assert len(np.unique(partition)) == 4


def test_procrustes_coordination_is_rotation_and_sign_invariant():
    rng = np.random.RandomState(19)
    reference = row_normalize(rng.randn(80, 6))
    score_rotation, _ = np.linalg.qr(rng.randn(6, 6))
    target = reference @ score_rotation
    aligned1, _, _ = orthogonal_procrustes_align(reference, target)
    another_rotation, _ = np.linalg.qr(rng.randn(6, 6))
    aligned2, _, _ = orthogonal_procrustes_align(reference, target @ another_rotation)
    aligned3, _, _ = orthogonal_procrustes_align(reference, target * np.asarray([-1, 1, -1, 1, -1, 1]))
    assert np.allclose(aligned1, aligned2, atol=2e-10)
    assert np.allclose(aligned1, aligned3, atol=2e-10)


def test_robust_standardization_handles_constant_columns():
    value = np.column_stack([np.arange(12), np.ones(12)])
    result = robust_standardize(value)
    assert np.isfinite(result).all() and np.allclose(result[:, 1], 0)
