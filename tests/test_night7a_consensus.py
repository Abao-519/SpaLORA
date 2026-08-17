import json
import ast
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from SpaLORA.night7a_consensus import (
    CANDIDATE_ORDER, G00, G04, VIEWS, affinity_audit,
    candidate_affinity, canonical_csr, canonical_partition,
    local_reliability, parse_registry, partition_sha,
    sparse_sha, sparse_snf10, spatial_affinity, _topk_rows,
)
from SpaLORA.night6c_pipeline import _neighbors, row_normalize, self_tuning_affinity, spectral


def symmetric(values):
    x = sp.csr_matrix(np.asarray(values, dtype=float))
    x = x.maximum(x.T); x.setdiag(0); x.eliminate_zeros()
    return x


def toy_base(n=12):
    ids = [f"s{i:02d}" for i in range(n)]
    a = np.zeros((n, n), dtype=float)
    b = np.zeros((n, n), dtype=float)
    for i in range(n):
        for d in range(1, 7):
            a[i, (i + d) % n] = 1 / d
            b[i, (i - d) % n] = 1 / (d + 1)
    s0, s4 = symmetric(a), symmetric(b)
    affinities = {G00: [s0 * .8, s0, s0 * 1.2],
                  G04: [s4 * .7, s4, s4 * 1.3]}
    reliability = {G00: np.linspace(.2, .9, n),
                   G04: np.linspace(.9, .2, n)}
    coords = np.column_stack([np.arange(n), np.arange(n) % 3])
    return ids, {"S_G00": s0, "S_G04": s4,
                 "affinities": affinities, "reliability": reliability,
                 "T_spatial": spatial_affinity(coords, ids)}


def test_registry_exact_order(tmp_path):
    registry = {"candidate_order": list(CANDIDATE_ORDER),
                "candidates": [{"id": x, "formula": x} for x in CANDIDATE_ORDER]}
    assert list(parse_registry(registry)) == list(CANDIDATE_ORDER)
    registry["candidate_order"] = list(reversed(CANDIDATE_ORDER))
    with pytest.raises(RuntimeError):
        parse_registry(registry)


def test_c02_is_six_view_mean():
    ids, base = toy_base()
    observed, _ = candidate_affinity("C02_DUAL_ARITHMETIC_MEAN", base, ids)
    matrices = base["affinities"][G00] + base["affinities"][G04]
    expected = canonical_csr(sum(matrices[1:], matrices[0]) * (1 / 6))
    diff = canonical_csr(observed - expected)
    assert diff.nnz == 0 or np.max(np.abs(diff.data)) <= 1e-12


def test_sparse_max_min_and_harmonic():
    ids, base = toy_base()
    s0, s4 = base["S_G00"], base["S_G04"]
    maximum, _ = candidate_affinity("C03_DUAL_ELEMENTWISE_MAX", base, ids)
    minimum, _ = candidate_affinity("C04_DUAL_ELEMENTWISE_MIN", base, ids)
    harmonic, _ = candidate_affinity("C05_DUAL_HARMONIC_INTERSECTION", base, ids)
    assert np.allclose(maximum.toarray(), np.maximum(s0.toarray(), s4.toarray()))
    assert np.allclose(minimum.toarray(), np.minimum(s0.toarray(), s4.toarray()))
    dense0, dense4 = s0.toarray(), s4.toarray()
    expected = np.where((dense0 > 0) & (dense4 > 0),
                        2 * dense0 * dense4 / (dense0 + dense4), 0)
    assert np.allclose(harmonic.toarray(), expected)


def test_c06_strict_zero_row_and_symmetry():
    ids, base = toy_base()
    observed, _ = candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN", base, ids)
    audit = affinity_audit(observed)
    assert audit["symmetry_max_error"] <= 1e-12
    assert audit["zero_degree_count"] == 0
    broken = dict(base); broken["S_G00"] = sp.csr_matrix(base["S_G00"].shape)
    with pytest.raises(ValueError):
        candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN", broken, ids)


def test_c07_jaccard_hand_calculation():
    sets = [
        np.array([[1, 2], [0, 2]]),
        np.array([[1, 3], [0, 3]]),
        np.array([[1, 2], [0, 2]]),
    ]
    # Jaccards per row are 1/3, 1, 1/3; mean = 5/9.
    assert np.allclose(local_reliability(sets), [5 / 9, 5 / 9])
    ids, base = toy_base()
    observed, extra = candidate_affinity("C07_DUAL_LOCAL_RELIABILITY", base, ids)
    assert sp.issparse(observed)
    assert np.all((extra["g00_weights"] > 0) & (extra["g00_weights"] < 1))


def test_c08_support_weighted_positive_median_hand_value():
    ids, base = toy_base()
    shape = base["S_G00"].shape
    mats = []
    for value in (1, 2, 3, 4, 0, 0):
        x = sp.csr_matrix(([value] if value else [],
                           ([0] if value else [], [1] if value else [])), shape=shape)
        mats.append(x)
    base["affinities"] = {G00: mats[:3], G04: mats[3:]}
    observed, _ = candidate_affinity("C08_SIX_VIEW_SUPPORT_MEDIAN", base, ids)
    # median(1,2,3,4)=2.5 and support is 4/6.
    assert observed[0, 1] == pytest.approx(2.5 * 4 / 6)


def test_c09_simultaneous_ten_iteration_determinism():
    ids, base = toy_base(12)
    first = sparse_snf10(base["S_G00"], base["S_G04"], ids)
    second = sparse_snf10(base["S_G00"], base["S_G04"], ids)
    assert sparse_sha(first) == sparse_sha(second)
    audit = affinity_audit(first)
    assert audit["symmetry_max_error"] <= 1e-12
    assert audit["diagonal_max_abs"] <= 1e-12


def test_c09_complete_graph_has_hand_computed_fixed_point():
    # For n=11, top-10 retains every off-diagonal edge.  With equal S_G00 and
    # S_G04, P has diagonal .5 and each of ten off-diagonals .05; K has each
    # off-diagonal .1.  K P K^T followed by the locked P convention returns
    # the same P, so all ten iterations and the final blend give .05 off diag.
    n = 11
    ids = [f"s{i:02d}" for i in range(n)]
    dense = np.ones((n, n), dtype=float) - np.eye(n)
    observed = sparse_snf10(sp.csr_matrix(dense), sp.csr_matrix(dense), ids)
    expected = np.full((n, n), .05, dtype=float)
    np.fill_diagonal(expected, 0.0)
    assert np.allclose(observed.toarray(), expected, atol=1e-14, rtol=0)


def test_c09_topk_uses_observation_id_tie_break():
    n = 12
    ids = ["z", "a", "y", "b", "x", "c", "w", "d", "v", "e", "u", "f"]
    dense = np.ones((n, n), dtype=float) - np.eye(n)
    observed = _topk_rows(sp.csr_matrix(dense), 10, ids, keep_diagonal=False)
    lexical = sorted(range(n), key=lambda i: ids[i])
    for row in range(n):
        assert observed.getrow(row).indices.tolist() == sorted(
            [index for index in lexical if index != row][:10]
        )


def test_c10_c11_spatial_blends_exact():
    ids, base = toy_base()
    c06, _ = candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN", base, ids)
    c10, _ = candidate_affinity("C10_DUAL_MEAN_SPATIAL05", base, ids)
    c11, _ = candidate_affinity("C11_DUAL_MEAN_SPATIAL10", base, ids)
    assert np.allclose(c10.toarray(), .95 * c06.toarray() + .05 * base["T_spatial"].toarray())
    assert np.allclose(c11.toarray(), .90 * c06.toarray() + .10 * base["T_spatial"].toarray())
    coords = np.column_stack([np.arange(12), np.arange(12) % 4]).astype(float)
    neighbors = _neighbors(coords, 6, "euclidean", ids)
    directed = sp.coo_matrix((np.ones(12 * 6),
                              (np.repeat(np.arange(12), 6), neighbors.reshape(-1))),
                             shape=(12, 12)).tocsr()
    binary = directed.maximum(directed.T)
    degree = np.asarray(binary.sum(axis=1)).ravel()
    p = sp.diags(1 / degree) @ binary
    expected_spatial = canonical_csr((p + p.T) * .5)
    expected_spatial.setdiag(0); expected_spatial.eliminate_zeros()
    assert sparse_sha(spatial_affinity(coords, ids)) == sparse_sha(expected_spatial)


def test_partition_canonical_relabeling():
    assert np.array_equal(canonical_partition([7, 7, 2, 9, 2]), [0, 0, 1, 2, 1])
    assert partition_sha([7, 7, 2, 9, 2]) == partition_sha([3, 3, 8, 4, 8])


def test_h05_row_l2_knn_tie_break_and_kernel_hand_formula():
    rng = np.random.RandomState(7)
    values = rng.normal(size=(14, 5))
    ids = [f"obs-{value:02d}" for value in reversed(range(14))]
    normalized = row_normalize(values)
    neighbors = _neighbors(normalized, 10, "euclidean", ids)
    assert all(row not in neighbors[row] for row in range(len(values)))
    # Independent construction of the directed local-sigma kernel followed by
    # maximum symmetrization and a zero diagonal.
    rows = np.repeat(np.arange(len(values)), 10)
    cols = neighbors.reshape(-1)
    distances = np.linalg.norm(normalized[rows] - normalized[cols], axis=1).reshape(len(values), 10)
    sigma = np.maximum(distances[:, -1], 1e-12)
    weights = np.exp(-(distances.reshape(-1) ** 2) /
                     np.maximum(sigma[rows] * sigma[cols], 1e-12))
    expected = sp.coo_matrix((weights, (rows, cols)), shape=(len(values), len(values))).tocsr()
    expected = expected.maximum(expected.T); expected.setdiag(0); expected.eliminate_zeros()
    observed = self_tuning_affinity(values, 10, ids)
    assert sparse_sha(observed) == sparse_sha(expected)
    assert affinity_audit(observed)["diagonal_max_abs"] == 0.0

    # Equidistant zero vectors prove that observation-ID lexical order, not
    # backend return order, resolves the k-neighbor boundary.
    zeros = np.zeros((12, 3), dtype=float)
    tied_ids = ["z", "a", "y", "b", "x", "c", "w", "d", "v", "e", "u", "f"]
    tied = _neighbors(zeros, 10, "euclidean", tied_ids)
    expected_indices = sorted(range(12), key=lambda i: tied_ids[i])
    for row in range(12):
        assert tied[row].tolist() == [i for i in expected_indices if i != row][:10]


def test_consensus_module_has_no_dense_sparse_matrix_conversion():
    source = (Path(__file__).resolve().parents[1] / "SpaLORA/night7a_consensus.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in {"toarray", "todense"}
                   for node in calls)


def test_nonfinite_affinity_is_explicitly_audited():
    matrix = sp.csr_matrix(([np.nan, np.nan], ([0, 1], [1, 0])), shape=(3, 3))
    assert affinity_audit(matrix)["finite"] is False


def test_spectral_fixed_parameters_and_repeat_determinism():
    first = np.ones((6, 6), dtype=float) - np.eye(6)
    second = np.ones((6, 6), dtype=float) - np.eye(6)
    matrix = sp.block_diag((first, second), format="csr")
    labels1 = spectral(matrix, 2)
    labels2 = spectral(matrix, 2)
    assert partition_sha(labels1) == partition_sha(labels2)
    source = (Path(__file__).resolve().parents[1] / "SpaLORA/night6c_pipeline.py").read_text()
    assert 'affinity="precomputed"' in source
    assert 'assign_labels="discretize"' in source
    assert "n_init=20" in source and "random_state=2020" in source


def test_fixed_transform_primary_key_cardinality():
    dataset_seed_count = 5 + 5 + 10 + 10
    assert dataset_seed_count * len(CANDIDATE_ORDER) == 360
