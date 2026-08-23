import pathlib
import inspect
import sys

import numpy as np
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15c_cluster_energy import (
    dynamic_prototype_icm,
    multiscale_edge_conductance,
    partition_bank_unary,
    potts_icm,
    potts_mean_field,
    pseudo_fisher_transform,
    raw_bimodal_edge_conductance,
    reduced_controlled,
    spectral_partition,
)
from scripts.night15c.night15c_build_handoff import classify_metric_delta


def chain_graph(n: int) -> sp.csr_matrix:
    row = np.arange(n - 1)
    return sp.coo_matrix(
        (np.ones(2 * (n - 1)), (np.r_[row, row + 1], np.r_[row + 1, row])),
        shape=(n, n),
    ).tocsr()


def test_potts_uses_unary_and_pairwise():
    graph = chain_graph(6)
    initial = np.array([0, 0, 1, 0, 1, 1])
    unary = np.full((6, 2), 1.0, dtype=np.float32)
    unary[np.arange(6), np.array([0, 0, 0, 1, 1, 1])] = 0.0
    icm = potts_icm(unary, graph, initial, pairwise_strength=0.5, iterations=5)
    mf = potts_mean_field(unary, graph, initial, pairwise_strength=0.5, iterations=8)
    assert np.array_equal(icm, np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(mf, np.array([0, 0, 0, 1, 1, 1]))


def test_dynamic_icm_preserves_requested_cardinality():
    graph = chain_graph(12)
    x = np.r_[np.zeros((6, 2)), np.ones((6, 2))].astype(np.float32)
    initial = np.array([0] * 5 + [1] + [0] + [1] * 5)
    observed, steps, collapse = dynamic_prototype_icm(x, graph, initial, 2, 1.0, 10)
    assert len(np.unique(observed)) == 2
    assert steps >= 1
    assert isinstance(collapse, bool)


def test_zero_steps_is_exact_noop():
    graph = chain_graph(12)
    x = np.random.default_rng(7).normal(size=(12, 4)).astype(np.float32)
    initial = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    observed, steps, collapse = dynamic_prototype_icm(x, graph, initial, 2, 0.0, 0)
    assert np.array_equal(observed, initial)
    assert steps == 0
    assert collapse is False


def test_anisotropic_conductance_is_sparse_and_deterministic():
    graph = chain_graph(8)
    view1 = np.column_stack((np.arange(8), np.arange(8) % 2)).astype(np.float32)
    view2 = np.column_stack((np.arange(8) * 2, np.arange(8) % 3)).astype(np.float32)
    first = multiscale_edge_conductance(graph, view1, view2, graph, graph)
    second = multiscale_edge_conductance(graph, view1, view2, graph, graph)
    assert sp.issparse(first)
    assert first.nnz <= graph.nnz
    assert np.array_equal(first.indptr, second.indptr)
    assert np.allclose(first.data, second.data)


def test_raw_bimodal_max_and_min_have_expected_order():
    graph = chain_graph(10)
    first = np.column_stack((np.arange(10), np.arange(10) % 2)).astype(np.float32)
    second = np.column_stack((np.arange(10) % 3, np.arange(10) * 2)).astype(np.float32)
    either = raw_bimodal_edge_conductance(graph, first, second, "either_similar", dim=2)
    both = raw_bimodal_edge_conductance(graph, first, second, "both_similar", dim=2)
    assert sp.issparse(either) and sp.issparse(both)
    assert np.all(either.data + 1e-7 >= both.data)


def test_full_solver_is_byte_deterministic():
    value = np.random.default_rng(11).normal(size=(80, 12)).astype(np.float32)
    first = reduced_controlled(value, 7, "full")
    second = reduced_controlled(value, 7, "full")
    assert np.array_equal(first, second)


def test_model_core_has_no_dataset_or_ground_truth_argument():
    for function in (dynamic_prototype_icm, raw_bimodal_edge_conductance):
        parameters = set(inspect.signature(function).parameters)
        assert "dataset" not in parameters
        assert "labels" not in parameters
        assert "ground_truth" not in parameters


def test_score_delta_classifier_rejects_float_noop_residue():
    assert classify_metric_delta(5.551115123125783e-17, 5.551115123125783e-17) == "NONE"
    assert classify_metric_delta(0.01, 0.02) == "DUAL"
    assert classify_metric_delta(0.01, -0.02) == "ARI_ONLY"


def test_partition_bank_unary_has_no_label_argument():
    graph = chain_graph(8)
    bank = [
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 0, 0, 0, 0]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1]),
    ]
    unary, consensus, diagnostics = partition_bank_unary(bank, graph, 2)
    assert unary.shape == (8, 2)
    assert consensus.shape == (8,)
    assert np.isfinite(unary).all()
    assert diagnostics["global_weights"].shape == (3,)


def test_sparse_spectral_and_pseudo_fisher_shapes():
    graph = chain_graph(30) + sp.eye(30, format="csr") * 0.1
    partition, vectors = spectral_partition(graph, 3, seed=0, n_init=2)
    transformed = pseudo_fisher_transform(
        np.random.default_rng(0).normal(size=(30, 6)), partition, 3
    )
    assert partition.shape == (30,)
    assert vectors.shape == (30, 3)
    assert transformed.shape == (30, 6)
    assert np.isfinite(transformed).all()
