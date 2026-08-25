import numpy as np
import scipy.sparse as sp

from SpaLORA.night18a_backbone import (
    BackboneConfig, decode_embedding, feature_graph, prepare_graph,
    reload_representation, sha256_array, train_backbone,
)


def toy():
    rng = np.random.default_rng(7); n = 42
    x1 = rng.normal(size=(n, 8)).astype(np.float32); x2 = rng.normal(size=(n, 6)).astype(np.float32)
    retained = rng.normal(size=(n, 10)).astype(np.float32)
    rows = np.arange(n); graph = sp.csr_matrix((np.ones(n * 2), (np.repeat(rows, 2), np.column_stack([(rows-1)%n,(rows+1)%n]).reshape(-1))), shape=(n,n))
    return x1, x2, retained, graph


def test_sparse_feature_graph_and_no_dense():
    x1, _, _, _ = toy(); graph = feature_graph(x1, k=4)
    assert sp.isspmatrix_csr(graph) and graph.nnz < len(x1) * 12


def test_real_train_parameter_change_and_strict_reload():
    x1, x2, retained, graph = toy(); config = BackboneConfig("T", hidden_dim=12, residual_scale=.05, steps=3)
    rep, state, diagnostics = train_backbone(x1, x2, retained, graph, 3, config, 0, "cpu")
    replay = reload_representation(x1, x2, retained, graph, 3, config, state, "cpu")
    assert diagnostics["parameter_changed"] and diagnostics["optimizer_steps"] == 3
    assert sha256_array(rep) == sha256_array(replay) and np.array_equal(rep, replay)


def test_seed_enters_training():
    x1, x2, retained, graph = toy(); config = BackboneConfig("T", hidden_dim=12, residual_scale=.05, steps=2)
    left, _, _ = train_backbone(x1, x2, retained, graph, 3, config, 0, "cpu")
    right, _, _ = train_backbone(x1, x2, retained, graph, 3, config, 1, "cpu")
    assert sha256_array(left) != sha256_array(right)


def test_decoder_exact_k_feasibility_and_shared_budget():
    x1, x2, retained, graph = toy(); graphs = [prepare_graph(graph), prepare_graph(graph), prepare_graph(graph)]
    partitions, records, selections = decode_embedding(retained, x1, x2, graphs, 3)
    assert partitions.shape == (15, 42)
    assert all(len(np.unique(row)) == 3 for row in partitions)
    assert set(selections) == {"COMMON_KMEANS", "COMMON_GMM", "FEASIBLE_MEDOID", "NIGHT16H_FIXED_STRUCTURED"}
    assert len({row["candidate_id"] for row in records}) == 15


def test_graph_order_determinism():
    _, _, _, graph = toy(); prepared = prepare_graph(graph)
    permuted = sp.coo_matrix((graph.data[::-1], (graph.tocoo().row[::-1], graph.tocoo().col[::-1])), shape=graph.shape).tocsr()
    other = prepare_graph(permuted)
    assert np.array_equal(prepared.indptr, other.indptr) and np.array_equal(prepared.indices, other.indices)
    assert np.allclose(prepared.data, other.data)
