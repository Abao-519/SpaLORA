import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night21a_amcf import common_kmeans_endpoint, make_config, multiscale_texture_bank, prepare_graph, reload_amcf, sha256_array, train_amcf


def toy():
    rng = np.random.default_rng(4); n = 36
    x1 = rng.normal(size=(n, 7)).astype(np.float32); x2 = rng.normal(size=(n, 5)).astype(np.float32); z = rng.normal(size=(n, 9)).astype(np.float32)
    row = np.repeat(np.arange(n), 2); col = np.column_stack(((np.arange(n)-1)%n, (np.arange(n)+1)%n)).reshape(-1)
    g = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n,n)); return x1, x2, z, [g, g, g]


def test_texture_bank_schema_and_sparse_graph():
    x1, _, _, graphs = toy(); bank, names = multiscale_texture_bank(x1, graphs, True)
    assert bank.shape == (7, 36, 7) and names[0] == "SELF" and names[-1] == "GRADIENT_S2"
    assert all(sp.isspmatrix_csr(prepare_graph(graph)) for graph in graphs)


def test_zero_start_train_change_reload_and_exact_k():
    x1, x2, z, graphs = toy(); config = make_config("FULL_COMPOSITION", steps=3)
    rep, state, diag = train_amcf(x1, x2, z, graphs, 3, config, 0, "cpu")
    replay = reload_amcf(x1, x2, z, graphs, 3, config, 0, state)
    assert diag["identity_initial_max_abs"] == 0.0 and diag["parameter_changed"]
    assert np.array_equal(rep, replay) and sha256_array(rep) == sha256_array(replay)
    assert len(np.unique(common_kmeans_endpoint(rep, 3))) == 3


def test_registered_graph_coordinate_rotation_semantics():
    x1, _, _, graphs = toy(); left, _ = multiscale_texture_bank(x1, graphs, True)
    # Coordinates are not consumed after graph registration; rigid coordinate
    # rotation/scaling cannot change the same registered sparse operators.
    right, _ = multiscale_texture_bank(x1, graphs, True)
    assert np.array_equal(left, right)


def test_node_permutation_equivariance_texture_bank():
    x1, _, _, graphs = toy(); order = np.random.default_rng(8).permutation(len(x1)); inverse = np.argsort(order)
    left, _ = multiscale_texture_bank(x1, graphs, True)
    permuted_graphs = [graph[order][:, order] for graph in graphs]
    right, _ = multiscale_texture_bank(x1[order], permuted_graphs, True)
    assert np.allclose(left, right[:, inverse], atol=1e-6)


def test_atomic_arms_enter_distinct_paths():
    x1, x2, z, graphs = toy(); hashes = []
    for arm in ("MEAN_ONLY_ANCHORED", "ANCHOR_WITHOUT_GRADIENT", "ANCHOR_WITHOUT_HIERARCHICAL_FUSION", "FULL_COMPOSITION"):
        rep, _, _ = train_amcf(x1, x2, z, graphs, 3, make_config(arm, steps=2), 0, "cpu"); hashes.append(sha256_array(rep))
    assert len(set(hashes)) == len(hashes)
