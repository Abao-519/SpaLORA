from __future__ import annotations

import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "night21c"))

from SpaLORA.night21c_official_math import (
    OfficialMathConfig,
    build_official_model,
    common_kmeans,
    official_dense_loss,
    official_fused_embedding,
    official_preprocess_graph,
    scipy_to_torch_sparse,
    set_determinism,
)
from night21c_endpoint_producer import mutual_knn, spectral_partition


def snapshot_root() -> str:
    return str(ROOT / "third_party" / "night21c_spamgcn_fixed")


def test_official_graph_preprocess_matches_dense_reference():
    graph = sp.csr_matrix(np.array([[0, 0.2, 0], [0.25, 0, 0.5], [0, 0, 0]], dtype=float))
    got = official_preprocess_graph(graph).toarray()
    support = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) + np.eye(3)
    degree = support.sum(axis=1)
    expected = np.diag(1 / np.sqrt(degree)) @ support @ np.diag(1 / np.sqrt(degree))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_tiny_dense_path_backward_and_encoder_equivalence():
    set_determinism(7)
    config = OfficialMathConfig("TEST", epochs=2, learning_rate=1e-4, sigma=0.7, loss_n=0.01)
    graph = official_preprocess_graph(sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0],
    ], dtype=float)))
    adj = scipy_to_torch_sparse(graph, torch.device("cpu"))
    dense = adj.to_dense()
    x1 = torch.randn(6, 5)
    x2 = torch.randn(6, 4)
    model = build_official_model(snapshot_root(), 5, 4, 3, config, torch.device("cpu"))
    before = [x.detach().clone() for x in model.parameters()]
    total, pieces, fused_forward = official_dense_loss(model, x1, x2, adj, dense, config, 1)
    assert total.isfinite() and all(value.isfinite() for value in pieces.values())
    np.testing.assert_allclose(
        fused_forward.detach().numpy(), official_fused_embedding(model, x1, x2, adj).detach().numpy(),
        rtol=1e-6, atol=1e-6,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad(); total.backward(); optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_strict_reload_and_exact_k():
    set_determinism(11)
    config = OfficialMathConfig("TEST", epochs=1, learning_rate=1e-4, sigma=0.7, loss_n=0.01)
    graph = official_preprocess_graph(sp.csr_matrix(np.eye(8, k=1) + np.eye(8, k=-1)))
    adj = scipy_to_torch_sparse(graph, torch.device("cpu"))
    x1, x2 = torch.randn(8, 3), torch.randn(8, 4)
    first = build_official_model(snapshot_root(), 3, 4, 2, config, torch.device("cpu"))
    state = first.state_dict()
    second = build_official_model(snapshot_root(), 3, 4, 2, config, torch.device("cpu"))
    incompatible = second.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    with torch.no_grad():
        a = official_fused_embedding(first, x1, x2, adj).numpy()
        b = official_fused_embedding(second, x1, x2, adj).numpy()
    assert np.array_equal(a, b)
    partition = common_kmeans(a, 2, 0, 20)
    assert np.unique(partition).size == 2


def test_sparse_endpoint_is_deterministic_exact_k():
    rng = np.random.default_rng(91)
    value = rng.normal(size=(36, 7))
    operator = mutual_knn(value, 7)
    assert sp.issparse(operator) and operator.shape == (36, 36)
    first = spectral_partition(operator, 3, 2, 0)
    second = spectral_partition(operator, 3, 2, 0)
    assert np.array_equal(first, second)
    assert np.unique(first).size == 3
