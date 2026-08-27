import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night22a_junction import (
    GeometryJunction,
    JunctionConfig,
    exact_k_repair,
    hard_partition,
    upper_edges,
)


def toy_graph(n=12):
    rows = np.arange(n - 1)
    cols = rows + 1
    return sp.csr_matrix((np.ones(n - 1), (rows, cols)), shape=(n, n)).maximum(
        sp.csr_matrix((np.ones(n - 1), (cols, rows)), shape=(n, n))
    )


def test_upper_edges_and_direct_parameter_update():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(-1, 0.2, (6, 4)), rng.normal(1, 0.2, (6, 4))]
    start = np.repeat([0, 1], 6)
    edge = upper_edges(toy_graph())
    model = GeometryJunction(x, start, 2, [edge, edge], JunctionConfig(steps=5), "FULL")
    before = model.logits.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss, _ = model.loss(0)
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(model.logits.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.logits.detach())


def test_checkpoint_reload_partition_exact(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(15, 3))
    start = np.arange(15) % 3
    edge = upper_edges(toy_graph(15))
    config = JunctionConfig(steps=2)
    first = GeometryJunction(x, start, 3, [edge], config, "FULL")
    with torch.no_grad():
        first.logits.add_(0.013)
    expected, _, _ = hard_partition(first)
    path = tmp_path / "checkpoint.pt"
    torch.save(first.state_dict(), path)
    second = GeometryJunction(x, start, 3, [edge], config, "FULL")
    second.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    observed, _, _ = hard_partition(second)
    assert np.array_equal(expected, observed)


def test_empty_cluster_repair_is_exact_k():
    partition = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    probabilities = np.array(
        [[.8, .1, .1], [.6, .1, .3], [.7, .2, .1], [.1, .8, .1], [.1, .7, .2], [.1, .6, .3]]
    )
    repaired, count = exact_k_repair(partition, probabilities, 3)
    assert count == 1
    assert np.unique(repaired).tolist() == [0, 1, 2]


def test_full_and_atomic_losses_are_distinct():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 5))
    start = np.arange(12) % 3
    edge = upper_edges(toy_graph())
    config = JunctionConfig(steps=3)
    values = {}
    for arm in ["EMISSION_ONLY", "SHARED_GRAPH_ONLY", "CLUSTER_GRAPH_ONLY", "ADDITIVE_SHARED", "FULL"]:
        torch.manual_seed(0)
        model = GeometryJunction(x, start, 3, [edge, edge], config, arm)
        values[arm] = float(model.loss(0)[0])
    assert values["FULL"] != values["EMISSION_ONLY"]
    assert values["FULL"] != values["SHARED_GRAPH_ONLY"]
    assert values["ADDITIVE_SHARED"] != values["EMISSION_ONLY"]
