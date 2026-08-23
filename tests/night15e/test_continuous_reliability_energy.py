import ast
import inspect
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from SpaLORA import night15e_continuous_reliability_energy as core


def fixture():
    rng = np.random.default_rng(20260824)
    n = 72
    rows = np.repeat(np.arange(n), 4)
    cols = np.column_stack(
        ((np.arange(n) + 1) % n, (np.arange(n) - 1) % n, (np.arange(n) + 6) % n, (np.arange(n) - 6) % n)
    ).reshape(-1)
    graph = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    latent = rng.normal(size=(n, 12)).astype(np.float32)
    view1 = latent[:, :8] + 0.1 * rng.normal(size=(n, 8))
    view2 = latent[:, 4:] + 0.1 * rng.normal(size=(n, 8))
    initial = (np.arange(n) % 4).astype(np.int32)
    evidence = core.prepare_continuous_evidence(graph, latent, view1, view2, retained_dim=8, view_dim=6, edge_dim=5)
    config = core.ContinuousEnergyConfig(
        beta=0.8,
        edge_floor=0.08,
        conflict_center=0.2,
        conflict_temperature=0.1,
        conflict_union_weight=0.55,
        conflict_penalty=0.25,
        mass_center=0.4,
        mass_temperature=0.12,
        neighbor_capacity=0.8,
        low_weight=0.6,
        twohop_weight=0.2,
        high_weight=0.5,
        unary_temperature=0.5,
        retained_bias=1.0,
        view_balance=0.0,
        trust_scale=0.7,
        trust_center=0.2,
        trust_temperature=0.1,
        move_threshold=0.01,
        move_fraction=0.1,
        sweeps=2,
    )
    return graph, initial, evidence, config


def test_transition_is_sparse_mass_preserving_and_has_self_return():
    _, _, evidence, config = fixture()
    transition, diagnostics = core.continuous_reliability_transition(evidence, config)
    assert sp.isspmatrix_csr(transition)
    assert np.allclose(np.asarray(transition.sum(axis=1)).ravel(), 1.0, atol=1e-6)
    assert np.all(transition.diagonal() > 0)
    assert 0 < diagnostics["mean_neighbor_mass"] < 1
    assert diagnostics["mean_self_return"] > 0


def test_deterministic_partition_and_cardinality():
    _, initial, evidence, config = fixture()
    first, first_diag = core.continuous_reliability_energy(initial, 4, evidence, config)
    second, second_diag = core.continuous_reliability_energy(initial, 4, evidence, config)
    assert np.array_equal(first, second)
    assert first_diag == second_diag
    assert len(np.unique(first)) == 4
    assert np.isfinite(first).all()


def test_every_feature_branch_has_positive_weight_and_changes_unary():
    _, initial, evidence, config = fixture()
    unary, diagnostics = core.continuous_prototype_unary(initial, 4, evidence, config)
    altered, _ = core.continuous_prototype_unary(initial, 4, evidence, replace(config, high_weight=2.0))
    assert unary.shape == (len(initial), 4)
    assert not np.allclose(unary, altered)
    assert diagnostics["mean_retained_weight"] > 0
    assert diagnostics["mean_view1_weight"] > 0
    assert diagnostics["mean_view2_weight"] > 0


def test_conflict_and_mass_controls_are_continuous_not_discrete_modes():
    _, _, evidence, config = fixture()
    left, _ = core.continuous_reliability_transition(evidence, replace(config, conflict_union_weight=0.40))
    right, _ = core.continuous_reliability_transition(evidence, replace(config, conflict_union_weight=0.41))
    assert np.linalg.norm(left.data - right.data) > 0
    assert np.linalg.norm(left.data - right.data) < 1.0


def test_core_signature_and_ast_are_identity_blind():
    source = inspect.getsource(core)
    tree = ast.parse(source)
    forbidden = {"dataset", "tissue", "path", "label", "ari", "nmi"}
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (names & forbidden)
    signature = inspect.signature(core.continuous_reliability_energy)
    assert set(signature.parameters) == {"initial", "k", "evidence", "config"}


def test_no_dense_n_by_n_allocation_in_core_source():
    source = inspect.getsource(core)
    assert "np.zeros((len(partition), len(partition)" not in source
    assert "np.ones((len(partition), len(partition)" not in source
    assert "np.eye(len(partition)" not in source


def test_empty_cluster_fails_closed():
    _, initial, evidence, config = fixture()
    broken = initial.copy()
    broken[broken == 3] = 2
    try:
        core.continuous_reliability_energy(broken, 4, evidence, config)
    except ValueError as error:
        assert "cardinality" in str(error)
    else:
        raise AssertionError("cardinality mismatch did not fail closed")
