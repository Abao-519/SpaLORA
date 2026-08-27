from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night22a_geometry import array_sha, generate_geometry_bank
from SpaLORA.night22a_junction import JunctionConfig, graph_bank


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "night22b" / "frozen_transfer_contract.json"
PARENT = ROOT / "configs" / "night22a" / "stage_b_junction_freeze_v2.json"


def evaluator_module():
    path = ROOT / "scripts" / "night22b" / "night22b_independent_evaluator.py"
    spec = importlib.util.spec_from_file_location("night22b_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_j01_matches_parent_exactly():
    contract = json.loads(CONTRACT.read_text())
    parent = json.loads(PARENT.read_text())
    observed = contract["profiles"]
    expected = [row for row in parent["profiles"] if row["profile_id"] == "J01_GRAPH_LEAN"]
    assert observed == expected
    assert contract["graph_neighbors"] == parent["graph_neighbors"] == [12, 24]
    assert contract["arms"] == parent["arms"]


def test_required_primary_and_sensitivity_are_preregistered():
    contract = json.loads(CONTRACT.read_text())
    assert contract["primary_start"] == "GEOM_LEIDEN_FEATURE"
    assert contract["sensitivity_starts"] == ["GEOM_FEATURE_NCUT_K24", "NIGHT16H_FROZEN_SELECTOR"]


def test_geometry_bank_contains_exact_k_on_nontrivial_sparse_graph():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(-2, 0.2, (10, 4)), rng.normal(2, 0.2, (10, 4))]
    rows = np.arange(19)
    graph = sp.csr_matrix((np.ones(38), (np.r_[rows, rows + 1], np.r_[rows + 1, rows])), shape=(20, 20))
    bank = generate_geometry_bank(x, graph, 2)
    names = set(bank.candidate_ids.tolist())
    resolved = bank.diagnostics["candidate_aliases"].get("GEOM_FEATURE_NCUT_K24", "GEOM_FEATURE_NCUT_K24")
    assert resolved in names
    assert all(np.unique(partition).size == 2 for partition in bank.partitions)


def test_five_graph_bank_is_sparse_nonnegative_and_mass_normalized():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 5))
    graph = sp.diags(np.ones(29), 1) + sp.diags(np.ones(29), -1)
    graphs = graph_bank(x, x[:, :3], x[:, 2:], graph, neighbors=4, secondary_neighbors=7)
    assert list(graphs) == [
        "retained_feature_k4", "retained_feature_k7", "view1_feature_k4", "view2_feature_k4", "registered_spatial"
    ]
    for value in graphs.values():
        assert sp.issparse(value)
        assert value.nnz > 0
        assert np.all(value.data >= 0)
        assert np.isclose(float(value.sum()), 1.0)


def test_internal_edge_support_detects_isolated_cluster():
    module = evaluator_module()
    graph = sp.csr_matrix(
        (np.ones(4), ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(5, 5)
    )
    partition = np.asarray([0, 0, 1, 1, 2], dtype=np.int32)
    assert module.internal_edge_support(partition, graph, 3) == [1, 1, 0]


def test_array_sha_is_label_permutation_sensitive_but_stable():
    value = np.asarray([0, 1, 0, 0], dtype=np.int32)
    assert array_sha(value) == array_sha(value.copy())
    assert array_sha(value) != array_sha(value[::-1].copy())


def test_junction_config_constructs_from_frozen_profile():
    contract = json.loads(CONTRACT.read_text())
    config = JunctionConfig(**contract["profiles"][0]["config"])
    assert config.steps == 240
    assert config.learning_rate == 0.025
    assert config.emission_weight == 0.2
    assert config.graph_weight == 1.3
