from __future__ import annotations

import ast
import argparse
import csv
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SpaLORA.night16d_cmbf_rl import (  # noqa: E402
    CMBFRLConfig,
    CMBFResidualLearner,
    array_sha256,
    build_tri_state_field,
    canonical_graph,
    encode_partition,
    retained_teacher_representation,
    teacher_representation,
)
from scripts.night16d.night16d_evaluator import metrics, select_family, spatial_metrics  # noqa: E402


def fixture():
    rng = np.random.default_rng(13)
    n = 30
    v1 = rng.normal(size=(n, 8)).astype(np.float32)
    v2 = (v1[:, :6] + 0.3 * rng.normal(size=(n, 6))).astype(np.float32)
    row, col = [], []
    for i in range(n):
        for j in {(i - 1) % n, (i + 1) % n, (i + 5) % n}:
            row.append(i); col.append(j)
    graph = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    initial = np.repeat(np.arange(3), 10).astype(np.int32)
    bank = np.stack([initial, np.roll(initial, 1), np.roll(initial, -1)])
    return v1, v2, graph, initial, bank


def test_tri_state_partition_and_sparse_boundary():
    v1, v2, graph, initial, bank = fixture()
    field = build_tri_state_field(v1, v2, graph, initial, bank, "global")
    assert np.allclose(field.support + field.boundary + field.conflict, 1.0, atol=1e-6)
    assert field.graph.nnz < len(initial) ** 2


def test_node_local_rank_is_distinct_and_finite():
    v1, v2, graph, initial, bank = fixture()
    global_field = build_tri_state_field(v1, v2, graph, initial, bank, "global")
    local_field = build_tri_state_field(v1, v2, graph, initial, bank, "node_local")
    assert np.isfinite(local_field.support).all()
    assert not np.array_equal(global_field.support, local_field.support)


def test_zero_degree_relation_messages_self_return_finite():
    v1, v2, graph, initial, bank = fixture()
    config = CMBFRLConfig(training_steps=1)
    field = build_tri_state_field(v1, v2, graph, initial, bank, config.rank_mode)
    teacher = teacher_representation(v1, v2, initial, config.latent_dim, config.teacher_scale, config.content_scale)
    model = CMBFResidualLearner(v1.shape[1], v2.shape[1], config)
    n = len(initial)
    row = torch.as_tensor(field.row, dtype=torch.long)
    col = torch.as_tensor(field.col, dtype=torch.long)
    base = torch.as_tensor(field.base_weight)
    zeros = torch.zeros_like(base)
    stats = torch.from_numpy(np.stack([field.node_support,field.node_boundary,field.node_conflict,field.rejected_mass,field.start_stability,field.prototype_confidence],axis=1))
    output = model(torch.from_numpy(v1),torch.from_numpy(v2),torch.from_numpy(teacher),stats,torch.from_numpy(field.trust),row,col,base,zeros,zeros,zeros)
    assert output["z"].shape == (n, config.latent_dim)
    assert torch.isfinite(output["z"]).all()


def test_core_has_no_dataset_or_label_routing_literals():
    import SpaLORA.night16d_cmbf_rl as module
    tree = ast.parse(inspect.getsource(module))
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for forbidden in ("A1", "D1", "P22", "MISAR", "tonsil", "labels_primary"):
        assert forbidden not in literals


def test_config_rejects_invalid_semantics():
    try:
        CMBFRLConfig(rank_mode="bad").validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid rank mode accepted")


def test_relations_are_nonnegative_and_not_collapsed_to_one_operator():
    v1, v2, graph, initial, bank = fixture()
    field = build_tri_state_field(v1, v2, graph, initial, bank, "global")
    for value in (field.support, field.boundary, field.conflict):
        assert np.min(value) >= 0.0
        assert np.max(value) <= 1.0
    assert not np.allclose(field.support, field.boundary)
    assert not np.allclose(field.support, field.conflict)


def test_edge_state_shuffle_is_seeded_and_changes_only_relation_assignment():
    v1, v2, graph, initial, bank = fixture()
    a = build_tri_state_field(v1, v2, graph, initial, bank, "global", shuffle_seed=991)
    b = build_tri_state_field(v1, v2, graph, initial, bank, "global", shuffle_seed=991)
    plain = build_tri_state_field(v1, v2, graph, initial, bank, "global")
    assert np.array_equal(a.support, b.support)
    assert np.array_equal(a.graph.indptr, plain.graph.indptr)
    assert np.array_equal(a.graph.indices, plain.graph.indices)
    assert not np.array_equal(a.support, plain.support)


def test_retained_teacher_is_deterministic_and_auditable():
    v1, _, _, initial, _ = fixture()
    retained = np.concatenate([v1, v1[:, :3]], axis=1)
    a = retained_teacher_representation(retained, initial, 16, 1.5, 0.6)
    b = retained_teacher_representation(retained.copy(), initial.copy(), 16, 1.5, 0.6)
    assert a.shape == (len(initial), 16)
    assert array_sha256(a) == array_sha256(b)


def test_partition_encoding_and_graph_shape_fail_closed():
    encoded = encode_partition(np.asarray([8, 8, 3, 9, 3]))
    assert np.array_equal(encoded, np.asarray([1, 1, 0, 2, 0], dtype=np.int32))
    try:
        canonical_graph(sp.eye(4, format="csr"), 5)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched graph shape accepted")


def test_label_arrays_are_unreachable_from_producer_source():
    source = (ROOT / "scripts" / "night16d" / "night16d_train_producer.py").read_text(encoding="utf-8")
    for forbidden in ("labels_primary", "label_mask", "adjusted_rand_score", "normalized_mutual_info_score"):
        assert forbidden not in source


def test_categorical_spatial_metrics_are_label_permutation_invariant():
    _, _, graph, initial, _ = fixture()
    mapping = np.asarray([2, 0, 1], dtype=np.int32)
    permuted = mapping[initial]
    a = spatial_metrics(initial, graph)
    b = spatial_metrics(permuted, graph)
    assert np.allclose(a, b, atol=0.0, rtol=0.0)


def test_full_and_evaluation_mask_cluster_sizes_are_distinct():
    _, _, graph, initial, _ = fixture()
    mask = np.arange(len(initial)) < 20
    result = metrics(initial, {"labels": initial.copy(), "mask": mask, "graph": graph})
    assert json.loads(result["cluster_sizes_full"]) == [10, 10, 10]
    assert json.loads(result["cluster_sizes_eval"]) == [10, 10, 0]
    assert result["min_cluster_size_full"] == 10
    assert result["min_cluster_size_eval"] == 0


def test_family_selector_uses_lane_specific_input_and_head_baselines(tmp_path):
    rows = []
    for lane, input_score, head_score, full_score in (
        ("lane_a", 0.80, 0.70, 0.81),
        ("lane_b", 0.20, 0.10, 0.21),
    ):
        for mode, score, suffix, config_sha in (
            ("teacher", input_score, "input", f"{lane}_input"),
            ("teacher_head", head_score, "head", f"{lane}_head"),
            ("full", full_score, "full", "shared_full"),
        ):
            cfg = {"operation_mode": mode, "training_steps": 0 if mode != "full" else 1}
            rows.append({
                "lane": lane, "candidate_id": f"{lane}_{suffix}", "status": "PASS",
                "config_json": json.dumps(cfg), "config_sha256": config_sha,
                "absolute_ari": score, "absolute_nmi": score,
            })
    source = tmp_path / "input"; source.mkdir()
    with (source / "evaluated_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output = tmp_path / "selected.json"
    select_family(argparse.Namespace(
        input=str(source), discovery_lanes="lane_a,lane_b", family="TEST", output=str(output)
    ))
    selected = json.loads(output.read_text(encoding="utf-8"))["selected"]
    assert np.isclose(selected["lanes"]["lane_a"]["delta_ari_vs_input_strong_start"], 0.01)
    assert np.isclose(selected["lanes"]["lane_b"]["delta_ari_vs_input_strong_start"], 0.01)
    assert np.isclose(selected["lanes"]["lane_a"]["delta_ari_vs_same_head_teacher"], 0.11)
    assert np.isclose(selected["lanes"]["lane_b"]["delta_ari_vs_same_head_teacher"], 0.11)
