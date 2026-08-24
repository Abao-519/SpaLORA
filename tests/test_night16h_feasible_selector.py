import ast
import argparse
import csv
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from SpaLORA.night16h_feasible_selector import (
    feasibility_record,
    graph_support,
    is_feasible,
    select_candidate,
)


def chain_graph(n: int) -> sp.csr_matrix:
    row = np.r_[np.arange(n - 1), np.arange(1, n)]
    col = np.r_[np.arange(1, n), np.arange(n - 1)]
    return sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))


def test_singleton_fails_smallest_scale_support():
    graph = chain_graph(6)
    record = feasibility_record([graph, graph, graph], np.array([0, 0, 0, 1, 1, 2]), 3)
    assert record["min_cluster_size"] == 1
    assert not is_feasible(record, "NO_SINGLETON")
    assert not is_feasible(record, "SMALLEST_SCALE_INTERNAL_EDGE")


def test_each_cluster_requires_internal_edge():
    graph = chain_graph(6)
    supported = graph_support(graph, np.array([0, 0, 1, 1, 2, 2]), 3)
    assert supported["min_internal_edges"] == 1
    record = feasibility_record([graph, graph, graph], np.array([0, 0, 1, 1, 2, 2]), 3)
    assert is_feasible(record, "SMALLEST_SCALE_INTERNAL_EDGE")


def test_cluster_label_permutation_invariant():
    graph = chain_graph(8)
    left = feasibility_record([graph] * 3, np.array([0, 0, 0, 1, 1, 2, 2, 2]), 3)
    right = feasibility_record([graph] * 3, np.array([7, 7, 7, 4, 4, 9, 9, 9]), 3)
    assert left == right


def test_candidate_order_tie_break_is_id_stable():
    records = [
        {"candidate_id": "B", "molecular_joint": 1, "topology_joint": 1, "microcluster_score": 1},
        {"candidate_id": "A", "molecular_joint": 1, "topology_joint": 1, "microcluster_score": 1},
    ]
    partitions = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    similarity = np.ones((2, 2))
    result = select_candidate(records, partitions, np.array([True, True]), similarity, "EQUAL_RANK")
    assert records[result.selected_index]["candidate_id"] == "A"


def test_content_adaptive_spearman_api_in_real_scipy():
    records = [
        {"candidate_id": "A", "molecular_joint": 1, "topology_joint": 3, "microcluster_score": 2},
        {"candidate_id": "B", "molecular_joint": 2, "topology_joint": 1, "microcluster_score": 3},
        {"candidate_id": "C", "molecular_joint": 3, "topology_joint": 2, "microcluster_score": 1},
    ]
    partitions = np.array(
        [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 2, 2], [0, 0, 1, 2, 1, 2]],
        dtype=np.int32,
    )
    similarity = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    result = select_candidate(
        records, partitions, np.ones(3, dtype=bool), similarity, "CONTENT_ADAPTIVE"
    )
    assert result.selected_index in (0, 1, 2)
    assert np.isclose(sum(result.axis_weights), 1.0)


def test_cross_evidence_arbitration_uses_molecular_when_topology_is_supported():
    records = [
        {"candidate_id": "M", "molecular_joint": 3, "topology_joint": 2, "microcluster_score": 1},
        {"candidate_id": "T", "molecular_joint": 1, "topology_joint": 3, "microcluster_score": 2},
        {"candidate_id": "L", "molecular_joint": 2, "topology_joint": 1, "microcluster_score": 3},
    ]
    partitions = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.int32)
    result = select_candidate(
        records,
        partitions,
        np.ones(3, dtype=bool),
        np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.2], [0.9, 0.2, 1.0]]),
        "CROSS_EVIDENCE_ARBITRATION",
    )
    assert records[result.selected_index]["candidate_id"] == "M"
    assert result.diagnostics["decision"] == "MOLECULAR_CHAMPION"


def test_cross_evidence_arbitration_falls_back_to_topology_and_is_order_invariant():
    records = [
        {"candidate_id": "M", "molecular_joint": 3, "topology_joint": 1, "microcluster_score": 1},
        {"candidate_id": "T", "molecular_joint": 1, "topology_joint": 3, "microcluster_score": 2},
        {"candidate_id": "B", "molecular_joint": 2, "topology_joint": 2, "microcluster_score": 3},
    ]
    partitions = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.int32)
    similarity = np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.2], [0.9, 0.2, 1.0]])
    first = select_candidate(
        records, partitions, np.ones(3, dtype=bool), similarity, "CROSS_EVIDENCE_ARBITRATION"
    )
    order = np.array([2, 0, 1])
    second = select_candidate(
        [records[i] for i in order],
        partitions[order],
        np.ones(3, dtype=bool),
        similarity[np.ix_(order, order)],
        "CROSS_EVIDENCE_ARBITRATION",
    )
    assert records[first.selected_index]["candidate_id"] == "T"
    assert [records[i] for i in order][second.selected_index]["candidate_id"] == "T"


def test_core_has_no_dataset_or_annotation_argument():
    path = Path(__file__).parents[1] / "SpaLORA/night16h_feasible_selector.py"
    tree = ast.parse(path.read_text())
    forbidden = {"dataset", "lane", "annotation", "label", "ari", "nmi"}
    args = {arg.arg.lower() for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for arg in node.args.args}
    assert not (args & forbidden)


def test_no_dense_observation_matrix_in_graph_support():
    graph = chain_graph(1000)
    partition = np.repeat(np.arange(10), 100)
    result = graph_support(graph, partition, 10)
    assert len(result["internal_edge_counts"]) == 10


def test_zero_feasible_candidates_fail_closed():
    records = [
        {"candidate_id": "A", "molecular_joint": 1, "topology_joint": 1, "microcluster_score": 1}
    ]
    with pytest.raises(ValueError, match="no structurally feasible candidate"):
        select_candidate(
            records,
            np.array([[0, 0, 1, 1]], dtype=np.int32),
            np.array([False]),
            np.ones((1, 1)),
            "CROSS_EVIDENCE_ARBITRATION",
        )


def test_scout_records_zero_feasible_mode_without_crashing(tmp_path):
    from scripts.night16h.scout_selectors import run

    feature = tmp_path / "features.csv"
    fields = [
        "candidate_id",
        "molecular_joint",
        "topology_joint",
        "microcluster_score",
    ] + [f"feasible_{mode}" for mode in (
        "UNCONSTRAINED",
        "NO_SINGLETON",
        "ANY_SCALE_INTERNAL_EDGE",
        "SMALLEST_SCALE_INTERNAL_EDGE",
        "DEGREE_DERIVED_INTERNAL_EDGE",
    )]
    with feature.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "candidate_id": "A",
            "molecular_joint": 1,
            "topology_joint": 1,
            "microcluster_score": 1,
            **{field: False for field in fields if field.startswith("feasible_")},
        })
    bank = tmp_path / "bank.npz"
    np.savez_compressed(
        bank,
        partitions=np.array([[0, 0, 1, 1]], dtype=np.int32),
        candidate_ids=np.array(["A"]),
    )
    evaluation = tmp_path / "evaluation.csv"
    with evaluation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id"])
        writer.writeheader(); writer.writerow({"candidate_id": "A"})
    output = tmp_path / "scout.csv"
    run(argparse.Namespace(lane=[f"X::{feature}::{bank}::{evaluation}"], output=str(output)))
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 45
    assert all(row["status"] == "NO_FEASIBLE_CANDIDATE" for row in rows)
