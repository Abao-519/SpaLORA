from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from SpaLORA.night16g_basin_selector import (
    BasinSelectorConfig,
    candidate_similarity,
    partition_sha256,
    plain_medoid_index,
    select_evidence_rank,
)


def records() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "p0",
            "start_id": "s0",
            "arm": "path",
            "molecular_joint": 0.3,
            "topology_joint": 0.8,
            "microcluster_score": 0.4,
            "path_id": "s0",
            "path_index": 0,
        },
        {
            "candidate_id": "p1",
            "start_id": "s0",
            "arm": "path",
            "molecular_joint": 0.4,
            "topology_joint": 0.9,
            "microcluster_score": 0.5,
            "path_id": "s0",
            "path_index": 1,
        },
        {
            "candidate_id": "p2",
            "start_id": "s0",
            "arm": "path",
            "molecular_joint": 0.2,
            "topology_joint": 0.7,
            "microcluster_score": 0.3,
            "path_id": "s0",
            "path_index": 2,
        },
        {
            "candidate_id": "other",
            "start_id": "s1",
            "arm": "input",
            "molecular_joint": 0.9,
            "topology_joint": 0.1,
            "microcluster_score": 0.2,
            "path_id": "",
            "path_index": "",
        },
    ]


def partitions() -> np.ndarray:
    return np.asarray(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 2, 2, 1],
            [0, 1, 0, 1, 2, 2],
        ],
        dtype=np.int32,
    )


def test_candidate_similarity_is_cluster_label_permutation_invariant() -> None:
    value = partitions()
    permuted = value.copy()
    permuted[1] = np.asarray([2, 2, 0, 0, 1, 1])
    assert np.array_equal(candidate_similarity(value), candidate_similarity(permuted))


def test_selector_is_candidate_order_invariant() -> None:
    value = partitions()
    rows = records()
    config = BasinSelectorConfig(
        molecular_weight=0.25,
        topology_weight=1.0,
        persistence_weight=0.25,
        risk_weight=0.0,
        ordered_path_weight=0.75,
    )
    first, _, _ = select_evidence_rank(rows, value, config)
    order = np.asarray([3, 1, 0, 2])
    reordered_rows = [rows[index] for index in order]
    second, _, _ = select_evidence_rank(reordered_rows, value[order], config)
    assert partition_sha256(value[first]) == partition_sha256(value[order][second])


def test_ordered_path_persistence_is_materialized() -> None:
    _, enriched, diagnostics = select_evidence_rank(
        records(),
        partitions(),
        BasinSelectorConfig(ordered_path_weight=0.75),
    )
    assert diagnostics["ordered_path_candidate_count"] == 3
    assert enriched[1]["ordered_path_persistence"] > 0
    assert enriched[1]["ordered_path_span"] >= 2 / 3


def test_no_ordered_path_ablation_changes_persistence_semantics() -> None:
    _, with_path, _ = select_evidence_rank(
        records(), partitions(), BasinSelectorConfig(ordered_path_weight=0.75)
    )
    _, without_path, _ = select_evidence_rank(
        records(), partitions(), BasinSelectorConfig(ordered_path_weight=0.0)
    )
    assert with_path[1]["persistence"] != without_path[1]["persistence"]


def test_plain_medoid_tie_break_is_candidate_id_deterministic() -> None:
    rows = [{"candidate_id": "z"}, {"candidate_id": "a"}]
    similarity = np.eye(2)
    assert plain_medoid_index(rows, similarity) == 1


def test_config_rejects_invalid_ordered_path_weight() -> None:
    with pytest.raises(ValueError):
        BasinSelectorConfig(ordered_path_weight=1.1).validate()


def test_core_has_no_annotation_or_dataset_router_argument() -> None:
    source = Path("SpaLORA/night16g_basin_selector.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"label", "labels", "annotation", "dataset", "lane"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert forbidden.isdisjoint(argument.arg for argument in node.args.args)


def test_numeric_scripts_pin_threads_before_numpy_import() -> None:
    for name in (
        "build_scout.py",
        "fit_strict_selector.py",
        "apply_frozen_selector.py",
        "generate_ordered_path.py",
    ):
        text = Path("scripts/night16g", name).read_text(encoding="utf-8")
        numpy_position = text.index("import numpy")
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert text.index(variable) < numpy_position


def test_candidate_similarity_is_small_candidate_by_candidate_only() -> None:
    value = candidate_similarity(partitions())
    assert value.shape == (4, 4)
    assert np.allclose(value, value.T)
