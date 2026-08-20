import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from SpaLORA.night8b_cardinality_safe_eval import (
    atomic_json_fsync,
    canonicalize_labels,
    independent_decision_metrics,
    independent_ari_nmi,
    paired_rows,
    paired_statistics,
    primary_metrics,
    symmetric_knn_adjacency,
)


def test_predicted_and_reference_cardinality_mismatch_is_valid():
    raw = np.asarray([[b"a"], [b"a"], [b"b"], [b"c"]])
    labels, contract = canonicalize_labels(raw, 4)
    assert contract["reference_K"] == 3
    assert labels.shape == (4,)
    assert labels.tolist() == ["a", "a", "b", "c"]
    assert contract["missing_like_count"] == 0


@pytest.mark.parametrize("raw", [
    np.zeros((2, 2), dtype=np.int64),
    np.asarray([b"a", b" ", b"b"]),
    np.asarray([0.0, np.nan, 1.0]),
    np.asarray([0.0, np.inf, 1.0]),
])
def test_malformed_or_missing_like_labels_fail_before_metrics(raw):
    with pytest.raises(ValueError):
        canonicalize_labels(raw, raw.size if raw.ndim == 1 else 4)


def test_length_mismatch_fails():
    with pytest.raises(ValueError, match="Y_LENGTH_MISMATCH"):
        canonicalize_labels(np.asarray([b"a", b"b", b"c"]), 4)


def test_contingency_ari_nmi_matches_library_with_unequal_k():
    truth = np.asarray(["a", "a", "b", "b", "c", "c", "c", "d"])
    prediction = np.asarray([0, 0, 1, 2, 2, 3, 3, 3])
    ari, nmi = independent_ari_nmi(truth, prediction)
    assert abs(ari - adjusted_rand_score(truth, prediction)) <= 1e-12
    assert abs(nmi - normalized_mutual_info_score(truth, prediction)) <= 1e-12


def test_primary_and_independent_decision_metrics_match():
    coordinates = np.column_stack((np.arange(30), np.arange(30) % 4)).astype(float)
    graph = symmetric_knn_adjacency(coordinates, 4)
    truth = np.asarray(["t%d" % (index % 7) for index in range(30)])
    prediction = np.asarray([index % 5 for index in range(30)])
    primary = primary_metrics(truth, prediction, graph)
    independent = independent_decision_metrics(truth, prediction, graph)
    for key in ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement"):
        assert abs(primary[key] - independent[key]) <= 1e-12, key


def test_paired_statistics_fixed_10_units():
    rows = []
    for method, offset in (("HR_U00", 0.0), ("HR_F00", 0.02)):
        for seed in range(10):
            value = seed / 100.0 + offset
            rows.append({
                "method": method, "seed": seed, "ari": value, "nmi": value,
                "q": value, "neighbor_agreement": value, "moran_i": value,
                "geary_c": value, "boundary_disagreement": value,
                "end_to_end_seconds": 1.0, "peak_gpu_mib": 1.0,
            })
    paired = paired_rows(rows)
    indices = np.random.default_rng(20260820).integers(0, 10, size=(100000, 10))
    statistics = paired_statistics(paired, indices)
    assert len(paired) == 10
    assert statistics["q_wins"] == 10
    assert statistics["exact_sign_flip_delta_q"]["enumerations"] == 1024


def test_atomic_reference_contract_is_complete(tmp_path):
    target = tmp_path / "reference_label_contract.json"
    payload = {"status": "PERSISTED_BEFORE_ANY_METRIC", "reference_K": 5,
               "lineage_raw_Y_read_count_after": 2}
    atomic_json_fsync(target, payload)
    assert json.loads(target.read_text()) == payload
    assert not target.with_name(target.name + ".tmp").exists()


def test_evaluator_contract_order_and_forbidden_cli():
    source = (Path(__file__).resolve().parents[1] / "scripts/night8b_cardinality_safe_evaluate.py").read_text()
    contract_write = source.index("atomic_json_fsync(raw_contract_path, reference_contract)")
    metric_start = source.index("result = evaluate_after_contract(truth, preloaded, reference_sha)")
    assert contract_write < metric_start
    assert "--execute-final-authorized-Y-read" in source
    parser_lines = [line.strip() for line in source.splitlines()
                    if "add_argument(" in line]
    assert parser_lines == [
        'parser.add_argument("--execute-final-authorized-Y-read", action="store_true", required=True)'
    ]
    assert "lineage_raw_Y_read_count_after\": 2" in source


def test_evaluation_contract_scope_is_zero():
    path = (Path(__file__).resolve().parents[1]
            / "protocols/night8b_cardinality_safe_eval/evaluation_contract_lock.json")
    contract = json.loads(path.read_text())
    assert contract["status"] == "LOCKED_BEFORE_FINAL_Y_READ"
    assert contract["reference_contract"]["predicted_K_must_equal_reference_K"] is False
    assert set(contract["scope_counts"].values()) == {0}
    assert contract["descriptive_sensitivity"]["terminal_decision_input"] is False
