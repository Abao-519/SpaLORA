from __future__ import annotations

import ast
import csv
from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night16b_unified_structured_decoder import (
    PreparedDecoderInput,
    RepairConfig,
    WeightedView,
    build_feature_bank,
    combine_weighted_views,
    generic_repair,
    partition_sha256,
)
from scripts.night16b.night16b_freeze import choose_headline


TEST_FILE = Path(__file__).resolve()
REPO = TEST_FILE.parents[1]
if not (REPO / "SpaLORA" / "night16b_unified_structured_decoder.py").exists():
    REPO = TEST_FILE.parents[2] / "night16b_local_work"
CORE = REPO / "SpaLORA" / "night16b_unified_structured_decoder.py"
WORK = REPO / "working"
HANDOFF = REPO / "outputs" / "night16b_handoff"


def ring_graph(n: int) -> sp.csr_matrix:
    row = np.arange(n)
    col = (row + 1) % n
    return sp.coo_matrix(
        (np.ones(2 * n), (np.r_[row, col], np.r_[col, row])), shape=(n, n)
    ).tocsr()


def test_core_has_no_dataset_identity_router():
    source = CORE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"A1", "D1", "P22", "MISAR", "tonsil_s1", "tonsil_s2", "tonsil_s3"}
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not forbidden.intersection(strings)


def test_core_api_has_no_reference_label_argument():
    import SpaLORA.night16b_unified_structured_decoder as module

    for function in (module.decode, module.generic_repair, module.combine_weighted_views):
        names = set(inspect.signature(function).parameters)
        assert not names.intersection({"label", "labels", "truth", "annotation", "ari", "nmi"})


def test_view_weight_is_not_cancelled_by_final_standardization():
    rng = np.random.default_rng(3)
    first = rng.normal(size=(60, 4))
    second = rng.normal(size=(60, 4))
    low = combine_weighted_views((WeightedView("a", first, 1.0), WeightedView("b", second, 0.01)), final_dim=None)
    high = combine_weighted_views((WeightedView("a", first, 1.0), WeightedView("b", second, 1.0)), final_dim=None)
    assert not np.allclose(low, high)


def test_missing_optional_view_is_byte_exact_fallback():
    rng = np.random.default_rng(4)
    n = 36
    base = PreparedDecoderInput(
        retained=rng.normal(size=(n, 8)),
        view1=rng.normal(size=(n, 7)),
        view2=rng.normal(size=(n, 6)),
        coordinates=rng.normal(size=(n, 2)),
        graph=ring_graph(n),
    )
    missing = replace(
        base,
        optional_views=(WeightedView("morphology", rng.normal(size=(n, 5)), 1.0, np.zeros(n)),),
    )
    a = build_feature_bank(base)
    b = build_feature_bank(missing)
    assert np.array_equal(a["molecular"], b["molecular_optional"])
    assert np.array_equal(a["molecular_coord"], b["molecular_optional_coord"])


def test_generic_repair_enforces_exact_k_and_minimum_size():
    rng = np.random.default_rng(7)
    n, k = 100, 4
    feature = rng.normal(size=(n, 6)).astype(np.float32)
    initial = np.r_[np.zeros(1), np.ones(19), np.full(40, 2), np.full(40, 3)].astype(np.int32)
    config = RepairConfig(
        enabled=True,
        min_cluster_fraction_of_equal=0.20,
        feature_mode="retained",
        merge_mode="pointwise",
        split_mode="pca_quantile",
        split_quantile=0.5,
    )
    result, detail = generic_repair(initial, feature, ring_graph(n), k, config)
    assert len(np.unique(result)) == k
    assert np.bincount(result).min() >= 5
    assert detail["repair_merged_clusters"] >= 1


def test_repair_is_deterministic():
    rng = np.random.default_rng(11)
    n, k = 80, 4
    feature = rng.normal(size=(n, 5)).astype(np.float32)
    initial = np.repeat(np.arange(k), n // k)
    config = RepairConfig(boundary_refine_beta=0.02, boundary_refine_sweeps=2)
    first, _ = generic_repair(initial, feature, ring_graph(n), k, config)
    second, _ = generic_repair(initial, feature, ring_graph(n), k, config)
    assert partition_sha256(first) == partition_sha256(second)


def test_headline_rule_prefers_ari_before_nmi_and_complexity():
    import pandas as pd

    frame = pd.DataFrame(
        [
            dict(status="PASS", exact_k=1, finite=1, min_cluster_size=20, absolute_ari=.4, absolute_nmi=.9, complexity=2, candidate_id="a"),
            dict(status="PASS", exact_k=1, finite=1, min_cluster_size=20, absolute_ari=.41, absolute_nmi=.1, complexity=2, candidate_id="b"),
            dict(status="PASS", exact_k=1, finite=1, min_cluster_size=20, absolute_ari=.41, absolute_nmi=.1, complexity=1, candidate_id="c"),
        ]
    )
    assert choose_headline(frame, 100, 4).candidate_id == "c"


def test_candidate_generation_manifests_opened_no_labels_and_used_no_dense_graph():
    manifests = list(WORK.glob("coarse*/generation_manifest.json")) + list(WORK.glob("refine*/generation_manifest.json"))
    if manifests:
        for path in manifests:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["label_arrays_opened"] == 0
            assert payload["dense_n_by_n_count"] == 0
            assert payload["failed"] == 0
    else:
        payload = json.loads((HANDOFF / "label_flow_audit.json").read_text(encoding="utf-8"))
        assert payload["labels_in_features"] == 0
        assert payload["labels_in_graph"] == 0
        assert payload["labels_in_prototype_or_energy"] == 0


def test_frozen_registry_has_all_primary_and_secondary_lanes():
    path = WORK / "frozen" / "night16b_frozen_registry.json"
    if not path.exists():
        path = HANDOFF / "night16b_frozen_registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["lanes"]) == 9
    assert payload["labels_in_candidate_producer"] == 0
    assert payload["labels_in_cross_run_hpo"] is True


def test_family_default_producer_did_not_read_heldout_metrics():
    path = WORK / "family_default" / "family_default_producer.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["labels_opened"] == 0
        assert len(payload["lanes"]) == 7
    else:
        with (HANDOFF / "family_default_table.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 7
        audit = json.loads((HANDOFF / "label_flow_audit.json").read_text(encoding="utf-8"))
        assert audit["labels_in_prototype_or_energy"] == 0
