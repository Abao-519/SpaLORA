"""Fail-closed semantic checks for the frozen Night-8B protocol."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "protocols/night8b/SpaLORA_Night8B_MISAR_Family_Policy_Registry_2026-08-20.json"
PIPELINE = ROOT / "SpaLORA/night8b_pipeline.py"
TRAIN = ROOT / "scripts/night8b_train.py"


def test_registry_is_frozen_two_method_confirmation():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert registry["candidate_search"] is False
    assert registry["dataset"]["seeds"] == list(range(10))
    assert registry["dataset"]["K"] == 12
    assert [x["id"] for x in registry["fixed_methods"]] == [
        "U00_UNIVERSAL_C00", "F00_FAMILY_R02"
    ]
    assert registry["training"]["scientific_training_total"] == 20
    assert registry["transforms"]["total"] == 20


def test_provenance_correction_is_explicit_and_gse_is_never_a_source():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    dataset = registry["dataset"]
    assert dataset["official_accession"] == "OEP003285"
    assert dataset["valid_raw_read_cross_reference"] == "SRP491963"
    assert dataset["official_processed_record"]["record_id"] == 7480069
    assert dataset["forbidden_incorrect_cross_reference"] == "GSE213264"
    source = PIPELINE.read_text(encoding="utf-8")
    assert "GSE213264" not in source


def test_training_module_never_references_annotation_values():
    for path in (PIPELINE, TRAIN):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "GSE213264" not in source
        assert not any(isinstance(node, ast.Constant) and node.value == "Y" for node in ast.walk(tree))


def test_fixed_family_and_graph_contracts_are_literal():
    source = PIPELINE.read_text(encoding="utf-8")
    assert '"n_clusters": 12' in source
    assert '"embedding_dim": 128' in source
    assert '"epochs": 1600' in source
    assert '"loss_factors": [1.5, 5.0, 1.5, 1.0]' in source
    assert '"learned_fraction_rho": 0.25' in source
    assert '"spatial_k": 18' in source and '"feature_k": 20' in source
    assert '"spatial_k": 10' in source and '"feature_k": 10' in source
    assert 'finite highly_variable_rank in [0,n_top)' in source


def test_worker_contract_is_identity_blind_and_no_fallback():
    source = TRAIN.read_text(encoding="utf-8")
    assert '"dataset_identity_used": False' in source
    assert '"retry": False' in source
    assert '"fallback": False' in source
    assert '"recipe_id":"R02"' in source
    assert '"losses":["RECON","MNN"]' in source
    assert '"fusion":"equal"' in source
    assert '"epochs":160' in source
    assert "atol=1e-6, rtol=1e-5" in source
