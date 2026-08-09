"""Protocol and numerical tests for the preregistered SpaLORA Night-3A-R study."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night1_pipeline import normalize_graph_sparse
from SpaLORA.night3a_ige import LOSS_KEYS, Night3ATrainer, required_record_steps
from SpaLORA.night3ar_ige import Night3ARTrainer, optimizer_state_sha256, weighted_gradient_influence
from SpaLORA.night3ar_protocol import (
    ScientificWindow, assert_training_payload_label_free, integrity_read, sha256_file,
    training_cfg,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/night3ar_ige_feasibility.json"


def synthetic_data():
    rng = np.random.RandomState(17)
    n, f1, f2 = 7, 5, 4
    graph = normalize_graph_sparse(sp.eye(n, format="csr", dtype=np.float32))
    return {
        "features_omics1": rng.normal(size=(n, f1)).astype(np.float32),
        "features_omics2": rng.normal(size=(n, f2)).astype(np.float32),
        "adj_spatial_omics1": graph,
        "adj_feature_omics1": graph,
        "adj_spatial_omics2": graph,
        "adj_feature_omics2": graph,
    }


def synthetic_cfg(epochs=2):
    return {"embedding_dim": 3, "epochs": epochs, "loss_factors": [1.9, 2.5, 1.5, 10.0]}


def fake_firewall_config(ground_truth: Path):
    return {
        "datasets": {"fake": {"ground_truth": str(ground_truth)}},
        "label_firewall": {"forbidden_modules_before_manifest_lock": ["fake_evaluator_module"]},
    }


def test_protocol_amendment_preserves_old_administrative_status():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["previous_night3a_status"] == "ADMINISTRATIVE_HARD_STOP_SCIENTIFIC_GO_NO_GO_NOT_EVALUATED"
    assert config["paths"]["output_root"] == "outputs/night3ar_handoff"
    assert config["paths"]["old_output_root"] == "outputs/night3a_handoff"


def test_integrity_read_returns_metadata_not_content(tmp_path):
    target = tmp_path / "locked.csv"
    target.write_bytes(b"id,label\na,secret\n")
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    observed = integrity_read(target, expected, "integrity only")
    assert observed["match"] and observed["actual_sha256"] == expected
    assert observed["size_bytes"] == target.stat().st_size
    assert observed["content_returned"] is False and observed["semantic_parse"] is False
    assert "secret" not in repr(observed)


def test_scientific_window_blocks_ground_truth_open_and_parser(tmp_path):
    ground_truth = tmp_path / "ground_truth.csv"
    ground_truth.write_text("id,label\na,secret\n", encoding="utf-8")
    # Integrity is deliberately verified before the scientific window.
    assert integrity_read(ground_truth, sha256_file(ground_truth), "integrity only")["match"]
    window = ScientificWindow(fake_firewall_config(ground_truth), tmp_path, "test").install()
    with pytest.raises(RuntimeError, match="Ground-truth CSV"):
        pd.read_csv(ground_truth)
    payload = window.close(passed=False)
    assert payload["semantic_label_values_read"] is False
    assert payload["ground_truth_parser_guard_trigger_count"] == 1


def test_training_config_and_payload_strip_label_semantics():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = training_cfg(config["datasets"]["a1"])
    assert set(cfg) == {
        "n_clusters", "hvg", "spatial_neighbors", "legacy_datatype", "embedding_dim",
        "epochs", "loss_factors", "locked_m_bad_expected",
    }
    assert_training_payload_label_free(synthetic_data(), cfg, [])
    with pytest.raises(AssertionError):
        assert_training_payload_label_free({"cell_type": ["x"]}, cfg, [])


def test_weighted_gradient_formula_and_state_neutrality_cpu():
    trainer = Night3ATrainer(synthetic_data(), synthetic_cfg(), "IGE", 2, torch.device("cpu"))
    model = trainer.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    parameter_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    rng_before = torch.get_rng_state().clone()
    optimizer_before = optimizer_state_sha256(optimizer)
    coefficients = {name: float(index + 1) for index, name in enumerate(LOSS_KEYS)}
    diagnostic = weighted_gradient_influence(model, trainer.forward, coefficients, 1e-12, optimizer)
    expected_q = {
        name: abs(coefficients[name]) * diagnostic["raw_rms_gradients"][name] for name in LOSS_KEYS
    }
    total = sum(expected_q.values())
    assert diagnostic["weighted_gradient_influence"] == expected_q
    assert np.allclose(
        [diagnostic["weighted_gradient_share"][name] for name in LOSS_KEYS],
        [expected_q[name] / total for name in LOSS_KEYS], rtol=0, atol=1e-15,
    )
    assert all(diagnostic[name] for name in (
        "parameter_state_unchanged", "grad_fields_unchanged", "rng_state_unchanged",
        "optimizer_state_unchanged",
    ))
    assert optimizer_state_sha256(optimizer) == optimizer_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(parameter_before[name], value) for name, value in model.named_parameters())


def test_ige_initial_weighted_gradient_influence_is_equalized():
    trainer = Night3ARTrainer(synthetic_data(), synthetic_cfg(epochs=1), "IGE", 3, torch.device("cpu"))
    result = trainer.train()
    initial = result.gradient_logs[0]
    q = np.asarray([initial[name.replace("L_", "").replace("_raw", "") + "_weighted_gradient_influence"]
                    for name in LOSS_KEYS])
    shares = np.asarray([initial[name.replace("L_", "").replace("_raw", "") + "_weighted_gradient_share"]
                         for name in LOSS_KEYS])
    assert np.allclose(q, q[0], rtol=2e-6, atol=2e-8)
    assert np.allclose(shares, 0.25, rtol=2e-6, atol=2e-8)


def test_night3ar_cpu_training_remains_exactly_night3a_math():
    old = Night3ATrainer(synthetic_data(), synthetic_cfg(epochs=2), "IGE", 1, torch.device("cpu")).train()
    new = Night3ARTrainer(synthetic_data(), synthetic_cfg(epochs=2), "IGE", 1, torch.device("cpu")).train()
    assert old.initial_state_sha256 == new.initial_state_sha256
    assert old.final_state_sha256 == new.final_state_sha256
    assert old.coefficients == new.coefficients
    for key in old.output:
        assert np.array_equal(old.output[key], new.output[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_gradient_diagnostic_preserves_state():
    trainer = Night3ATrainer(synthetic_data(), synthetic_cfg(), "IGE", 0, torch.device("cuda:0"))
    model = trainer.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    diagnostic = weighted_gradient_influence(
        model, trainer.forward, {name: 1.0 for name in LOSS_KEYS}, 1e-12, optimizer
    )
    assert all(diagnostic[name] for name in (
        "parameter_state_unchanged", "grad_fields_unchanged", "rng_state_unchanged",
        "optimizer_state_unchanged",
    ))


def test_scalar_contribution_is_descriptive_and_gradient_share_is_gate():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = config["gate"]
    assert gate["scalar_contribution_fraction_is_descriptive_only"] is True
    assert "contribution_fraction_min" not in gate and "contribution_fraction_max" not in gate
    assert gate["weighted_gradient_share_min"] == 0.01
    assert gate["weighted_gradient_share_max"] == 0.90
    evaluator = (REPO / "scripts/night3ar_evaluate.py").read_text(encoding="utf-8")
    assert "gradient_share_persistent_below_001" in evaluator
    assert "gradient_share_persistent_above_090" in evaluator


def test_runner_has_no_evaluator_or_ground_truth_parser():
    runner = (REPO / "scripts/night3ar_runner.py").read_text(encoding="utf-8")
    assert "SpaLORA.night1_evaluation" not in runner
    assert "load_evaluation_labels" not in runner
    assert "pd.read_csv" not in runner and "read_h5ad" not in runner
    assert "locked_60_run_manifest.json" in runner
    assert "ScientificWindow" in runner


def test_evaluator_import_is_after_locked_manifest_preconditions():
    evaluator = (REPO / "scripts/night3ar_evaluate.py").read_text(encoding="utf-8")
    assert evaluator.index("verify_preconditions(config, output)") < evaluator.index(
        "from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels"
    )


def test_resume_hash_and_failure_contracts_are_explicit():
    runner = (REPO / "scripts/night3ar_runner.py").read_text(encoding="utf-8")
    for phrase in ("failure.json", "run_manifest.json", "artifact_sha256", "Partial/mismatched"):
        assert phrase in runner
    finalize = (REPO / "scripts/night3ar_finalize.py").read_text(encoding="utf-8")
    for phrase in ("night3ar_report.md", "gradient_influence_trajectories.csv", "SHA256SUMS"):
        assert phrase in finalize


def test_actual_p0ar_matches_old_fingerprints_and_order():
    output = REPO / "outputs/night3ar_handoff"
    p0ar = json.loads((output / "night3ar_p0a.json").read_text(encoding="utf-8"))
    assert p0ar["passed"] and p0ar["label_values_read"] is False
    assert p0ar["all_preprocessing_fingerprints_exact"] and p0ar["run_order_exact"]
    assert all(row["exact_match"] for row in p0ar["night3a_comparison"].values())
    new = json.loads((output / "preregistered_run_order.json").read_text(encoding="utf-8"))
    old = json.loads((REPO / "outputs/night3a_handoff/preregistered_run_order.json").read_text(encoding="utf-8"))
    assert new == old and len(new["runs"]) == 60


def test_config_source_data_and_old_evidence_locks_are_complete():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    lock = json.loads((REPO / "outputs/night3ar_handoff/config_lock.json").read_text(encoding="utf-8"))
    assert sha256_file(CONFIG_PATH) == lock["config_sha256"]
    assert set(config["source_lock_files"]) == set(lock["source_sha256"])
    assert all(sha256_file(REPO / name) == expected for name, expected in lock["source_sha256"].items())
    assert sha256_file(REPO / "outputs/night3a_handoff/night3a_p0b.json")
    assert Path(config["paths"]["protected_night3a_manifest"]).is_file()
    assert Path(config["paths"]["protected_night2c_manifest"]).is_file()


def test_no_asr_seed_search_or_formula_expansion():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["variants"] == ["C0", "C1", "IGE", "ILN"]
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert "ASR" not in json.dumps(config) and "rescue" not in config["variants"]


def test_required_checkpoint_schedule_is_unchanged():
    assert required_record_steps(200) == [0, 1, 5, 10, 20, 40, 80, 100, 200]
    assert required_record_steps(1600) == [0, 1, 5, 10, 20, 40, 80, 320, 800, 1600]
