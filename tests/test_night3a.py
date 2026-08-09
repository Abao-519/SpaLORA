"""Pre-main hard tests for the preregistered SpaLORA Night-3A study."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from SpaLORA.model_corrected import GraphLinear
from SpaLORA.night1_evaluation import evaluate
from SpaLORA.night1_pipeline import normalize_graph_sparse, weighted_gene_mse
from SpaLORA.night3a_ige import (
    LOSS_KEYS,
    VARIANTS,
    Night3ATrainer,
    coefficients_for_variant,
    ige_weights_from_gradients,
    legacy_m_bad,
    model_state_sha256,
    raw_losses,
    registered_variant_contract,
    required_record_steps,
    rms_gradient_probe,
    run_initial_probe,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/night3a_ige_feasibility.json"


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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


def test_raw_loss_and_legacy_uniform_reconstruction_algebra():
    torch.manual_seed(1)
    difference = torch.randn(9, 11)
    raw = torch.mean(difference ** 2)
    weighted_ones = weighted_gene_mse(difference, torch.ones(11))
    assert torch.equal(raw, weighted_ones)
    m_bad = legacy_m_bad(difference)
    factors = [1.9, 2.5, 1.5, 10.0]
    c0 = coefficients_for_variant("C0", {name: 1.0 for name in LOSS_KEYS},
                                  {name: 1.0 for name in LOSS_KEYS}, factors, float(m_bad), 1e-12)
    c1 = coefficients_for_variant("C1", {name: 1.0 for name in LOSS_KEYS},
                                  {name: 1.0 for name in LOSS_KEYS}, factors, float(m_bad), 1e-12)
    assert c0[LOSS_KEYS[0]] == factors[0]
    assert np.isclose(c1[LOSS_KEYS[0]] * raw.item(), factors[0] * m_bad.item() * raw.item())
    assert all(c0[name] == c1[name] for name in LOSS_KEYS[1:])


def test_raw_losses_preserve_mean_reduction_and_shapes():
    data = synthetic_data()
    trainer = Night3ATrainer(data, synthetic_cfg(), "C0", 0, torch.device("cpu"))
    model = trainer.new_model()
    result = trainer.forward(model)
    losses = raw_losses(result, trainer.features1, trainer.features2)
    assert tuple(losses) == LOSS_KEYS
    assert all(value.ndim == 0 for value in losses.values())
    assert torch.equal(losses[LOSS_KEYS[0]], F.mse_loss(trainer.features1, result["emb_recon_omics1"]))


def test_ige_weight_formula_and_sum():
    gradients = dict(zip(LOSS_KEYS, [1.0, 2.0, 4.0, 8.0]))
    observed = ige_weights_from_gradients(gradients, 1e-12)
    raw = 1.0 / np.asarray([1.0, 2.0, 4.0, 8.0])
    expected = 4.0 * raw / raw.sum()
    assert np.allclose([observed[name] for name in LOSS_KEYS], expected, rtol=0, atol=1e-12)
    assert abs(sum(observed.values()) - 4.0) <= 1e-12


def test_unused_parameter_and_zero_gradient_handling():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
            self.unused = torch.nn.Parameter(torch.tensor([4.0]))

    model = Tiny()
    losses = {name: (model.used ** 2).mean() for name in LOSS_KEYS}
    gradients, rows = rms_gradient_probe(model, {}, losses, 1e-12)
    assert all(gradients[name] > 0 for name in LOSS_KEYS)
    assert all(any(row["parameter"] == "unused" and not row["has_gradient"] for row in rows if row["loss"] == name)
               for name in LOSS_KEYS)
    zero_losses = {name: (model.used * 0).sum() for name in LOSS_KEYS}
    zero_gradients, _ = rms_gradient_probe(model, {}, zero_losses, 1e-12)
    assert all(value == 0.0 for value in zero_gradients.values())
    with pytest.raises(ValueError):
        ige_weights_from_gradients(zero_gradients, 1e-12)


def test_probe_does_not_change_parameter_gradient_or_rng_state():
    trainer = Night3ATrainer(synthetic_data(), synthetic_cfg(), "IGE", 2, torch.device("cpu"))
    model = trainer.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model_before = model_state_sha256(model)
    optimizer_before = repr(optimizer.state_dict())
    probe = run_initial_probe(model, trainer.forward, 1e-12)
    assert probe["state_unchanged"] and probe["grad_fields_unchanged"] and probe["rng_unchanged"]
    assert probe["repeat_exact"] and probe["reload_forward_within_envelope"]
    assert model_state_sha256(model) == model_before
    assert repr(optimizer.state_dict()) == optimizer_before


def test_ige_coefficients_are_frozen_plain_scalars():
    result = Night3ATrainer(synthetic_data(), synthetic_cfg(), "IGE", 3, torch.device("cpu")).train()
    assert all(isinstance(value, float) for value in result.coefficients.values())
    assert all(name not in dict(result.model.named_parameters()) for name in result.coefficients)
    assert result.probe["weights"] == result.coefficients


def test_label_firewall_static_and_evaluator_import_order():
    runner = (REPO / "scripts/night3a_runner.py").read_text(encoding="utf-8")
    p0b = (REPO / "scripts/night3a_p0b.py").read_text(encoding="utf-8")
    evaluator = (REPO / "scripts/night3a_evaluate.py").read_text(encoding="utf-8")
    assert "SpaLORA.night1_evaluation" not in runner
    assert "load_evaluation_labels" not in runner
    assert "load_evaluation_labels" not in p0b
    assert evaluator.index("verify_preconditions(config, output)") < evaluator.index(
        "from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels"
    )


def test_four_variants_differ_only_by_registered_calibration():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert tuple(config["variants"]) == VARIANTS
    contracts = registered_variant_contract(config)
    assert set(contracts) == set(VARIANTS)
    assert all(contract["gene_shape"] is False for contract in contracts.values())
    data, cfg = synthetic_data(), synthetic_cfg()
    hashes = {}
    for variant in VARIANTS:
        trainer = Night3ATrainer(data, cfg, variant, 4, torch.device("cpu"))
        hashes[variant] = model_state_sha256(trainer.new_model())
    assert len(set(hashes.values())) == 1


def test_corrected_sparse_adjacency_never_densifies():
    graph = normalize_graph_sparse(sp.csr_matrix([[0, 1], [1, 0]], dtype=np.float32))
    assert graph.is_sparse
    layer = GraphLinear(3, 2)
    output = layer(torch.ones(2, 3), graph)
    assert output.shape == (2, 2)
    assert graph.is_sparse


def test_actual_p0a_spot_feature_and_graph_orders_are_locked():
    output = REPO / "outputs/night3a_handoff"
    p0a = json.loads((output / "night3a_p0a.json").read_text(encoding="utf-8"))
    assert p0a["passed"] and not p0a["label_values_read"]
    for dataset in ("a1", "placenta", "p22"):
        audit = p0a["datasets"][dataset]
        assert audit["prepared_n_obs"] > 0 and audit["selected_gene_count"] in (2000, 3000)
        assert len(audit["prepared_observation_order_sha256"]) == 64
        assert len(audit["selected_gene_order_sha256"]) == 64
        assert all(graph["is_sparse"] and graph["finite"] for graph in audit["graphs"].values())
        assert audit["ground_truth_identifier_audit"]["training_identifier_set_equal"]


def test_cpu_repeat_forward_loss_gradient_and_update_exact():
    data, cfg = synthetic_data(), synthetic_cfg(epochs=2)
    first = Night3ATrainer(data, cfg, "IGE", 1, torch.device("cpu")).train()
    second = Night3ATrainer(data, cfg, "IGE", 1, torch.device("cpu")).train()
    assert first.initial_state_sha256 == second.initial_state_sha256
    assert first.final_state_sha256 == second.final_state_sha256
    assert first.coefficients == second.coefficients
    for key in first.output:
        assert np.array_equal(first.output[key], second.output[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the registered GPU envelope")
def test_gpu_probe_envelope():
    trainer = Night3ATrainer(synthetic_data(), synthetic_cfg(), "IGE", 0, torch.device("cuda:0"))
    model = trainer.new_model()
    probe = run_initial_probe(model, trainer.forward, 1e-12)
    assert probe["repeat_within_gpu_envelope"]
    assert probe["reload_forward_within_envelope"]


def test_evaluator_regression_on_fixed_synthetic_case():
    true = np.asarray(["a", "a", "b", "b"])
    predicted = np.asarray([1, 1, 2, 2])
    embedding = np.asarray([[0, 0], [0, .1], [2, 2], [2, 2.1]], dtype=np.float32)
    coordinates = np.asarray([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=np.float32)
    metrics = evaluate(true, predicted, predicted, embedding, coordinates, spatial_neighbors=1)
    assert metrics["ari"] == 1.0 and metrics["nmi"] == 1.0
    assert metrics["hungarian_macro_f1"] == 1.0
    assert metrics["spatial_neighbor_agreement"] == 1.0
    again = evaluate(true, predicted, predicted, embedding, coordinates, spatial_neighbors=1)
    assert metrics == again


def test_failure_resume_manifest_hash_and_report_schema_contracts():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["paths"]["output_root"] == "outputs/night3a_handoff"
    assert config["label_firewall"]["evaluation_requires_locked_60_run_manifest"]
    finalize = (REPO / "scripts/night3a_finalize.py").read_text(encoding="utf-8")
    runner = (REPO / "scripts/night3a_runner.py").read_text(encoding="utf-8")
    for phrase in ("failure.json", "run_manifest.json", "artifact_sha256", "Partial/mismatched"):
        assert phrase in runner
    for phrase in ("night3a_report.md", "night3a_completion.json", "SHA256SUMS"):
        assert phrase in finalize


def test_protected_night2c_files_have_no_night3a_write_target():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protected = config["paths"]["protected_root"]
    assert str(REPO) != protected and not str(REPO).startswith(protected + "/")
    manifest = Path(config["paths"]["protected_manifest"])
    assert manifest.is_file() and sha256_file(manifest) == json.loads(
        (REPO / "outputs/night3a_handoff/config_lock.json").read_text(encoding="utf-8")
    )["protected_manifest_sha256"]
    for script in (REPO / "scripts").glob("night3a_*.py"):
        text = script.read_text(encoding="utf-8")
        assert "write_text(" + repr(protected) not in text


def test_m_bad_locked_values_and_no_gene_shape_variant():
    for n_genes, expected in ((2000, 2.290322960), (3000, 2.289938091)):
        features = torch.arange(float(n_genes)).reshape(1, n_genes)
        assert abs(float(legacy_m_bad(features)) - expected) <= 1e-6
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "ASR" not in config["variants"] and "rescue" not in config["variants"]


def test_preregistered_run_order_is_exact_and_locked():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / "outputs/night3a_handoff"
    order_path = REPO / config["run_order"]["manifest"]
    order = json.loads(order_path.read_text(encoding="utf-8"))
    cells = [(row["dataset"], row["variant"], row["seed"]) for row in order["runs"]]
    expected = {(d, v, s) for d in config["datasets"] for v in VARIANTS for s in config["seeds"]}
    assert len(cells) == len(set(cells)) == 60 and set(cells) == expected
    lock = json.loads((output / "config_lock.json").read_text(encoding="utf-8"))
    assert sha256_file(order_path) == lock["run_order_sha256"]


def test_config_source_and_data_lock_is_complete():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    lock = json.loads((REPO / "outputs/night3a_handoff/config_lock.json").read_text(encoding="utf-8"))
    assert sha256_file(CONFIG_PATH) == lock["config_sha256"]
    assert set(config["source_lock_files"]) == set(lock["source_sha256"])
    assert all(sha256_file(REPO / name) == expected for name, expected in lock["source_sha256"].items())
    assert config["gate"]["placenta_recovery_fraction"] == 0.60
    assert config["gate"]["a1_p22_ari_floor"] == -0.02
    assert config["gate"]["spatial_joint_decline_limit"] == 0.03


def test_required_checkpoint_schedule_covers_absolute_and_relative_points():
    assert required_record_steps(200) == [0, 1, 5, 10, 20, 40, 80, 100, 200]
    assert required_record_steps(1600) == [0, 1, 5, 10, 20, 40, 80, 320, 800, 1600]
