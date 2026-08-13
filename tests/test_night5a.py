"""Focused Night-5A engineering and scientific-integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from SpaLORA.model_corrected import EncoderOverallCorrected
from SpaLORA.night3a_ige import LOSS_KEYS, raw_losses
from SpaLORA.night3b_ablation import Night3BTrainer
from SpaLORA.night5a_rnd import (
    LEGACY_DELEGATES, Night5AModel, Night5ATrainer, ResidualSparseEncoder,
    active_mask_corr2_off, anchor_graph, canonical_sha256, candidate_model_policy,
    frozen_contrast_pairs, frozen_triplets, hybrid_coefficients, load_registry,
    local_reliability_weights, registry_contracts, run_identity,
)
from SpaLORA.night5a_strict_ids import d1_strict_positions, strict_id_positions


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/night5a_authoritative_inputs/SpaLORA_Night5A_Candidate_Registry_2026-08-13.json"


def sparse_identity(n):
    index = torch.arange(n)
    return torch.sparse_coo_tensor(torch.stack((index, index)), torch.ones(n), (n, n)).coalesce()


def sparse_ring(n):
    rows, cols = [], []
    for value in range(n):
        for other in ((value - 1) % n, (value + 1) % n):
            rows.append(value); cols.append(other)
    return torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.ones(len(rows)), (n, n)).coalesce()


def synthetic_data(n=32, first_dim=12, second_dim=7):
    rng = np.random.default_rng(17)
    ring = sparse_ring(n)
    return {
        "features_omics1": rng.normal(size=(n, first_dim)).astype(np.float32),
        "features_omics2": rng.normal(size=(n, second_dim)).astype(np.float32),
        "rna_pca_scores": rng.normal(size=(n, min(8, first_dim))).astype(np.float32),
        "adj_spatial_omics1": ring, "adj_spatial_omics2": ring,
        "adj_feature_omics1": ring, "adj_feature_omics2": ring,
    }


def cfg():
    return {"embedding_dim": 6, "epochs": 2, "loss_factors": [1.9, 2.5, 1.5, 10.0]}


def registry():
    return load_registry(REGISTRY)


def candidate(prefix):
    return next(row for row in registry()["candidates"] if row["id"].startswith(prefix))


def artifacts(data):
    n = len(data["features_omics1"])
    reliability = local_reliability_weights(data["rna_pca_scores"], data["features_omics2"], k=5)
    triplets, _ = frozen_triplets(data["rna_pca_scores"], data["features_omics2"], 29, k=3)
    contrast, _ = frozen_contrast_pairs(data["adj_spatial_omics1"], 29)
    anchor05, _ = anchor_graph(data["adj_spatial_omics1"], data["adj_feature_omics1"], 0.5)
    anchor10, _ = anchor_graph(data["adj_spatial_omics1"], data["adj_feature_omics1"], 1.0)
    return {"reliability": reliability, "triplets": triplets, "contrast": contrast,
            "dgi_permutation": np.random.default_rng(29).permutation(n),
            "anchor05": anchor05, "anchor10": anchor10}


def test_01_registry_exact_17():
    assert len(registry()["candidates"]) == 17


def test_02_registry_ids_exact_c00_c16():
    assert [row["id"].split("_", 1)[0] for row in registry()["candidates"]] == ["C%02d" % i for i in range(17)]


def test_03_registry_unique_serialized_sha():
    contracts = registry_contracts(registry())
    assert len({row["config_sha256"] for row in contracts.values()}) == 17


def test_04_registry_roundtrip_reproducible():
    payload = registry()
    assert canonical_sha256(json.loads(json.dumps(payload))) == canonical_sha256(payload)


@pytest.mark.parametrize("candidate_id,legacy", sorted(LEGACY_DELEGATES.items()))
def test_05_07_locked_candidate_delegate(candidate_id, legacy):
    assert candidate_model_policy(candidate(candidate_id.split("_", 1)[0]))["delegate"] == legacy


def test_08_uniform_attention_exact():
    model = Night5AModel(4, 3, 5, 3, "uniform_all")
    result = model(torch.randn(9, 4), torch.randn(9, 5), *([sparse_identity(9)] * 4))
    for key in ("alpha", "alpha_omics1", "alpha_omics2"):
        assert torch.equal(result[key], torch.full((9, 2), 0.5))


@pytest.mark.parametrize("rho", [0.25, 0.5])
def test_09_10_shrink_attention_contract(rho):
    model = Night5AModel(4, 3, 5, 3, "shrink_to_uniform", learned_fraction=rho)
    result = model(torch.randn(9, 4), torch.randn(9, 5), *([sparse_identity(9)] * 4))
    assert torch.all(result["alpha"] >= 0.5 * (1.0 - rho))
    assert torch.all(result["alpha"] <= 0.5 * (1.0 - rho) + rho)
    assert torch.allclose(result["alpha"].sum(1), torch.ones(9))


def test_11_reliability_swap_equivariance():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(40, 8)), rng.normal(size=(40, 6))
    first = local_reliability_weights(a, b, k=7)
    swapped = local_reliability_weights(b, a, k=7)
    assert np.allclose(first, swapped[:, ::-1], atol=1e-6)


def test_12_reliability_constant_is_uniform():
    weights = local_reliability_weights(np.ones((30, 5)), np.ones((30, 3)), k=5)
    assert np.array_equal(weights, np.full((30, 2), 0.5, dtype=np.float32))


def test_13_reliability_finite_rows_sum_one():
    data = synthetic_data()
    weights = local_reliability_weights(data["rna_pca_scores"], data["features_omics2"], k=5)
    assert np.isfinite(weights).all() and np.allclose(weights.sum(1), 1.0)


@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_14_15_anchor_preserves_spatial_and_is_symmetric(eta):
    graph, stats = anchor_graph(sparse_ring(20), sparse_ring(20), eta)
    dense = graph.to_dense()
    assert stats["original_spatial_support_preserved"]
    assert torch.allclose(dense, dense.T)
    assert torch.all(dense.diag() > 0)


def test_16_triplet_manifest_seed_stable():
    data = synthetic_data()
    first, _ = frozen_triplets(data["rna_pca_scores"], data["features_omics2"], 9)
    second, _ = frozen_triplets(data["rna_pca_scores"], data["features_omics2"], 9)
    assert np.array_equal(first, second)


def test_17_contrast_manifest_seed_stable():
    first, _ = frozen_contrast_pairs(sparse_ring(30), 9)
    second, _ = frozen_contrast_pairs(sparse_ring(30), 9)
    assert np.array_equal(first, second)


def test_18_contrast_negative_is_nonedge():
    pairs, stats = frozen_contrast_pairs(sparse_ring(30), 9)
    assert len(pairs) > 0 and not stats["contains_positive_as_negative"]


def test_19_hybrid_corr2_zero_sum_four():
    gradients = {name: value for name, value in zip(LOSS_KEYS, [1.0, 2.0, 3.0, 4.0])}
    result = hybrid_coefficients(gradients, [1.9, 2.5, 1.5, 10.0], 0.25, 1e-12)
    assert result["L_corr2_raw"] == 0.0 and sum(result.values()) == pytest.approx(4.0)


def test_20_active_mask_corr2_exact_off():
    mask = active_mask_corr2_off()
    assert mask["L_corr2_raw"] is False and sum(mask.values()) == 3


def test_21_residual_encoder_forward_backward_finite():
    encoder = ResidualSparseEncoder(5, 4)
    output = encoder(torch.randn(12, 5), sparse_ring(12))
    output.square().mean().backward()
    assert torch.isfinite(output).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_22_residual_parameter_envelope():
    baseline = Night5AModel(12, 6, 7, 6, "uniform_all")
    residual = Night5AModel(12, 6, 7, 6, "uniform_all", residual_encoder=True)
    assert sum(p.numel() for p in residual.parameters()) <= 2 * sum(p.numel() for p in baseline.parameters())


@pytest.mark.parametrize("prefix", ["C03", "C04", "C06", "C08", "C10", "C12", "C13", "C15", "C16"])
def test_23_candidate_forward_backward_finite_and_active_gradients(prefix):
    data = synthetic_data()
    trainer = Night5ATrainer(data, cfg(), candidate(prefix), 0, torch.device("cpu"), artifacts(data))
    model = trainer.new_model()
    result = trainer.forward(model)
    coefficients = {name: (0.0 if name == "L_corr2_raw" else 4.0 / 3.0) for name in LOSS_KEYS}
    total = sum(raw_losses(result, trainer.features1, trainer.features2)[name] * coefficients[name] for name in LOSS_KEYS)
    auxiliary, _ = trainer._auxiliary_loss(model, result)
    (total + auxiliary).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert torch.isfinite(result["emb_latent_combined"]).all()
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_24_dgi_uses_locked_permutation_and_finite():
    data = synthetic_data()
    trainer = Night5ATrainer(data, cfg(), candidate("C16"), 0, torch.device("cpu"), artifacts(data))
    model = trainer.new_model(); result = trainer.forward(model)
    loss, payload = trainer._auxiliary_loss(model, result)
    assert torch.isfinite(loss) and payload["name"] == "dgi"


def test_25_run_identity_resume_exact():
    first = run_identity("a1", "C03_SIMPLE_BASE", 2, "abc", "def")
    second = run_identity("a1", "C03_SIMPLE_BASE", 2, "abc", "def")
    changed = run_identity("a1", "C03_SIMPLE_BASE", 3, "abc", "def")
    assert first == second and first["run_identity_sha256"] != changed["run_identity_sha256"]


def test_26_no_label_or_evaluator_import_in_method_source():
    source = (ROOT / "SpaLORA/night5a_rnd.py").read_text(encoding="utf-8").lower()
    forbidden = ("night1_evaluation", "night5a_evaluate", "ground_truth", "cell_type")
    assert not any(token in source for token in forbidden)


def test_27_unregistered_candidate_rejected():
    data = synthetic_data()
    with pytest.raises((KeyError, ValueError)):
        Night5ATrainer(data, cfg(), {"id": "C17_FORBIDDEN"}, 0, torch.device("cpu"), artifacts(data)).new_model()


def test_28_checkpoint_identity_depends_on_artifact_hash():
    first = run_identity("a1", "C03_SIMPLE_BASE", 0, "abc", "def")
    second = run_identity("a1", "C03_SIMPLE_BASE", 0, "abc", "different")
    assert first["run_identity_sha256"] != second["run_identity_sha256"]


def test_29_strict_ids_reorders_synthetic_labels():
    assert np.array_equal(strict_id_positions(["b", "a"], ["a", "b"]), [1, 0])


def test_30_strict_ids_reject_duplicates_and_missing():
    with pytest.raises(ValueError):
        strict_id_positions(["a", "a"], ["a", "b"])
    with pytest.raises(ValueError):
        strict_id_positions(["a", "c"], ["a", "b"])


def test_31_d1_interface_requires_exact_3359_synthetic_ids():
    ids = ["synthetic_%04d" % value for value in range(3359)]
    assert np.array_equal(d1_strict_positions(ids, ids), np.arange(3359))
    with pytest.raises(ValueError):
        d1_strict_positions(ids[:-1], ids[:-1])
