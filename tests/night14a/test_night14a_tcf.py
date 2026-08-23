import ast
import inspect

import torch

from SpaLORA.night14a_tcf import (
    UnifiedGraphAutoencoder, apply_tcf, loss_components, sparse_aggregate,
    unsupervised_loss,
)


def config(backbone="CR_SAGE_AE"):
    return {
        "backbone": backbone, "hidden_dim": 16, "latent_dim": 8,
        "depth": 2, "dropout": 0.0,
    }


def weights():
    return {
        "private_recon": 1.0, "cross_recon": 1.0, "alignment": 0.1,
        "graph_smooth": 0.01, "topology_agreement": 0.05,
        "variance": 0.1, "covariance": 0.001,
    }


def graph():
    edge = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                         [0, 1, 1, 2, 2, 3, 3, 0]])
    weight = torch.full((8,), 0.5)
    return edge, weight


def filter_config(variant):
    return {
        "variant": variant, "support_scale": 8.0, "support_center": 0.5,
        "conflict_scale": 6.0, "max_low": 0.4, "max_high": 0.1,
        "node_scale": 8.0, "node_center": 0.5, "high_center": 0.3,
        "global_scale": 10.0, "global_center": 0.4,
        "roughness_scale": 30.0, "roughness_center": 0.23,
    }


def test_real_contract_shape_gradient_and_registered_parameters():
    x1, x2 = torch.randn(4, 5), torch.randn(4, 7)
    edge, weight = graph()
    model = UnifiedGraphAutoencoder(5, 7, config("CR_BALANCED_XREC"))
    output = model(x1, x2, edge, weight)
    assert output["fused"].shape == (4, 8)
    loss, _ = unsupervised_loss(model, loss_components(output, x1, x2, edge), weights())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    names = dict(model.named_parameters())
    assert any(name.startswith("shared_blocks.0") for name in names)


def test_sparse_aggregate_matches_hand_calculation_without_dense_matrix():
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    edge, weight = graph()
    out = sparse_aggregate(x, edge, weight)
    assert out.shape == x.shape
    assert torch.allclose(out[0], 0.5 * x[0] + 0.5 * x[1])


def test_tcf_identity_and_all_variants_are_finite():
    z1, z2, fused = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    edge, _ = graph()
    identity, _ = apply_tcf(z1, z2, fused, edge, filter_config("IDENTITY"))
    assert torch.equal(identity, fused)
    for variant in ("FIXED_LOW", "SUPPORT_LOW", "TCF_LOW_HIGH"):
        value, audit = apply_tcf(z1, z2, fused, edge, filter_config(variant))
        assert value.shape == fused.shape
        assert torch.isfinite(value).all()
        assert audit["dense_n_by_n_count"] == 0


def test_tcf_fail_safe_floor_is_byte_exact_identity():
    z1, z2, fused = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    edge, _ = graph()
    cfg = filter_config("TCF_LOW_HIGH")
    cfg["max_high"] = 0.0
    cfg["global_center"] = 10.0
    cfg["roughness_center"] = 10.0
    cfg["global_gate_floor"] = 0.05
    value, audit = apply_tcf(z1, z2, fused, edge, cfg)
    assert torch.equal(value, fused)
    assert audit["exact_identity_fallback"] == 1.0
    assert audit["global_gate"] == 0.0


def test_tcf_trust_mixture_boundaries_are_finite_and_distinct():
    z1, z2, fused = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    edge, _ = graph()
    values = []
    for fraction in (0.0, 0.5, 1.0):
        cfg = filter_config("TCF_LOW_HIGH")
        cfg["max_high"] = 0.0
        cfg["trust_evidence_fraction"] = fraction
        value, audit = apply_tcf(z1, z2, fused, edge, cfg)
        assert torch.isfinite(value).all()
        assert audit["trust_evidence_fraction"] == fraction
        values.append(value)
    assert not torch.equal(values[0], values[-1])


def test_integrity_gate_can_force_fail_safe_identity():
    z1, z2, fused = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    edge, _ = graph()
    cfg = filter_config("TCF_LOW_HIGH")
    cfg.update({
        "max_high": 0.0,
        "global_gate_floor": 0.05,
        "integrity_scale": 100.0,
        "integrity_center": 1.0,
    })
    value, audit = apply_tcf(z1, z2, fused, edge, cfg)
    assert torch.equal(value, fused)
    assert audit["integrity_gate"] < 0.05
    assert audit["exact_identity_fallback"] == 1.0


def test_model_and_filter_api_are_identity_blind():
    source = inspect.getsource(UnifiedGraphAutoencoder) + inspect.getsource(apply_tcf)
    tree = ast.parse(source)
    forbidden = {"dataset", "tissue", "family", "path", "label"}
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (names & forbidden)
