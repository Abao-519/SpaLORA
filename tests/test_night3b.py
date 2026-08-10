import copy

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.model_corrected import EncoderOverallCorrected
from SpaLORA.night3a_ige import LOSS_KEYS
from SpaLORA.night3b_ablation import (
    ATTENTION_MODE_BY_VARIANT,
    VARIANTS,
    active_ige_coefficients,
    active_loss_mask,
)
from SpaLORA.night3b_metrics import geary_c, mean_one_vs_rest_geary


def inputs():
    torch.manual_seed(13)
    f1 = torch.randn(7, 5)
    f2 = torch.randn(7, 3)
    idx = torch.arange(7).repeat(2, 1)
    eye = torch.sparse_coo_tensor(idx, torch.ones(7), (7, 7)).coalesce()
    return f1, f2, eye


def test_default_learned_forward_is_exact():
    f1, f2, graph = inputs()
    torch.manual_seed(7)
    default = EncoderOverallCorrected(5, 4, 3, 4)
    explicit = copy.deepcopy(default).set_attention_mode("learned")
    first = default(f1, f2, graph, graph, graph, graph)
    second = explicit(f1, f2, graph, graph, graph, graph)
    assert set(first) == set(second)
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_all_modes_keep_identical_parameter_contract():
    torch.manual_seed(19)
    baseline = EncoderOverallCorrected(5, 4, 3, 4)
    names = [(name, tuple(value.shape)) for name, value in baseline.named_parameters()]
    for mode in sorted(set(ATTENTION_MODE_BY_VARIANT.values())):
        model = copy.deepcopy(baseline).set_attention_mode(mode)
        assert [(name, tuple(value.shape)) for name, value in model.named_parameters()] == names


def test_uniform_modes_emit_exact_half_and_keep_output_contract():
    f1, f2, graph = inputs()
    for mode, uniform_keys in (
        ("uniform_within", ("alpha_omics1", "alpha_omics2")),
        ("uniform_cross", ("alpha",)),
        ("uniform_all", ("alpha_omics1", "alpha_omics2", "alpha")),
    ):
        torch.manual_seed(23)
        output = EncoderOverallCorrected(5, 4, 3, 4).set_attention_mode(mode)(
            f1, f2, graph, graph, graph, graph
        )
        assert set(output) == {
            "emb_latent_omics1", "emb_latent_omics2", "emb_latent_combined",
            "emb_recon_omics1", "emb_recon_omics2",
            "emb_latent_omics1_across_recon", "emb_latent_omics2_across_recon",
            "alpha_omics1", "alpha_omics2", "alpha",
        }
        for key in uniform_keys:
            assert torch.equal(output[key], torch.full_like(output[key], 0.5))
            assert torch.equal(output[key].sum(dim=1), torch.ones(len(f1)))


def test_registered_active_set_coefficients():
    gradients = {name: value for name, value in zip(LOSS_KEYS, (1.0, 2.0, 4.0, 8.0))}
    assert len(VARIANTS) == 8
    for variant in VARIANTS:
        mask = active_loss_mask(variant)
        coefficients = active_ige_coefficients(gradients, mask, 1e-12)
        active = [name for name in LOSS_KEYS if mask[name]]
        dropped = [name for name in LOSS_KEYS if not mask[name]]
        assert abs(sum(coefficients[name] for name in active) - 4.0) <= 1e-12
        assert all(np.isfinite(coefficients[name]) and coefficients[name] > 0 for name in active)
        assert all(coefficients[name] == 0.0 for name in dropped)
        influence = np.asarray([coefficients[name] * gradients[name] for name in active])
        assert np.max(np.abs(influence / influence.sum() - 1.0 / len(active))) < 1e-12


def test_geary_dense_sparse_equivalence_and_cluster_count():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    dense = np.zeros((6, 6), dtype=float)
    for i in range(5):
        dense[i, i + 1] = dense[i + 1, i] = 1.0
    dense_mean, dense_parts = mean_one_vs_rest_geary(labels, dense)
    sparse_mean, sparse_parts = mean_one_vs_rest_geary(labels, sp.csr_matrix(dense))
    assert dense_mean == sparse_mean
    assert dense_parts == sparse_parts
    assert len(dense_parts) == len(np.unique(labels)) == 3


def test_geary_boundary_contracts():
    graph = sp.csr_matrix(np.asarray([[0, 1], [1, 0]], dtype=float))
    assert np.isnan(geary_c(np.ones(2), graph))
    assert np.isnan(geary_c(np.asarray([0.0, 1.0]), sp.csr_matrix((2, 2))))
    try:
        geary_c(np.asarray([0.0, 1.0]), sp.csr_matrix((3, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch was not rejected")
