import json
from pathlib import Path

import numpy as np
import pytest
import torch

from SpaLORA.night5b_rnd import (Night5BModel, assert_development_dataset,
                                 laplacian_loss, latent_reliability_weights,
                                 load_registry, registry_contracts,
                                 single_step_diffusion, undirected_edges)


REGISTRY = Path(__file__).resolve().parents[1] / "docs/night5b_authoritative_inputs/SpaLORA_Night5B_Candidate_Registry_2026-08-13.json"


def test_registry_exact_unique_b00_b24():
    contracts = registry_contracts(load_registry(REGISTRY))
    assert len(contracts) == 25
    assert len({row["config_sha256"] for row in contracts.values()}) == 25


@pytest.mark.parametrize("name", ["p22", "d1", "gse198353", "night4b"])
def test_withheld_datasets_rejected(name):
    with pytest.raises(RuntimeError):
        assert_development_dataset(name)


def test_latent_reliability_swap_row_sum_and_resume():
    rng = np.random.default_rng(4)
    first, second = rng.normal(size=(48, 9)), rng.normal(size=(48, 7))
    weights = latent_reliability_weights(first, second)
    swapped = latent_reliability_weights(second, first)
    assert np.allclose(weights.sum(1), 1.0)
    assert np.allclose(weights, swapped[:, ::-1])
    left = Night5BModel(5, 4, 3, 4, attention_policy="shrink_to_uniform", learned_fraction=.25,
                        reliability_weights=np.full((48, 2), .5, np.float32))
    left.activate_latent_reliability(weights)
    right = Night5BModel(5, 4, 3, 4, attention_policy="shrink_to_uniform", learned_fraction=.25,
                         reliability_weights=np.full((48, 2), .5, np.float32))
    right.load_state_dict(left.state_dict())
    assert torch.equal(left.reliability_weights, right.reliability_weights)
    assert bool(right.latent_reliability_active_flag.item())


def test_diffusion_alpha_zero_bitwise_and_sparse():
    rng = np.random.default_rng(5)
    embedding = rng.normal(size=(12, 6)).astype(np.float32)
    indices = torch.arange(12)
    adjacency = torch.sparse_coo_tensor(torch.stack((indices, indices)), torch.ones(12), (12, 12)).coalesce()
    assert np.array_equal(embedding, single_step_diffusion(embedding, adjacency, 0.0))


def test_laplacian_edge_contracts():
    raw = torch.tensor([[0, 1], [0, 1], [1, 0], [2, 2]], dtype=torch.long)
    adjacency = torch.sparse_coo_tensor(raw.t(), torch.ones(len(raw)), (4, 4)).coalesce()
    edges = undirected_edges(adjacency)
    assert edges.tolist() == [[0, 1]]
    z = torch.randn(4, 3, requires_grad=True)
    assert laplacian_loss(z, torch.empty((0, 2), dtype=torch.long)).item() == 0.0
    value = laplacian_loss(z, edges)
    value.backward()
    assert torch.isfinite(value) and z.grad is not None and torch.isfinite(z.grad).all()


def test_laplacian_candidates_keep_locked_uniform_attention():
    contracts = registry_contracts(load_registry(REGISTRY))
    for candidate_id in ("B21_C09_LAPLACIAN005", "B22_C09_LAPLACIAN010",
                         "B23_C10_LAPLACIAN005", "B24_C10_LAPLACIAN010"):
        assert contracts[candidate_id]["attention"] == "uniform_all"
