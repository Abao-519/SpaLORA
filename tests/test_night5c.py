import numpy as np
import pytest
import torch

from SpaLORA.night5c_semantic import ParameterlessAttention, assert_contract_match, declared_contract


def test_parameterless_uniform_attention_exact_and_no_parameters():
    layer = ParameterlessAttention()
    output = layer(torch.randn(7, 2, 5))
    assert torch.equal(output, torch.full((7, 2), 0.5))
    assert list(layer.parameters()) == []


def test_parameterless_attention_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ParameterlessAttention()(torch.randn(7, 3, 5))


def test_declared_contract_maps_registered_mechanisms():
    row = {"id":"B23_C10_LAPLACIAN005","attention":"uniform_all","triplet_weight":0.1,
           "triplet_margin":0.5,"laplacian_target_gradient_fraction":0.05}
    contract = declared_contract(row)
    assert contract["actual_attention_policy"] == "uniform_all"
    assert contract["actual_learned_fraction"] is None
    assert contract["enabled_mechanisms"] == ["ige_base", "laplacian", "mnn_triplet"]


def test_negative_runtime_contract_mismatch_fails_closed():
    declared = {"candidate_id":"B21","actual_attention_policy":"uniform_all","actual_learned_fraction":None,
                "enabled_mechanisms":["ige_base","laplacian","rna_anchor"],
                "learnable_attention_parameter_count":0,"uniform_weight":[0.5,0.5]}
    resolved = dict(declared); resolved["actual_attention_policy"] = "shrink_to_uniform"
    with pytest.raises(RuntimeError):
        assert_contract_match(declared, resolved)
