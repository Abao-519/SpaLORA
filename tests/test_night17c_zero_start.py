import numpy as np
import torch

from SpaLORA.night17b_sfrd import RelationPosterior
from SpaLORA.night17c_zero_start import (
    balanced_soft_relation_loss,
    node_trust_gate,
    stratified_permute_relation,
    train_zero_start,
)
from scripts.night17c.night17c_select import select


def posterior(probability):
    probability = np.asarray(probability, dtype=np.float32)
    uncertainty = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)[: probability.size]
    return RelationPosterior(probability, uncertainty, probability, 1 - probability, np.ones(2), 2)


def test_node_gate_is_bounded_and_self_return_exists():
    pair_i = np.asarray([0, 1, 2, 0])
    pair_j = np.asarray([1, 2, 3, 3])
    gate, _ = node_trust_gate(4, pair_i, pair_j, np.asarray([1, 1, 0, 0], dtype=bool), posterior([0.95, 0.51, 0.1, 0.49]))
    assert np.all((gate >= 0) & (gate <= 1))
    assert np.any(gate == 0)


def test_zero_start_and_zero_gate_are_exact_after_training():
    rng = np.random.RandomState(7)
    n = 12
    view1 = rng.normal(size=(n, 5)).astype(np.float32)
    view2 = rng.normal(size=(n, 4)).astype(np.float32)
    retained = rng.normal(size=(n, 6)).astype(np.float32)
    smooth = rng.normal(size=(n, 6)).astype(np.float32)
    pair_i = np.arange(8) % n
    pair_j = (pair_i + 1) % n
    post = posterior(np.linspace(0.1, 0.9, 8))
    post = RelationPosterior(post.probability_same, np.linspace(0.1, 0.8, 8).astype(np.float32), post.positive_weight, post.negative_weight, post.candidate_weights, 2)
    gate = np.linspace(0, 1, n).astype(np.float32)
    config = dict(hidden_dim=8, residual_scale=.1, learning_rate=.001, steps=2, relation_weight=1., anchor_weight=2., self_return_weight=3., consistency_weight=.5, variance_weight=.1)
    result = train_zero_start(view1, view2, retained, smooth, gate, pair_i, pair_j, np.ones(8, dtype=bool), post, config, seed=0)
    assert result.diagnostics["step0_exact_smooth"]
    assert result.diagnostics["parameter_l2_change"] > 0
    assert result.diagnostics["max_gradient_norm"] > 0
    assert np.array_equal(result.representation[gate == 0], smooth[gate == 0])


def test_strict_gate_uses_strongest_control_and_permutation():
    rows = []
    for lane, base in (("A", .2), ("B", .3), ("C", .4)):
        for arm, offset in (("FROZEN_RETAINED_SAME_HEAD", 0), ("FULL_BANK_SMOOTH_REFERENCE", .01), ("UNBIASED_SMOOTH_REFERENCE", .02), ("ZERO_RESIDUAL_CONTROL", .02)):
            rows.append(dict(lane=lane, arm=arm, config_id="BASELINE", ari=str(base+offset), nmi=str(base+offset)))
        rows.append(dict(lane=lane, arm="UNBIASED_FULL", config_id="Z", ari=str(base+.03), nmi=str(base+.03)))
        rows.append(dict(lane=lane, arm="PERMUTED_RELATION", config_id="Z", ari=str(base+.01), nmi=str(base+.01)))
    result = select(rows)
    assert result["gate_passed"] and result["strict_pass_lanes"] == 3
    rows[-1]["ari"] = "0.9"
    rows[-1]["nmi"] = "0.9"
    result = select(rows)
    assert result["strict_pass_lanes"] == 2


def test_model_has_exact_zero_final_layer_at_construction():
    from SpaLORA.night17c_zero_start import ZeroStartRelationEncoder
    model = ZeroStartRelationEncoder(3, 4, 5, 6, .1)
    assert torch.count_nonzero(model.residual_out.weight).item() == 0
    assert torch.count_nonzero(model.residual_out.bias).item() == 0


def test_stratified_permutation_preserves_each_edge_type_distribution():
    probability = np.asarray([.1, .2, .8, .9, .3, .7], dtype=np.float32)
    uncertainty = np.asarray([.2, .3, .4, .5, .6, .7], dtype=np.float32)
    post = RelationPosterior(probability, uncertainty, probability, 1-probability, np.ones(2), 2)
    strata = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    permuted = stratified_permute_relation(post, strata)
    for value in (False, True):
        assert np.array_equal(np.sort(permuted.probability_same[strata == value]), np.sort(probability[strata == value]))
        assert np.array_equal(np.sort(permuted.uncertainty[strata == value]), np.sort(uncertainty[strata == value]))
    assert not np.array_equal(permuted.probability_same, probability)


def test_balanced_soft_relation_loss_resists_class_mass_imbalance():
    prediction = torch.tensor([.8, .8, .8, .8], dtype=torch.float32)
    target = torch.tensor([.99, .99, .99, .01], dtype=torch.float32)
    weight = torch.ones(4)
    loss, positive, negative, positive_mass, negative_mass = balanced_soft_relation_loss(prediction, target, weight)
    assert torch.isfinite(loss)
    assert positive_mass > negative_mass
    assert torch.allclose(loss, .5 * (positive + negative))
    assert negative > positive


def test_degenerate_high_confidence_iqr_does_not_zero_all_nodes():
    pair_i = np.asarray([0, 1, 2, 3])
    pair_j = np.asarray([1, 2, 3, 0])
    post = RelationPosterior(
        np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32), np.ones(2), 2,
    )
    gate, diagnostics = node_trust_gate(4, pair_i, pair_j, np.ones(4, dtype=bool), post)
    assert np.array_equal(gate, np.ones(4, dtype=np.float32))
    assert diagnostics["calibration_mode"] == "RAW_CONFIDENCE_DEGENERATE_IQR"
