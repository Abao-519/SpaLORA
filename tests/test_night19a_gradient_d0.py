import numpy as np
import torch

from SpaLORA.night17b_sfrd import RelationPosterior, sha256_array
from SpaLORA.night19a_gradient_d0 import (
    build_model,
    evidence_support_strata,
    flatten_loss_gradient,
    loss_components,
    ordered_trainable_parameters,
    prepare_tensors,
    safe_gradient_pair,
    train_standard_sum_d0,
)
from scripts.night19a.build_d0_gate import evaluate_seed
from scripts.night19a.build_d0_gate_rev1 import evaluate_seed as evaluate_seed_rev1
from SpaLORA.night19a_sparse_arbitration import (
    deterministic_stratified_permutation,
    global_min_norm_gradient,
    make_topology_disabled_tensors,
    projection_groups_and_strengths,
    train_stage_a_arm,
    vanilla_pcgrad_gradient,
)


CONFIG = {
    "hidden_dim": 4,
    "residual_scale": 0.05,
    "learning_rate": 0.0007,
    "steps": 40,
    "relation_weight": 1.0,
    "anchor_weight": 4.0,
    "self_return_weight": 4.0,
    "consistency_weight": 0.5,
    "variance_weight": 0.1,
}


def posterior(values):
    p = np.asarray(values, dtype=np.float32)
    u = np.clip(1.0 - np.abs(2.0 * p - 1.0), 0.0, 1.0).astype(np.float32)
    return RelationPosterior(
        probability_same=p,
        uncertainty=u,
        positive_weight=np.maximum(2.0 * p - 1.0, 0).astype(np.float32),
        negative_weight=np.maximum(1.0 - 2.0 * p, 0).astype(np.float32),
        candidate_weights=np.ones(4, dtype=np.float64) / 4.0,
        selected_candidate_count=4,
    )


def toy_inputs():
    rng = np.random.RandomState(7)
    n = 9
    view1 = rng.normal(size=(n, 3)).astype(np.float32)
    view2 = rng.normal(size=(n, 2)).astype(np.float32)
    retained = rng.normal(size=(n, 4)).astype(np.float32)
    anchor = retained / np.maximum(np.linalg.norm(retained, axis=1, keepdims=True), 1e-8)
    pair_i = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6], dtype=np.int64)
    pair_j = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 8, 8], dtype=np.int64)
    spatial = np.asarray([True] * 8 + [False] * 4)
    post = posterior([0.95, 0.05, 0.8, 0.2, 0.65, 0.35, 0.9, 0.1, 0.7, 0.3, 0.58, 0.42])
    strata, _ = evidence_support_strata(post, spatial)
    gate = np.linspace(0, 1, n, dtype=np.float32)
    tensors = prepare_tensors(
        view1, view2, retained, anchor.astype(np.float32), gate,
        pair_i, pair_j, spatial, post, strata, "cpu",
    )
    return view1, view2, retained, anchor.astype(np.float32), tensors


def test_unused_gradient_coordinates_are_zero_filled_and_aligned():
    class Split(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
            self.right = torch.nn.Parameter(torch.tensor([5.0]))

    model = Split()
    ordered = ordered_trainable_parameters(model)
    left_loss = (model.left ** 2).sum()
    right_loss = (model.right ** 2).sum()
    left, left_active = flatten_loss_gradient(left_loss, ordered)
    right, right_active = flatten_loss_gradient(right_loss, ordered)
    assert left.tolist() == [4.0, 6.0, 0.0]
    assert right.tolist() == [0.0, 0.0, 10.0]
    assert left_active == ("left",)
    assert right_active == ("right",)
    pair = safe_gradient_pair(left, right)
    assert pair["dot"] == 0.0
    assert pair["cosine"] == 0.0


def test_zero_norm_cosine_is_none_not_zero():
    pair = safe_gradient_pair(torch.zeros(3), torch.ones(3))
    assert pair["cosine"] is None
    assert pair["negative_inner_product"] is False


def test_tied_evidence_receives_identical_stratum():
    post = posterior([0.9, 0.9, 0.1, 0.1, 0.7, 0.3])
    strata, diagnostics = evidence_support_strata(post, np.ones(6, dtype=bool))
    assert strata[0] == strata[1]
    assert strata[2] == strata[3]
    assert diagnostics["support_unique_count"] < 6


def test_relation_strata_sum_to_weighted_relation():
    view1, view2, retained, anchor, tensors = toy_inputs()
    model = build_model(view1.shape[1], view2.shape[1], retained.shape[1], CONFIG, 0, "cpu")
    _, _, weighted, strata = loss_components(model, tensors, CONFIG, 0)
    error = abs(float(sum(strata.values()).detach()) - float(weighted["relation"].detach()))
    assert error < 1e-6


def test_real_train_parameter_change_and_strict_reload_exact():
    view1, view2, retained, anchor, tensors = toy_inputs()
    model = build_model(view1.shape[1], view2.shape[1], retained.shape[1], CONFIG, 3, "cpu")
    result = train_standard_sum_d0(model, tensors, CONFIG)
    assert result["parameter_l2_change"] > 0
    assert [row["completed_steps"] for row in result["trajectory"]] == [0, 1, 5, 20, 40]
    assert result["trajectory"][0]["zero_start_boundary"] is True
    assert np.array_equal(result["initial_representation"], anchor)
    rebuilt = build_model(view1.shape[1], view2.shape[1], retained.shape[1], CONFIG, 999, "cpu")
    rebuilt.load_state_dict(result["state_dict"], strict=True)
    from SpaLORA.night19a_gradient_d0 import zero_start_representation
    replay = zero_start_representation(rebuilt, tensors)
    assert np.array_equal(replay, result["final_representation"])
    assert sha256_array(replay) == result["final_representation_sha256"]


def test_d0_gate_excludes_step0_zero_start_boundary():
    def row(step, cosine):
        pair = {"dot": -1.0 if cosine is not None and cosine < 0 else 0.0,
                "cosine": cosine, "norm_a": 1.0, "norm_b": 1.0,
                "negative_inner_product": bool(cosine is not None and cosine < 0)}
        return {
            "completed_steps": step,
            "gradient_norms": {"anchor": 0.0 if step == 0 else 1.0},
            "pair_metrics": {
                "relation__anchor": pair,
                "relation__consistency": pair,
                "relation__variance": pair,
            },
            "relation_stratum_vs_anchor": {
                "relation_low": {"cosine": None if step == 0 else -0.2},
                "relation_mid": {"cosine": None if step == 0 else 0.0},
                "relation_high": {"cosine": None if step == 0 else 0.2},
            },
        }
    manifest = {"trajectory": [row(0, -1.0), row(1, -0.2), row(5, -0.2), row(20, 0.2), row(40, 0.2)]}
    result = evaluate_seed(manifest)
    assert result["pair_summary"]["relation__anchor"]["conflict_steps"] == [1, 5]
    assert result["step0_excluded_from_persistence_gate"] is True


def test_d0_rev1_excludes_step1_ramp_but_requires_two_operational_conflicts():
    def row(step, cosine, ratio):
        norm_a = 1.0
        norm_b = ratio
        pair = {
            "dot": float(cosine or 0.0) * norm_a * norm_b,
            "cosine": cosine,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "negative_inner_product": bool(cosine is not None and cosine < 0),
        }
        return {
            "completed_steps": step,
            "gradient_norms": {"anchor": 0.0 if step == 0 else norm_b},
            "pair_metrics": {
                "relation__anchor": pair,
                "relation__consistency": {**pair, "cosine": 0.2, "negative_inner_product": False},
                "relation__variance": {**pair, "cosine": 0.2, "negative_inner_product": False},
            },
            "relation_stratum_vs_anchor": {
                "relation_low": {"cosine": None if step == 0 else -0.2},
                "relation_mid": {"cosine": None if step == 0 else 0.0},
                "relation_high": {"cosine": None if step == 0 else 0.2},
            },
        }

    manifest = {"trajectory": [
        row(0, None, 0.0),
        row(1, -0.8, 2e-4),
        row(5, -0.7, 2e-3),
        row(20, -0.6, 4e-3),
        row(40, 0.1, 8e-3),
    ]}
    result = evaluate_seed_rev1(manifest)
    summary = result["pair_summary"]["relation__anchor"]
    assert summary["directional_negative_steps_including_ramp"] == [1, 5, 20]
    assert summary["operational_conflict_steps"] == [5, 20]
    assert summary["seed_pair_pass"] is True
    assert summary["step_diagnostics"]["1"]["zero_start_ramp_excluded"] is True


def test_permuted_evidence_is_deterministic_bijective_and_mass_matched():
    ids = np.asarray([f"s{i}" for i in range(12)])
    pair_i = np.arange(9, dtype=np.int64)
    pair_j = np.arange(1, 10, dtype=np.int64)
    spatial = np.asarray([True] * 6 + [False] * 3)
    strata = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    weight = np.linspace(0.1, 1.0, 9)
    first = deterministic_stratified_permutation(ids, pair_i, pair_j, spatial, strata)
    second = deterministic_stratified_permutation(ids, pair_i, pair_j, spatial, strata)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, strata)
    for flag in (False, True):
        assert np.array_equal(np.sort(first[spatial == flag]), np.sort(strata[spatial == flag]))
    _, _, diagnostic = projection_groups_and_strengths(ids, pair_i, pair_j, spatial, weight, strata, True)
    assert diagnostic["maximum_absolute_mass_error"] < 1e-8


def test_pcgrad_and_global_min_norm_fixed_coordinates_finite():
    gradients = (
        torch.tensor([1.0, -1.0, 0.0]),
        torch.tensor([-0.5, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.25]),
    )
    pcgrad = vanilla_pcgrad_gradient(gradients)
    minimum, weights = global_min_norm_gradient(gradients)
    assert torch.all(torch.isfinite(pcgrad))
    assert torch.all(torch.isfinite(minimum))
    assert torch.all(weights >= 0)
    assert abs(float(weights.sum()) - 1.0) < 1e-6


def test_global_min_norm_excludes_inactive_zero_gradient():
    gradients = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([-0.25, 1.0]),
        torch.zeros(2),
    )
    result, weights = global_min_norm_gradient(gradients)
    assert weights[2] == 0
    assert abs(float(weights[:2].sum()) - 1.0) < 1e-6
    assert torch.all(torch.isfinite(result))


def test_arbitration_disabled_matches_standard_sum_representation():
    view1, view2, retained, _, tensors = toy_inputs()
    spatial = np.asarray([True] * 8 + [False] * 4)
    topology_disabled = make_topology_disabled_tensors(tensors, spatial)
    groups = spatial.astype(np.int64) * 3 + tensors.strata.cpu().numpy()
    strengths = np.tile(np.asarray([5.0 / 6.0, 0.5, 1.0 / 6.0]), 2)
    standard = train_stage_a_arm(
        "STANDARD_WEIGHTED_SUM", tensors, topology_disabled, CONFIG, 11,
        groups, strengths, groups, strengths, "cpu",
    )
    disabled = train_stage_a_arm(
        "ARBITRATION_DISABLED_SAME_LOSSES", tensors, topology_disabled, CONFIG, 11,
        groups, strengths, groups, strengths, "cpu",
    )
    assert np.allclose(
        standard["final_representation"], disabled["final_representation"], rtol=1e-6, atol=1e-6
    )
