import numpy as np

from SpaLORA.night17b_sfrd import RelationPosterior
from SpaLORA.night17d_learned_evidence import (
    SelectorWeights,
    fit_weight_config,
    permutation_order,
    posterior_without_candidate,
    relation_alignment,
    select_candidate,
    weight_grid,
)


def posterior(probability, weights):
    p = np.asarray(probability, dtype=np.float32)
    return RelationPosterior(
        probability_same=p,
        uncertainty=np.zeros_like(p),
        positive_weight=np.ones_like(p),
        negative_weight=np.zeros_like(p),
        candidate_weights=np.asarray(weights, dtype=np.float64),
        selected_candidate_count=int(np.count_nonzero(weights)),
    )


def test_leave_one_candidate_out_exact():
    same0 = np.asarray([1.0, 0.0, 1.0])
    same1 = np.asarray([0.0, 1.0, 1.0])
    probability = 0.25 * same0 + 0.75 * same1
    result, removed = posterior_without_candidate(probability, 0.25, same0)
    assert removed
    assert np.allclose(result, same1)


def test_candidate_not_in_posterior_is_unchanged():
    probability = np.asarray([0.2, 0.8])
    result, removed = posterior_without_candidate(probability, 0.0, np.ones(2))
    assert not removed
    assert np.array_equal(result, probability)


def test_alignment_self_inclusion_sensitivity_is_exposed():
    partition = np.asarray([0, 0, 1])
    pair_i = np.asarray([0, 0])
    pair_j = np.asarray([1, 2])
    result = relation_alignment(
        partition,
        pair_i,
        pair_j,
        np.asarray([True, False]),
        posterior([0.75, 0.25], [0.25, 0.75]),
        0,
        np.asarray([1.0, 0.0]),
    )
    assert result["relation_self_removed"] is True
    assert "relation_alignment_full" in result
    assert "relation_alignment_loo" in result


def test_permuted_candidate_contribution_uses_same_stratified_order():
    strata = np.asarray([False, True, False, True, True, False])
    order = permutation_order(strata, seed=17)
    contribution = np.asarray([1, 0, 1, 1, 0, 0], dtype=np.float64)
    assert sorted(order[~strata].tolist()) == np.flatnonzero(~strata).tolist()
    assert sorted(order[strata].tolist()) == np.flatnonzero(strata).tolist()
    assert np.array_equal(contribution[order], np.take(contribution, order))


def test_selector_uses_feasible_candidates_only_and_is_order_invariant():
    rows = [
        {"candidate_id": "b", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "true", "molecular_rank": 0.2, "topology_rank": 0.2, "LEARNED_evidence": 0.2, "LEARNED_uncertainty_rank": 0.0},
        {"candidate_id": "a", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "true", "molecular_rank": 0.8, "topology_rank": 0.8, "LEARNED_evidence": 0.8, "LEARNED_uncertainty_rank": 0.0},
        {"candidate_id": "z", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "false", "molecular_rank": 1.0, "topology_rank": 1.0, "LEARNED_evidence": 1.0, "LEARNED_uncertainty_rank": 0.0},
    ]
    weights = SelectorWeights(1, 1, 1, 0.5)
    assert select_candidate(rows, weights)["candidate_id"] == "a"
    assert select_candidate(list(reversed(rows)), weights)["candidate_id"] == "a"


def test_fit_uses_only_supplied_training_lanes_and_mechanical_tiebreak():
    rows = {
        "train_a": [
            {"candidate_id": "x", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "true", "molecular_rank": 1.0, "topology_rank": 0.0, "LEARNED_evidence": 0.0, "LEARNED_uncertainty_rank": 0.0},
            {"candidate_id": "y", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "true", "molecular_rank": 0.0, "topology_rank": 1.0, "LEARNED_evidence": 1.0, "LEARNED_uncertainty_rank": 0.0},
        ]
    }
    evaluation = {"train_a": {"x": {"absolute_ari": 0.1, "absolute_nmi": 0.1}, "y": {"absolute_ari": 0.2, "absolute_nmi": 0.2}}}
    authority = {"train_a": {"absolute_ari": 0.15, "absolute_nmi": 0.15}}
    grid = weight_grid({"molecular": [0.0, 1.0], "topology": [1.0], "learned": [1.0], "uncertainty": [0.0]})
    winner, summaries, detail = fit_weight_config(rows, evaluation, authority, grid)
    assert winner.molecular == 0.0
    assert len(summaries) == 2
    assert {row["lane"] for row in detail} == {"train_a"}
