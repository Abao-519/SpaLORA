import numpy as np
import scipy.sparse as sp

from SpaLORA.night17b_sfrd import (
    candidate_mask,
    canonical_pair_bank,
    exceeds_all_matched_controls,
    relation_posterior,
    same_head_partition,
    training_seed_from_run_ids,
    train_residual,
)


def records():
    rows = []
    for index, candidate_id in enumerate(
        ["PRIMARY_AUTHORITY__INPUT", "KMEANS_RETAINED_S0__INPUT_START", "PATH_UNIFORM__KMEANS_RETAINED_S0__L01"]
    ):
        rows.append(
            {
                "candidate_id": candidate_id,
                "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "True",
                "molecular_joint": str(0.1 + index),
                "topology_joint": str(0.2 + index),
                "persistence": str(0.3 + index),
            }
        )
    return rows


def test_unbiased_bank_excludes_authority():
    mask = candidate_mask(records(), "UNBIASED_BANK")
    assert mask.tolist() == [False, True, True]


def test_relation_posterior_preserves_uncertain_abstention():
    partitions = np.asarray([[0, 0], [0, 1], [0, 1]], dtype=np.int32)
    posterior = relation_posterior(partitions, records(), np.asarray([0]), np.asarray([1]), weighted=False)
    assert 0.0 < posterior.probability_same[0] < 1.0
    assert posterior.uncertainty[0] > 0.8
    assert posterior.positive_weight[0] < 0.1


def test_sparse_pair_union_is_canonical_and_bounded():
    graph = sp.csr_matrix((np.ones(6), ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])), shape=(4, 4))
    retained = np.arange(16, dtype=np.float32).reshape(4, 4)
    pair_i, pair_j, spatial = canonical_pair_bank(graph, retained, feature_neighbors=1)
    assert np.all(pair_i < pair_j)
    assert len(set(zip(pair_i.tolist(), pair_j.tolist()))) == pair_i.size
    assert spatial.sum() == 3


def test_trainable_residual_has_gradients_and_parameter_change():
    rng = np.random.RandomState(4)
    n = 20
    partitions = np.vstack([np.repeat([0, 1], 10), np.repeat([0, 1], 10), np.tile([0, 1], 10)]).astype(np.int32)
    pair_i = np.arange(n - 1, dtype=np.int64)
    pair_j = pair_i + 1
    posterior = relation_posterior(partitions, records(), pair_i, pair_j, weighted=False)
    config = {
        "hidden_dim": 8,
        "residual_scale": 0.1,
        "learning_rate": 0.001,
        "steps": 2,
        "relation_weight": 1.0,
        "anchor_weight": 1.0,
        "consistency_weight": 0.1,
        "variance_weight": 0.1,
        "margin": 1.0,
    }
    result = train_residual(
        rng.normal(size=(n, 5)).astype(np.float32),
        rng.normal(size=(n, 6)).astype(np.float32),
        rng.normal(size=(n, 7)).astype(np.float32),
        pair_i,
        pair_j,
        np.ones(n - 1, dtype=bool),
        posterior,
        config,
        seed=0,
        device="cpu",
    )
    assert result.diagnostics["actual_optimizer_steps"] == 2
    assert result.diagnostics["max_gradient_norm"] > 0
    assert result.diagnostics["parameter_l2_change"] > 0
    assert np.isfinite(result.representation).all()


def test_same_head_returns_exact_k():
    rng = np.random.RandomState(5)
    partition = same_head_partition(rng.normal(size=(40, 6)), k=4, seed=0)
    assert np.unique(partition).size == 4


def test_evaluator_seed_is_recovered_from_locked_run_ids():
    run_ids = ["BASELINE__FROZEN_RETAINED", "F03__FULL_WEIGHTED__S2"]
    assert training_seed_from_run_ids(run_ids[0], run_ids) == 2
    assert training_seed_from_run_ids(run_ids[1], run_ids) == 2


def test_strict_gate_cannot_ignore_stronger_feasible_head_control():
    passed, strongest_ari, strongest_nmi = exceeds_all_matched_controls(
        0.15, 0.22, [(0.13, 0.19), (0.20, 0.28)]
    )
    assert not passed
    assert strongest_ari == 0.20
    assert strongest_nmi == 0.28
