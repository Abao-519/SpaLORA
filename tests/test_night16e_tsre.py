from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    add_current_label_stay_cost,
    alpha_expansion_move,
    prepare_expansion_evidence,
)
from SpaLORA.night16e_tsre import (
    TSREConfig,
    _tri_state_from_similarities,
    boundary_exclusion_unary,
    private_modality_unary,
    prepare_tsre_evidence,
    tsre_expansion,
)


def test_tri_state_is_finite_nonnegative_partition() -> None:
    relation = _tri_state_from_similarities(
        np.asarray([0, 0, 1]),
        np.asarray([1, 2, 2]),
        np.asarray([0.9, 0.2, 0.8]),
        np.asarray([0.8, 0.1, 0.3]),
    )
    total = relation.support + relation.boundary + relation.conflict
    assert np.all(np.isfinite(total))
    assert np.all(relation.support >= 0)
    assert np.all(relation.boundary >= 0)
    assert np.all(relation.conflict >= 0)
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_edge_order_permutation_preserves_associated_states() -> None:
    rows = np.asarray([0, 0, 1, 2])
    cols = np.asarray([1, 2, 2, 3])
    first = np.asarray([0.9, 0.2, 0.7, 0.1])
    second = np.asarray([0.8, 0.3, 0.1, 0.2])
    direct = _tri_state_from_similarities(rows, cols, first, second)
    order = np.asarray([2, 0, 3, 1])
    permuted = _tri_state_from_similarities(rows[order], cols[order], first[order], second[order])
    inverse = np.argsort(order)
    np.testing.assert_allclose(direct.support, permuted.support[inverse])
    np.testing.assert_allclose(direct.boundary, permuted.boundary[inverse])
    np.testing.assert_allclose(direct.conflict, permuted.conflict[inverse])


def test_boundary_is_legal_frozen_unary_not_negative_potts() -> None:
    partition = np.asarray([0, 1], dtype=np.int32)
    boundary = sp.csr_matrix(np.asarray([[0.0, 0.7], [0.7, 0.0]]))
    unary = boundary_exclusion_unary(partition, 2, boundary, 2.0)
    assert np.all(unary >= 0)
    np.testing.assert_allclose([unary[0, 1], unary[1, 0]], [1.4, 1.4])
    assert unary[0, 0] == unary[1, 1] == 0.0


def test_private_unary_prefers_numerically_supported_modality() -> None:
    view1 = np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    view2 = np.asarray([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    margins = np.asarray([[10.0, 0.1], [10.0, 0.1]], dtype=np.float32)
    first = sp.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    second = sp.csr_matrix((2, 2), dtype=np.float32)
    unary, diagnostics = private_modality_unary(view1, view2, margins, first, second, 1.0)
    assert diagnostics["mean_private_view1_weight"] > 0.99
    assert np.all(unary[:, 0] < unary[:, 1])


def test_categorical_spatial_metrics_are_label_permutation_invariant() -> None:
    evaluator_path = Path(__file__).parents[1] / "scripts" / "night16e" / "night16e_evaluator.py"
    spec = importlib.util.spec_from_file_location("night16e_evaluator", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    graph = sp.csr_matrix(
        np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    first = np.asarray([0, 0, 1, 1])
    second = np.asarray([7, 7, 3, 3])
    np.testing.assert_allclose(
        module.categorical_spatial_metrics(first, graph),
        module.categorical_spatial_metrics(second, graph),
        atol=0,
        rtol=0,
    )


def test_all_rejected_mass_is_exact_current_state_stay_cost() -> None:
    unary = np.zeros((3, 2), dtype=np.float32)
    current = np.asarray([0, 1, 0], dtype=np.int32)
    rejected = np.ones(3, dtype=np.float32)
    value = add_current_label_stay_cost(unary, current, rejected, 2.5)
    np.testing.assert_allclose(value[np.arange(3), current], 0.0)
    np.testing.assert_allclose(value[np.arange(3), 1 - current], 2.5)


def test_accepted_alpha_move_is_frozen_energy_monotone_and_exact_k() -> None:
    initial = np.asarray([0, 0, 1, 1], dtype=np.int32)
    unary = np.asarray(
        [[0.0, 3.0], [2.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
    )
    rows = np.asarray([0, 1, 2], dtype=np.int32)
    cols = np.asarray([1, 2, 3], dtype=np.int32)
    weights = np.asarray([0.1, 0.1, 0.1], dtype=np.float64)
    proposal, accepted, before, after = alpha_expansion_move(
        initial, 1, unary, rows, cols, weights, 100000.0, 1e-9
    )
    assert accepted
    assert after < before
    assert len(np.unique(proposal)) == 2


def test_full_and_evaluation_mask_cluster_sizes_are_separate() -> None:
    evaluator_path = Path(__file__).parents[1] / "scripts" / "night16e" / "night16e_evaluator.py"
    spec = importlib.util.spec_from_file_location("night16e_evaluator_sizes", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    full, evaluated = module.cluster_sizes_full_eval(
        np.asarray([0, 0, 1, 1]), np.asarray([True, False, True, True]), 2
    )
    np.testing.assert_array_equal(full, [2, 2])
    np.testing.assert_array_equal(evaluated, [1, 2])


def _small_config(boundary: float, private: float) -> TSREConfig:
    local = ContinuousEnergyConfig(
        beta=1.0,
        edge_floor=0.1,
        conflict_center=0.3,
        conflict_temperature=0.2,
        conflict_union_weight=0.5,
        conflict_penalty=0.1,
        mass_center=0.0,
        mass_temperature=0.2,
        neighbor_capacity=0.7,
        low_weight=0.2,
        twohop_weight=0.1,
        high_weight=0.2,
        unary_temperature=0.5,
        retained_bias=2.0,
        view_balance=0.0,
        trust_scale=1.0,
        trust_center=0.0,
        trust_temperature=0.2,
        move_threshold=0.1,
        move_fraction=0.1,
        sweeps=1,
    )
    return TSREConfig(
        base=ExpansionEnergyConfig(
            local=local,
            scale_fine=1.0,
            scale_registered=1.0,
            scale_broad=1.0,
            pairwise_beta=1.0,
            self_return_strength=1.0,
            size_prior=0.0,
            expansion_cycles=1,
            capacity_scale=100000.0,
            energy_tolerance=1e-9,
        ),
        support_mix=1.0,
        relation_temperature=1.0,
        boundary_strength=boundary,
        private_strength=private,
        relation_stay_strength=1.0,
    )


def test_candidate_call_order_does_not_change_partitions() -> None:
    n = 12
    row = np.arange(n, dtype=np.int32)
    col = (row + 1) % n
    graph = sp.csr_matrix(
        (np.ones(2 * n), (np.concatenate((row, col)), np.concatenate((col, row)))),
        shape=(n, n),
    )
    rng = np.random.default_rng(7)
    retained = rng.normal(size=(n, 5)).astype(np.float32)
    view1 = rng.normal(size=(n, 4)).astype(np.float32)
    view2 = rng.normal(size=(n, 4)).astype(np.float32)
    initial = np.repeat(np.arange(3, dtype=np.int32), 4)
    evidence = prepare_tsre_evidence(
        prepare_expansion_evidence((graph, graph, graph), retained, view1, view2, 4, 3, 3)
    )
    first = _small_config(0.2, 0.1)
    second = _small_config(0.8, 0.4)
    a1, _ = tsre_expansion(initial, 3, evidence, first)
    b1, _ = tsre_expansion(initial, 3, evidence, second)
    b2, _ = tsre_expansion(initial, 3, evidence, second)
    a2, _ = tsre_expansion(initial, 3, evidence, first)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)


def test_noop_replay_registry_has_one_unique_input_candidate() -> None:
    builder_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "night16e"
        / "build_score_frontier_summary.py"
    )
    spec = importlib.util.spec_from_file_location("night16e_score_frontier_builder", builder_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    registry = {
        "INPUT_STRONG_START": {
            "candidate_id": "INPUT_STRONG_START",
            "profile_id": "INPUT_STRONG_START",
            "variant": "INPUT_STRONG_START",
            "config": None,
        },
        "TSRE_FULL_example": {
            "candidate_id": "TSRE_FULL_example",
            "profile_id": "profile_example",
            "variant": "TSRE_FULL",
            "config": {"pairwise_beta": 1.0},
        },
    }
    chosen = module.build_replay_candidates("INPUT_STRONG_START", registry)
    assert [candidate["candidate_id"] for candidate in chosen] == ["INPUT_STRONG_START"]
    assert len({candidate["candidate_id"] for candidate in chosen}) == len(chosen)
