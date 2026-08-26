from __future__ import annotations

import itertools

import numpy as np

from SpaLORA.night18e_ccsr import (
    certified_leave_penalty,
    exhaustive_certificate_check,
    exhaustive_quantized_alpha_check,
    incident_capacity,
    margin_against_target,
    stable_random_mask,
)


def test_exhaustive_negative_margin_and_multiple_certified_nodes():
    initial = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    unary = np.array(
        [
            [3.0, 0.0, 2.0],  # negative margin for target 0
            [1.0, 0.0, 1.5],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
    )
    rows = np.array([0, 1, 2, 3], dtype=np.int32)
    cols = np.array([1, 2, 3, 4], dtype=np.int32)
    weights = np.array([0.4, 0.8, 0.2, 0.6])
    trusted = np.array([True, False, True, True, False])
    result = exhaustive_certificate_check(initial, unary, rows, cols, weights, trusted)
    assert result["status"] == "PASS"
    assert result["enumerated_states"] == 3**5


def test_exhaustive_isolated_zero_edge_and_rounding_guard():
    initial = np.array([0, 1, 0, 1], dtype=np.int32)
    unary = np.array([[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.0, 0.2]])
    rows = np.array([0], dtype=np.int32)
    cols = np.array([1], dtype=np.int32)
    weights = np.array([3.0e-7])
    trusted = np.array([True, True, True, False])
    result = exhaustive_certificate_check(
        initial, unary, rows, cols, weights, trusted, capacity_scale=1.0e5
    )
    assert result["minimum_certificate_slack"] > 0


def test_incident_capacity_and_margin_exact():
    rows = np.array([0, 0, 1], dtype=np.int32)
    cols = np.array([1, 2, 2], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0])
    capacity, degree = incident_capacity(3, rows, cols, weights)
    np.testing.assert_allclose(capacity, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(degree, [2, 2, 2])
    unary = np.array([[0.0, 2.0], [3.0, 1.0], [0.0, -1.0]])
    np.testing.assert_allclose(margin_against_target(unary, np.array([0, 1, 0])), [2, 2, -1])


def test_random_mask_is_id_stable_and_count_matched():
    ids = np.array(["c", "a", "d", "b"])
    mask = stable_random_mask(ids, 2)
    assert mask.sum() == 2
    perm = np.array([1, 3, 0, 2])
    mask2 = stable_random_mask(ids[perm], 2)
    selected1 = set(ids[mask])
    selected2 = set(ids[perm][mask2])
    assert selected1 == selected2


def _certified_tiny(
    target: np.ndarray,
    unary: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    trusted: np.ndarray,
    capacity_scale: float,
):
    incident, degree = incident_capacity(len(target), rows, cols, weights)
    certified, _ = certified_leave_penalty(
        unary, target, trusted, incident, degree, capacity_scale, 1.0e-6
    )
    return certified


def test_quantized_alpha_subspaces_negative_margin_tiny_capacity_and_isolate():
    target = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    unary = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.2, 0.0, 0.3],
            [0.3, 0.4, 0.0],
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.1],
            [0.5, 0.4, 0.0],
        ]
    )
    rows = np.array([0, 1, 2, 3], dtype=np.int32)
    cols = np.array([1, 2, 3, 4], dtype=np.int32)  # node 5 isolated
    weights = np.array([3e-7, 0.7, 0.2, 5e-7], dtype=np.float64)
    trusted = np.array([True, False, True, True, False, True])
    scale = 1.0e5
    certified = _certified_tiny(target, unary, rows, cols, weights, trusted, scale)
    result = exhaustive_quantized_alpha_check(
        target, target, certified, rows, cols, weights, trusted, scale
    )
    assert result["status"] == "PASS"
    assert result["enumerated_binary_states"] == 3 * 2**6


def test_cycle_refresh_induction_with_fixed_initial_target():
    target = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    rows = np.array([0, 1, 2, 3], dtype=np.int32)
    cols = np.array([1, 2, 3, 4], dtype=np.int32)
    weights = np.array([0.5, 0.7, 0.4, 0.6], dtype=np.float64)
    trusted = np.array([True, False, True, False, True])
    currents = [target.copy(), np.array([0, 2, 2, 2, 1], dtype=np.int32)]
    unaries = [
        np.array([[1.0, 0.0, 2.0], [1.0, 0.0, 1.2], [2.0, 1.0, 0.0], [0.0, 0.2, 0.3], [0.4, 0.0, 0.1]]),
        np.array([[3.0, 0.0, 2.0], [0.7, 0.0, 0.8], [0.2, 0.0, 1.0], [0.0, 0.5, 0.2], [0.0, 1.0, 0.4]]),
    ]
    for current, unary in zip(currents, unaries):
        certified = _certified_tiny(target, unary, rows, cols, weights, trusted, 1e6)
        exhaustive_quantized_alpha_check(
            current, target, certified, rows, cols, weights, trusted, 1e6
        )
        # The induction premise is the only cross-cycle condition: the unary
        # may change arbitrarily, while every trusted label remains the fixed
        # input target under the freshly recomputed certificate.
        assert np.array_equal(current[trusted], target[trusted])


def test_quantized_zero_edge_graph():
    target = np.array([0, 1, 0, 1], dtype=np.int32)
    unary = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    rows = np.array([], dtype=np.int32)
    cols = np.array([], dtype=np.int32)
    weights = np.array([], dtype=np.float64)
    trusted = np.array([True, True, True, False])
    certified = _certified_tiny(target, unary, rows, cols, weights, trusted, 1e5)
    result = exhaustive_quantized_alpha_check(
        target, target, certified, rows, cols, weights, trusted, 1e5
    )
    assert result["status"] == "PASS"


def test_random_quantized_cut_equals_exhaustive_optimum_and_is_persistent():
    rng = np.random.default_rng(1805)
    for case in range(40):
        n = int(rng.integers(3, 7))
        k = int(rng.integers(2, min(3, n) + 1))
        target = (np.arange(n) % k).astype(np.int32)
        unary = rng.normal(0.0, 0.7, size=(n, k))
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
        rows = np.asarray([x[0] for x in pairs], dtype=np.int32)
        cols = np.asarray([x[1] for x in pairs], dtype=np.int32)
        weights = np.exp(rng.uniform(-15.0, 0.5, size=len(pairs)))
        if case % 7 == 0 and len(weights):
            weights[0] = 0.0
        trusted = rng.random(n) < 0.55
        trusted[case % n] = True
        scale = float([1e4, 1e5, 1e6][case % 3])
        certified = _certified_tiny(
            target, unary, rows, cols, weights, trusted, scale
        )
        result = exhaustive_quantized_alpha_check(
            target, target, certified, rows, cols, weights, trusted, scale
        )
        assert result["formal_cut_integer_energy_matches_exhaustive"] is True
