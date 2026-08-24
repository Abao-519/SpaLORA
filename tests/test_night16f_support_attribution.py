from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    add_current_label_stay_cost,
    prepare_expansion_evidence,
)
from SpaLORA.night16e_tsre import TSREConfig, prepare_tsre_evidence
from SpaLORA.night16f_support_attribution import (
    _mass_match,
    attribution_pairwise,
    deterministic_permuted_support,
    run_attribution_arm,
)


def small_config() -> TSREConfig:
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
        trust_scale=0.2,
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
            scale_registered=0.7,
            scale_broad=0.3,
            pairwise_beta=1.0,
            self_return_strength=1.3,
            size_prior=0.0,
            expansion_cycles=1,
            capacity_scale=100000.0,
            energy_tolerance=1e-9,
        ),
        support_mix=0.6,
        relation_temperature=1.1,
        boundary_strength=0.2,
        private_strength=0.1,
        relation_stay_strength=0.8,
    )


def small_evidence(n: int = 15):
    rows = np.arange(n, dtype=np.int32)
    cols = (rows + 1) % n
    graph = sp.csr_matrix(
        (
            np.ones(2 * n),
            (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
        ),
        shape=(n, n),
    )
    rng = np.random.default_rng(17)
    retained = rng.normal(size=(n, 6)).astype(np.float32)
    view1 = rng.normal(size=(n, 5)).astype(np.float32)
    view2 = rng.normal(size=(n, 5)).astype(np.float32)
    return prepare_tsre_evidence(
        prepare_expansion_evidence((graph, graph, graph), retained, view1, view2, 4, 3, 3)
    )


def test_weighted_mass_match_is_exact() -> None:
    base = np.asarray([0.1, 0.4, 0.7, 1.2])
    factor = np.asarray([0.9, 0.2, 0.8, 0.3])
    target = 0.713
    matched, _ = _mass_match(base, factor, target)
    assert abs(float(np.dot(base, matched)) - target) <= 1e-12


def test_deterministic_permutation_is_edge_order_invariant() -> None:
    ids = np.asarray(["a", "b", "c", "d", "e"])
    rows = np.asarray([0, 0, 1, 2, 3])
    cols = np.asarray([1, 2, 3, 4, 4])
    factor = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    direct = deterministic_permuted_support(ids, rows, cols, 2, factor)
    order = np.asarray([3, 0, 4, 1, 2])
    reordered = deterministic_permuted_support(
        ids, rows[order], cols[order], 2, factor[order]
    )
    np.testing.assert_array_equal(direct, reordered[np.argsort(order)])


def test_all_matched_arms_have_exact_bimodal_capacity() -> None:
    evidence = small_evidence()
    ids = np.asarray([f"id-{index}" for index in range(15)])
    for arm in (
        "BIMODAL_SUPPORT",
        "UNIFORM_MASS_MATCHED",
        "PERMUTED_SUPPORT",
        "RNA_ONLY_SUPPORT",
        "ATAC_ONLY_SUPPORT",
    ):
        _, _, weights, _, diagnostics = attribution_pairwise(
            evidence, small_config(), ids, arm
        )
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0)
        for scale in diagnostics["per_scale_mass_ledger"]:
            assert scale["absolute_mass_error"] <= max(
                1e-10, abs(scale["bimodal_target_mass"]) * 1e-12
            )


def test_all_rejected_is_exact_base_stay_cost() -> None:
    unary = np.zeros((4, 3), dtype=np.float32)
    current = np.asarray([0, 1, 2, 0], dtype=np.int32)
    rejected = np.ones(4, dtype=np.float32)
    value = add_current_label_stay_cost(unary, current, rejected, 2.25)
    np.testing.assert_allclose(value[np.arange(4), current], 0.0)
    for node, label in enumerate(current):
        np.testing.assert_allclose(np.delete(value[node], label), 2.25)


def test_relation_stay_ablation_keeps_base_stay_on() -> None:
    initial = np.repeat(np.arange(3, dtype=np.int32), 5)
    ids = np.asarray([f"id-{index}" for index in range(15)])
    _, diagnostics = run_attribution_arm(
        initial,
        3,
        small_evidence(),
        small_config(),
        ids,
        "RELATION_STAY_OFF_BASE_STAY_ON",
    )
    assert diagnostics["config"]["relation_stay_strength"] == 0.0
    assert diagnostics["config"]["base"]["self_return_strength"] == 1.3


def test_every_registered_primary_arm_is_exact_k_and_finite() -> None:
    initial = np.repeat(np.arange(3, dtype=np.int32), 5)
    ids = np.asarray([f"id-{index}" for index in range(15)])
    evidence = small_evidence()
    for arm in (
        "DIRECT_BASE",
        "BIMODAL_SUPPORT",
        "UNIFORM_MASS_MATCHED",
        "PERMUTED_SUPPORT",
        "RNA_ONLY_SUPPORT",
        "ATAC_ONLY_SUPPORT",
    ):
        partition, diagnostics = run_attribution_arm(
            initial, 3, evidence, small_config(), ids, arm
        )
        assert len(np.unique(partition)) == 3
        assert np.min(np.bincount(partition, minlength=3)) > 0
        assert diagnostics["producer_label_reads"] == 0


def test_categorical_spatial_metrics_are_label_permutation_invariant() -> None:
    path = Path(__file__).parents[1] / "scripts" / "night16f" / "night16f_evaluator.py"
    spec = importlib.util.spec_from_file_location("night16f_evaluator_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    graph = sp.csr_matrix(
        np.asarray(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            dtype=np.float64,
        )
    )
    first = np.asarray([0, 0, 1, 1])
    second = np.asarray([9, 9, 3, 3])
    np.testing.assert_allclose(
        module.categorical_spatial_metrics(first, graph),
        module.categorical_spatial_metrics(second, graph),
        atol=0,
        rtol=0,
    )


def test_candidate_call_order_does_not_change_output() -> None:
    initial = np.repeat(np.arange(3, dtype=np.int32), 5)
    ids = np.asarray([f"id-{index}" for index in range(15)])
    evidence = small_evidence()
    first, _ = run_attribution_arm(
        initial, 3, evidence, small_config(), ids, "PERMUTED_SUPPORT"
    )
    run_attribution_arm(initial, 3, evidence, small_config(), ids, "RNA_ONLY_SUPPORT")
    second, _ = run_attribution_arm(
        initial, 3, evidence, small_config(), ids, "PERMUTED_SUPPORT"
    )
    np.testing.assert_array_equal(first, second)


def test_numeric_carrier_preserves_graph_dtype_and_records_thread_limit(tmp_path) -> None:
    path = Path(__file__).parents[1] / "scripts" / "night16f" / "build_numeric_carrier.py"
    spec = importlib.util.spec_from_file_location("night16f_carrier_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    graph = sp.csr_matrix(
        (
            np.asarray([0.123456789012345, 0.123456789012345], dtype=np.float64),
            (np.asarray([0, 1]), np.asarray([1, 0])),
        ),
        shape=(2, 2),
    )
    output = tmp_path / "carrier.npz"
    module.save_carrier(
        output,
        ids=np.asarray(["a", "b"]),
        view1=np.zeros((2, 1), dtype=np.float32),
        view2=np.zeros((2, 1), dtype=np.float32),
        retained=np.zeros((2, 1), dtype=np.float32),
        graphs=(graph, graph, graph),
        starts=[np.asarray([0, 1], dtype=np.int32)],
        start_ids=["S0"],
        metadata={"k": 2},
    )
    with np.load(output, allow_pickle=False) as carrier:
        assert carrier["graph0__data"].dtype == np.float64
    manifest = json.loads(output.with_suffix(".carrier.json").read_text())
    assert manifest["deterministic_thread_limit"] == 1
    assert manifest["graph_shapes_nnz"][0]["data_dtype"] == "float64"
