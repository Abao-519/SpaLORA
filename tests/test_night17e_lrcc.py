from __future__ import annotations

import json
import csv
from argparse import Namespace

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig, prepare_expansion_evidence
from SpaLORA.night17e_lrcc import (
    LRCCConfig,
    _stable_stratified_permutation,
    lrcc_expansion,
    prepare_relation_evidence,
    relation_conditioned_pairwise,
)
from scripts.night17e.night17e_loso import fit


def config() -> LRCCConfig:
    local = ContinuousEnergyConfig(
        beta=1.0,
        edge_floor=0.05,
        conflict_center=0.4,
        conflict_temperature=0.2,
        conflict_union_weight=0.5,
        conflict_penalty=0.2,
        mass_center=0.0,
        mass_temperature=0.2,
        neighbor_capacity=0.8,
        low_weight=0.2,
        twohop_weight=0.1,
        high_weight=0.2,
        unary_temperature=0.5,
        retained_bias=0.5,
        view_balance=0.0,
        trust_scale=0.1,
        trust_center=0.2,
        trust_temperature=0.5,
        move_threshold=0.0,
        move_fraction=0.1,
        sweeps=2,
    )
    base = ExpansionEnergyConfig(
        local=local,
        scale_fine=0.5,
        scale_registered=0.3,
        scale_broad=0.2,
        pairwise_beta=2.0,
        self_return_strength=1.0,
        size_prior=0.0,
        expansion_cycles=2,
        capacity_scale=100000.0,
    )
    return LRCCConfig(base, 0.6, 0.05, 1.0, 1.0)


def evidence():
    rng = np.random.RandomState(4)
    n = 18
    graphs = []
    for hop in (1, 2, 3):
        rows, cols = [], []
        for index in range(n):
            for step in range(1, hop + 1):
                rows.extend((index, index))
                cols.extend(((index + step) % n, (index - step) % n))
        graphs.append(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)))
    retained = rng.normal(size=(n, 7)).astype(np.float32)
    view1 = rng.normal(size=(n, 5)).astype(np.float32)
    view2 = rng.normal(size=(n, 6)).astype(np.float32)
    prepared = prepare_expansion_evidence(graphs, retained, view1, view2, retained_dim=5, view_dim=4, edge_dim=3)
    learned = [retained + 0.02 * rng.normal(size=retained.shape) for _ in range(3)]
    zero = [np.roll(retained, shift, axis=0) for shift in (0, 1, 2)]
    ids = np.asarray([f"spot-{index:02d}" for index in range(n)])
    return prepared, learned, zero, ids


def test_stratified_permutation_is_fixed_bijection_and_nonidentity():
    rows = np.arange(16, dtype=np.int32)
    cols = rows + 1
    ids = np.asarray([f"s{x}" for x in range(18)])
    weights = np.linspace(0.1, 1.0, len(rows))
    first = _stable_stratified_permutation(ids, rows, cols, weights, 1)
    second = _stable_stratified_permutation(ids, rows, cols, weights, 1)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(len(rows)))
    assert not np.array_equal(first, np.arange(len(rows)))


def test_relation_arms_are_nonnegative_and_mass_matched():
    prepared, learned, zero, ids = evidence()
    value = config()
    relation = prepare_relation_evidence(prepared, learned, zero, value)
    results = {}
    for arm in (
        "LEARNED_RELATION",
        "ZERO_RELATION",
        "UNIFORM_MASS_MATCHED",
        "PERMUTED_RELATION",
        "RELATION_DISABLED",
    ):
        rows, cols, weights, _, diagnostics = relation_conditioned_pairwise(
            prepared, relation, value, arm, ids
        )
        assert len(rows) == len(cols) == len(weights)
        assert np.all(np.isfinite(weights)) and np.all(weights >= 0)
        results[arm] = weights
        if arm in {"ZERO_RELATION", "UNIFORM_MASS_MATCHED", "PERMUTED_RELATION"}:
            assert max(x["mass_match_absolute_error"] for x in diagnostics["scale_diagnostics"]) < 1e-9
    assert not np.array_equal(results["LEARNED_RELATION"], results["ZERO_RELATION"])
    assert not np.array_equal(results["LEARNED_RELATION"], results["PERMUTED_RELATION"])
    assert not np.array_equal(results["LEARNED_RELATION"], results["RELATION_DISABLED"])


def test_edge_set_alignment_exact_k_and_frozen_cycle_monotonicity():
    prepared, learned, zero, ids = evidence()
    value = config()
    relation = prepare_relation_evidence(prepared, learned, zero, value)
    initial = np.repeat(np.arange(3, dtype=np.int32), 6)
    partition, diagnostics = lrcc_expansion(
        initial, 3, ids, prepared, relation, value, "LEARNED_RELATION"
    )
    assert np.unique(partition).size == 3
    assert np.bincount(partition, minlength=3).min() > 0
    cycles = json.loads(diagnostics["cycle_energy_ledger_json"])
    assert cycles
    assert all(row["end_energy"] <= row["start_energy"] + 1e-9 for row in cycles)


def _write_loso_rows(path, lane, baseline_ari):
    fields = ["lane", "start_id", "arm", "config_id", "absolute_ari", "absolute_nmi"]
    rows = [
        {
            "lane": lane,
            "start_id": "METHOD_NIGHT16H_START",
            "arm": "INPUT_START",
            "config_id": "INPUT",
            "absolute_ari": baseline_ari,
            "absolute_nmi": 0.5,
        }
    ]
    for index, config_id in enumerate(("L01_MIX025", "L02_MIX050", "L03_MIX075", "L04_MIX050_POWER2")):
        rows.append(
            {
                "lane": lane,
                "start_id": "METHOD_NIGHT16H_START",
                "arm": "LEARNED_RELATION",
                "config_id": config_id,
                "absolute_ari": baseline_ari + 0.001 * index,
                "absolute_nmi": 0.5 + 0.001 * index,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_strict_loso_fit_reads_only_declared_training_lanes(tmp_path):
    first = tmp_path / "lane_a.csv"
    second = tmp_path / "lane_b.csv"
    output = tmp_path / "fit.json"
    _write_loso_rows(first, "LANE_A", 0.2)
    _write_loso_rows(second, "LANE_B", 0.3)
    fit(
        Namespace(
            training_evaluation=[str(first), str(second)],
            expected_training_lane=["LANE_A", "LANE_B"],
            heldout_lane="LANE_C",
            output=str(output),
        )
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["heldout_evaluation_files_read"] == 0
    assert payload["training_lanes"] == ["LANE_A", "LANE_B"]


def test_strict_loso_fit_fails_closed_if_heldout_is_present(tmp_path):
    value = tmp_path / "heldout.csv"
    _write_loso_rows(value, "LANE_C", 0.2)
    try:
        fit(
            Namespace(
                training_evaluation=[str(value)],
                expected_training_lane=["LANE_C"],
                heldout_lane="LANE_C",
                output=str(tmp_path / "invalid.json"),
            )
        )
    except ValueError as error:
        assert "physically present" in str(error)
    else:
        raise AssertionError("held-out evaluation was not rejected")
