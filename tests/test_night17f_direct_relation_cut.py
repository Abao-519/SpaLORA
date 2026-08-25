import numpy as np
import scipy.sparse as sp

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig, prepare_expansion_evidence
from SpaLORA.night17e_lrcc import LRCCConfig, _stable_stratified_permutation, relation_conditioned_pairwise
from SpaLORA.night17f_direct_relation_cut import (
    candidate_mask,
    direct_relation_posterior,
    posterior_factor,
    prepare_direct_relation_evidence,
)


def records():
    return [
        {"candidate_id": "KMEANS_RETAINED_S0__X", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "True", "molecular_joint": 0.1, "topology_joint": 0.2, "persistence": 0.3},
        {"candidate_id": "PATH_UNIFORM__KMEANS_RETAINED_S1__X", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "True", "molecular_joint": 0.3, "topology_joint": 0.1, "persistence": 0.2},
        {"candidate_id": "AUTHORITY", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "True", "molecular_joint": 0.9, "topology_joint": 0.9, "persistence": 0.9},
        {"candidate_id": "KMEANS_RETAINED_S2__X", "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "True", "molecular_joint": 0.2, "topology_joint": 0.4, "persistence": 0.1},
    ]


def test_unbiased_mask_excludes_authority():
    assert candidate_mask(records(), "UNBIASED_BANK").tolist() == [True, True, False, True]


def test_weighted_posterior_is_finite_and_ordered():
    partitions = np.asarray([[0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.int32)
    result = direct_relation_posterior(
        partitions, records(), np.asarray([0, 1]), np.asarray([1, 2]),
        bank_mode="UNBIASED_BANK", weighted=True,
    )
    assert result.selected_candidate_count == 3
    assert np.all(np.isfinite(result.probability_same))
    assert np.all((result.probability_same >= 0) & (result.probability_same <= 1))
    unweighted = direct_relation_posterior(
        partitions, records(), np.asarray([0, 1]), np.asarray([1, 2]),
        bank_mode="UNBIASED_BANK", weighted=False,
    )
    assert np.unique(result.probability_same).size > 1
    assert np.unique(unweighted.probability_same).size > 1


def test_posterior_factor_nonnegative(monkeypatch):
    class C:
        relation_mix = 0.5
        relation_floor = 0.05
        relation_power = 1.0
        uncertainty_scale = 1.0
    partitions = np.asarray([[0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.int32)
    posterior = direct_relation_posterior(
        partitions, records(), np.asarray([0, 1]), np.asarray([1, 2]),
        bank_mode="UNBIASED_BANK", weighted=False,
    )
    factor = posterior_factor(posterior, C())
    assert np.all(np.isfinite(factor))
    assert np.all(factor >= 0.5)
    assert np.all(factor <= 1.0)


def _config():
    local = ContinuousEnergyConfig(
        beta=1.0, edge_floor=0.05, conflict_center=0.4, conflict_temperature=0.2,
        conflict_union_weight=0.5, conflict_penalty=0.2, mass_center=0.0,
        mass_temperature=0.2, neighbor_capacity=0.8, low_weight=0.2,
        twohop_weight=0.1, high_weight=0.2, unary_temperature=0.5,
        retained_bias=0.5, view_balance=0.0, trust_scale=0.1, trust_center=0.2,
        trust_temperature=0.5, move_threshold=0.0, move_fraction=0.1, sweeps=2,
    )
    base = ExpansionEnergyConfig(
        local=local, scale_fine=0.5, scale_registered=0.3, scale_broad=0.2,
        pairwise_beta=2.0, self_return_strength=1.0, size_prior=0.0,
        expansion_cycles=1, capacity_scale=100000.0,
    )
    return LRCCConfig(base, 0.5, 0.05, 1.0, 1.0)


def test_edges_permutation_and_mass_matching_runtime_gate():
    rng = np.random.RandomState(9)
    n = 18
    graphs = []
    for hop in (1, 2, 3):
        row, col = [], []
        for index in range(n):
            for step in range(1, hop + 1):
                row.extend((index, index)); col.extend(((index + step) % n, (index - step) % n))
        graphs.append(sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)))
    prepared = prepare_expansion_evidence(
        graphs, rng.normal(size=(n, 6)), rng.normal(size=(n, 5)), rng.normal(size=(n, 4)),
        retained_dim=4, view_dim=3, edge_dim=2,
    )
    candidate_records = []
    partitions = []
    for index in range(5):
        candidate_records.append({
            "candidate_id": f"KMEANS_RETAINED_S{index}__X",
            "feasible_SMALLEST_SCALE_INTERNAL_EDGE": "True",
            "molecular_joint": 0.1 + index, "topology_joint": 0.2 + index / 2,
            "persistence": 0.3 + index / 3,
        })
        partitions.append(np.roll(np.repeat(np.arange(3), 6), index))
    relation, _ = prepare_direct_relation_evidence(
        prepared, np.asarray(partitions, dtype=np.int32), candidate_records, _config()
    )
    ids = np.asarray([f"id-{index}" for index in range(n)])
    values = {}
    for arm in ("LEARNED_RELATION", "ZERO_RELATION", "PERMUTED_RELATION", "UNIFORM_MASS_MATCHED"):
        rows, cols, weights, _, diagnostics = relation_conditioned_pairwise(
            prepared, relation, _config(), arm, ids
        )
        assert np.all(rows >= 0) and np.all(cols >= 0) and np.all(rows != cols)
        assert np.all(np.isfinite(weights)) and np.all(weights >= 0)
        values[arm] = weights
        if arm != "LEARNED_RELATION":
            assert max(x["mass_match_absolute_error"] for x in diagnostics["scale_diagnostics"]) < 1e-9
    first = _stable_stratified_permutation(ids, relation[0].rows, relation[0].cols, np.ones(len(relation[0].rows)), 0)
    second = _stable_stratified_permutation(ids, relation[0].rows, relation[0].cols, np.ones(len(relation[0].rows)), 0)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.arange(len(first)))
