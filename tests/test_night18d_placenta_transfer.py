from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig
from SpaLORA.night18d_placenta_transfer import arm_config, medoid_index, structure_feasibility
from scripts.night18d.evaluate_placenta import categorical_spatial


def config():
    local = ContinuousEnergyConfig(
        beta=1.0, sweeps=2, view_balance=.5, retained_bias=.5, low_weight=.5,
        twohop_weight=.5, high_weight=.5, conflict_center=.5,
        conflict_temperature=.2, conflict_penalty=.2, conflict_union_weight=.2,
        mass_center=.5, mass_temperature=.2, edge_floor=.01,
        trust_center=.5, trust_temperature=.2, trust_scale=.2,
        move_threshold=.1, move_fraction=.2, unary_temperature=.5,
        neighbor_capacity=.5,
    )
    return ExpansionEnergyConfig(local, .2, .6, .2, 2.0, 1.0, .1, 1)


def test_arm_semantics_are_exact():
    base = config()
    assert arm_config(base, "FULL_FROZEN_ENERGY") == base
    assert arm_config(base, "REGISTERED_SCALE_ONLY").scale_registered == 1.0
    assert arm_config(base, "REGISTERED_SCALE_ONLY").scale_fine == 0.0
    assert arm_config(base, "NO_SELF_RETURN_STAY").self_return_strength == 0.0
    assert arm_config(base, "PAIRWISE_ZERO_KEEP_STAY").pairwise_beta == 0.0
    pure = arm_config(base, "PURE_DYNAMIC_UNARY")
    assert pure.pairwise_beta == pure.self_return_strength == 0.0


def test_structure_feasibility_uses_real_internal_edges():
    rows = np.array([0, 1, 2, 3, 4, 5]); cols = np.array([1, 0, 3, 2, 5, 4])
    graph = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))
    feasible = structure_feasibility(np.array([0, 0, 1, 1, 2, 2]), graph, 3)
    assert feasible["feasible"] and feasible["min_internal_edges"] == 1
    singleton = structure_feasibility(np.array([0, 1, 1, 2, 2, 2]), graph, 3)
    assert not singleton["feasible"] and singleton["min_cluster_size"] == 1


def test_medoid_is_deterministic_and_candidate_local():
    partitions = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)
    assert medoid_index(partitions) == 0
    assert medoid_index(partitions, [1, 2]) == 1


def test_exact_k_invalid_partition_fails_closed():
    graph = sp.eye(4, format="csr")
    result = structure_feasibility(np.zeros(4, dtype=np.int32), graph, 2)
    assert not result["feasible"] and not result["exact_k"]


def test_categorical_spatial_metrics_are_label_permutation_invariant():
    rows = np.array([0, 1, 1, 2, 2, 3, 3, 0]); cols = np.array([1, 0, 2, 1, 3, 2, 0, 3])
    graph = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(4, 4))
    left = categorical_spatial(np.array([0, 0, 1, 1]), graph)
    right = categorical_spatial(np.array([9, 9, 3, 3]), graph)
    assert np.allclose(left, right, rtol=0, atol=1e-12)
