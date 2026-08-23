from __future__ import annotations

import ast
import itertools
import json
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    add_current_label_stay_cost,
    alpha_expansion_move,
    continuous_multiscale_expansion,
    continuous_multiscale_single_site,
    normalized_scale_weights,
    potts_energy,
    prepare_expansion_evidence,
)
from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig


def local_config() -> ContinuousEnergyConfig:
    return ContinuousEnergyConfig(
        beta=1.0,
        edge_floor=0.01,
        conflict_center=0.2,
        conflict_temperature=0.1,
        conflict_union_weight=0.5,
        conflict_penalty=0.1,
        mass_center=0.3,
        mass_temperature=0.1,
        neighbor_capacity=0.8,
        low_weight=0.5,
        twohop_weight=0.2,
        high_weight=0.2,
        unary_temperature=0.5,
        retained_bias=0.0,
        view_balance=0.0,
        trust_scale=0.0,
        trust_center=0.0,
        trust_temperature=0.2,
        move_threshold=0.0,
        move_fraction=1.0,
        sweeps=1,
    )


class ExpansionP0Test(unittest.TestCase):
    def test_scale_weights_are_continuous_nonnegative_simplex(self) -> None:
        config = ExpansionEnergyConfig(local_config(), 2.0, 3.0, 5.0, 1.0, 0.4, 0.0, 1)
        value = normalized_scale_weights(config)
        np.testing.assert_allclose(value, [0.2, 0.3, 0.5])
        self.assertAlmostEqual(float(value.sum()), 1.0)
        with self.assertRaises(ValueError):
            normalized_scale_weights(
                ExpansionEnergyConfig(local_config(), -1.0, 1.0, 1.0, 1.0, 0.4, 0.0, 1)
            )

    def test_rejected_mass_is_explicit_current_label_stay_unary(self) -> None:
        unary = np.zeros((4, 3), dtype=np.float32)
        partition = np.asarray([0, 2, 1, 0], dtype=np.int32)
        rejected = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
        value = add_current_label_stay_cost(unary, partition, rejected, 2.0)
        for index, label in enumerate(partition):
            self.assertEqual(float(value[index, label]), 0.0)
            other = [candidate for candidate in range(3) if candidate != label]
            np.testing.assert_allclose(value[index, other], 2.0 * rejected[index])
        refreshed = add_current_label_stay_cost(unary, (partition + 1) % 3, rejected, 2.0)
        self.assertFalse(np.array_equal(value, refreshed))

    def test_random_sparse_alpha_moves_equal_exhaustive_subspace_optimum(self) -> None:
        # Arbitrary sparse graphs (including isolated nodes and holes), n <= 8.
        rng = np.random.default_rng(20260824)
        checked = 0
        for n in range(4, 9):
            k = min(3, n)
            for _ in range(12):
                partition = np.arange(n, dtype=np.int32) % k
                rng.shuffle(partition)
                unary = rng.uniform(0.0, 0.8, size=(n, k))
                # Keep at least one strong anchor per label so the unconstrained
                # binary optimum retains cardinality and the core's guard is inert.
                for label in range(k):
                    anchor = int(np.flatnonzero(partition == label)[0])
                    unary[anchor, :] += 3.0
                    unary[anchor, label] = 0.0
                pairs = []
                for left in range(n):
                    for right in range(left + 1, n):
                        if rng.random() < 0.22:
                            pairs.append((left, right))
                if not pairs:
                    pairs = [(0, n - 1)]
                rows = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
                cols = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
                weights = rng.uniform(0.0, 0.35, size=len(pairs))
                for alpha in range(k):
                    proposal, _, _, _ = alpha_expansion_move(
                        partition,
                        alpha,
                        unary,
                        rows,
                        cols,
                        weights,
                        capacity_scale=100000000.0,
                        energy_tolerance=1e-12,
                    )
                    actual = potts_energy(proposal, unary, rows, cols, weights)
                    optimum = np.inf
                    optimum_cardinality = 0
                    for bits in itertools.product((0, 1), repeat=n):
                        candidate = partition.copy()
                        candidate[np.asarray(bits, dtype=bool)] = alpha
                        energy = potts_energy(candidate, unary, rows, cols, weights)
                        if energy < optimum:
                            optimum = energy
                            optimum_cardinality = len(np.unique(candidate))
                    self.assertEqual(optimum_cardinality, k)
                    self.assertLessEqual(actual, optimum + 2e-7)
                    self.assertAlmostEqual(actual, optimum, delta=2e-7)
                    checked += 1
        self.assertGreaterEqual(checked, 100)

    def test_move_never_increases_float_energy(self) -> None:
        partition = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32)
        unary = np.asarray(
            [[0.1, 1.0, 1.1], [1.2, 0.1, 1.0], [1.1, 1.0, 0.1]] * 2,
            dtype=np.float64,
        )
        rows = np.asarray([0, 0, 2, 4], dtype=np.int32)
        cols = np.asarray([1, 5, 3, 5], dtype=np.int32)
        weights = np.asarray([0.2, 0.7, 0.1, 0.6], dtype=np.float64)
        for alpha in range(3):
            proposal, _, before, after = alpha_expansion_move(
                partition, alpha, unary, rows, cols, weights, 1000000.0, 1e-10
            )
            self.assertLessEqual(potts_energy(proposal, unary, rows, cols, weights), before + 1e-12)
            self.assertTrue(np.isfinite(after))

    def test_each_dynamic_cycle_has_its_own_monotone_energy_ledger(self) -> None:
        rng = np.random.default_rng(11)
        n = 15
        rows = np.arange(n, dtype=np.int32)
        cols = (rows + 1) % n
        graph = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(n, n)).maximum(
            sp.csr_matrix((np.ones(n), (cols, rows)), shape=(n, n))
        )
        evidence = prepare_expansion_evidence(
            (graph, graph, graph),
            rng.normal(size=(n, 7)).astype(np.float32),
            rng.normal(size=(n, 5)).astype(np.float32),
            rng.normal(size=(n, 6)).astype(np.float32),
            retained_dim=5,
            view_dim=4,
            edge_dim=3,
        )
        initial = np.arange(n, dtype=np.int32) % 3
        config = ExpansionEnergyConfig(local_config(), 1.0, 1.0, 1.0, 0.8, 0.4, 0.05, 2)
        _, diagnostics = continuous_multiscale_expansion(initial, 3, evidence, config)
        ledger = json.loads(diagnostics["cycle_energy_ledger_json"])
        self.assertGreaterEqual(len(ledger), 1)
        for cycle in ledger:
            self.assertLessEqual(cycle["end_energy"], cycle["start_energy"] + 1e-9)
        self.assertEqual(diagnostics["cross_dynamic_cycle_global_monotonicity_claimed"], 0.0)

    def test_matched_single_site_solver_is_deterministic_and_monotone(self) -> None:
        rng = np.random.default_rng(23)
        n = 18
        rows = np.arange(n - 1, dtype=np.int32)
        cols = rows + 1
        graph = sp.csr_matrix((np.ones(n - 1), (rows, cols)), shape=(n, n)).maximum(
            sp.csr_matrix((np.ones(n - 1), (cols, rows)), shape=(n, n))
        )
        evidence = prepare_expansion_evidence(
            (graph, graph, graph),
            rng.normal(size=(n, 8)).astype(np.float32),
            rng.normal(size=(n, 5)).astype(np.float32),
            rng.normal(size=(n, 6)).astype(np.float32),
            retained_dim=6,
            view_dim=4,
            edge_dim=3,
        )
        initial = np.arange(n, dtype=np.int32) % 3
        config = ExpansionEnergyConfig(local_config(), 0.4, 0.4, 0.2, 0.5, 0.3, 0.0, 2)
        first, diagnostics = continuous_multiscale_single_site(initial, 3, evidence, config)
        second, _ = continuous_multiscale_single_site(initial, 3, evidence, config)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 3)
        for cycle in json.loads(diagnostics["cycle_energy_ledger_json"]):
            self.assertLessEqual(cycle["end_energy"], cycle["start_energy"] + 1e-9)

    def test_core_has_no_dataset_or_metric_identity_reads(self) -> None:
        source_path = Path(__file__).parents[2] / "SpaLORA" / "night15f_multiscale_expansion.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = {"dataset", "study", "label", "labels", "ari", "nmi", "metric", "annotation"}
        names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & forbidden)


if __name__ == "__main__":
    unittest.main()
