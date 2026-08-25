import unittest

import numpy as np
import scipy.sparse as sp

from SpaLORA.night17a_ceup import (
    MaskedLinearPredictor,
    canonical_undirected_graph,
    deterministic_permutation,
    directed_leave_one_edge_out_utilities,
    row_normalized_message_fixed_degree,
    scalar_edge_utility_recompute,
    signed_energy,
    signed_icm_partition,
    symmetric_signed_evidence,
)


class TestNight17ACEUP(unittest.TestCase):
    def setUp(self):
        self.ids = np.asarray([f"spot-{i}" for i in range(7)])
        row = np.asarray([0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6])
        col = np.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 0, 6, 5])
        data = np.asarray([1.0, 1.0, 0.7, 0.7, 1.3, 1.3, 0.5, 0.5, 0.4, 0.4, 0.8, 0.8])
        self.graph = sp.csr_matrix((data, (row, col)), shape=(7, 7))
        self.edge_i, self.edge_j, self.edge_w, self.graph = canonical_undirected_graph(self.graph)

    def test_fixed_degree_vectorized_matches_scalar(self):
        import torch

        rng = np.random.RandomState(3)
        target = rng.normal(size=(7, 4)).astype(np.float32)
        source = rng.normal(size=(7, 3)).astype(np.float32)
        model = MaskedLinearPredictor(4, 3)
        held = np.asarray([1, 3], dtype=np.int64)
        vector = directed_leave_one_edge_out_utilities(
            model, [held], target, source, self.graph, self.edge_i, self.edge_j, self.edge_w
        )
        for edge in range(min(5, self.edge_i.size)):
            scalar_ij = scalar_edge_utility_recompute(
                model, held, target, source, self.graph,
                int(self.edge_i[edge]), int(self.edge_j[edge]), float(self.edge_w[edge]),
            )
            scalar_ji = scalar_edge_utility_recompute(
                model, held, target, source, self.graph,
                int(self.edge_j[edge]), int(self.edge_i[edge]), float(self.edge_w[edge]),
            )
            self.assertAlmostEqual(vector[edge, 0, 0], scalar_ij, places=6)
            self.assertAlmostEqual(vector[edge, 1, 0], scalar_ji, places=6)

    def test_edge_deletion_keeps_full_degree_denominator(self):
        source = np.arange(14, dtype=np.float32).reshape(7, 2)
        message, degree = row_normalized_message_fixed_degree(self.graph, source)
        receiver, sender = 1, 2
        weight = float(self.graph[receiver, sender])
        expected = message[receiver] - (weight / degree[receiver]) * source[sender]
        modified = self.graph.copy().tolil()
        modified[receiver, sender] = 0.0
        modified = modified.tocsr()
        missing_placeholder = modified.dot(source)[receiver] / degree[receiver]
        np.testing.assert_allclose(expected, missing_placeholder, rtol=0, atol=5e-7)
        renormalized = modified.dot(source)[receiver] / float(modified.sum(axis=1)[receiver, 0])
        self.assertFalse(np.allclose(expected, renormalized))

    def test_four_direction_discordance_is_reject(self):
        q = np.asarray([[0.8, 0.7, 0.9, 0.6], [-0.8, -0.7, -0.9, -0.6], [0.8, -0.7, 0.9, 0.6]])
        evidence = symmetric_signed_evidence(q)
        self.assertGreater(evidence["q_positive"][0], 0)
        self.assertEqual(evidence["q_negative"][0], 0)
        self.assertGreater(evidence["q_negative"][1], 0)
        self.assertEqual(evidence["q_positive"][1], 0)
        self.assertEqual(evidence["q_positive"][2], 0)
        self.assertEqual(evidence["q_negative"][2], 0)
        self.assertEqual(evidence["relation"][2], 0)

    def test_permutation_is_edge_order_invariant(self):
        values = np.linspace(0.0, 1.0, self.edge_i.size)
        reference = deterministic_permutation(self.ids, self.edge_i, self.edge_j, values)
        reorder = np.arange(self.edge_i.size)[::-1]
        permuted = deterministic_permutation(
            self.ids, self.edge_i[reorder], self.edge_j[reorder], values[reorder]
        )
        restored = np.empty_like(permuted)
        restored[reorder] = permuted
        np.testing.assert_array_equal(reference, restored)

    def test_signed_icm_monotonic_exact_k_and_all_rejected_self_return(self):
        start = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        unary = np.full((7, 2), 2.0)
        unary[np.arange(7), start] = 0.0
        zeros = np.zeros(self.edge_i.size)
        same, diag = signed_icm_partition(start, unary, self.edge_i, self.edge_j, zeros, zeros)
        np.testing.assert_array_equal(same, start)
        self.assertEqual(diag["changed_spots"], 0)
        attraction = np.ones(self.edge_i.size) * 0.3
        repulsion = np.ones(self.edge_i.size) * 0.2
        labels, diag = signed_icm_partition(start, unary, self.edge_i, self.edge_j, attraction, repulsion)
        self.assertEqual(np.unique(labels).size, 2)
        self.assertTrue(all(b <= a + 1e-7 for a, b in zip(diag["energy_trace"], diag["energy_trace"][1:])))
        self.assertAlmostEqual(
            diag["energy_trace"][-1],
            signed_energy(labels, unary, self.edge_i, self.edge_j, attraction, repulsion),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
