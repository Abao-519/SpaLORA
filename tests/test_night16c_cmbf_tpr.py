from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SpaLORA.night16c_cmbf_tpr import (  # noqa: E402
    CMBFTPRConfig,
    array_sha256,
    canonical_graph,
    cmbf_tpr,
    partition_sha256,
    prepare_boundary_evidence,
)


def fixture():
    rng = np.random.default_rng(7)
    n = 24
    view1 = rng.normal(size=(n, 6)).astype(np.float32)
    view2 = (view1[:, :4] + 0.25 * rng.normal(size=(n, 4))).astype(np.float32)
    row = []
    col = []
    for i in range(n):
        for j in {(i - 1) % n, (i + 1) % n, (i + 4) % n}:
            row.append(i); col.append(j)
    graph = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    initial = np.repeat(np.arange(4), 6).astype(np.int32)
    bank = np.stack([initial, np.roll(initial, 1), np.roll(initial, -1)])
    return view1, view2, graph, initial, bank


class Night16CTests(unittest.TestCase):
    def test_01_tri_state_is_probability_partition(self):
        v1, v2, graph, initial, bank = fixture()
        evidence = prepare_boundary_evidence(v1, v2, graph, initial, bank, CMBFTPRConfig())
        self.assertTrue(np.allclose(evidence.support + evidence.boundary + evidence.conflict, 1.0, atol=1e-6))
        self.assertTrue(np.all((evidence.conductance >= 0) & (evidence.conductance <= 1)))

    def test_02_sparse_contract(self):
        v1, v2, graph, initial, bank = fixture()
        evidence = prepare_boundary_evidence(v1, v2, graph, initial, bank, CMBFTPRConfig())
        self.assertTrue(sp.isspmatrix_csr(evidence.graph))
        self.assertLess(evidence.graph.nnz, graph.shape[0] ** 2)

    def test_03_noop_is_byte_exact(self):
        v1, v2, graph, initial, bank = fixture()
        config = CMBFTPRConfig(sweeps=0, pairwise_strength=0, self_return_strength=0, anchor_strength=0)
        partition, _, detail = cmbf_tpr(initial, v1, v2, graph, bank, config)
        self.assertTrue(np.array_equal(partition, initial))
        self.assertEqual(detail["changed_observations"], 0)

    def test_04_deterministic_replay(self):
        v1, v2, graph, initial, bank = fixture()
        config = CMBFTPRConfig(directional_mix=.25, pairwise_strength=.08, sweeps=2)
        first = cmbf_tpr(initial, v1, v2, graph, bank, config)[0]
        second = cmbf_tpr(initial, v1, v2, graph, bank, config)[0]
        self.assertEqual(partition_sha256(first), partition_sha256(second))

    def test_05_missing_dataset_and_label_arguments(self):
        signature = inspect.signature(cmbf_tpr)
        for forbidden in ("dataset", "study", "label", "annotation", "family"):
            self.assertNotIn(forbidden, signature.parameters)

    def test_06_no_dataset_name_core_literals(self):
        source = inspect.getsource(sys.modules["SpaLORA.night16c_cmbf_tpr"])
        tree = ast.parse(source)
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        for forbidden in ("A1", "D1", "P22", "MISAR", "tonsil"):
            self.assertNotIn(forbidden, literals)

    def test_07_invalid_shapes_fail_closed(self):
        v1, v2, graph, initial, bank = fixture()
        with self.assertRaises(ValueError):
            prepare_boundary_evidence(v1[:-1], v2, graph, initial, bank, CMBFTPRConfig())

    def test_08_invalid_config_fails_closed(self):
        with self.assertRaises(ValueError):
            CMBFTPRConfig(conflict_pass=1.5).validate()

    def test_09_partition_hash_includes_shape_and_dtype(self):
        value = np.asarray([0, 1, 1, 0], dtype=np.int32)
        self.assertEqual(partition_sha256(value), partition_sha256(value.copy()))
        self.assertNotEqual(array_sha256(value), array_sha256(value.astype(np.int64)))

    def test_10_graph_canonicalization_removes_diagonal(self):
        graph = sp.eye(5, format="csr") + sp.diags(np.ones(4), 1)
        canonical = canonical_graph(graph)
        self.assertEqual(int(canonical.diagonal().sum()), 0)
        self.assertTrue((canonical != canonical.T).nnz == 0)

    def test_11_stable_start_bank_anchors_all_nodes(self):
        v1, v2, graph, initial, _ = fixture()
        bank = np.stack([initial, initial, initial])
        config = CMBFTPRConfig(
            pairwise_strength=1.0,
            self_return_strength=0.0,
            anchor_strength=0.0,
            trust_threshold=0.55,
            sweeps=2,
        )
        partition, _, detail = cmbf_tpr(initial, v1, v2, graph, bank, config)
        self.assertTrue(np.array_equal(partition, initial))
        self.assertEqual(detail["changed_observations"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
