from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15d_reliability_energy import (
    centroid_unary,
    fused_dynamic_unary,
    multiscale_feature_bank,
    reliability_energy_icm,
    reliability_transition,
)


class ReliabilityEnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.binary = sp.csr_matrix(
            np.asarray(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        self.weighted = sp.csr_matrix(
            np.asarray(
                [
                    [0.0, 0.2, 0.4],
                    [0.2, 0.0, 0.6],
                    [0.4, 0.6, 0.0],
                ],
                dtype=np.float32,
            )
        )

    def test_mass_retains_absolute_conductance(self) -> None:
        row, _ = reliability_transition(self.binary, self.weighted, "row")
        mass, _ = reliability_transition(self.binary, self.weighted, "mass")
        np.testing.assert_allclose(np.asarray(row.sum(1)).ravel(), 1.0, atol=1e-6)
        self.assertTrue(np.all(np.asarray(mass.sum(1)).ravel() < 1.0))

    def test_rejected_mass_returns_to_self(self) -> None:
        mass, _ = reliability_transition(self.binary, self.weighted, "mass")
        returned, diag = reliability_transition(self.binary, self.weighted, "self")
        np.testing.assert_allclose(np.asarray(returned.sum(1)).ravel(), 1.0, atol=1e-6)
        expected = 1.0 - np.asarray(mass.sum(1)).ravel()
        np.testing.assert_allclose(returned.diagonal(), expected, atol=1e-6)
        self.assertGreater(diag["mean_rejected_mass"], 0.0)

    def test_core_stays_sparse(self) -> None:
        returned, _ = reliability_transition(self.binary, self.weighted, "self")
        self.assertTrue(sp.isspmatrix_csr(returned))
        self.assertLessEqual(returned.nnz, self.weighted.nnz + returned.shape[0])

    def test_modality_margin_unary_is_finite(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        retained = np.asarray([[0, 0], [0.1, 0], [2, 2], [2.1, 2]], dtype=np.float32)
        first = retained.copy()
        second = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        unary, diag = fused_dynamic_unary(
            labels, 2, "retained_dual_margin", retained, first, second, 0.5, 0.35
        )
        self.assertEqual(unary.shape, (4, 2))
        self.assertTrue(np.isfinite(unary).all())
        self.assertAlmostEqual(
            diag["mean_view1_weight"]
            + diag["mean_view2_weight"]
            + diag["mean_retained_weight"],
            1.0,
            places=6,
        )

    def test_multiscale_bank_has_registered_shapes(self) -> None:
        rng = np.random.default_rng(7)
        retained = rng.normal(size=(12, 8)).astype(np.float32)
        view1 = rng.normal(size=(12, 5)).astype(np.float32)
        view2 = rng.normal(size=(12, 4)).astype(np.float32)
        graph = sp.csr_matrix(np.eye(12, k=1) + np.eye(12, k=-1), dtype=np.float32)
        bank = multiscale_feature_bank(retained, view1, view2, graph)
        self.assertEqual(
            set(bank),
            {
                "retained",
                "retained_low_high",
                "views_low_high",
                "views_multiscale",
                "retained_plus_views",
            },
        )
        self.assertTrue(all(value.shape[0] == 12 for value in bank.values()))
        self.assertTrue(all(np.isfinite(value).all() for value in bank.values()))

    def test_zero_steps_is_exact_noop(self) -> None:
        initial = np.asarray([0, 0, 1], dtype=np.int32)
        returned, completed, collapse, _ = reliability_energy_icm(
            initial,
            sp.eye(3, format="csr", dtype=np.float32),
            2,
            1.0,
            0,
            "retained",
            np.asarray([[0], [0.1], [1]], dtype=np.float32),
            np.asarray([[0], [0.1], [1]], dtype=np.float32),
            np.asarray([[0], [0.1], [1]], dtype=np.float32),
        )
        np.testing.assert_array_equal(returned, initial)
        self.assertEqual(completed, 0)
        self.assertFalse(collapse)

    def test_centroid_unary_fails_closed_on_empty_cluster(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty cluster"):
            centroid_unary(
                np.asarray([[0.0], [1.0]], dtype=np.float32),
                np.asarray([0, 0], dtype=np.int32),
                2,
            )

    def test_core_signature_has_no_dataset_or_label_argument(self) -> None:
        source = Path(__file__).resolve().parents[2] / "SpaLORA" / "night15d_reliability_energy.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        forbidden = {"dataset", "dataset_name", "labels", "ground_truth", "annotation"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = {arg.arg for arg in node.args.args}
                self.assertFalse(arguments & forbidden, (node.name, arguments & forbidden))

    def test_frozen_registry_contains_nine_dual_positive_lanes(self) -> None:
        registry = Path(__file__).resolve().parents[2] / "configs" / "night15d" / "night15d_frozen_config_registry.json"
        value = json.loads(registry.read_text(encoding="utf-8"))
        self.assertEqual(len(value["lanes"]), 9)
        for lane in value["lanes"].values():
            self.assertGreater(lane["delta_vs_night15c_ari"], 1e-10)
            self.assertGreater(lane["delta_vs_night15c_nmi"], 1e-10)

    def test_two_replays_are_byte_exact(self) -> None:
        work = Path(__file__).resolve().parents[2] / "working"
        first = json.loads(
            (work / "final_replay1" / "replay.json").read_text(encoding="utf-8")
        )
        second = json.loads(
            (work / "final_replay2" / "replay.json").read_text(encoding="utf-8")
        )
        self.assertEqual(first["lane_count"], 9)
        self.assertEqual(second["lane_count"], 9)
        mapped = {row["lane"]: row for row in second["rows"]}
        for row in first["rows"]:
            other = mapped[row["lane"]]
            self.assertEqual(row["partition_sha256"], other["partition_sha256"])
            self.assertEqual(row["absolute_ari"], other["absolute_ari"])
            self.assertEqual(row["absolute_nmi"], other["absolute_nmi"])


if __name__ == "__main__":
    unittest.main()
