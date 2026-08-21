from __future__ import annotations

import unittest

import numpy as np

from metric_expansion_reference import (
    SKLEARN_AVAILABLE,
    embedding_cluster_metrics,
    paired_cross_modal_metrics,
    supervised_clustering_metrics,
)


class MetricExpansionReferenceTests(unittest.TestCase):
    @unittest.skipUnless(SKLEARN_AVAILABLE, "local minimal runtime lacks scikit-learn")
    def test_perfect_label_agreement(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2])
        metrics = supervised_clustering_metrics(labels, labels.copy())
        for name in ("ari", "nmi", "ami", "fmi", "homogeneity", "completeness", "v_measure"):
            self.assertAlmostEqual(metrics[name], 1.0, places=14)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "local minimal runtime lacks scikit-learn")
    def test_internal_metrics_are_finite(self) -> None:
        embedding = np.array(
            [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]], dtype=float
        )
        metrics = embedding_cluster_metrics(embedding, np.array([0, 0, 1, 1]))
        self.assertGreater(metrics["silhouette"], 0.9)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_perfect_cross_modal_pairing(self) -> None:
        embedding = np.eye(5, dtype=float)
        metrics = paired_cross_modal_metrics(embedding, embedding.copy(), top_k=(1, 3))
        self.assertEqual(metrics["foscttm_mean"], 0.0)
        self.assertEqual(metrics["recall_at_1_mean"], 1.0)
        self.assertEqual(metrics["median_paired_rank_mod1_to_mod2"], 1.0)

    def test_cross_modal_shape_mismatch_fails(self) -> None:
        with self.assertRaises(ValueError):
            paired_cross_modal_metrics(np.ones((3, 2)), np.ones((4, 2)))


if __name__ == "__main__":
    unittest.main()
