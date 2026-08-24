from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15f_multiscale_expansion import continuous_multiscale_expansion
from SpaLORA.night16a_self_calibrating_energy import (
    CalibrationConstants,
    calibrate_from_statistics,
    prepare_and_calibrate,
    run_calibrated_energy,
)
from scripts.night16a.night16a_balanced_repair_probe import repair


def ring_graph(n: int, hop: int = 1) -> sp.csr_matrix:
    rows, cols = [], []
    for index in range(n):
        for delta in range(1, hop + 1):
            rows.extend((index, index))
            cols.extend(((index - delta) % n, (index + delta) % n))
    value = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    value.setdiag(0); value.eliminate_zeros()
    return value


def test_core_api_has_no_dataset_or_reference_identity():
    source = Path(__file__).resolve().parents[1] / "SpaLORA" / "night16a_self_calibrating_energy.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {"dataset", "tissue", "timepoint", "label", "ground_truth", "ari", "nmi"}
    arguments = {
        argument.arg.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (*node.args.args, *node.args.kwonlyargs)
    }
    assert not arguments.intersection(forbidden)


def test_calibration_is_deterministic_and_finite():
    statistics = {
        "registered_conflict_median": 0.2,
        "registered_conflict_scale": 0.1,
        "registered_neighbor_overlap": 0.55,
        "registered_similarity_correlation": 0.3,
        "start_stability_median_ari": 0.7,
        "view1_edge_coherence": 0.45,
        "view2_edge_coherence": 0.4,
        "observations_per_cluster": 300.0,
        "retained_effective_rank": 20.0,
        "view1_effective_rank": 15.0,
        "view2_effective_rank": 12.0,
        "scale_scores": [0.2, 0.4, 0.3],
        "optional_reliability": 0.5,
    }
    first = calibrate_from_statistics(statistics, CalibrationConstants())
    second = calibrate_from_statistics(statistics, CalibrationConstants())
    assert asdict(first.config) == asdict(second.config)
    values = np.asarray([
        first.config.pairwise_beta,
        first.config.self_return_strength,
        first.config.size_prior,
        first.optional_config.morphology_unary_weight,
    ])
    assert np.all(np.isfinite(values)) and np.all(values >= 0)
    scales = np.asarray([first.config.scale_fine, first.config.scale_registered, first.config.scale_broad])
    assert np.isclose(scales.sum(), 1.0)


def test_missing_optional_view_is_exact_molecular_fallback():
    rng = np.random.default_rng(20260824)
    n, k = 36, 3
    retained = rng.normal(size=(n, 8)).astype(np.float32)
    view1 = rng.normal(size=(n, 6)).astype(np.float32)
    view2 = rng.normal(size=(n, 5)).astype(np.float32)
    graphs = (ring_graph(n, 1), ring_graph(n, 2), ring_graph(n, 3))
    initial = np.repeat(np.arange(k), n // k).astype(np.int32)
    evidence, calibrated = prepare_and_calibrate(
        graphs, retained, view1, view2, (initial,), k, CalibrationConstants(), None
    )
    expected, _ = continuous_multiscale_expansion(initial, k, evidence, calibrated.config)
    observed, diagnostics = run_calibrated_energy(
        initial, k, evidence, calibrated, graphs, retained, view1, view2, None, None
    )
    assert np.array_equal(observed, expected)
    assert diagnostics["optional_missing_exact_molecular_fallback"] == 1.0


def test_balanced_repair_removes_tiny_clusters_without_changing_k():
    rng = np.random.default_rng(7)
    feature = np.concatenate(
        [rng.normal(loc=float(group), scale=0.2, size=(20, 5)) for group in range(4)], axis=0
    ).astype(np.float32)
    initial = np.repeat(np.arange(4), 20).astype(np.int32)
    initial[0] = 4
    initial[1] = 5
    repaired, threshold, removed, sizes = repair(initial, feature, 4, 0.15, 0.5, "mean")
    assert len(np.unique(repaired)) == 4
    assert sizes.min() >= threshold
    assert removed >= 0


def test_source_contains_no_dense_n_by_n_materialization():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "SpaLORA").rglob("*.py")) + list((root / "scripts").rglob("*.py"))
    content = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert ".to" + "array(" not in content
    assert ".to" + "dense(" not in content
