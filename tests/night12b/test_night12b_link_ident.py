import ast
import csv
from pathlib import Path

import numpy as np

from SpaLORA.night12b_link_ident import (
    AXES,
    ZERO_COUNTS,
    _scores,
    fisher_pool,
    optimized_crossfit,
    reference_crossfit,
    safe_spearman,
    spatial_blocks,
    unit_bootstrap,
    utility_and_tail,
)


def test_spatial_blocks_deterministic_nonempty_and_exact_tie_fallback():
    coords = np.column_stack([np.arange(20), np.arange(20)]).astype(float)
    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0))
    ids = [f"s{i:02d}" for i in range(20)]
    a, audit_a = spatial_blocks(coords, ids)
    b, audit_b = spatial_blocks(coords, ids)
    assert np.array_equal(a, b)
    assert sorted(np.bincount(a).tolist()) == [4] * 5
    square = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, float)
    _, audit = spatial_blocks(square, ids)
    assert audit["exact_tie_fallback_to_x"] is True


def test_reference_ols_shapes_finite_and_decoy_ties_count_against_true():
    rng = np.random.default_rng(7)
    n, m = 100, 4
    coords = rng.random((n, 2))
    ids = [f"x{i}" for i in range(n)]
    blocks, _ = spatial_blocks(coords, ids)
    rna = rng.normal(size=(n, m))
    target = rna + rng.normal(scale=.1, size=(n, m))
    base, linked, zr, zy = reference_crossfit(rna, target, coords, blocks)
    assert base.shape == (5, m) and linked.shape == (5, m, m)
    assert np.isfinite(linked).all() and not zr.any() and not zy.any()
    tied = np.zeros((5, m, m))
    utility, p = utility_and_tail(np.ones((5, m)), tied)
    assert np.array_equal(p, np.ones(m))


def test_optimized_crossfit_matches_literal_candidate_ols_to_frozen_tolerance():
    rng = np.random.default_rng(19)
    n, m = 125, 5
    coords = rng.random((n, 2))
    blocks, _ = spatial_blocks(coords, [f"u{i:03d}" for i in range(n)])
    rna = rng.normal(size=(n, m))
    target = rng.normal(size=(n, m)) + .25 * rna
    ref = reference_crossfit(rna, target, coords, blocks)
    opt = optimized_crossfit(rna, target, coords, blocks)
    assert np.allclose(ref[0], opt[0], atol=1e-10, rtol=1e-10)
    assert np.allclose(ref[1], opt[1], atol=1e-10, rtol=1e-10)
    assert np.array_equal(ref[2], opt[2]) and np.array_equal(ref[3], opt[3])


def test_block_bootstrap_is_exactly_reproducible():
    rng = np.random.default_rng(3)
    base = rng.uniform(1, 2, (5, 3))
    linked = rng.uniform(.5, 2.5, (5, 3, 3))
    a = unit_bootstrap("P5S1", base, linked)
    b = unit_bootstrap("P5S1", base, linked)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_discovery_scores_do_not_accept_heldout_values():
    p1 = np.asarray([.1, .2, .3]); p2 = np.asarray([.2, .3, .4])
    u1 = np.asarray([.1, .3, .2]); u2 = np.asarray([.4, .2, .1])
    a = _scores(p1, u1, p2, u2)
    heldout_a = np.asarray([.9, .2, .1])
    heldout_b = np.asarray([.01, .5, .7])
    b = _scores(p1, u1, p2, u2)
    assert all(np.array_equal(a[k], b[k]) for k in AXES)
    assert not np.array_equal(heldout_a, heldout_b)


def test_score_formulas_pooling_and_constant_fail_closed():
    p1 = np.asarray([.1, .4]); p2 = np.asarray([.2, .3])
    u1 = np.asarray([.3, .2]); u2 = np.asarray([.2, .4])
    s = _scores(p1, u1, p2, u2)
    assert np.allclose(s["combined"], -np.log10(np.maximum.reduce([p1, p2, u1, u2])))
    assert -1 < fisher_pool([.2, .3, .4]) < 1
    assert safe_spearman(np.ones(3), np.arange(3))[0] == 0.0


def test_firewall_counts_all_zero_and_no_metric_imports_or_label_arguments():
    assert ZERO_COUNTS and all(v == 0 for v in ZERO_COUNTS.values())
    source = Path(__file__).parents[2] / "SpaLORA" / "night12b_link_ident.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {"adjusted_rand_score", "normalized_mutual_info_score", "adjusted_mutual_info_score", "fowlkes_mallows_score"}
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not (names & forbidden)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = {x.arg.lower() for x in node.args.args}
            assert not (args & {"label", "labels", "ground_truth", "dataset", "family", "tissue", "timepoint"})


def test_no_lexical_256_or_dense_n_by_n_scientific_path():
    source = (Path(__file__).parents[2] / "SpaLORA" / "night12b_link_ident.py").read_text(encoding="utf-8")
    assert "fixed_linked_features" not in source
    assert "pairwise_distances" not in source
    assert "np.zeros((len(coords), len(coords)))" not in source
