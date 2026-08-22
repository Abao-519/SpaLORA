import ast
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from SpaLORA.night11b_discordance import (
    bootstrap_stability, crossfit_shared, diffused_perturbation, evidence_axes,
    make_permutations, safe_h5ad, scanpy_scale, seurat_clr, spatial_blocks,
)


def test_safe_h5_boundary_does_not_return_obs_annotation(tmp_path):
    path = tmp_path / "tiny.h5ad"
    with h5py.File(path, "w") as h:
        x = h.create_group("X"); x.attrs["encoding-type"] = "csr_matrix"; x.attrs["shape"] = (3, 2)
        m = sp.csr_matrix([[1, 0], [0, 2], [3, 1]])
        x.create_dataset("data", data=m.data); x.create_dataset("indices", data=m.indices); x.create_dataset("indptr", data=m.indptr)
        obs = h.create_group("obs"); obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.asarray([b"s0", b"s1", b"s2"])); obs.create_dataset("manual_annotation", data=np.asarray([b"SECRET"] * 3))
        var = h.create_group("var"); var.attrs["_index"] = "_index"; var.create_dataset("_index", data=np.asarray([b"g0", b"g1"]))
        obsm = h.create_group("obsm"); obsm.create_dataset("spatial", data=np.asarray([[0, 0], [1, 0], [2, 0]], dtype=float))
    value = safe_h5ad(path)
    assert value["obs"] == ["s0", "s1", "s2"]
    assert value["obs_keys_not_opened"] == ["manual_annotation"]
    assert "manual_annotation" not in repr(value["matrix"])


def test_registered_transforms_match_frozen_algebra():
    x = sp.csr_matrix([[1.0, 0.0], [3.0, 1.0]])
    clr = seurat_clr(x)
    expected0 = np.log1p(np.asarray([1.0, 0.0]) / np.exp(np.log(2.0) / 2.0))
    assert np.allclose(clr[0], expected0)
    scaled = scanpy_scale(clr)
    assert np.allclose(scaled.mean(axis=0), 0.0)
    assert np.allclose(scaled.std(axis=0, ddof=1), [1.0, 1.0])


def test_spatial_blocks_and_crossfit_have_disjoint_holdouts():
    coords = np.stack([np.arange(50), np.zeros(50)], axis=1)
    blocks = spatial_blocks(coords, 5)
    assert np.bincount(blocks).tolist() == [10] * 5
    rng = np.random.RandomState(2); x = rng.normal(size=(50, 4)); y = x[:, :2] + rng.normal(scale=0.1, size=(50, 2))
    result = crossfit_shared(x, y, blocks, [0.1, 1.0])
    assert result["prediction"].shape == y.shape
    assert np.isfinite(result["shared_r2"]).all()
    assert len(result["outer_alphas"]) == 5


def test_permutations_deterministic_and_have_no_fixed_points():
    first = make_permutations(101, 5, 17); second = make_permutations(101, 5, 17)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
        assert not np.any(a == np.arange(101))


def test_sparse_axes_and_four_control_boundaries():
    n = 80; rng = np.random.RandomState(3)
    row = np.arange(n); col = (row + 1) % n
    graph = sp.csr_matrix((np.ones(n * 2), (np.r_[row, col], np.r_[col, row])), shape=(n, n))
    blocks = np.arange(n) % 5
    iid = rng.normal(size=(n, 3 * 4)); null = bootstrap_stability(iid, graph, blocks, 3, 11).reshape(4, 3)
    structured = diffused_perturbation(rng.normal(size=(n, 3)), sp.diags(1 / np.asarray(graph.sum(1)).ravel()).dot(graph), 9, 3)
    patterns = np.concatenate([structured, rng.normal(size=(n, 3)), structured[::-1], rng.normal(size=(n, 3))], axis=1)
    axes = evidence_axes(patterns, graph, blocks, null, make_permutations(n, 7, 5), 3, 7)
    assert axes["combined"].shape == (12,)
    assert np.allclose(axes["combined"], np.sqrt(axes["p_boot"] * axes["p_space"]))
    assert sp.issparse(graph)


def test_core_module_has_no_dataset_or_metric_routing():
    source = Path(__file__).parents[1] / "SpaLORA/night11b_discordance.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden_calls = {"ari", "nmi", "ami", "fmi", "adjusted_rand_score", "normalized_mutual_info_score"}
    calls = {getattr(node.func, "id", "").lower() for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not (calls & forbidden_calls)
    assert "dataset ==" not in source.read_text(encoding="utf-8").lower()

