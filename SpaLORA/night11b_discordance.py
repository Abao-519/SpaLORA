"""Night-11B label-free RNA-protein discordance P0 utilities.

This module deliberately reads only HDF5 X, ordered identifiers and spatial
coordinates. Observation annotation columns are never opened.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import scipy.sparse as sp


ALLOWED_H5_PATHS = ("/X", "/obs/_index", "/var/_index", "/obsm/spatial")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def ordered_text_sha(values: Sequence[str]) -> str:
    return sha256_bytes("\n".join(values).encode("utf-8"))


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    header = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}, sort_keys=True)
    return sha256_bytes(header.encode("utf-8") + b"\0" + arr.tobytes())


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(temp), str(path))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(str(temp), **arrays)
    os.replace(str(temp), str(path))


def _decode(values: np.ndarray) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.atleast_1d(values).tolist()]


def _index_key(group: h5py.Group) -> str:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return str(key)


def read_ids(handle: h5py.File, group_name: str) -> List[str]:
    group = handle[group_name]
    key = _index_key(group)
    if key not in group:
        raise ValueError("missing H5AD index: %s/%s" % (group_name, key))
    return _decode(group[key][()])


def read_h5_matrix(node: object) -> sp.csr_matrix:
    if isinstance(node, h5py.Dataset):
        value = np.asarray(node[()])
        if value.ndim != 2:
            raise ValueError("H5AD dense X must be rank two")
        return sp.csr_matrix(value)
    if not isinstance(node, h5py.Group):
        raise TypeError("unsupported H5AD X node")
    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    shape = tuple(int(x) for x in np.asarray(node.attrs["shape"]).tolist())
    data = np.asarray(node["data"][()])
    indices = np.asarray(node["indices"][()], dtype=np.int64)
    indptr = np.asarray(node["indptr"][()], dtype=np.int64)
    if encoding == "csc_matrix":
        return sp.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    if encoding != "csr_matrix":
        raise ValueError("unsupported H5AD sparse encoding: %s" % encoding)
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def safe_h5ad(path: Path) -> Dict[str, object]:
    """Load only the four preregistered label-free HDF5 boundaries."""
    path = path.resolve()
    with h5py.File(str(path), "r") as handle:
        obs = read_ids(handle, "obs")
        var = read_ids(handle, "var")
        matrix = read_h5_matrix(handle["X"])
        spatial = np.asarray(handle["obsm"]["spatial"][()])
        obs_keys_not_opened = sorted(k for k in handle["obs"].keys() if k != _index_key(handle["obs"]))
    if matrix.shape != (len(obs), len(var)):
        raise ValueError("H5AD X/identifier shape mismatch")
    if spatial.shape != (len(obs), 2):
        raise ValueError("registered spatial coordinates must have shape N x 2")
    return {
        "path": str(path), "matrix": matrix, "obs": obs, "var": var,
        "spatial": spatial, "obs_keys_not_opened": obs_keys_not_opened,
        "read_paths": list(ALLOWED_H5_PATHS),
        "file_size": int(path.stat().st_size), "file_mtime_ns": int(path.stat().st_mtime_ns),
    }


def seurat_clr(matrix: sp.csr_matrix) -> np.ndarray:
    dense = matrix.toarray().astype(np.float64, copy=False)
    logged = np.log1p(dense)
    denominator = np.exp(logged.sum(axis=1) / float(dense.shape[1]))
    return np.log1p(dense / denominator[:, None])


def scanpy_scale(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    mean = value.mean(axis=0, dtype=np.float64)
    std = value.std(axis=0, ddof=1, dtype=np.float64)
    std[std == 0] = 1.0
    return (value - mean) / std


def row_normalize_graph(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = graph.tocsr().astype(np.float64)
    graph = graph.maximum(graph.T)
    graph = graph - sp.diags(graph.diagonal())
    graph.eliminate_zeros()
    graph.data[:] = 1.0
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("registered sparse graph contains isolated spots")
    return sp.diags(1.0 / degree).dot(graph).tocsr()


def load_registered_input(spec: Dict[str, object]) -> Dict[str, object]:
    rna = safe_h5ad(Path(str(spec["rna"])))
    adt = safe_h5ad(Path(str(spec["adt"])))
    if rna["obs"] != adt["obs"]:
        raise ValueError("paired ordered spot identifiers differ")
    if not np.array_equal(rna["spatial"], adt["spatial"]):
        raise ValueError("paired coordinates differ")
    graph_path = Path(str(spec["graph"])).resolve()
    graph = sp.load_npz(str(graph_path)).tocsr()
    if graph.shape != (len(rna["obs"]), len(rna["obs"])):
        raise ValueError("registered graph shape differs from paired inputs")
    if graph.nnz >= graph.shape[0] * graph.shape[0]:
        raise ValueError("dense N x N graph is forbidden")
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0); graph.eliminate_zeros()
    rna_features = list(rna["var"])
    adt_features = list(adt["var"])
    rna_index = {name: i for i, name in enumerate(rna_features)}
    exact = [name for name in adt_features if name in rna_index]
    unmatched = [name for name in adt_features if name not in rna_index]
    if len(exact) != 29 or unmatched != ["HLA-DRA", "PTPRC-1"]:
        raise ValueError("exact deposited mapping contract is not 29 + two frozen exclusions")
    if len(set(exact)) != 29:
        raise ValueError("exact linked identifiers are not unique")
    rna_counts = rna["matrix"].tocsr()
    if rna_counts.data.size and not np.allclose(rna_counts.data, np.rint(rna_counts.data)):
        raise ValueError("RNA input is not raw count-like")
    library = np.asarray(rna_counts.sum(axis=1)).ravel().astype(np.float64)
    if np.any(library <= 0):
        raise ValueError("RNA input contains a zero library")
    linked_counts = rna_counts[:, [rna_index[x] for x in exact]].astype(np.float64)
    linked_rna = linked_counts.multiply((10000.0 / library)[:, None]).tocsr()
    linked_rna.data = np.log1p(linked_rna.data)
    adt_index = {name: i for i, name in enumerate(adt_features)}
    clr_all = seurat_clr(adt["matrix"].tocsr())
    scaled_all = scanpy_scale(clr_all)
    linked_adt = scaled_all[:, [adt_index[x] for x in exact]]
    w = row_normalize_graph(graph)
    spatial_linked = w.dot(linked_rna).toarray().astype(np.float64, copy=False)
    design = np.concatenate([linked_rna.toarray(), spatial_linked], axis=1)
    obs = list(rna["obs"])
    return {
        "dataset": str(spec["dataset"]), "obs": obs, "feature_ids": exact,
        "rna": np.asarray(linked_rna.toarray(), dtype=np.float64),
        "adt": np.asarray(linked_adt, dtype=np.float64),
        "design": np.asarray(design, dtype=np.float64),
        "coordinates": np.asarray(rna["spatial"], dtype=np.float64),
        "graph": graph, "w": w, "unmatched": unmatched,
        "audit": {
            "dataset": str(spec["dataset"]), "spot_count": len(obs),
            "rna_raw_shape": list(rna_counts.shape), "adt_raw_shape": list(adt["matrix"].shape),
            "linked_rna_shape": list(linked_rna.shape), "linked_adt_shape": list(linked_adt.shape),
            "design_shape": list(design.shape), "coordinates_shape": list(rna["spatial"].shape),
            "graph_shape": list(graph.shape), "graph_nnz": int(graph.nnz),
            "rna_dtype": str(rna_counts.dtype), "adt_dtype": str(adt["matrix"].dtype),
            "processed_dtype": str(linked_adt.dtype), "ordered_observation_sha256": ordered_text_sha(obs),
            "coordinate_sha256": array_sha(np.asarray(rna["spatial"])),
            "linked_feature_sha256": ordered_text_sha(exact),
            "graph_file": str(graph_path), "graph_file_sha256": file_sha(graph_path),
            "rna_file": str(rna["path"]), "adt_file": str(adt["path"]),
            "rna_file_size": rna["file_size"], "adt_file_size": adt["file_size"],
            "rna_file_mtime_ns": rna["file_mtime_ns"], "adt_file_mtime_ns": adt["file_mtime_ns"],
            "rna_obs_keys_not_opened": rna["obs_keys_not_opened"],
            "adt_obs_keys_not_opened": adt["obs_keys_not_opened"],
            "h5_read_allowlist": list(ALLOWED_H5_PATHS), "label_values_opened": False,
            "preprocessing": {
                "rna": "Night-1 log1p(10000 * raw_count / full_library_size)",
                "protein": "Night-1 Seurat CLR per spot over all 31 deposited targets then Scanpy scale ddof=1",
                "spatial_context": "registered G04 symmetric sparse row-normalized one-hop linked RNA",
            },
        },
    }


def spatial_blocks(coordinates: np.ndarray, n_folds: int) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    projection = centered.dot(axis)
    order = np.lexsort((np.arange(len(coords)), coords[:, 1], coords[:, 0], projection))
    blocks = np.empty(len(coords), dtype=np.int64)
    for fold, chunk in enumerate(np.array_split(order, n_folds)):
        blocks[chunk] = fold
    if set(blocks.tolist()) != set(range(n_folds)):
        raise ValueError("spatial block construction failed")
    return blocks


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    ym = y.mean(axis=0)
    gram = z.T.dot(z) + float(alpha) * np.eye(z.shape[1], dtype=np.float64)
    coef = np.linalg.solve(gram, z.T.dot(y - ym))
    return mean, std, ym, coef


def _predict_ridge(model: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    mean, std, ym, coef = model
    return ((x - mean) / std).dot(coef) + ym


def choose_alpha(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, alphas: Sequence[float]) -> float:
    losses = []
    for alpha in alphas:
        fold_losses = []
        for fold in sorted(set(blocks.tolist())):
            train = blocks != fold; test = ~train
            model = _fit_ridge(x[train], y[train], float(alpha))
            pred = _predict_ridge(model, x[test])
            fold_losses.append(float(np.mean((y[test] - pred) ** 2)))
        losses.append((float(np.mean(fold_losses)), float(alpha)))
    return min(losses, key=lambda item: (item[0], item[1]))[1]


def crossfit_shared(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, alphas: Sequence[float]) -> Dict[str, object]:
    prediction = np.empty_like(y, dtype=np.float64)
    selected = []
    folds = sorted(set(blocks.tolist()))
    for outer in folds:
        train = blocks != outer; test = ~train
        inner_blocks = blocks[train]
        alpha = choose_alpha(x[train], y[train], inner_blocks, alphas)
        prediction[test] = _predict_ridge(_fit_ridge(x[train], y[train], alpha), x[test])
        selected.append(alpha)
    residual = y - prediction
    denominator = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    shared_r2 = 1.0 - np.sum(residual ** 2, axis=0) / np.maximum(denominator, 1e-15)
    full_alpha = choose_alpha(x, y, blocks, alphas)
    full_prediction = _predict_ridge(_fit_ridge(x, y, full_alpha), x)
    full_residual = y - full_prediction
    return {
        "prediction": prediction, "residual": residual, "shared_r2": shared_r2,
        "outer_alphas": selected, "full_alpha": full_alpha,
        "full_prediction": full_prediction, "full_residual": full_residual,
    }


def _column_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0); yc = y - y.mean(axis=0)
    num = np.sum(xc * yc, axis=0)
    den = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc, axis=0))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def bootstrap_stability(patterns: np.ndarray, graph: sp.csr_matrix, blocks: np.ndarray,
                        n_boot: int, seed: int) -> np.ndarray:
    patterns = np.asarray(patterns, dtype=np.float64)
    w = row_normalize_graph(graph)
    reference = w.dot(patterns)
    rng = np.random.RandomState(int(seed))
    values = []
    for _ in range(int(n_boot)):
        counts = np.zeros(patterns.shape[0], dtype=np.float64)
        for block in sorted(set(blocks.tolist())):
            idx = np.flatnonzero(blocks == block)
            sampled = rng.choice(idx, size=len(idx), replace=True)
            counts += np.bincount(sampled, minlength=len(counts))
        numerator = graph.dot(counts[:, None] * patterns)
        denominator = np.asarray(graph.dot(counts)).ravel()
        estimate = np.divide(numerator, denominator[:, None], out=np.zeros_like(numerator), where=denominator[:, None] > 0)
        values.append(_column_corr(reference, estimate))
    return np.median(np.stack(values, axis=0), axis=0)


def moran_columns(patterns: np.ndarray, graph: sp.csr_matrix) -> np.ndarray:
    x = np.asarray(patterns, dtype=np.float64)
    a = graph.tocsr().astype(np.float64)
    a = a.maximum(a.T); a.setdiag(0); a.eliminate_zeros()
    centered = x - x.mean(axis=0)
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(centered * a.dot(centered), axis=0)
    scale = float(x.shape[0]) / float(a.sum())
    return np.divide(scale * numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def evidence_axes(patterns: np.ndarray, graph: sp.csr_matrix, blocks: np.ndarray,
                  iid_null_stability: np.ndarray, permutations: Sequence[np.ndarray],
                  n_boot: int, seed: int) -> Dict[str, np.ndarray]:
    patterns = np.asarray(patterns, dtype=np.float64)
    stability = bootstrap_stability(patterns, graph, blocks, n_boot, seed)
    null_boot = np.asarray(iid_null_stability, dtype=np.float64)
    if null_boot.ndim != 2 or patterns.shape[1] % null_boot.shape[1] != 0:
        raise ValueError("IID bootstrap null must tile the registered 29-feature blocks")
    feature_n = null_boot.shape[1]
    boot_reference = np.tile(null_boot, (1, patterns.shape[1] // feature_n))
    p_boot = (1.0 + np.sum(boot_reference <= stability[None, :], axis=0)) / (boot_reference.shape[0] + 1.0)
    observed_moran = moran_columns(patterns, graph)
    null_moran = np.stack([moran_columns(patterns[p], graph) for p in permutations], axis=0)
    p_space = (1.0 + np.sum(null_moran <= observed_moran[None, :], axis=0)) / (len(permutations) + 1.0)
    combined = np.sqrt(p_boot * p_space)
    return {
        "stability": stability, "p_boot": p_boot, "moran": observed_moran,
        "moran_null_mean": null_moran.mean(axis=0),
        "spatial_excess": observed_moran - null_moran.mean(axis=0),
        "p_space": p_space, "combined": combined,
    }


def make_permutations(n: int, count: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(int(seed))
    result = []
    base = np.arange(n)
    for _ in range(int(count)):
        value = rng.permutation(n)
        fixed = np.flatnonzero(value == base)
        if len(fixed) == 1:
            i = int(fixed[0]); j = (i + 1) % n
            value[i], value[j] = value[j], value[i]
        elif len(fixed) > 1:
            value[fixed] = np.roll(value[fixed], 1)
        if np.any(value == base):
            raise AssertionError("permutation contains a fixed point")
        result.append(value)
    return result


def matched_iid(reference: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    scale = np.std(reference, axis=0, ddof=1)
    values = rng.normal(size=(reference.shape[0], reference.shape[1] * int(draws)))
    tiled = np.tile(scale, int(draws))
    values *= tiled[None, :]
    return values


def diffused_perturbation(reference: np.ndarray, w: sp.csr_matrix, seed: int, steps: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    target_std = np.std(reference, axis=0, ddof=1)
    value = rng.normal(size=reference.shape)
    for _ in range(int(steps)):
        value = w.dot(value)
    value -= value.mean(axis=0)
    value_std = np.std(value, axis=0, ddof=1)
    return value * np.divide(target_std, value_std, out=np.ones_like(target_std), where=value_std > 0)[None, :]
