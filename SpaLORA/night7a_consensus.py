"""Night-7A CPU-only, zero-training cross-graph consensus primitives.

The module deliberately contains no ground-truth paths and never imports an
evaluator.  It reuses the exact Night-6C/6D H05 affinity and spectral
implementations and adds only the SHA-locked sparse consensus formulas.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .night6c_pipeline import (
    _neighbors,
    array_sha,
    self_tuning_affinity,
    sparse_sha,
    spectral,
)


G00 = "G00_SP18_F20_CORR_UNION"
G04 = "G04_SP10_F10_EUC_UNION"
VIEWS = ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused")
DATASETS = ("a1", "tonsil", "d1", "p22")
K_BY_DATASET = {"a1": 10, "tonsil": 4, "d1": 10, "p22": 9}
CANDIDATE_ORDER = (
    "C00_G04_H05_CONFIRMED",
    "C01_G00_H05",
    "C02_DUAL_ARITHMETIC_MEAN",
    "C03_DUAL_ELEMENTWISE_MAX",
    "C04_DUAL_ELEMENTWISE_MIN",
    "C05_DUAL_HARMONIC_INTERSECTION",
    "C06_DUAL_ROW_STOCHASTIC_MEAN",
    "C07_DUAL_LOCAL_RELIABILITY",
    "C08_SIX_VIEW_SUPPORT_MEDIAN",
    "C09_DUAL_SPARSE_SNF10",
    "C10_DUAL_MEAN_SPATIAL05",
    "C11_DUAL_MEAN_SPATIAL10",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_sparse(path: Path, value: sp.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    sp.save_npz(tmp, canonical_csr(value), compressed=True)
    os.replace(tmp, path)


def canonical_csr(value: sp.spmatrix) -> sp.csr_matrix:
    result = value.tocsr().astype(np.float64)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def canonical_partition(labels: Sequence[object]) -> np.ndarray:
    mapping: Dict[object, int] = {}
    result = np.empty(len(labels), dtype=np.int64)
    for index, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = len(mapping)
        result[index] = mapping[label]
    return result


def partition_sha(labels: Sequence[object]) -> str:
    return array_sha(canonical_partition(labels))


def parse_registry(registry: Mapping[str, object]) -> Dict[str, dict]:
    order = tuple(registry["candidate_order"])
    if order != CANDIDATE_ORDER:
        raise RuntimeError("candidate order differs from the locked contract")
    rows = {str(row["id"]): dict(row) for row in registry["candidates"]}
    if tuple(rows) != CANDIDATE_ORDER or len(rows) != 12:
        raise RuntimeError("candidate registry is not exactly 12/12 in order")
    hashes = [canonical_json_sha(rows[candidate]) for candidate in order]
    if len(set(hashes)) != 12:
        raise RuntimeError("candidate canonical SHA collision")
    return rows


def _row_normalize_strict(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = canonical_csr(matrix)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0):
        raise ValueError("zero or non-finite sparse row")
    return canonical_csr(sp.diags(1.0 / degree) @ matrix)


def _sym_zero(matrix: sp.spmatrix) -> sp.csr_matrix:
    result = canonical_csr((matrix + matrix.T) * 0.5)
    result.setdiag(0.0)
    result.eliminate_zeros()
    result.sort_indices()
    return result


def spatial_affinity(coords: np.ndarray, ids: Sequence[str]) -> sp.csr_matrix:
    neighbors = _neighbors(np.asarray(coords, dtype=np.float64), 6,
                           "euclidean", ids)
    n = len(ids)
    directed = sp.coo_matrix(
        (np.ones(n * 6, dtype=np.float64),
         (np.repeat(np.arange(n), 6), neighbors.reshape(-1))),
        shape=(n, n),
    ).tocsr()
    binary_union = directed.maximum(directed.T)
    return _sym_zero(_row_normalize_strict(binary_union))


def local_reliability(neighbor_sets: Sequence[np.ndarray]) -> np.ndarray:
    if len(neighbor_sets) != 3:
        raise ValueError("local reliability requires exactly three views")
    n = neighbor_sets[0].shape[0]
    if any(x.shape != neighbor_sets[0].shape for x in neighbor_sets):
        raise ValueError("neighbor-set shape mismatch")
    result = np.empty(n, dtype=np.float64)
    for row in range(n):
        scores = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            a = set(map(int, neighbor_sets[left][row]))
            b = set(map(int, neighbor_sets[right][row]))
            scores.append(len(a & b) / len(a | b))
        result[row] = float(np.mean(scores))
    return result


def build_base_affinities(views_g00: Mapping[str, np.ndarray],
                          views_g04: Mapping[str, np.ndarray],
                          ids: Sequence[str], coords: np.ndarray) -> dict:
    all_views = {G00: views_g00, G04: views_g04}
    affinities = {}
    neighbor_sets = {}
    reliability = {}
    for graph_id in (G00, G04):
        affinities[graph_id] = []
        neighbor_sets[graph_id] = []
        for key in VIEWS:
            values = np.asarray(all_views[graph_id][key])
            affinities[graph_id].append(self_tuning_affinity(values, 10, ids))
            normalized = values.astype(np.float64)
            norms = np.linalg.norm(normalized, axis=1, keepdims=True)
            normalized = normalized / np.maximum(norms, 1e-12)
            neighbor_sets[graph_id].append(
                _neighbors(normalized, 10, "euclidean", ids)
            )
        reliability[graph_id] = local_reliability(neighbor_sets[graph_id])
    s00 = canonical_csr(sum(affinities[G00][1:], affinities[G00][0]) * (1.0 / 3.0))
    s04 = canonical_csr(sum(affinities[G04][1:], affinities[G04][0]) * (1.0 / 3.0))
    return {
        "affinities": affinities,
        "neighbor_sets": neighbor_sets,
        "reliability": reliability,
        "S_G00": s00,
        "S_G04": s04,
        "T_spatial": spatial_affinity(coords, ids),
    }


def _support_weighted_positive_median(matrices: Sequence[sp.spmatrix]) -> sp.csr_matrix:
    coos = [canonical_csr(matrix).tocoo() for matrix in matrices]
    rows = np.concatenate([matrix.row for matrix in coos])
    cols = np.concatenate([matrix.col for matrix in coos])
    vals = np.concatenate([matrix.data for matrix in coos])
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    boundaries = np.r_[0, np.flatnonzero((rows[1:] != rows[:-1]) |
                                        (cols[1:] != cols[:-1])) + 1, len(vals)]
    out_r, out_c, out_v = [], [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        positive = vals[start:stop][vals[start:stop] > 0]
        if len(positive):
            out_r.append(int(rows[start])); out_c.append(int(cols[start]))
            out_v.append(float(np.median(positive)) * len(positive) / 6.0)
    n = matrices[0].shape[0]
    result = sp.coo_matrix((out_v, (out_r, out_c)), shape=(n, n)).tocsr()
    result.setdiag(0.0)
    return canonical_csr(result)


def _topk_rows(matrix: sp.spmatrix, k: int, ids: Sequence[str],
               keep_diagonal: bool) -> sp.csr_matrix:
    matrix = canonical_csr(matrix)
    names = np.asarray(ids, dtype=str)
    out_rows, out_cols, out_vals = [], [], []
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        cols = matrix.indices[start:stop]
        vals = matrix.data[start:stop]
        diagonal = vals[cols == row]
        mask = cols != row
        cols_off, vals_off = cols[mask], vals[mask]
        if len(cols_off):
            order = np.lexsort((names[cols_off], -vals_off))[:k]
            out_rows.extend([row] * len(order))
            out_cols.extend(cols_off[order].tolist())
            out_vals.extend(vals_off[order].tolist())
        if keep_diagonal and len(diagonal):
            out_rows.append(row); out_cols.append(row)
            out_vals.append(float(diagonal[-1]))
    return canonical_csr(sp.coo_matrix((out_vals, (out_rows, out_cols)),
                                       shape=matrix.shape))


def _p_convention(matrix: sp.spmatrix, ids: Sequence[str],
                  retain_top10: bool) -> sp.csr_matrix:
    work = canonical_csr(matrix)
    if retain_top10:
        work = _topk_rows(work, 10, ids, keep_diagonal=True)
    work.setdiag(0.0)
    work.eliminate_zeros()
    off = _row_normalize_strict(work) * 0.5
    return canonical_csr(off + sp.eye(work.shape[0], format="csr") * 0.5)


def sparse_snf10(s00: sp.spmatrix, s04: sp.spmatrix,
                 ids: Sequence[str]) -> sp.csr_matrix:
    p00 = _p_convention(s00, ids, retain_top10=False)
    p04 = _p_convention(s04, ids, retain_top10=False)
    k00 = _row_normalize_strict(_topk_rows(s00, 10, ids, keep_diagonal=False))
    k04 = _row_normalize_strict(_topk_rows(s04, 10, ids, keep_diagonal=False))
    for _ in range(10):
        u00 = canonical_csr(k00 @ p04 @ k00.T)
        u04 = canonical_csr(k04 @ p00 @ k04.T)
        p00_new = _p_convention(u00, ids, retain_top10=True)
        p04_new = _p_convention(u04, ids, retain_top10=True)
        p00, p04 = p00_new, p04_new
    result = _sym_zero((p00 + p04) * 0.5)
    return result


def candidate_affinity(candidate_id: str, base: Mapping[str, object],
                       ids: Sequence[str]) -> Tuple[sp.csr_matrix, dict]:
    s00 = canonical_csr(base["S_G00"])
    s04 = canonical_csr(base["S_G04"])
    diagnostics = {}
    if candidate_id == "C00_G04_H05_CONFIRMED":
        result = s04
    elif candidate_id == "C01_G00_H05":
        result = s00
    elif candidate_id == "C02_DUAL_ARITHMETIC_MEAN":
        result = (s00 + s04) * 0.5
    elif candidate_id == "C03_DUAL_ELEMENTWISE_MAX":
        result = s00.maximum(s04)
    elif candidate_id == "C04_DUAL_ELEMENTWISE_MIN":
        result = s00.minimum(s04)
    elif candidate_id == "C05_DUAL_HARMONIC_INTERSECTION":
        numerator = s00.multiply(s04) * 2.0
        denominator = s00 + s04
        inverse = denominator.copy()
        inverse.data = 1.0 / inverse.data
        result = numerator.multiply(inverse)
    elif candidate_id == "C06_DUAL_ROW_STOCHASTIC_MEAN":
        result = _sym_zero((_row_normalize_strict(s00) +
                            _row_normalize_strict(s04)) * 0.5)
    elif candidate_id == "C07_DUAL_LOCAL_RELIABILITY":
        r00 = np.asarray(base["reliability"][G00], dtype=np.float64)
        r04 = np.asarray(base["reliability"][G04], dtype=np.float64)
        w00 = (r00 + 1e-12) / (r00 + r04 + 2e-12)
        w04 = (r04 + 1e-12) / (r00 + r04 + 2e-12)
        directed = sp.diags(w00) @ s00 + sp.diags(w04) @ s04
        result = _sym_zero(directed)
        diagnostics["g00_weight_quantiles"] = np.quantile(
            w00, [0, .01, .05, .25, .5, .75, .95, .99, 1]
        ).tolist()
        diagnostics["g04_weight_quantiles"] = np.quantile(
            w04, [0, .01, .05, .25, .5, .75, .95, .99, 1]
        ).tolist()
        diagnostics["g00_weights"] = w00
    elif candidate_id == "C08_SIX_VIEW_SUPPORT_MEDIAN":
        result = _support_weighted_positive_median(
            list(base["affinities"][G00]) + list(base["affinities"][G04])
        )
    elif candidate_id == "C09_DUAL_SPARSE_SNF10":
        result = sparse_snf10(s00, s04, ids)
    elif candidate_id == "C10_DUAL_MEAN_SPATIAL05":
        c06 = _sym_zero((_row_normalize_strict(s00) +
                         _row_normalize_strict(s04)) * 0.5)
        result = c06 * .95 + canonical_csr(base["T_spatial"]) * .05
    elif candidate_id == "C11_DUAL_MEAN_SPATIAL10":
        c06 = _sym_zero((_row_normalize_strict(s00) +
                         _row_normalize_strict(s04)) * 0.5)
        result = c06 * .90 + canonical_csr(base["T_spatial"]) * .10
    else:
        raise KeyError(candidate_id)
    result = canonical_csr(result)
    result.setdiag(0.0)
    result.eliminate_zeros()
    result.sort_indices()
    return result, diagnostics


def affinity_audit(matrix: sp.spmatrix) -> dict:
    value = canonical_csr(matrix)
    diagonal = value.diagonal()
    diff = canonical_csr(value - value.T)
    symmetry_error = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    degree = np.asarray(value.sum(axis=1)).ravel()
    component_count, component = connected_components(value, directed=False)
    sizes = np.bincount(component, minlength=component_count)
    return {
        "shape": list(value.shape),
        "nnz": int(value.nnz),
        "sparse_csr": True,
        "finite": bool(np.isfinite(value.data).all()),
        "min": float(value.data.min()) if value.nnz else None,
        "max": float(value.data.max()) if value.nnz else None,
        "symmetry_max_error": symmetry_error,
        "diagonal_max_abs": float(np.max(np.abs(diagonal))) if len(diagonal) else 0.0,
        "zero_degree_count": int(np.sum(degree == 0)),
        "connected_component_count": int(component_count),
        "connected_component_sizes": sorted(map(int, sizes), reverse=True),
        "canonical_sparse_sha256": sparse_sha(value),
    }


def run_spectral(matrix: sp.spmatrix, dataset: str) -> Tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    labels = spectral(matrix, K_BY_DATASET[dataset])
    runtime = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return labels, runtime, peak
