"""Pure evaluation utilities for the Night-8B cardinality-safe recovery.

This module deliberately exposes no training, checkpoint, forward, affinity,
head-transform, or clustering entry point.  It operates only on frozen
partitions and an in-memory reference-label vector.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_fsync(path: Path, payload: dict) -> None:
    """Atomically persist JSON and fsync both the file and its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
               + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _canonical_scalar(value: object) -> Tuple[str, str]:
    if isinstance(value, (bytes, np.bytes_)):
        text = bytes(value).decode("utf-8", errors="strict").strip()
        if text == "":
            raise ValueError("EMPTY_STRING_AFTER_TRIM")
        return "string", text
    if isinstance(value, (str, np.str_)):
        text = str(value).strip()
        if text == "":
            raise ValueError("EMPTY_STRING_AFTER_TRIM")
        return "string", text
    if isinstance(value, (bool, np.bool_)):
        return "boolean", "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return "integer", str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("NONFINITE_NUMERIC_LABEL")
        # Python's repr is a lossless decimal round-trip for IEEE-754 doubles.
        return "float", repr(numeric)
    raise ValueError("UNSUPPORTED_LABEL_DTYPE:%s" % type(value).__name__)


def canonicalize_labels(raw: np.ndarray, expected_n: int) -> Tuple[np.ndarray, dict]:
    """Canonicalize without merging, dropping, binning, or renaming labels."""
    values = np.asarray(raw)
    original_shape = list(values.shape)
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError("Y_STRUCTURE_NOT_1D_OR_SINGLETON_2D")
    if values.size != int(expected_n):
        raise ValueError("Y_LENGTH_MISMATCH:%d!=%d" % (values.size, expected_n))
    kinds: List[str] = []
    canonical: List[str] = []
    for value in values.tolist():
        kind, text = _canonical_scalar(value)
        kinds.append(kind)
        canonical.append(kind + ":" + text)
    if len(set(kinds)) != 1:
        raise ValueError("MIXED_LABEL_TYPES_FORBIDDEN")
    encoded = "\n".join(canonical).encode("utf-8")
    unique, counts = np.unique(np.asarray(canonical, dtype=str), return_counts=True)
    if not (2 <= unique.size < int(expected_n)):
        raise ValueError("REFERENCE_K_OUT_OF_RANGE:%d" % unique.size)
    contract = {
        "original_shape": original_shape,
        "flattened_shape": [int(values.size)],
        "canonical_kind": kinds[0],
        "canonical_Y_sha256": hashlib.sha256(encoded).hexdigest(),
        "reference_K": int(unique.size),
        "category_values_and_counts": [
            {"canonical_value": str(value), "count": int(count)}
            for value, count in zip(unique.tolist(), counts.tolist())
        ],
        "missing_like_count": 0,
    }
    return np.asarray(canonical, dtype=str), contract


def symmetric_knn_adjacency(coordinates: np.ndarray, k: int = 18) -> sp.csr_matrix:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("invalid coordinates")
    neighbors = min(max(1, int(k)), coordinates.shape[0] - 1)
    indices = NearestNeighbors(n_neighbors=neighbors + 1).fit(coordinates).kneighbors(
        return_distance=False
    )
    rows = np.repeat(np.arange(coordinates.shape[0]), neighbors)
    cols = indices[:, 1:].reshape(-1)
    graph = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(coordinates.shape[0], coordinates.shape[0]),
    ).maximum(sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(coordinates.shape[0], coordinates.shape[0]),
    ).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _primary_spatial(prediction: np.ndarray, graph: sp.csr_matrix) -> Dict[str, float]:
    prediction = np.asarray(prediction)
    graph = graph.tocsr().astype(np.float64)
    row, col = graph.nonzero()
    neighbor = float(np.mean(prediction[row] == prediction[col]))
    total_weight = float(graph.sum())
    n = prediction.size
    moran_values: List[float] = []
    geary_values: List[float] = []
    weights = np.asarray(graph[row, col]).reshape(-1)
    for level in np.unique(prediction):
        binary = (prediction == level).astype(np.float64)
        centered = binary - binary.mean()
        denominator = float(np.dot(centered, centered))
        if denominator > 0.0:
            moran_values.append(
                (n / total_weight) * float(centered @ graph.dot(centered)) / denominator
            )
            numerator = float(np.sum(weights * (binary[row] - binary[col]) ** 2))
            geary_values.append(
                (n - 1) * numerator / (2.0 * total_weight * denominator)
            )
    return {
        "neighbor_agreement": neighbor,
        "moran_i": float(np.mean(moran_values)),
        "geary_c": float(np.mean(geary_values)),
        "boundary_disagreement": 1.0 - neighbor,
    }


def primary_metrics(truth: np.ndarray, prediction: np.ndarray,
                    graph: sp.csr_matrix) -> Dict[str, float]:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    ari = float(adjusted_rand_score(truth, prediction))
    nmi = float(normalized_mutual_info_score(truth, prediction))
    result = {
        "ari": ari,
        "nmi": nmi,
        "q": (ari + nmi) / 2.0,
        "ami": float(adjusted_mutual_info_score(truth, prediction)),
        "fmi": float(fowlkes_mallows_score(truth, prediction)),
        "homogeneity": float(homogeneity_score(truth, prediction)),
        "completeness": float(completeness_score(truth, prediction)),
        "v_measure": float(v_measure_score(truth, prediction)),
    }
    result.update(_primary_spatial(prediction, graph))
    return result


def _contingency(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    _, true_codes = np.unique(np.asarray(truth), return_inverse=True)
    _, pred_codes = np.unique(np.asarray(prediction), return_inverse=True)
    table = np.zeros((true_codes.max() + 1, pred_codes.max() + 1), dtype=np.int64)
    np.add.at(table, (true_codes, pred_codes), 1)
    return table


def _comb2(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sum(values * (values - 1.0) / 2.0))


def independent_ari_nmi(truth: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    table = _contingency(truth, prediction)
    n = float(table.sum())
    sum_cells = _comb2(table)
    sum_rows = _comb2(table.sum(axis=1))
    sum_cols = _comb2(table.sum(axis=0))
    total_pairs = n * (n - 1.0) / 2.0
    expected = (sum_rows * sum_cols / total_pairs) if total_pairs else 0.0
    denominator = 0.5 * (sum_rows + sum_cols) - expected
    ari = (sum_cells - expected) / denominator if denominator else 1.0

    proportions = table.astype(np.float64) / n
    row_p = proportions.sum(axis=1)
    col_p = proportions.sum(axis=0)
    nz_row, nz_col = np.nonzero(proportions)
    mutual_information = float(np.sum(
        proportions[nz_row, nz_col]
        * np.log(proportions[nz_row, nz_col] / (row_p[nz_row] * col_p[nz_col]))
    ))
    entropy_truth = float(-np.sum(row_p[row_p > 0] * np.log(row_p[row_p > 0])))
    entropy_pred = float(-np.sum(col_p[col_p > 0] * np.log(col_p[col_p > 0])))
    average_entropy = 0.5 * (entropy_truth + entropy_pred)
    nmi = mutual_information / average_entropy if average_entropy else 1.0
    return float(ari), float(nmi)


def _independent_spatial(prediction: np.ndarray,
                         graph: sp.csr_matrix) -> Dict[str, float]:
    prediction = np.asarray(prediction)
    coo = graph.tocoo(copy=True)
    order = np.lexsort((coo.col, coo.row))
    row = coo.row[order]
    col = coo.col[order]
    weight = coo.data[order].astype(np.float64)
    neighbor = float(np.count_nonzero(prediction[row] == prediction[col]) / row.size)
    total_weight = float(weight.sum())
    n = prediction.size
    moran_values: List[float] = []
    geary_values: List[float] = []
    for level in np.unique(prediction):
        indicator = np.equal(prediction, level).astype(np.float64)
        centered = indicator - float(indicator.sum()) / float(n)
        denominator = float(np.sum(centered * centered))
        if denominator > 0.0:
            cross = float(np.sum(weight * centered[row] * centered[col]))
            moran_values.append(float(n) * cross / (total_weight * denominator))
            squared_difference = (indicator[row] - indicator[col]) ** 2
            geary_values.append(
                float(n - 1) * float(np.sum(weight * squared_difference))
                / (2.0 * total_weight * denominator)
            )
    return {
        "neighbor_agreement": neighbor,
        "moran_i": float(np.sum(moran_values) / len(moran_values)),
        "geary_c": float(np.sum(geary_values) / len(geary_values)),
        "boundary_disagreement": 1.0 - neighbor,
    }


def independent_decision_metrics(truth: np.ndarray, prediction: np.ndarray,
                                 graph: sp.csr_matrix) -> Dict[str, float]:
    ari, nmi = independent_ari_nmi(truth, prediction)
    result = {"ari": ari, "nmi": nmi, "q": (ari + nmi) / 2.0}
    result.update(_independent_spatial(prediction, graph))
    return result


def paired_rows(rows: Sequence[dict], methods: Tuple[str, str] = ("HR_U00", "HR_F00")) -> List[dict]:
    reference, treatment = methods
    by_key = {(str(row["method"]), int(row["seed"])): row for row in rows}
    keys = (
        "ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c",
        "boundary_disagreement", "end_to_end_seconds", "peak_gpu_mib",
    )
    output: List[dict] = []
    for seed in range(10):
        left = by_key[(reference, seed)]
        right = by_key[(treatment, seed)]
        output.append({
            "seed": seed,
            **{("delta_" + key): float(right[key]) - float(left[key]) for key in keys},
        })
    return output


def exact_signflip(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    observed = float(values.mean())
    enumerated = np.asarray([
        np.mean(values * np.asarray(signs, dtype=np.float64))
        for signs in itertools.product((-1.0, 1.0), repeat=values.size)
    ])
    tail = int(np.sum(enumerated >= observed - 1e-15))
    return {
        "observed_mean": observed,
        "enumerations": int(enumerated.size),
        "tail_count": tail,
        "p_one_sided": float(tail / enumerated.size),
    }


def paired_bootstrap(values: np.ndarray, indices: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    means = values[np.asarray(indices, dtype=np.int64)].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "replicates": int(means.size),
        "seed": 20260820,
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def paired_statistics(paired: Sequence[dict], bootstrap_indices: np.ndarray) -> dict:
    output: Dict[str, object] = {
        "paired_units": len(paired),
        "inference_note": "seeds measure algorithmic stability, not independent biological replication",
    }
    for name in ("ari", "nmi", "q"):
        values = np.asarray([row["delta_" + name] for row in paired], dtype=np.float64)
        output["mean_delta_" + name] = float(values.mean())
        output["std_delta_" + name] = float(values.std(ddof=1))
        output[name + "_wins"] = int(np.sum(values > 0.0))
        output["bootstrap_delta_" + name] = paired_bootstrap(values, bootstrap_indices)
    output["exact_sign_flip_delta_q"] = exact_signflip(
        np.asarray([row["delta_q"] for row in paired], dtype=np.float64)
    )
    return output


def spatial_summary(paired: Sequence[dict]) -> dict:
    result = {
        "mean_delta_neighbor": float(np.mean([row["delta_neighbor_agreement"] for row in paired])),
        "mean_delta_moran": float(np.mean([row["delta_moran_i"] for row in paired])),
        "mean_delta_geary": float(np.mean([row["delta_geary_c"] for row in paired])),
        "mean_delta_boundary": float(np.mean([row["delta_boundary_disagreement"] for row in paired])),
        "thresholds": {
            "neighbor_min": -0.01,
            "moran_min": -0.02,
            "geary_max": 0.02,
            "boundary_max": 0.01,
        },
    }
    result["pass"] = bool(
        result["mean_delta_neighbor"] >= -0.01
        and result["mean_delta_moran"] >= -0.02
        and result["mean_delta_geary"] <= 0.02
        and result["mean_delta_boundary"] <= 0.01
    )
    return result


__all__ = [
    "atomic_json_fsync", "canonicalize_labels", "exact_signflip",
    "independent_decision_metrics", "paired_rows", "paired_statistics",
    "primary_metrics", "sha256_file", "spatial_summary",
    "symmetric_knn_adjacency",
]
