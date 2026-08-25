#!/usr/bin/env python3
"""Fail-closed authority audit for the Night-18D human placenta assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


EXPECTED_OFFICIAL_ATAC_SHA = "f53ef887ec5b96a79d19ded1ac21a7e4ef44ae0d07ab572706bc076350f1460d"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def string_sha(values) -> str:
    payload = b"\0".join(str(value).encode("utf-8") for value in values)
    return hashlib.sha256(payload).hexdigest()


def normalized_obs_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    series = frame[column]
    return np.asarray(["__NA__" if pd.isna(value) else str(value) for value in series.to_numpy()], dtype="U")


def matrix_compare(left, right) -> dict[str, object]:
    left = sp.csr_matrix(left, dtype=np.float64) if sp.issparse(left) else np.asarray(left, dtype=np.float64)
    right = sp.csr_matrix(right, dtype=np.float64) if sp.issparse(right) else np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_equal": False, "numeric_equal": False, "max_abs_difference": None}
    if sp.issparse(left) or sp.issparse(right):
        left, right = sp.csr_matrix(left), sp.csr_matrix(right)
        difference = (left - right).tocsr(); difference.eliminate_zeros()
        maximum = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
        return {"shape_equal": True, "numeric_equal": bool(difference.nnz == 0), "max_abs_difference": maximum,
                "left_nnz": int(left.nnz), "right_nnz": int(right.nnz), "difference_nnz": int(difference.nnz)}
    difference = np.asarray(left - right)
    return {"shape_equal": True, "numeric_equal": bool(np.array_equal(left, right)),
            "max_abs_difference": float(np.max(np.abs(difference))) if difference.size else 0.0}


def run(args: argparse.Namespace) -> None:
    paths = {"rna": Path(args.rna), "local_atac": Path(args.local_atac), "official_atac": Path(args.official_atac)}
    hashes = {name: file_sha(path) for name, path in paths.items()}
    if hashes["official_atac"] != EXPECTED_OFFICIAL_ATAC_SHA:
        raise RuntimeError("official repository ATAC SHA mismatch")
    rna = ad.read_h5ad(paths["rna"])
    local = ad.read_h5ad(paths["local_atac"])
    official = ad.read_h5ad(paths["official_atac"])
    expected_shapes = {"rna": [1662, 36601], "atac": [1662, 63]}
    if list(rna.shape) != expected_shapes["rna"] or list(local.shape) != expected_shapes["atac"] or list(official.shape) != expected_shapes["atac"]:
        raise RuntimeError("authority matrix shape mismatch")
    ids_rna = np.asarray(rna.obs_names.astype(str)); ids_local = np.asarray(local.obs_names.astype(str)); ids_official = np.asarray(official.obs_names.astype(str))
    ids_equal = bool(np.array_equal(ids_rna, ids_local) and np.array_equal(ids_local, ids_official))
    features_equal = bool(np.array_equal(np.asarray(local.var_names.astype(str)), np.asarray(official.var_names.astype(str))))
    if not ids_equal or not features_equal:
        raise RuntimeError("ordered IDs or ATAC feature names mismatch")
    if "cell_type" not in rna.obs or "cell_type" not in local.obs:
        raise KeyError("cell_type")
    labels_rna = normalized_obs_column(rna.obs, "cell_type"); labels_local = normalized_obs_column(local.obs, "cell_type")
    if not np.array_equal(labels_rna, labels_local) or len(np.unique(labels_rna)) != 10:
        raise RuntimeError("RNA/ATAC cell_type authority mismatch")
    for object_name, value in (("rna", rna), ("local_atac", local), ("official_atac", official)):
        if "spatial" not in value.obsm:
            raise RuntimeError(f"{object_name} lacks spatial coordinates")
    coordinates_rna = np.asarray(rna.obsm["spatial"], dtype=np.float64)
    coordinates_local = np.asarray(local.obsm["spatial"], dtype=np.float64)
    coordinates_official = np.asarray(official.obsm["spatial"], dtype=np.float64)
    coordinates_equal = bool(np.array_equal(coordinates_rna, coordinates_local) and np.array_equal(coordinates_local, coordinates_official))
    if not coordinates_equal or not np.isfinite(coordinates_rna).all():
        raise RuntimeError("spatial coordinate authority mismatch")
    x_comparison = matrix_compare(local.X, official.X)
    if not x_comparison["numeric_equal"]:
        raise RuntimeError("local and official ATAC X differ")
    obs_columns_equal = list(local.obs.columns) == list(official.obs.columns)
    obs_value_mismatches = []
    if obs_columns_equal:
        for column in local.obs.columns:
            if not np.array_equal(normalized_obs_column(local.obs, column), normalized_obs_column(official.obs, column)):
                obs_value_mismatches.append(str(column))
    if not obs_columns_equal or obs_value_mismatches:
        raise RuntimeError("local and official ATAC obs differ")
    unique, counts = np.unique(labels_rna, return_counts=True)
    audit = {
        "status": "PASS",
        "schema": "night18d-human-placenta-authority-audit-v1",
        "files": {name: {"path": str(path), "size": path.stat().st_size, "sha256": hashes[name]} for name, path in paths.items()},
        "expected_official_atac_sha256": EXPECTED_OFFICIAL_ATAC_SHA,
        "rna_shape": list(rna.shape), "atac_shape": list(local.shape),
        "rna_x_sparse": bool(sp.issparse(rna.X)), "atac_x_sparse": bool(sp.issparse(local.X)),
        "atac_semantic_name": "official processed ATAC-derived TF-associated regulatory features",
        "ordered_ids_equal_all_three": ids_equal, "ordered_ids_sha256": string_sha(ids_rna),
        "atac_features_equal": features_equal, "atac_feature_names_sha256": string_sha(local.var_names.astype(str)),
        "local_vs_official_atac_x": x_comparison,
        "local_vs_official_atac_obs_columns_equal": obs_columns_equal,
        "local_vs_official_atac_obs_value_mismatches": obs_value_mismatches,
        "coordinates_equal_all_three": coordinates_equal, "coordinates_sha256": array_sha(coordinates_rna),
        "rna_atac_cell_type_equal": True, "reference_k": int(len(unique)),
        "cell_type_counts": {str(label): int(count) for label, count in zip(unique, counts)},
        "reference_semantics": "original-author manually annotated cell-type partition after joint RNA/ATAC QC",
        "n_joint_qc_cells": len(ids_rna),
        "authority_gap": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--rna", required=True); parser.add_argument("--local-atac", required=True)
    parser.add_argument("--official-atac", required=True); parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
