#!/usr/bin/env python3
"""Read-only schema and pairing audit for the official MultiGATE hippocampus files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def sha_strings(values: np.ndarray) -> str:
    payload = b"\0".join(str(value).encode("utf-8") for value in values.tolist())
    return hashlib.sha256(payload).hexdigest()


def describe(path: Path) -> tuple[ad.AnnData, dict[str, object]]:
    data = ad.read_h5ad(path)
    obs_names = np.asarray(data.obs_names.astype(str), dtype=str)
    var_names = np.asarray(data.var_names.astype(str), dtype=str)
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "shape": [int(data.n_obs), int(data.n_vars)],
        "x_dtype": str(data.X.dtype),
        "x_sparse": bool(sp.issparse(data.X)),
        "x_nnz": int(data.X.nnz) if sp.issparse(data.X) else int(np.count_nonzero(data.X)),
        "obs_columns": [str(value) for value in data.obs.columns],
        "var_columns": [str(value) for value in data.var.columns],
        "obsm": {key: list(np.asarray(value).shape) for key, value in data.obsm.items()},
        "varm": {key: list(np.asarray(value).shape) for key, value in data.varm.items()},
        "layers": {key: list(value.shape) for key, value in data.layers.items()},
        "uns_keys": sorted(str(key) for key in data.uns.keys()),
        "obs_id_unique": bool(len(np.unique(obs_names)) == len(obs_names)),
        "obs_id_ordered_sha256": sha_strings(obs_names),
        "obs_id_set_sha256": sha_strings(np.sort(obs_names)),
        "var_id_unique": bool(len(np.unique(var_names)) == len(var_names)),
        "var_id_ordered_sha256": sha_strings(var_names),
    }
    for key in data.obs.columns:
        series = data.obs[key]
        result.setdefault("obs_column_audit", {})[str(key)] = {
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "sample_values": [str(value) for value in series.dropna().astype(str).unique()[:20]],
        }
    for key, value in data.obsm.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            result.setdefault("obsm_numeric_audit", {})[str(key)] = {
                "dtype": str(array.dtype),
                "finite": bool(np.isfinite(array).all()),
                "minimum": float(np.nanmin(array)),
                "maximum": float(np.nanmax(array)),
            }
    return data, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--atac", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rna, rna_audit = describe(Path(args.rna))
    atac, atac_audit = describe(Path(args.atac))
    rna_ids = np.asarray(rna.obs_names.astype(str), dtype=str)
    atac_ids = np.asarray(atac.obs_names.astype(str), dtype=str)
    intersection = np.intersect1d(rna_ids, atac_ids)
    aligned_atac = atac[rna_ids].copy() if len(intersection) == len(rna_ids) == len(atac_ids) else None
    audit = {
        "schema": "night16e-human-hippocampus-schema-audit-v1",
        "rna": rna_audit,
        "atac": atac_audit,
        "pairing": {
            "intersection": int(len(intersection)),
            "same_set": bool(set(rna_ids.tolist()) == set(atac_ids.tolist())),
            "same_order": bool(np.array_equal(rna_ids, atac_ids)),
            "rna_ordered_ids_sha256": sha_strings(rna_ids),
            "atac_ordered_ids_sha256": sha_strings(atac_ids),
            "aligned_atac_ids_sha256": sha_strings(np.asarray(aligned_atac.obs_names.astype(str), dtype=str)) if aligned_atac is not None else None,
            "explicit_string_id_alignment": "PASS" if aligned_atac is not None else "FAIL",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
