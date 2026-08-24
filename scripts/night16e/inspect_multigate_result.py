#!/usr/bin/env python3
"""Audit official MultiGATE result carriers against the processed input files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np


def sha_strings(values: np.ndarray) -> str:
    return hashlib.sha256(b"\0".join(str(v).encode() for v in values.tolist())).hexdigest()


def column_summary(data: ad.AnnData) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in data.obs.columns:
        series = data.obs[key]
        summary = {
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if summary["unique"] <= 50:
            summary["value_counts"] = {
                str(name): int(count)
                for name, count in series.value_counts(dropna=False).items()
            }
        else:
            summary["value_counts"] = "OMITTED_HIGH_CARDINALITY"
        result[str(key)] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rna", required=True)
    parser.add_argument("--input-atac", required=True)
    parser.add_argument("--result-rna", required=True)
    parser.add_argument("--result-atac", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_rna = ad.read_h5ad(args.input_rna)
    input_atac = ad.read_h5ad(args.input_atac)
    result_rna = ad.read_h5ad(args.result_rna)
    result_atac = ad.read_h5ad(args.result_atac)
    datasets = {
        "input_rna": input_rna,
        "input_atac": input_atac,
        "result_rna": result_rna,
        "result_atac": result_atac,
    }
    audit: dict[str, object] = {
        "schema": "night16e-multigate-official-result-audit-v1",
        "datasets": {},
    }
    for key, data in datasets.items():
        ids = np.asarray(data.obs_names.astype(str), dtype=str)
        audit["datasets"][key] = {
            "shape": [int(data.n_obs), int(data.n_vars)],
            "ordered_id_sha256": sha_strings(ids),
            "obs_columns": column_summary(data),
            "obsm": {name: list(np.asarray(value).shape) for name, value in data.obsm.items()},
            "spatial_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(data.obsm["spatial"])).tobytes()
            ).hexdigest() if "spatial" in data.obsm else None,
        }
    input_ids = np.asarray(input_rna.obs_names.astype(str), dtype=str)
    result_ids = np.asarray(result_rna.obs_names.astype(str), dtype=str)
    input_spatial = np.asarray(input_rna.obsm["spatial"])
    result_spatial = np.asarray(result_rna[input_ids].obsm["spatial"])
    audit["alignment"] = {
        "result_rna_same_set_as_input": bool(set(input_ids) == set(result_ids)),
        "result_atac_same_set_as_input": bool(set(input_ids) == set(result_atac.obs_names.astype(str))),
        "result_rna_same_order_as_input": bool(np.array_equal(input_ids, result_ids)),
        "result_atac_same_order_as_input": bool(np.array_equal(input_ids, result_atac.obs_names.astype(str))),
        "result_spatial_equals_input": bool(np.array_equal(result_spatial, input_spatial)),
        "result_spatial_equals_y_negated_input": bool(
            np.array_equal(result_spatial, input_spatial * np.asarray([1, -1]))
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
