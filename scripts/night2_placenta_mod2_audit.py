#!/usr/bin/env python3
"""Read-only numerical and provenance audit of placenta modality 2."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


REPO = Path(__file__).resolve().parents[1]
PATH = "/root/autodl-fs/Human placenta architecture/humanplacenta_atac.h5ad"


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> None:
    obj = ad.read_h5ad(PATH)
    matrix = obj.X.toarray() if sp.issparse(obj.X) else np.asarray(obj.X)
    finite = np.isfinite(matrix)
    if not finite.all():
        raise AssertionError("Placenta modality-2 matrix contains non-finite values")
    feature_summary = []
    for index, name in enumerate(obj.var_names.astype(str)):
        values = matrix[:, index]
        feature_summary.append(
            {
                "feature": name,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "fraction_zero": float(np.mean(values == 0)),
                "fraction_negative": float(np.mean(values < 0)),
                "fraction_integer_like": float(np.mean(np.isclose(values, np.rint(values)))),
            }
        )
    var_columns = {}
    for column in obj.var.columns:
        values = obj.var[column]
        unique = values.drop_duplicates().tolist()
        var_columns[str(column)] = {
            "dtype": str(values.dtype),
            "unique_count": int(len(unique)),
            "unique_values": [jsonable(value) for value in unique[:200]],
            "truncated": len(unique) > 200,
        }
    keywords = ("chromvar", "motif", "deviation", "enrichment", "activity", "gene activity", "pca", "lsi")
    searchable = " ".join(
        list(map(str, obj.var_names))
        + list(map(str, obj.var.columns))
        + list(map(str, obj.uns.keys()))
        + list(map(str, obj.obsm.keys()))
        + list(map(str, obj.layers.keys()))
    ).lower()
    report = {
        "source_path": PATH,
        "source_study": {
            "citation": "Ounadjela et al., Nature Medicine 2024",
            "doi": "10.1038/s41591-024-03073-9",
            "taskbook_context": "Slide-tags analysis includes TF motif activity/deviation via ChromVAR",
        },
        "shape": [int(obj.n_obs), int(obj.n_vars)],
        "feature_names": list(map(str, obj.var_names)),
        "var_columns": var_columns,
        "feature_types_unique": (
            [jsonable(value) for value in obj.var["feature_types"].drop_duplicates().tolist()]
            if "feature_types" in obj.var
            else None
        ),
        "x": {
            "storage": obj.X.getformat() if sp.issparse(obj.X) else "dense",
            "dtype": str(obj.X.dtype),
            "min": float(matrix.min()),
            "max": float(matrix.max()),
            "mean": float(matrix.mean()),
            "std": float(matrix.std(ddof=0)),
            "fraction_negative": float(np.mean(matrix < 0)),
            "fraction_zero": float(np.mean(matrix == 0)),
            "fraction_integer_like": float(np.mean(np.isclose(matrix, np.rint(matrix)))),
            "finite": True,
        },
        "per_feature_summary": feature_summary,
        "layers": list(map(str, obj.layers.keys())),
        "raw_present": obj.raw is not None,
        "obsm": {str(key): list(map(int, np.asarray(value).shape)) for key, value in obj.obsm.items()},
        "varm": {str(key): list(map(int, np.asarray(value).shape)) for key, value in obj.varm.items()},
        "uns_keys": list(map(str, obj.uns.keys())),
        "provenance_keyword_presence": {keyword: keyword in searchable for keyword in keywords},
        "conservative_description": "ATAC-derived / TF-associated regulatory features",
        "raw_peak_claim_supported": False,
        "clr_assessment": (
            "The matrix is nonnegative, so CLR is numerically defined, but it is continuous and mostly non-integer. "
            "Without provenance proving raw counts/compositional input, CLR appropriateness is not established and must not be changed based on ARI tonight."
        ),
    }
    output = REPO / "reports" / "placenta_mod2_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
