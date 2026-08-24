#!/usr/bin/env python3
"""Create a one-variable human carrier with float64 spatial edges.

This diagnostic keeps every existing carrier array byte-identical except the
three sparse graph data arrays, which are recomputed from the registered RNA
coordinates with the parent Night-16E float64 graph construction.  It exists
only to attribute the Night-16E/Night-16F DIRECT_BASE discrepancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from scripts.night16f.build_numeric_carrier import ordered_id_sha256, spatial_graph


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-carrier", required=True)
    parser.add_argument("--rna", required=True)
    parser.add_argument("--output-carrier", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()

    with np.load(args.input_carrier, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    rna = ad.read_h5ad(args.rna)
    ids = np.asarray(rna.obs_names.astype(str))
    if ordered_id_sha256(ids) != ordered_id_sha256(payload["ids"]):
        raise ValueError("registered RNA IDs do not match carrier IDs")
    coordinates = np.asarray(rna.obsm["spatial"], dtype=np.float64)
    graphs = tuple(spatial_graph(coordinates, value) for value in (4, 8, 18))

    rows = []
    for index, graph in enumerate(graphs):
        old_data = np.asarray(payload[f"graph{index}__data"])
        if not np.array_equal(payload[f"graph{index}__indices"], graph.indices):
            raise ValueError(f"graph {index} indices changed")
        if not np.array_equal(payload[f"graph{index}__indptr"], graph.indptr):
            raise ValueError(f"graph {index} indptr changed")
        payload[f"graph{index}__data"] = np.asarray(graph.data, dtype=np.float64)
        rows.append(
            {
                "scale_index": index,
                "nnz": int(graph.nnz),
                "old_dtype": str(old_data.dtype),
                "new_dtype": str(graph.data.dtype),
                "old_sha256": array_sha256(old_data),
                "new_sha256": array_sha256(graph.data),
                "max_abs_float64_minus_float32": float(
                    np.max(np.abs(graph.data - old_data.astype(np.float64)))
                ),
                "changed_after_float32_roundtrip": int(
                    np.sum(graph.data != old_data.astype(np.float64))
                ),
            }
        )

    output = Path(args.output_carrier)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    audit = {
        "schema": "night16f-human-graph-precision-diagnostic-v1",
        "input_carrier": str(Path(args.input_carrier).resolve()),
        "output_carrier": str(output.resolve()),
        "ordered_id_sha256": ordered_id_sha256(ids),
        "all_non_graph_arrays_preserved": True,
        "rows": rows,
    }
    Path(args.audit).write_text(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
