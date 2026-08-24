#!/usr/bin/env python3
"""Audit the Night-16E/Night-16F human DIRECT_BASE discrepancy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from scripts.night16f.build_numeric_carrier import spatial_graph


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def manifest_row(
    path: Path, key: str, value: str, *, primary_start: bool = False
) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    matches = [
        row
        for row in manifest["rows"]
        if str(row[key]) == value
        and (not primary_start or int(row.get("start_index", 0)) == 0)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {key}={value} in {path}")
    return matches[0]


def carrier_summary(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: {
                "dtype": str(archive[name].dtype),
                "shape": list(archive[name].shape),
                "sha256": array_sha(archive[name]),
            }
            for name in ("view1", "view2", "retained")
        }
        graphs = []
        for index in range(3):
            data = archive[f"graph{index}__data"]
            indices = archive[f"graph{index}__indices"]
            indptr = archive[f"graph{index}__indptr"]
            graphs.append(
                {
                    "scale_index": index,
                    "data_dtype": str(data.dtype),
                    "data_sha256": array_sha(data),
                    "indices_sha256": array_sha(indices),
                    "indptr_sha256": array_sha(indptr),
                    "nnz": int(len(data)),
                }
            )
    return {"path": str(path.resolve()), "arrays": arrays, "graphs": graphs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--rna", required=True)
    parser.add_argument("--parent-producer", required=True)
    parser.add_argument("--superseded-carrier", required=True)
    parser.add_argument("--superseded-producer", required=True)
    parser.add_argument("--graph64-diagnostic-producer", required=True)
    parser.add_argument("--final-carrier", required=True)
    parser.add_argument("--final-producer", required=True)
    parser.add_argument("--parent-thread1-rerun-producer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registry = json.loads(Path(args.registry).read_text())
    by_variant = {row["variant"]: row for row in registry["candidates"]}
    base_shas = {
        variant: canonical_sha(by_variant[variant]["config"]["base"])
        for variant in ("NIGHT15F_DIRECT", "TSRE_FULL", "SUPPORT_MODULATION_ONLY")
    }
    if len(set(base_shas.values())) != 1:
        raise RuntimeError("Night-16E direct/full/support base configs differ")

    parent_input = manifest_row(Path(args.parent_producer), "variant", "INPUT_STRONG_START")
    parent_direct = manifest_row(Path(args.parent_producer), "variant", "NIGHT15F_DIRECT")
    old_input = manifest_row(
        Path(args.superseded_producer), "arm", "INPUT_START", primary_start=True
    )
    old_direct = manifest_row(
        Path(args.superseded_producer), "arm", "DIRECT_BASE", primary_start=True
    )
    graph64_direct = manifest_row(
        Path(args.graph64_diagnostic_producer), "arm", "DIRECT_BASE", primary_start=True
    )
    final_input = manifest_row(
        Path(args.final_producer), "arm", "INPUT_START", primary_start=True
    )
    final_direct = manifest_row(
        Path(args.final_producer), "arm", "DIRECT_BASE", primary_start=True
    )
    thread1_direct = manifest_row(
        Path(args.parent_thread1_rerun_producer), "variant", "NIGHT15F_DIRECT"
    )

    old_carrier = carrier_summary(Path(args.superseded_carrier))
    final_carrier = carrier_summary(Path(args.final_carrier))
    with np.load(args.superseded_carrier, allow_pickle=False) as old, np.load(
        args.final_carrier, allow_pickle=False
    ) as final:
        array_deltas = {}
        for name in ("view1", "view2", "retained"):
            left = np.asarray(old[name], dtype=np.float64)
            right = np.asarray(final[name], dtype=np.float64)
            array_deltas[name] = {
                "max_abs": float(np.max(np.abs(left - right))),
                "rms": float(np.sqrt(np.mean(np.square(left - right)))),
                "byte_exact": bool(np.array_equal(old[name], final[name])),
            }

    rna = ad.read_h5ad(args.rna)
    coordinates = np.asarray(rna.obsm["spatial"], dtype=np.float64)
    parent_graphs = tuple(spatial_graph(coordinates, value) for value in (4, 8, 18))
    recomputed_graphs = [
        {
            "scale_index": index,
            "data_dtype": str(graph.data.dtype),
            "data_sha256": array_sha(graph.data),
            "indices_sha256": array_sha(graph.indices),
            "indptr_sha256": array_sha(graph.indptr),
            "nnz": int(graph.nnz),
        }
        for index, graph in enumerate(parent_graphs)
    ]
    topology_exact = all(
        old_carrier["graphs"][i]["indices_sha256"]
        == final_carrier["graphs"][i]["indices_sha256"]
        == recomputed_graphs[i]["indices_sha256"]
        and old_carrier["graphs"][i]["indptr_sha256"]
        == final_carrier["graphs"][i]["indptr_sha256"]
        == recomputed_graphs[i]["indptr_sha256"]
        for i in range(3)
    )
    start_exact = (
        parent_input["partition_sha256"]
        == old_input["partition_sha256"]
        == final_input["partition_sha256"]
    )
    direct_restored = (
        parent_direct["partition_sha256"]
        == thread1_direct["partition_sha256"]
        == final_direct["partition_sha256"]
    )
    graph_precision_not_causal = (
        graph64_direct["partition_sha256"] == old_direct["partition_sha256"]
    )
    if not (topology_exact and start_exact and direct_restored and graph_precision_not_causal):
        raise RuntimeError("human DIRECT_BASE reproduction audit did not close")

    audit = {
        "schema": "night16f-human-direct-reproduction-audit-v1",
        "status": "PASS",
        "root_cause": (
            "Night-16F omitted the Night-16E final-replay OMP/MKL/OPENBLAS=1 "
            "boundary during carrier reduction and producer evidence preparation"
        ),
        "engineering_correction": (
            "pin all three BLAS thread controls to 1 before numeric imports and "
            "enforce threadpool_limits(1) during carrier and producer computation"
        ),
        "base_config_sha256": base_shas,
        "same_start_byte_exact": start_exact,
        "same_graph_topology_exact": topology_exact,
        "parent_recomputed_graphs": recomputed_graphs,
        "superseded_carrier": old_carrier,
        "final_carrier": final_carrier,
        "continuous_array_deltas_superseded_vs_final": array_deltas,
        "direct_partition_comparison": {
            "parent_night16e": {
                "sha256": parent_direct["partition_sha256"],
                "changed": parent_direct["changed_from_unlabeled_start"],
                "mean_retained_weight": parent_direct["diagnostics"]["mean_retained_weight"],
                "mean_view1_weight": parent_direct["diagnostics"]["mean_view1_weight"],
                "mean_view2_weight": parent_direct["diagnostics"]["mean_view2_weight"],
            },
            "night16f_unpinned_superseded": {
                "sha256": old_direct["partition_sha256"],
                "changed": old_direct["changed_from_initial"],
                "mean_retained_weight": old_direct["diagnostics"]["mean_retained_weight"],
                "mean_view1_weight": old_direct["diagnostics"]["mean_view1_weight"],
                "mean_view2_weight": old_direct["diagnostics"]["mean_view2_weight"],
            },
            "night16f_unpinned_with_float64_graph_only": {
                "sha256": graph64_direct["partition_sha256"],
                "changed": graph64_direct["changed_from_initial"],
                "mean_retained_weight": graph64_direct["diagnostics"]["mean_retained_weight"],
                "mean_view1_weight": graph64_direct["diagnostics"]["mean_view1_weight"],
                "mean_view2_weight": graph64_direct["diagnostics"]["mean_view2_weight"],
            },
            "parent_source_rerun_thread1": {
                "sha256": thread1_direct["partition_sha256"],
                "changed": thread1_direct["changed_from_unlabeled_start"],
                "mean_retained_weight": thread1_direct["diagnostics"]["mean_retained_weight"],
                "mean_view1_weight": thread1_direct["diagnostics"]["mean_view1_weight"],
                "mean_view2_weight": thread1_direct["diagnostics"]["mean_view2_weight"],
            },
            "night16f_corrected_thread1": {
                "sha256": final_direct["partition_sha256"],
                "changed": final_direct["changed_from_initial"],
                "mean_retained_weight": final_direct["diagnostics"]["mean_retained_weight"],
                "mean_view1_weight": final_direct["diagnostics"]["mean_view1_weight"],
                "mean_view2_weight": final_direct["diagnostics"]["mean_view2_weight"],
            },
        },
        "graph_float32_hypothesis": "FALSIFIED",
        "graph_precision_not_causal": graph_precision_not_causal,
        "direct_partition_restored_byte_exact": direct_restored,
        "superseded_scope": (
            "all four first-pass carriers, producer outputs, evaluations and replays"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
