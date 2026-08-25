#!/usr/bin/env python3
"""Sample real edges and compare vectorized CEUP utility with scalar LOO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

from SpaLORA.night17a_ceup import canonical_undirected_graph, scalar_edge_utility_recompute, standardize
from scripts.night17a.night17a_producer import load_numeric_carrier, read_config, strict_reload_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_config(Path(args.config))
    numeric = load_numeric_carrier(Path(args.carrier), str(config["graph_prefix"]))
    view1 = standardize(numeric["view1"])[0]
    view2 = standardize(numeric["view2"])[0]
    edge_i, edge_j, edge_w, graph = canonical_undirected_graph(numeric["graph"])
    _, models, metadata = strict_reload_checkpoint(Path(args.checkpoint), view1.shape[1], view2.shape[1], args.device)
    with np.load(args.artifact, allow_pickle=False) as artifact:
        raw = np.asarray(artifact["utility_raw_four"], dtype=np.float64)
        node_folds = np.asarray(artifact["node_folds"], dtype=np.int32)
    sample_edges = np.unique(np.linspace(0, edge_i.size - 1, num=min(8, edge_i.size), dtype=np.int64))
    comparisons = []
    specs = (
        ("cross_v1_from_v2", view1, view2, 0),
        ("cross_v2_from_v1", view2, view1, 2),
    )
    max_abs = 0.0
    for model_name, target, source, channel_offset in specs:
        for edge in sample_edges.tolist():
            for orientation, (receiver, sender) in enumerate(
                ((int(edge_i[edge]), int(edge_j[edge])), (int(edge_j[edge]), int(edge_i[edge])))
            ):
                fold = int(node_folds[receiver])
                model = models[model_name][fold]
                for mask_index, held in enumerate(metadata[model_name][fold]["masks"]):
                    scalar = scalar_edge_utility_recompute(
                        model, held, target, source, graph, receiver, sender, float(edge_w[edge])
                    )
                    vector = float(raw[edge, channel_offset + orientation, mask_index])
                    difference = abs(scalar - vector)
                    max_abs = max(max_abs, difference)
                    comparisons.append(
                        {
                            "edge": edge,
                            "orientation": orientation,
                            "modality_direction": model_name,
                            "fold": fold,
                            "mask": mask_index,
                            "scalar": scalar,
                            "vector": vector,
                            "abs_difference": difference,
                        }
                    )
    status = "PASS" if max_abs <= 2e-5 else "FAIL"
    record = {
        "schema": "night17a-real-fixed-degree-loo-audit-v1",
        "lane": args.lane,
        "status": status,
        "sampled_edges": sample_edges.tolist(),
        "comparisons": len(comparisons),
        "max_abs_difference": max_abs,
        "tolerance": 2e-5,
        "fixed_full_degree_denominator": True,
        "renormalization_control_used": False,
        "details": comparisons,
    }
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(f"real scalar/vector LOO mismatch: {max_abs}")
    print(json.dumps({"lane": args.lane, "status": status, "max_abs_difference": max_abs}, sort_keys=True))


if __name__ == "__main__":
    main()

