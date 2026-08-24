#!/usr/bin/env python3
"""Evaluate already locked Night-16A partitions in a separate process."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive
from scripts.night15g.night15g_optional_view_search import evaluate


LANE_DATASET = {
    "A1": "A1", "D1": "D1", "tonsil_s1": "tonsil_s1", "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3", "P22": "P22", "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1", "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
}

CURRENT_BEST = {
    "A1": (0.2760026589984753, 0.42173983941218224),
    "D1": (0.28887600833487515, 0.41376160498731795),
    "tonsil_s1": (0.23622005623915696, 0.3168843080420388),
    "tonsil_s2": (0.2575602926237894, 0.31187103018718415),
    "tonsil_s3": (0.349880567011789, 0.3092665192529602),
    "P22": (0.5939121542899773, 0.7142428513607495),
    "P22_3DOT_K18": (0.7396852743147485, 0.7542182135683022),
    "MISAR_E15_5_S1": (0.5414237853091904, 0.6667977615565593),
    "MISAR_E15_5_S1_K12": (0.45317629302756757, 0.5986289047196983),
}


def semantics(data, lane):
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"]), np.ones(len(data["ids"]), bool)
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)
    return int(data["k_primary"][0]), np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((args.frozen / "frozen_crossfit_registry.json").read_text(encoding="utf-8"))
    rows = []
    for lane, locked in registry["lanes"].items():
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        partition = np.load(args.frozen / "partitions" / f"{lane}.npy", allow_pickle=False)
        k, labels, mask = semantics(data, lane)
        graph = csr_from_archive(data, "graph")
        metric = evaluate(labels, mask, partition, graph)
        sizes = np.bincount(partition, minlength=k)
        old = CURRENT_BEST[lane]
        rows.append({
            "lane": lane,
            "held_out_study": locked["held_out_study"],
            "family": locked["family"],
            "total_observations": len(partition),
            "evaluated_observations": int(mask.sum()),
            "k": k,
            "absolute_ari": metric["absolute_ari"],
            "absolute_nmi": metric["absolute_nmi"],
            "ami": metric["ami"],
            "fmi": metric["fmi"],
            "morans_i": metric["morans_i"],
            "gearys_c": metric["gearys_c"],
            "delta_ari_vs_current_best": metric["absolute_ari"] - old[0],
            "delta_nmi_vs_current_best": metric["absolute_nmi"] - old[1],
            "min_cluster_size": int(sizes.min()),
            "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
            "partition_sha256": locked["partition_sha256"],
            "candidate_key": locked["candidate_key"],
            "parameter_source": locked["parameter_source"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
