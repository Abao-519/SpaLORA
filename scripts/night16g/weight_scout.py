#!/usr/bin/env python3
"""Transparent label-assisted scout of a compact global selector grid."""

from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

from SpaLORA.night16g_basin_selector import (
    BasinSelectorConfig,
    candidate_similarity,
    select_basin,
)


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7", "MELANOMA_TUMOR_K2")


def run() -> None:
    root = Path("/root/night16g_working/scout")
    data = {}
    for lane in LANES:
        rows = list(csv.DictReader((root / f"{lane}.features.csv").open()))
        with np.load(root / f"{lane}.npz", allow_pickle=False) as archive:
            partitions = np.asarray(archive["partitions"], dtype=np.int32)
        evaluation = (
            Path("/root/night16f_working")
            / ("formal" if lane.startswith("MELANOMA") else "development")
            / lane
            / "evaluation.csv"
        )
        metrics = {
            row["candidate_id"]: (float(row["absolute_ari"]), float(row["absolute_nmi"]))
            for row in csv.DictReader(evaluation.open())
        }
        data[lane] = (rows, partitions, candidate_similarity(partitions), metrics)

    thresholds = (0.65, 0.72, 0.78, 0.82, 0.86, 0.90)
    weights = (
        (0.0, 4.0, 1.0, 1.0),
        (0.25, 2.0, 0.5, 0.5),
        (0.5, 4.0, 1.0, 1.0),
        (1.0, 4.0, 1.0, 1.0),
        (1.0, 3.0, 1.0, 1.0),
        (1.0, 2.0, 1.0, 1.0),
        (0.5, 2.0, 0.5, 1.0),
    )
    representatives = (0.35, 0.75, 1.5, 3.0)
    results = []
    for threshold, weight, representative in itertools.product(
        thresholds, weights, representatives
    ):
        picks = []
        for lane in LANES:
            rows, partitions, similarity, metrics = data[lane]
            config = BasinSelectorConfig(
                basin_ari_threshold=threshold,
                persistent_ari_threshold=max(threshold, 0.90),
                molecular_weight=weight[0],
                topology_weight=weight[1],
                persistence_weight=weight[2],
                risk_weight=weight[3],
                representative_evidence_weight=representative,
            )
            index, _, diagnostics = select_basin(
                rows, partitions, config, similarity=similarity
            )
            ari, nmi = metrics[rows[index]["candidate_id"]]
            picks.append(
                {
                    "lane": lane,
                    "candidate_id": rows[index]["candidate_id"],
                    "absolute_ari": ari,
                    "absolute_nmi": nmi,
                    "winning_basin_id": diagnostics["winning_basin_id"],
                }
            )
        results.append(
            {
                "threshold": threshold,
                "weights": list(weight),
                "representative_weight": representative,
                "mean_ari": float(np.mean([row["absolute_ari"] for row in picks])),
                "mean_nmi": float(np.mean([row["absolute_nmi"] for row in picks])),
                "picks": picks,
            }
        )
    results.sort(key=lambda row: (row["mean_ari"], row["mean_nmi"]), reverse=True)
    (root / "weight_scout.json").write_text(
        json.dumps(
            {
                "schema": "night16g-label-assisted-global-weight-scout-v1",
                "candidate_partitions_locked_before_metrics": True,
                "grid_size": len(results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
