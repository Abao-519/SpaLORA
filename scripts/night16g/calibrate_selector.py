#!/usr/bin/env python3
"""Transparent family-global and leave-one-study-out selector calibration."""

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
    select_evidence_rank,
)


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7", "MELANOMA_TUMOR_K2")


def complexity(config: BasinSelectorConfig) -> tuple[float, ...]:
    return (
        sum(value > 0 for value in (
            config.molecular_weight,
            config.topology_weight,
            config.persistence_weight,
            config.risk_weight,
        )),
        config.molecular_weight
        + config.topology_weight
        + config.persistence_weight
        + config.risk_weight,
        config.basin_ari_threshold,
    )


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
        similarity = candidate_similarity(partitions)
        plain_medoid = int(
            np.argmax((similarity.sum(axis=1) - 1.0) / max(len(partitions) - 1, 1))
        )
        data[lane] = {
            "rows": rows,
            "partitions": partitions,
            "similarity": similarity,
            "metrics": metrics,
            "plain_medoid": metrics[rows[plain_medoid]["candidate_id"]],
        }

    values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    configs = []
    for molecular, topology, persistence, risk in itertools.product(values, repeat=4):
        if molecular + topology + persistence + risk == 0:
            continue
        configs.append(
            BasinSelectorConfig(
                basin_ari_threshold=0.82,
                persistent_ari_threshold=0.90,
                molecular_weight=molecular,
                topology_weight=topology,
                persistence_weight=persistence,
                risk_weight=risk,
                representative_evidence_weight=0.0,
            )
        )

    results = []
    for config_index, config in enumerate(configs):
        picks = {}
        for lane in LANES:
            lane_data = data[lane]
            index, _, diagnostics = select_evidence_rank(
                lane_data["rows"],
                lane_data["partitions"],
                config,
                similarity=lane_data["similarity"],
            )
            candidate_id = lane_data["rows"][index]["candidate_id"]
            ari, nmi = lane_data["metrics"][candidate_id]
            medoid_ari, medoid_nmi = lane_data["plain_medoid"]
            picks[lane] = {
                "candidate_id": candidate_id,
                "absolute_ari": ari,
                "absolute_nmi": nmi,
                "delta_medoid_ari": ari - medoid_ari,
                "delta_medoid_nmi": nmi - medoid_nmi,
                "partition_sha256": diagnostics["selected_partition_sha256"],
            }
        results.append(
            {
                "config_index": config_index,
                "config": config.__dict__,
                "picks": picks,
            }
        )

    def ranking(row: dict[str, object], training: tuple[str, ...]) -> tuple[object, ...]:
        picked = [row["picks"][lane] for lane in training]
        return (
            sum(value["delta_medoid_ari"] > 0 and value["delta_medoid_nmi"] > 0 for value in picked),
            min(value["delta_medoid_ari"] for value in picked),
            float(np.mean([value["delta_medoid_ari"] for value in picked])),
            float(np.mean([value["delta_medoid_nmi"] for value in picked])),
            tuple(-value for value in complexity(BasinSelectorConfig(**row["config"]))),
            -int(row["config_index"]),
        )

    global_best = max(results, key=lambda row: ranking(row, LANES))
    loso = {}
    for heldout in LANES:
        training = tuple(lane for lane in LANES if lane != heldout)
        best = max(results, key=lambda row: ranking(row, training))
        loso[heldout] = {
            "training_lanes": list(training),
            "config_index": best["config_index"],
            "config": best["config"],
            "heldout_pick": best["picks"][heldout],
            "training_picks": {lane: best["picks"][lane] for lane in training},
        }
    output = {
        "schema": "night16g-label-assisted-selector-calibration-v1",
        "candidate_partitions_locked_before_metric_join": True,
        "selection_rule": "dual-win-count_then_worst-delta-ari_then_mean-delta-ari_then-mean-delta-nmi_then-lower-complexity",
        "grid_size": len(configs),
        "global": global_best,
        "loso": loso,
        "all_results": results,
    }
    (root / "selector_calibration.json").write_text(
        json.dumps(output, indent=2, sort_keys=True)
    )
    print(json.dumps({"global": global_best, "loso": loso}, indent=2, sort_keys=True))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
