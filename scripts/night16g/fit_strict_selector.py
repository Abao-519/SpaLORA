#!/usr/bin/env python3
"""Fit a Night-16G selector without opening the held-out study evaluation.

Each --training-lane value is:
LANE::FEATURES_CSV::PARTITION_BANK_NPZ::EVALUATION_CSV
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    plain_medoid_index,
    select_evidence_rank,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_spec(value: str) -> tuple[str, Path, Path, Path]:
    fields = value.split("::")
    if len(fields) != 4:
        raise ValueError("training-lane must have four :: separated fields")
    return fields[0], Path(fields[1]), Path(fields[2]), Path(fields[3])


def complexity(config: BasinSelectorConfig) -> tuple[float, ...]:
    weights = (
        config.molecular_weight,
        config.topology_weight,
        config.persistence_weight,
        config.risk_weight,
    )
    return sum(value > 0 for value in weights), sum(weights)


def run(args: argparse.Namespace) -> None:
    lane_specs = [parse_spec(value) for value in args.training_lane]
    lane_names = [value[0] for value in lane_specs]
    if args.heldout_lane in lane_names:
        raise ValueError("held-out lane must not be a training lane")
    if len(set(lane_names)) != len(lane_names) or not lane_names:
        raise ValueError("training lane names must be nonempty and unique")

    data: dict[str, dict[str, object]] = {}
    opened_evaluation_paths: list[str] = []
    input_hashes: dict[str, str] = {}
    for lane, feature_path, partition_path, evaluation_path in lane_specs:
        rows = list(csv.DictReader(feature_path.open(encoding="utf-8")))
        with np.load(partition_path, allow_pickle=False) as archive:
            partitions = np.asarray(archive["partitions"], dtype=np.int32)
            candidate_ids = np.asarray(archive["candidate_ids"]).astype("U")
        if len(rows) != len(partitions) or not np.array_equal(
            candidate_ids, np.asarray([row["candidate_id"] for row in rows])
        ):
            raise ValueError(f"locked candidate mismatch for {lane}")
        metrics = {
            row["candidate_id"]: (
                float(row["absolute_ari"]),
                float(row["absolute_nmi"]),
            )
            for row in csv.DictReader(evaluation_path.open(encoding="utf-8"))
            if row.get("status") == "PASS"
        }
        if set(candidate_ids) - set(metrics):
            raise ValueError(f"evaluation rows missing for {lane}")
        similarity = candidate_similarity(partitions)
        medoid_index = plain_medoid_index(rows, similarity)
        _, enriched, _ = select_evidence_rank(
            rows,
            partitions,
            BasinSelectorConfig(
                basin_ari_threshold=args.basin_threshold,
                persistent_ari_threshold=args.persistence_threshold,
                molecular_weight=1.0,
                topology_weight=1.0,
                persistence_weight=1.0,
                risk_weight=1.0,
                representative_evidence_weight=0.0,
                ordered_path_weight=args.ordered_path_weight,
            ),
            similarity=similarity,
        )
        data[lane] = {
            "rows": rows,
            "partitions": partitions,
            "similarity": similarity,
            "metrics": metrics,
            "plain_medoid": metrics[rows[medoid_index]["candidate_id"]],
            "axis_matrix": np.asarray(
                [
                    [
                        float(row["molecular_rank"]),
                        float(row["topology_rank"]),
                        float(row["persistence_rank"]),
                        float(row["risk_rank"]),
                    ]
                    for row in enriched
                ],
                dtype=np.float64,
            ),
        }
        opened_evaluation_paths.append(str(evaluation_path.resolve()))
        for path in (feature_path, partition_path, evaluation_path):
            input_hashes[str(path.resolve())] = sha256(path)

    values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    configs: list[BasinSelectorConfig] = []
    for molecular, topology, persistence, risk in itertools.product(values, repeat=4):
        if molecular + topology + persistence + risk == 0:
            continue
        configs.append(
            BasinSelectorConfig(
                basin_ari_threshold=args.basin_threshold,
                persistent_ari_threshold=args.persistence_threshold,
                molecular_weight=molecular,
                topology_weight=topology,
                persistence_weight=persistence,
                risk_weight=risk,
                representative_evidence_weight=0.0,
                ordered_path_weight=args.ordered_path_weight,
            )
        )

    ledger: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for config_index, config in enumerate(configs):
        picks: dict[str, dict[str, object]] = {}
        for lane in lane_names:
            lane_data = data[lane]
            weights = np.asarray(
                [
                    config.molecular_weight,
                    config.topology_weight,
                    config.persistence_weight,
                    config.risk_weight,
                ],
                dtype=np.float64,
            )
            scores = lane_data["axis_matrix"] @ weights / float(weights.sum())
            selected = min(
                range(len(scores)),
                key=lambda index: (
                    -float(scores[index]),
                    str(lane_data["rows"][index]["candidate_id"]),
                ),
            )
            candidate_id = lane_data["rows"][selected]["candidate_id"]
            ari, nmi = lane_data["metrics"][candidate_id]
            medoid_ari, medoid_nmi = lane_data["plain_medoid"]
            picks[lane] = {
                "candidate_id": candidate_id,
                "absolute_ari": ari,
                "absolute_nmi": nmi,
                "delta_medoid_ari": ari - medoid_ari,
                "delta_medoid_nmi": nmi - medoid_nmi,
                "partition_sha256": lane_data["rows"][selected]["candidate_sha256"],
            }
        picked = list(picks.values())
        meaningful = sum(
            row["delta_medoid_ari"] > args.meaningful_gain
            and row["delta_medoid_nmi"] > args.meaningful_gain
            for row in picked
        )
        noninferior = sum(
            row["delta_medoid_ari"] >= -args.noninferiority_tolerance
            and row["delta_medoid_nmi"] >= -args.noninferiority_tolerance
            for row in picked
        )
        mean_ari = float(np.mean([row["delta_medoid_ari"] for row in picked]))
        mean_nmi = float(np.mean([row["delta_medoid_nmi"] for row in picked]))
        worst_joint = float(
            min(min(row["delta_medoid_ari"], row["delta_medoid_nmi"]) for row in picked)
        )
        rank = (
            meaningful,
            noninferior,
            mean_ari,
            mean_nmi,
            worst_joint,
            tuple(-value for value in complexity(config)),
            -config_index,
        )
        candidates.append(
            {
                "config_index": config_index,
                "selector_config": config.__dict__,
                "picks": picks,
                "ranking_tuple": rank,
            }
        )
        row: dict[str, object] = {
            "config_index": config_index,
            **config.__dict__,
            "meaningful_gain_count": meaningful,
            "noninferior_count": noninferior,
            "mean_delta_ari": mean_ari,
            "mean_delta_nmi": mean_nmi,
            "worst_joint_delta": worst_joint,
        }
        for lane, pick in picks.items():
            for key, value in pick.items():
                row[f"{lane}__{key}"] = value
        ledger.append(row)

    winner = max(candidates, key=lambda row: row["ranking_tuple"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = output.with_suffix(".grid.csv")
    columns = sorted({key for row in ledger for key in row})
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ledger)
    result = {
        "schema": "night16g-strict-training-only-selector-fit-v1",
        "heldout_lane": args.heldout_lane,
        "training_lanes": lane_names,
        "selector_config": winner["selector_config"],
        "config_index": winner["config_index"],
        "training_picks": winner["picks"],
        "selection_rule": (
            "meaningful_gain_count_then_noninferior_count_then_mean_delta_ari_"
            "then_mean_delta_nmi_then_worst_joint_delta_then_lower_complexity"
        ),
        "meaningful_gain": args.meaningful_gain,
        "noninferiority_tolerance": args.noninferiority_tolerance,
        "candidate_grid_size": len(configs),
        "candidate_partitions_locked_before_metric_join": True,
        "opened_training_evaluation_paths": opened_evaluation_paths,
        "heldout_evaluation_opened": False,
        "input_sha256": input_hashes,
        "grid_ledger": str(ledger_path.resolve()),
        "grid_ledger_sha256": sha256(ledger_path),
        "thread_limit": 1,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-lane", action="append", required=True)
    parser.add_argument("--heldout-lane", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basin-threshold", type=float, default=0.82)
    parser.add_argument("--persistence-threshold", type=float, default=0.90)
    parser.add_argument("--meaningful-gain", type=float, default=0.005)
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.01)
    parser.add_argument("--ordered-path-weight", type=float, default=0.75)
    with threadpool_limits(limits=1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
