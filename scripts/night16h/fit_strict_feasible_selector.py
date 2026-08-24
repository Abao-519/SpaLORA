#!/usr/bin/env python3
"""Fit Night-16H rank weights using training studies only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night16g_basin_selector import candidate_similarity, plain_medoid_index
from SpaLORA.night16h_feasible_selector import select_weighted_rank


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def parse(value: str) -> tuple[str, Path, Path, Path]:
    fields = value.split("::")
    if len(fields) != 4:
        raise ValueError("training-lane must be LANE::FEATURES::BANK::EVALUATION")
    return fields[0], Path(fields[1]), Path(fields[2]), Path(fields[3])


def run(args: argparse.Namespace) -> None:
    specs = [parse(value) for value in args.training_lane]
    names = [value[0] for value in specs]
    if args.heldout_lane in names or len(names) != len(set(names)):
        raise ValueError("held-out lane leaked into training or duplicate lane")
    data = {}
    hashes = {}
    opened = []
    for lane, feature_path, bank_path, evaluation_path in specs:
        rows = list(csv.DictReader(feature_path.open(encoding="utf-8")))
        evaluation = {row["candidate_id"]: row for row in csv.DictReader(evaluation_path.open(encoding="utf-8"))}
        with np.load(bank_path, allow_pickle=False) as archive:
            partitions = np.asarray(archive["partitions"], dtype=np.int32)
            candidate_ids = archive["candidate_ids"].astype("U").tolist()
        if candidate_ids != [row["candidate_id"] for row in rows] or set(candidate_ids) != set(evaluation):
            raise ValueError(f"locked candidate/evaluation mismatch in {lane}")
        feasible = np.asarray([truth(row[f"feasible_{args.feasibility_mode}"]) for row in rows])
        indices = np.flatnonzero(feasible)
        similarity = candidate_similarity(partitions)
        medoid_local = plain_medoid_index(
            [rows[i] for i in indices], similarity[np.ix_(indices, indices)]
        )
        medoid_id = candidate_ids[int(indices[medoid_local])]
        data[lane] = {
            "rows": rows,
            "partitions": partitions,
            "feasible": feasible,
            "evaluation": evaluation,
            "medoid": (
                float(evaluation[medoid_id]["absolute_ari"]),
                float(evaluation[medoid_id]["absolute_nmi"]),
            ),
        }
        opened.append(str(evaluation_path.resolve()))
        for path in (feature_path, bank_path, evaluation_path):
            hashes[str(path.resolve())] = sha256(path)
    values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    ledger = []
    for config_index, weights in enumerate(itertools.product(values, repeat=3)):
        if sum(weights) == 0:
            continue
        picks = {}
        for lane in names:
            datum = data[lane]
            result = select_weighted_rank(datum["rows"], datum["feasible"], *weights)
            candidate_id = datum["rows"][result.selected_index]["candidate_id"]
            metric = datum["evaluation"][candidate_id]
            ari, nmi = float(metric["absolute_ari"]), float(metric["absolute_nmi"])
            medoid_ari, medoid_nmi = datum["medoid"]
            picks[lane] = {
                "candidate_id": candidate_id,
                "absolute_ari": ari,
                "absolute_nmi": nmi,
                "delta_medoid_ari": ari - medoid_ari,
                "delta_medoid_nmi": nmi - medoid_nmi,
                "partition_sha256": metric["partition_sha256"],
            }
        values_pick = list(picks.values())
        meaningful = sum(
            row["delta_medoid_ari"] > args.meaningful_gain
            and row["delta_medoid_nmi"] > args.meaningful_gain
            for row in values_pick
        )
        noninferior = sum(
            row["delta_medoid_ari"] >= -args.noninferiority_tolerance
            and row["delta_medoid_nmi"] >= -args.noninferiority_tolerance
            for row in values_pick
        )
        mean_ari = float(np.mean([row["delta_medoid_ari"] for row in values_pick]))
        mean_nmi = float(np.mean([row["delta_medoid_nmi"] for row in values_pick]))
        worst_joint = float(min(min(row["delta_medoid_ari"], row["delta_medoid_nmi"]) for row in values_pick))
        complexity = sum(value > 0 for value in weights)
        magnitude = sum(weights)
        rank = (meaningful, noninferior, mean_ari, mean_nmi, worst_joint, -complexity, -magnitude, -config_index)
        ledger.append({
            "config_index": config_index,
            "molecular_weight": weights[0],
            "topology_weight": weights[1],
            "risk_weight": weights[2],
            "meaningful_gain_count": meaningful,
            "noninferior_count": noninferior,
            "mean_delta_ari": mean_ari,
            "mean_delta_nmi": mean_nmi,
            "worst_joint_delta": worst_joint,
            "ranking_tuple": rank,
            "picks": picks,
        })
    winner = max(ledger, key=lambda row: row["ranking_tuple"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid_path = output.with_suffix(".grid.csv")
    flat = []
    for row in ledger:
        out = {key: value for key, value in row.items() if key not in ("ranking_tuple", "picks")}
        for lane, pick in row["picks"].items():
            for key, value in pick.items():
                out[f"{lane}__{key}"] = value
        flat.append(out)
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(flat)
    result = {
        "schema": "night16h-strict-training-only-feasible-selector-fit-v1",
        "heldout_lane": args.heldout_lane,
        "training_lanes": names,
        "formal_feasibility_mode": args.feasibility_mode,
        "selector": "FROZEN_WEIGHTED_RANK",
        "selector_weights": {
            "molecular_weight": winner["molecular_weight"],
            "topology_weight": winner["topology_weight"],
            "risk_weight": winner["risk_weight"],
        },
        "training_picks": winner["picks"],
        "selection_rule": "meaningful_gain_count_then_noninferior_count_then_mean_delta_ari_then_mean_delta_nmi_then_worst_joint_then_lower_complexity",
        "candidate_grid_size": len(ledger),
        "candidate_partitions_locked_before_metric_join": True,
        "opened_training_evaluation_paths": opened,
        "heldout_evaluation_opened": False,
        "input_sha256": hashes,
        "grid_ledger": str(grid_path.resolve()),
        "grid_ledger_sha256": sha256(grid_path),
        "producer_label_reads": 0,
        "thread_limit": 1,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-lane", action="append", required=True)
    parser.add_argument("--heldout-lane", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feasibility-mode", default="SMALLEST_SCALE_INTERNAL_EDGE")
    parser.add_argument("--meaningful-gain", type=float, default=0.005)
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.01)
    with threadpool_limits(1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
