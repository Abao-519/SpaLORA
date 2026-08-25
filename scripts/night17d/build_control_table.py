#!/usr/bin/env python
"""Evaluate preregistered strong selector controls after feature freeze."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night16g_basin_selector import partition_sha256  # noqa: E402
from SpaLORA.night17d_learned_evidence import SelectorWeights, select_candidate  # noqa: E402


def choose_max(rows, key):
    feasible = [row for row in rows if str(row["feasible_SMALLEST_SCALE_INTERNAL_EDGE"]).lower() == "true"]
    return max(feasible, key=lambda row: (float(row[key]), str(row["candidate_id"])))


def medoid(rows, partitions, candidate_ids):
    feasible_ids = {
        row["candidate_id"] for row in rows if str(row["feasible_SMALLEST_SCALE_INTERNAL_EDGE"]).lower() == "true"
    }
    indices = [index for index, value in enumerate(candidate_ids) if str(value) in feasible_ids]
    similarity = np.eye(len(indices), dtype=np.float64)
    for a in range(len(indices)):
        for b in range(a):
            value = adjusted_rand_score(partitions[indices[a]], partitions[indices[b]])
            similarity[a, b] = similarity[b, a] = value
    scores = similarity.mean(axis=1)
    best = max(range(len(indices)), key=lambda p: (float(scores[p]), str(candidate_ids[indices[p]])))
    return str(candidate_ids[indices[best]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--fixed-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.features.open(encoding="utf-8")))
    evaluation = {row["candidate_id"]: row for row in csv.DictReader(args.evaluation.open(encoding="utf-8"))}
    authority = next(row for row in csv.DictReader(args.authority.open(encoding="utf-8")) if row["lane"] == args.lane)
    config = json.loads(args.fixed_config.read_text(encoding="utf-8"))["fixed_global"]
    weights = SelectorWeights(**{key: float(config[key]) for key in ("molecular", "topology", "learned", "uncertainty")})
    with np.load(args.bank, allow_pickle=False) as archive:
        partitions = np.asarray(archive["partitions"])
        candidate_ids = np.asarray(archive["candidate_ids"])
    feature_ids = {str(row["candidate_id"]) for row in rows}
    evaluation_ids = set(evaluation)
    bank_ids = {str(value) for value in candidate_ids}
    if feature_ids != evaluation_ids or feature_ids != bank_ids:
        raise ValueError("feature/evaluation/candidate-bank candidate sets differ")
    feature_map = {str(row["candidate_id"]): row for row in rows}
    for index, value in enumerate(candidate_ids):
        candidate_id = str(value)
        bank_sha = partition_sha256(np.asarray(partitions[index]))
        if str(feature_map[candidate_id]["candidate_sha256"]) != bank_sha:
            raise ValueError(f"feature partition SHA mismatch for {candidate_id}")
        if str(evaluation[candidate_id]["partition_sha256"]) != bank_sha:
            raise ValueError(f"evaluation partition SHA mismatch for {candidate_id}")
    selections = {
        "NIGHT16H_FIXED_SELECTOR": str(authority["candidate_id"]),
        "MAX_TOPOLOGY": str(choose_max(rows, "topology_rank")["candidate_id"]),
        "MAX_ORIGINAL_MOLECULAR": str(choose_max(rows, "molecular_rank")["candidate_id"]),
        "MAX_LEARNED": str(choose_max(rows, "LEARNED_evidence")["candidate_id"]),
        "FEASIBLE_MEDOID": medoid(rows, partitions, candidate_ids),
        "FIXED_LEARNED": str(select_candidate(rows, weights, "LEARNED")["candidate_id"]),
        "FIXED_ZERO": str(select_candidate(rows, weights, "ZERO")["candidate_id"]),
        "FIXED_PERMUTED": str(select_candidate(rows, weights, "PERMUTED")["candidate_id"]),
    }
    output = []
    for method, candidate_id in selections.items():
        metric = evaluation[candidate_id]
        output.append(
            {
                "lane": args.lane,
                "method": method,
                "candidate_id": candidate_id,
                "absolute_ari": metric["absolute_ari"],
                "absolute_nmi": metric["absolute_nmi"],
                "ami": metric["ami"],
                "fmi": metric["fmi"],
                "morans_i_macro": metric["morans_i_macro"],
                "gearys_c_macro": metric["gearys_c_macro"],
                "min_cluster_size_full": metric["min_cluster_size_full"],
                "cluster_sizes_full": metric["cluster_sizes_full"],
                "partition_sha256": metric["partition_sha256"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(json.dumps({"lane": args.lane, "methods": len(output)}, indent=2))


if __name__ == "__main__":
    main()
