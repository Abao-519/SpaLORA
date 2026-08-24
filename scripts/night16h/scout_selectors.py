#!/usr/bin/env python3
"""Label-assisted scout over locked Night-16H feasibility features."""

from __future__ import annotations

import argparse
import csv
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
from SpaLORA.night16h_feasible_selector import FEASIBILITY_MODES, select_candidate


METHODS = (
    "PARETO_MAXIMIN",
    "EQUAL_RANK",
    "CONTENT_ADAPTIVE",
    "PARETO_CENTRAL",
    "CROSS_EVIDENCE_ARBITRATION",
)
CONTROLS = (
    "FEASIBLE_PLAIN_MEDOID",
    "FEASIBLE_MIN_INERTIA",
    "FEASIBLE_MAX_CH",
    "FEASIBLE_MAX_SPATIAL",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return str(value).strip().lower() == "true"


def run(args: argparse.Namespace) -> None:
    rows = []
    for spec in args.lane:
        lane, feature_text, bank_text, evaluation_text = spec.split("::", 3)
        records = read_csv(Path(feature_text))
        evaluation = read_csv(Path(evaluation_text))
        by_id = {row["candidate_id"]: row for row in evaluation}
        with np.load(bank_text, allow_pickle=False) as locked:
            partitions = np.asarray(locked["partitions"], dtype=np.int32)
            candidate_ids = locked["candidate_ids"].astype("U").tolist()
        if candidate_ids != [row["candidate_id"] for row in records] or set(candidate_ids) != set(by_id):
            raise ValueError(f"candidate identity mismatch for {lane}")
        similarity = candidate_similarity(partitions)
        for mode in FEASIBILITY_MODES:
            feasible = np.asarray([truth(row[f"feasible_{mode}"]) for row in records])
            indices = np.flatnonzero(feasible)
            if len(indices) == 0:
                for method in CONTROLS + METHODS:
                    rows.append(
                        {
                            "lane": lane,
                            "feasibility_mode": mode,
                            "selector": method,
                            "candidate_id": "",
                            "absolute_ari": "",
                            "absolute_nmi": "",
                            "ami": "",
                            "fmi": "",
                            "min_cluster_size_full": "",
                            "cluster_sizes_full": "",
                            "morans_i_macro": "",
                            "gearys_c_macro": "",
                            "feasible_count": 0,
                            "status": "NO_FEASIBLE_CANDIDATE",
                        }
                    )
                continue
            controls = {
                "FEASIBLE_PLAIN_MEDOID": int(indices[plain_medoid_index([records[i] for i in indices], similarity[np.ix_(indices, indices)])]),
                "FEASIBLE_MIN_INERTIA": min(indices, key=lambda i: (float(records[i]["retained_within_per_observation"]), records[i]["candidate_id"])),
                "FEASIBLE_MAX_CH": max(indices, key=lambda i: (float(records[i]["retained_ch"]), records[i]["candidate_id"])),
                "FEASIBLE_MAX_SPATIAL": max(indices, key=lambda i: (float(records[i]["topology_joint"]), records[i]["candidate_id"])),
            }
            for method in METHODS:
                result = select_candidate(records, partitions, feasible, similarity, method)
                controls[method] = result.selected_index
            for method, index in controls.items():
                metric = by_id[candidate_ids[index]]
                rows.append(
                    {
                        "lane": lane,
                        "feasibility_mode": mode,
                        "selector": method,
                        "candidate_id": candidate_ids[index],
                        "absolute_ari": metric["absolute_ari"],
                        "absolute_nmi": metric["absolute_nmi"],
                        "ami": metric["ami"],
                        "fmi": metric["fmi"],
                        "min_cluster_size_full": metric["min_cluster_size_full"],
                        "cluster_sizes_full": metric["cluster_sizes_full"],
                        "morans_i_macro": metric["morans_i_macro"],
                        "gearys_c_macro": metric["gearys_c_macro"],
                        "feasible_count": int(feasible.sum()),
                        "status": "PASS",
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="append", required=True)
    parser.add_argument("--output", required=True)
    with threadpool_limits(1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
