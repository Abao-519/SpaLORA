#!/usr/bin/env python
"""Independent evaluator opens candidate metrics only after selection is locked."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = {row["candidate_id"]: row for row in csv.DictReader(args.evaluation.open(encoding="utf-8"))}
    metric = rows[str(selection["candidate_id"])]
    expected_partition_sha = str(metric["partition_sha256"])
    if str(selection["candidate_sha256"]) != expected_partition_sha:
        raise ValueError("selection candidate SHA differs from evaluation partition authority")
    result = {
        **selection,
        **{key: metric[key] for key in (
            "absolute_ari", "absolute_nmi", "ami", "fmi", "homogeneity", "v_measure",
            "morans_i_macro", "gearys_c_macro", "neighbor_agreement", "min_cluster_size_full",
            "min_cluster_size_eval", "cluster_sizes_full", "cluster_sizes_eval", "partition_sha256",
        )},
        "independent_evaluator_label_reads": 1,
        "evaluation_sha256": hashlib.sha256(args.evaluation.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
