#!/usr/bin/env python3
"""Independent metric join for one already locked Night-16H partition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-manifest", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    producer_path = Path(args.producer_manifest)
    evaluation_path = Path(args.evaluation)
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(evaluation_path.open(encoding="utf-8")))
    matches = [row for row in rows if row["candidate_id"] == producer["selected_candidate_id"]]
    if len(matches) != 1:
        raise ValueError("selected candidate missing or duplicated in evaluator")
    row = matches[0]
    if row["partition_sha256"] != producer["selected_partition_sha256"]:
        raise ValueError("producer/evaluator partition SHA mismatch")
    result = {
        **row,
        "lane": producer["lane"],
        "selector": producer["selector"],
        "feasibility_mode": producer["feasibility_mode"],
        "feasible_candidate_count": producer["feasible_candidate_count"],
        "producer_manifest_sha256": sha256(producer_path),
        "evaluation_sha256": sha256(evaluation_path),
        "independent_evaluator_label_reads": 1,
        "producer_label_reads": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
