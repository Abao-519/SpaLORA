#!/usr/bin/env python3
"""Join locked Night-16G candidate hashes with Night-16F evaluator rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    features = list(csv.DictReader(Path(args.features).open()))
    evaluation = {row["candidate_id"]: row for row in csv.DictReader(Path(args.evaluation).open())}
    rows = []
    for row in features:
        metric = evaluation[row["candidate_id"]]
        rows.append({**row, "absolute_ari": metric["absolute_ari"], "absolute_nmi": metric["absolute_nmi"]})
    by_id = {row["candidate_id"]: row for row in rows}
    selections = {
        "BASIN_PARETO": manifest["selected"]["candidate_id"],
        **{name.upper(): value["candidate_id"] for name, value in manifest["controls"].items()},
    }
    oracle = max(rows, key=lambda row: (float(row["absolute_ari"]), float(row["absolute_nmi"])))
    selections["LABEL_ASSISTED_ORACLE"] = oracle["candidate_id"]
    summary = []
    for name, candidate_id in selections.items():
        row = by_id[candidate_id]
        summary.append({
            "selector": name,
            "candidate_id": candidate_id,
            "absolute_ari": float(row["absolute_ari"]),
            "absolute_nmi": float(row["absolute_nmi"]),
            "candidate_sha256": row["candidate_sha256"],
        })
    Path(args.output).write_text(json.dumps({"lane": manifest["lane"], "selections": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
