#!/usr/bin/env python3
"""Compare two final fresh-process replays with the frozen registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--replay1", type=Path, required=True)
    parser.add_argument("--replay2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    producers = [
        json.loads((path / "producer_replay.json").read_text(encoding="utf-8"))
        for path in (args.replay1, args.replay2)
    ]
    metrics = [pd.read_csv(path / "metrics.csv") for path in (args.replay1, args.replay2)]
    rows = []
    for lane, record in registry["lanes"].items():
        producer_rows = [next(item for item in payload["lanes"] if item["lane"] == lane) for payload in producers]
        metric_rows = [frame[frame.lane == lane].iloc[0] for frame in metrics]
        partition_exact = (
            producer_rows[0]["partition_sha256"]
            == producer_rows[1]["partition_sha256"]
            == record["expected_partition_sha256"]
        )
        metric_exact = all(
            float(metric_rows[0][key]) == float(metric_rows[1][key])
            for key in ("absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c")
        )
        rows.append(
            {
                "lane": lane,
                "partition_exact": partition_exact,
                "metrics_exact": metric_exact,
                "cluster_sizes_exact": metric_rows[0].cluster_sizes == metric_rows[1].cluster_sizes,
                "expected_partition_sha256": record["expected_partition_sha256"],
                "absolute_ari": float(metric_rows[0].absolute_ari),
                "absolute_nmi": float(metric_rows[0].absolute_nmi),
            }
        )
    status = all(row["partition_exact"] and row["metrics_exact"] and row["cluster_sizes_exact"] for row in rows)
    payload = {
        "status": "PASS" if status else "FAILED",
        "fresh_process_replays": 2,
        "lanes_exact": sum(row["partition_exact"] and row["metrics_exact"] for row in rows),
        "lanes_expected": len(rows),
        "producer_label_reads": [item["labels_opened"] for item in producers],
        "dense_n_by_n_counts": [item["dense_n_by_n_count"] for item in producers],
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
