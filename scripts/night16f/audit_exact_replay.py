#!/usr/bin/env python3
"""Audit fresh-process exactness for frozen Night-16F producer artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from SpaLORA.night16e_tsre import partition_sha256


def load(prefix: Path) -> tuple[dict[str, object], np.ndarray]:
    manifest = json.loads(prefix.with_suffix(".producer.json").read_text())
    with np.load(prefix, allow_pickle=False) as archive:
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
    return manifest, partitions


def run(args: argparse.Namespace) -> None:
    rows = []
    for specification in args.pair:
        lane, left_text, right_text = specification.split("::", 2)
        left, left_bank = load(Path(left_text))
        right, right_bank = load(Path(right_text))
        if left["frozen_config_sha256"] != right["frozen_config_sha256"]:
            raise RuntimeError(f"{lane}: frozen config differs")
        left_rows = {row["candidate_id"]: row for row in left["rows"]}
        right_rows = {row["candidate_id"]: row for row in right["rows"]}
        if set(left_rows) != set(right_rows):
            raise RuntimeError(f"{lane}: candidate sets differ")
        mismatches = []
        for identifier in sorted(left_rows):
            first = left_rows[identifier]
            second = right_rows[identifier]
            if first["status"] != second["status"]:
                mismatches.append(f"{identifier}:status")
                continue
            if first["status"] == "PASS":
                first_partition = left_bank[int(first["partition_index"])]
                second_partition = right_bank[int(second["partition_index"])]
                first_sha = partition_sha256(first_partition)
                second_sha = partition_sha256(second_partition)
                if not (
                    first_sha == second_sha == first["partition_sha256"] == second["partition_sha256"]
                ):
                    mismatches.append(f"{identifier}:partition")
                if first["cluster_sizes_full"] != second["cluster_sizes_full"]:
                    mismatches.append(f"{identifier}:cluster_sizes")
        rows.append(
            {
                "lane": lane,
                "candidate_count": len(left_rows),
                "partition_exact_count": len(left_rows) - len(mismatches),
                "mismatches": mismatches,
                "status": "PASS" if not mismatches else "FAILED",
            }
        )
    result = {
        "schema": "night16f-fresh-process-replay-audit-v1",
        "rows": rows,
        "passed_lanes": sum(row["status"] == "PASS" for row in rows),
        "total_lanes": len(rows),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAILED",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, help="LANE::LEFT_NPZ::RIGHT_NPZ")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
