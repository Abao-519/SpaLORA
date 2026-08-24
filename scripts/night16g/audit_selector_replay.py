#!/usr/bin/env python3
"""Audit two independent frozen-selector process outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from SpaLORA.night16g_basin_selector import partition_sha256


def run(args: argparse.Namespace) -> None:
    rows = []
    for spec in args.pair:
        lane, left_text, right_text = spec.split("::", 2)
        left_path, right_path = Path(left_text), Path(right_text)
        with np.load(left_path, allow_pickle=False) as left, np.load(
            right_path, allow_pickle=False
        ) as right:
            ids_equal = np.array_equal(left["ids"], right["ids"])
            partitions_equal = np.array_equal(left["partitions"], right["partitions"])
            left_partition = np.asarray(left["partitions"][0], dtype=np.int32)
            right_partition = np.asarray(right["partitions"][0], dtype=np.int32)
        left_manifest = json.loads(left_path.with_suffix(".producer.json").read_text())
        right_manifest = json.loads(right_path.with_suffix(".producer.json").read_text())
        row = {
            "lane": lane,
            "ids_exact": bool(ids_equal),
            "partitions_exact": bool(partitions_equal),
            "left_partition_sha256": partition_sha256(left_partition),
            "right_partition_sha256": partition_sha256(right_partition),
            "candidate_id_exact": left_manifest["selected_candidate_id"]
            == right_manifest["selected_candidate_id"],
            "selector_config_sha256_exact": left_manifest["selector_config_sha256"]
            == right_manifest["selector_config_sha256"],
            "producer_label_reads": int(left_manifest["producer_label_reads"])
            + int(right_manifest["producer_label_reads"]),
        }
        row["status"] = "PASS" if all(
            row[key]
            for key in (
                "ids_exact",
                "partitions_exact",
                "candidate_id_exact",
                "selector_config_sha256_exact",
            )
        ) and row["producer_label_reads"] == 0 else "FAIL"
        rows.append(row)
    output = {
        "schema": "night16g-fresh-process-selector-replay-audit-v1",
        "rows": rows,
        "passed": sum(row["status"] == "PASS" for row in rows),
        "total": len(rows),
        "dense_n_by_n_count": 0,
    }
    if output["passed"] != output["total"]:
        raise RuntimeError("selector replay mismatch")
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
