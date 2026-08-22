#!/usr/bin/env python3
"""Stream one registered Night-12A unit and write a label-free schema record."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from SpaLORA.night12a_schema_p0 import (
    atomic_json,
    inspect_csv_matrix,
    read_coordinates,
    scan_fragments,
    text_sha256,
)


def compact_matrix(value: dict) -> dict:
    return {k: v for k, v in value.items()
            if k not in {"ordered_observation_ids", "feature_ids"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--other", type=Path)
    parser.add_argument("--other-kind", choices=["ADT", "ATAC_fragments"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    coordinate = read_coordinates(args.coordinates)
    rna = inspect_csv_matrix(args.rna, coordinate["ordered_ids"])
    record = {
        "unit_id": args.unit_id,
        "coordinate": {k: v for k, v in coordinate.items()
                       if k not in {"ordered_ids", "records"}},
        "rna": compact_matrix(rna),
        "rna_feature_ids": rna["feature_ids"],
        "rna_ordered_observation_ids": rna["ordered_observation_ids"],
        "label_reads": 0,
        "metric_calls": 0,
    }
    if args.other_kind == "ADT":
        other = inspect_csv_matrix(args.other, coordinate["ordered_ids"])
        if other["ordered_observation_ids"] != rna["ordered_observation_ids"]:
            raise ValueError("RNA and ADT ordered IDs differ")
        record["adt"] = compact_matrix(other)
        record["adt_feature_ids"] = other["feature_ids"]
        record["paired_spot_identity"] = "byte-exact after frozen terminal -1 transform"
    elif args.other_kind == "ATAC_fragments":
        fragment = scan_fragments(args.other, rna["ordered_observation_ids"], [])
        record["fragments"] = {k: v for k, v in fragment.items()
                               if k not in {"counts", "registered_fragment_depth"}}
        record["paired_spot_identity"] = "byte-exact after frozen terminal -1 transform"
    record["wall_seconds"] = time.monotonic() - start
    atomic_json(args.output, record)
    print(json.dumps({
        "unit_id": args.unit_id,
        "rna_shape": rna["observation_by_feature_shape"],
        "other_shape": (record.get("adt", {}).get("observation_by_feature_shape")
                        or [record.get("fragments", {}).get("fragment_rows")]),
        "ordered_id_sha256": text_sha256(rna["ordered_observation_ids"]),
        "wall_seconds": record["wall_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
