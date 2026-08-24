#!/usr/bin/env python3
"""Merge label-free partition banks while preserving provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run(args: argparse.Namespace) -> None:
    all_partitions: list[np.ndarray] = []
    all_rows: list[dict[str, object]] = []
    authority_ids: np.ndarray | None = None
    seen: set[str] = set()
    sources: list[str] = []
    for spec in args.bank:
        bank_text, producer_text = spec.split("::", 1)
        bank_path, producer_path = Path(bank_text), Path(producer_text)
        with np.load(bank_path, allow_pickle=False) as archive:
            ids = np.asarray(archive["ids"]).astype("U")
            partitions = np.asarray(archive["partitions"], dtype=np.int32)
        if authority_ids is None:
            authority_ids = ids
        elif not np.array_equal(authority_ids, ids):
            raise ValueError("bank ordered-ID mismatch")
        producer = json.loads(producer_path.read_text(encoding="utf-8"))
        passed = sorted(
            (row for row in producer["rows"] if row["status"] == "PASS"),
            key=lambda row: int(row["partition_index"]),
        )
        if len(passed) != len(partitions):
            raise ValueError("producer/partition count mismatch")
        for row, partition in zip(passed, partitions):
            candidate_id = str(row["candidate_id"])
            if candidate_id in seen:
                raise ValueError(f"duplicate candidate ID: {candidate_id}")
            seen.add(candidate_id)
            updated = dict(row)
            updated["partition_index"] = len(all_partitions)
            all_rows.append(updated)
            all_partitions.append(partition)
        sources.extend((str(bank_path.resolve()), str(producer_path.resolve())))
    if authority_ids is None or not all_partitions:
        raise ValueError("no candidate bank supplied")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=authority_ids, partitions=np.stack(all_partitions))
    manifest = {
        "schema": "night16g-merged-locked-candidate-bank-v1",
        "lane": args.lane,
        "k": int(args.k),
        "n": int(len(authority_ids)),
        "candidate_count": len(all_rows),
        "candidate_ids_unique": True,
        "sources": sources,
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "rows": all_rows,
    }
    output.with_suffix(".producer.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
