#!/usr/bin/env python3
"""Apply a frozen label-free selector to one locked candidate bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

from SpaLORA.night16g_basin_selector import (
    BasinSelectorConfig,
    candidate_similarity,
    partition_sha256,
    select_evidence_rank,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    feature_path = Path(args.features)
    bank_path = Path(args.partition_bank)
    config_path = Path(args.config)
    rows = list(csv.DictReader(feature_path.open(encoding="utf-8")))
    with np.load(bank_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype("U")
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
        candidate_ids = np.asarray(archive["candidate_ids"]).astype("U")
    if len(rows) != len(partitions) or not np.array_equal(
        candidate_ids, np.asarray([row["candidate_id"] for row in rows])
    ):
        raise ValueError("locked feature/partition candidate mismatch")
    if any(len(np.unique(partition)) != args.k for partition in partitions):
        raise ValueError("candidate bank violates exact K")
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    config = BasinSelectorConfig(**frozen["selector_config"])
    selected, enriched, diagnostics = select_evidence_rank(
        rows, partitions, config, similarity=candidate_similarity(partitions)
    )
    partition = np.asarray(partitions[selected], dtype=np.int32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=ids, partitions=partition[None, :])
    evidence_path = output.with_suffix(".evidence.csv")
    columns = sorted({key for row in enriched for key in row})
    with evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(enriched)
    selected_row = rows[selected]
    manifest = {
        "schema": "night16g-frozen-label-free-selector-v1",
        "lane": args.lane,
        "k": int(args.k),
        "n": int(len(ids)),
        "selector_config": config.__dict__,
        "selector_config_path": str(config_path.resolve()),
        "selector_config_sha256": sha256(config_path),
        "candidate_bank_sha256": sha256(bank_path),
        "candidate_features_sha256": sha256(feature_path),
        "candidate_count": int(len(rows)),
        "selected_candidate_id": selected_row["candidate_id"],
        "selected_source_partition_sha256": partition_sha256(partition),
        "selected_partition_sha256": partition_sha256(partition),
        "diagnostics": diagnostics,
        "producer_label_reads": 0,
        "dense_observation_by_observation_count": 0,
        "thread_limit": 1,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "rows": [
            {
                "candidate_id": selected_row["candidate_id"],
                "start_id": selected_row.get("start_id", ""),
                "start_index": 0,
                "start_role": "LOCKED_CANDIDATE_BANK",
                "arm": selected_row.get("arm", ""),
                "status": "PASS",
                "failure": "",
                "partition_sha256": partition_sha256(partition),
                "initial_partition_sha256": partition_sha256(partition),
                "changed_from_initial": 0,
                "partition_index": 0,
                "wall_seconds": float(time.perf_counter() - started),
            }
        ],
    }
    producer_path = output.with_suffix(".producer.json")
    producer_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["ids"].astype("U"), ids) or not np.array_equal(
            replay["partitions"][0].astype(np.int32), partition
        ):
            raise RuntimeError("atomic artifact reload mismatch")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
